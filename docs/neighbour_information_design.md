# How our drones know about each other

*October 2026. Design note. Status: Stage 1 of the plan (sensing radius, input scaling) and the neighbour source layer (section 7) are built, in `modules/observation_processor.py` and `modules/neighbour_source.py`. Stages 2 and 3 are still to do.*

While planning the deconfliction training run, a question came up that we hadn't really thought through: if every drone makes its own decisions, how does it actually know where the other drones are? In the simulator it's easy, because PyBullet knows every position. On real drones that information has to come from somewhere.

This note is what we worked out. It covers where that information would come from on real drones, what our model currently assumes, what's wrong with that today, and the plan for fixing it in stages, from PyBullet now to Webots later. It's meant to get everyone on the same page before we touch the code.

---

## 1. The short version

- **We are not simulating communication right now.** No radios, no Remote ID packets, no MAVLink, no UTM network. For our current scope that's the right call.
- **Our model already assumes the drones can share information**, because every step each drone is handed its neighbours' positions and velocities. So the question isn't *whether* to add communication. It's whether the information we hand the drone is realistic.
- **Today it isn't, in three ways.** Every drone sees its 5 nearest neighbours no matter how far away they are. The values aren't scaled. And the information is perfect: no delay, no noise, nothing lost.
- **The plan has three stages:**
  1. Now, in PyBullet: add a **sensing radius** (drones further than R are invisible) and **scale the neighbour inputs**. This goes into the deconfliction run.
  2. Before Webots, in PyBullet, evaluation only: check whether the trained model still avoids other drones when the neighbour information is **delayed, noisy or missing**.
  3. In Webots: drones **actually broadcast** their state to each other using Webots' built-in radio devices, and each drone decides only from what it has received.
- **One design rule ties it together:** the policy never cares where neighbour information comes from. A small "neighbour source" layer supplies it, and only that layer changes between stages.

---

## 2. Where this information comes from on real drones

On real drones there are four ways one drone can learn about another. None of these need to be built for our project. They're here so we know what our simulator is standing in for.

| Source | How it works | How fast | Good for |
|---|---|---|---|
| **Remote ID** (standard ASTM F3411) | Every drone broadcasts its ID, position, altitude, direction and speed over Bluetooth or Wi-Fi. Anything nearby can listen. Already legally required in the US and EU | about 1 message per second | identification. It's the closest real thing to our neighbour input, but slow |
| **MAVLink over a radio or mesh network** | The standard messaging protocol for PX4 / ArduPilot drones. Drones send each other position and velocity telemetry over a shared radio | 10+ messages per second | real swarms. This is what drone swarms typically use |
| **UTM network** | Drones report their position to a traffic management service over cellular or internet, and the service shares nearby traffic back | varies | traffic management. The drones don't talk directly, the network relays |
| **Onboard sensors** (LiDAR, radar, cameras) | The drone detects other drones itself | sensor rate | backup. Works even for drones that don't broadcast, but can't easily tell a drone from a building and doesn't measure velocity directly |

ADS-B, which manned aircraft use, is out. In the US small drones aren't allowed to transmit it, because millions of drones would flood the frequencies.

The important point for us: what our model expects as input (relative position and velocity of nearby drones) is exactly what a drone can build from Remote ID or MAVLink broadcasts. So our design maps onto real hardware without changing the model.

---

## 3. What our model actually sees today

Each drone's observation has two parts:

```
own state        27 numbers   position, velocity, orientation, direction and
                              distance to goal, 14 LiDAR rays (all zero for now)
                              -> scaled by the running normalizer

neighbour state  5 x 6        for each of the 5 nearest drones:
                              relative position (3) + relative velocity (3)
                              -> NOT scaled, raw metres and m/s
                 + a mask     marks empty slots when there are fewer than 5 neighbours
```

This is built in `modules/observation_processor.py`, in `_build_neighbor_states()`.

The actor reads the neighbour part through a small **attention module**. It scores each neighbour, combines them into one fixed-size summary (32 numbers), and ignores masked slots. That's why the same network works with 0, 1 or 5 neighbours, and why it didn't break when we tested it with 1 to 10 drones. K = 5 is just the maximum number of slots (`k_neighbors_max` in the config).

A reminder from the evaluation: the trained model currently ignores this whole neighbour part (blind and normal versions behave identically). That's bad for avoidance, but it has one upside: we can redesign the neighbour inputs now at no cost, because nothing has been learned from them yet.

---

## 4. What's wrong with it today

### 4.1 No range limit

The code always picks the 5 nearest drones, however far away they are. In our current tests that's harmless, because all drones are within a 20 x 20 m area. But picture a realistic case: 10 drones spread over a 1 km radius. The 5 "nearest" might be 300-800 m away. Each of them still fills a slot, and the drone is told to pay attention to them.

At our top speed (2 m/s), two drones flying straight at each other close at about 4 m/s. A drone 800 m away is more than 3 minutes from reaching us, and the policy makes a new decision 31 times a second. Those drones are irrelevant to collision avoidance, and in the real world we wouldn't even receive most of their broadcasts.

### 4.2 Unscaled inputs

Neighbour positions go into the network as raw metres. During training neighbours were always within a few metres, so the network has only ever seen values like 2.5 or -4.0. A drone 800 m away would show up as 800. The network has never seen anything like it, and there's no telling what it would do. (The drone's own state doesn't have this problem, because the normalizer scales it and clips extreme values.)

### 4.3 Perfect information

Every step, each drone gets the exact, current state of its neighbours. Real broadcasts arrive once a second (Remote ID) or ten times a second (MAVLink), carry GPS error of around a metre, and sometimes don't arrive at all. A model that only ever trained on perfect information might fall apart when it gets realistic information. Right now we don't know either way.

---

## 5. How it should work

On a real drone, every control step would look like this:

```
1. RECEIVE   collect the latest broadcast from every drone in radio range
             (could be 10, could be 100, up to ~1 km away)
                      │
2. FILTER    keep only drones within the sensing radius R
             (with many drones, H3 makes this fast, see below)
                      │
3. RANK      order what's left by how dangerous each one is
             (for now: by distance; later: by time to closest approach)
                      │
4. TOP K     put the K most important ones into the neighbour slots,
             mask the rest
                      │
5. DECIDE    the policy outputs a velocity command
```

**Why only a few neighbours are enough.** You rarely need to react to more than a handful of drones at the same moment. If 10 drones are within 1 km, maybe one or two are close enough to matter. If more than 5 drones are within a few metres of each other, the airspace is badly managed, and preventing that is the UTM layer's job, not the individual drone's.

**Where H3 fits.** In a simulation with hundreds or thousands of drones, step 2 is the expensive part: comparing every drone against every other drone is about 500,000 distance checks per step for 1,000 drones. H3 splits the map into hexagonal cells. If we pick a cell size around R, each drone only has to check drones in its own cell and the ring of cells around it, which turns the filter into a quick lookup. This is the job H3 does in our system. On real drones the radio range already does a rough version of this filtering for free.

---

## 6. The plan, in three stages

### Stage 1: sensing radius (PyBullet, now, part of the deconfliction run)

**What changes**
- Each drone only sees drones within **R metres**. Anything further away is masked, exactly as if we never received its broadcast.
- Neighbour inputs are **scaled to a fixed range**: relative position divided by R (so every visible neighbour lands between -1 and 1), and relative velocity divided by a fixed speed scale (for example 4 m/s, the fastest two drones can close on each other).
- Still perfect information inside the radius. This stage is only about *who* the drone can see, not how good the information is.

**Choosing R.** It should give a drone enough warning to react. Two drones closing at 4 m/s with, say, 5-8 seconds of warning works out to roughly 20-30 m. We'd start with **R = 30 m** and make it a config value. One caveat for our current test scenarios: they all fit inside a 20 x 20 m area, so with R = 30 m almost everyone is still visible. The radius only really starts doing something in larger worlds, which is fine. We want it in place before we scale up, not after.

**Code involved**
- `modules/observation_processor.py`: apply the radius and the scaling in `_build_neighbor_states()`.
- `configs/`: new options `sensing_radius` and the velocity scale.
- `eval_suite.py`: no changes needed, the evaluation runs the same way.

**How we know it worked:** the evaluation suite still passes on the normal task, and after the deconfliction run the blanking and interaction tests show the model actually using its neighbours.

### Stage 2: robustness check (PyBullet, evaluation only, before Webots)

**What changes:** only the information handed to the trained model during evaluation. Training is untouched. We degrade the neighbour information in a few controlled ways:

| Setting | Values to try | What it imitates |
|---|---|---|
| Update rate | every step (31 Hz), 10 Hz, 1 Hz | perfect, MAVLink-style, Remote ID-style |
| Position noise | 0, 0.1, 0.5, 1.0 m | GPS error |
| Messages lost | 0%, 10%, 30% | radio packets that never arrive |

Between updates, the drone keeps using the last message it got for each neighbour. An optional improvement is to move that old position forward using the neighbour's last known velocity ("dead reckoning"), which is what real systems do.

One thing worth keeping in mind: real GPS error (around a metre) is larger than our collision distance (0.2 m) and our drone size (0.3 m). Under realistic noise, drones would need to keep separation margins of metres, not tens of centimetres. This test will show us how big that problem is.

**What we do with the result**
- The model still avoids collisions → go to Webots as is.
- The model breaks down → a short retrain with mild degradation switched on during training, then check again.

**Why do this in PyBullet first:** if the model breaks for the first time in Webots, we'd be debugging the Webots setup, the drone physics and the communication all at once, and we wouldn't know which one is the cause. Checking here takes about a day and isolates the question.

### Stage 3: real broadcasting (Webots)

Webots has built-in **Emitter** and **Receiver** devices that simulate radio communication between robots. This lets us make the drones genuinely decentralized: each drone only knows what it has actually received.

**How each drone would work**

```
every step:
  read own GPS + IMU
  every N steps:  Emitter.send( drone_id, time, position, velocity )
  Receiver:       read everything in the queue,
                  keep the latest message from each drone (+ how old it is)
  -> filter by range -> top K -> policy -> velocity command
```

**What Webots gives us**
- **Range:** the emitter's `range` field means a receiver only gets messages from drones within that distance. That's our sensing radius, handled by the simulator.
- **Channels:** we put every drone on one shared broadcast channel, like Remote ID. Multiple channels don't add anything for us.
- **Raw bytes:** messages are just bytes, so we can send our own simple format, or real encoded Remote ID / MAVLink messages if someone has time later.
- **Extras:** the receiver can also report signal strength and the direction the signal came from. We don't need these, but they're there.

**What Webots doesn't give us.** The Webots docs say it doesn't model packet loss, delays or noise. Messages always arrive, one step later. So we add those ourselves in the drone controller, using the same settings as Stage 2: send only every N steps, add noise to the position before sending, and randomly drop some messages.

**Update rates to run:** 10 Hz (MAVLink-style, the normal case) and 1 Hz (Remote ID-style, the hard case).

**How this connects to the existing Webots plan:** in Level 2 of the Webots transfer plan, each drone already runs its own controller (for the propeller PID). The broadcasting fits into that same controller.

---

## 7. The one design rule: a neighbour source layer

The policy should never know or care where neighbour information comes from. Between the simulator and the observation code, we put one small layer whose only job is to answer: *"what does this drone currently know about other drones?"*

```
                  Stage 1:  PyBullet's true positions, within radius R
                  Stage 2:  true positions, degraded (rate, noise, loss)
 neighbour  ─────┤
 source           Stage 3:  latest Webots Receiver message from each drone
                        │
                        │   for each known drone: position, velocity, age
                        ▼
      same code from here on:  filter by range -> rank -> top K -> policy
```

This keeps the stages cheap:
- Moving from Stage 1 to 2 means adding degradation inside the source. Nothing downstream changes.
- Moving to Webots means writing one new source that reads the Receiver. The filtering, ranking, top-K selection, policy and evaluation code all stay the same.
- It also makes testing easy, because we can feed any source into the same evaluation suite and compare.

We'd build this layer in Stage 1 already, even though at that point it just passes the true positions through.

---

## 8. Later improvements (after basic avoidance works)

These all make the system more realistic, but each one changes the training, so they should come one at a time, after the deconfliction run shows real avoidance:

- **Rank by threat, not distance.** The nearest drone isn't always the most dangerous one. A drone 20 m away flying straight at us matters more than one 5 m away flying off. We can compute when and how close each neighbour will pass (the "closest point of approach"), using its relative position *p* and relative velocity *v*:
  - time of closest approach: *t = -(p · v) / |v|²* (or 0 if it's moving away)
  - distance at that moment: *|p + v·t|*

  Rank by that distance and time. This is the same idea aircraft collision avoidance systems use.
- **Vary the number of drones during training,** for example 2 to 10 per episode, so the model sees crowded and empty airspace and not always exactly 3 drones.
- **Train with degraded information,** only if Stage 2 shows we need it.
- **LiDAR as a fallback.** Once obstacles and real LiDAR exist (Stage 2 of the curriculum), other drones will also show up in the LiDAR rays. That gives a backup for drones that don't broadcast.

---

## 9. What we decided not to do

- **Simulate the radio or network itself** (signal propagation, interference, network protocols, tools like ns-3). That's a separate research field and far outside our scope.
- **Use multiple radio channels.** One shared channel is enough.
- **Encode real Remote ID / MAVLink messages.** Optional, only if someone has spare time in the Webots phase. It doesn't change any results.

---

## 10. Open questions

Things we still need to settle as a team:
- **Is R = 30 m right?** It depends on top speed and how much warning we want. Easy to change, since it'll be a config value.
- **Is K = 5 enough?** It has been so far. If dense swap tests later show drones crowding beyond 5 visible neighbours, we can raise it. The attention module doesn't care how many slots there are.
- **How stale is too stale?** At 1 Hz, information can be up to a second old, which at our speeds means a neighbour may have moved up to 2 m since its last message. Stage 2 will tell us whether that's a problem.
- **How to handle GPS-scale noise** when it's larger than the drones themselves. This may push us towards larger safety margins in the reward (for example a bigger danger radius for the proximity penalty).
