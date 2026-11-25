/* track_logic.js: what dashboard/track.html computes, with no DOM, so tests/test_tracking.py can run it under node.
   One implementation for the live view (scripts/utm_server.py's /api/state) and the replays (data/<scene>.json,
   GET /api/track): both are turned into the same "view" (position, route, status at a time) first.

   Times are simulation seconds. A run's clock time is read off its `clock` knots ([t, wall epoch], a live run's record
   of when each sim second really happened); without them it is start_epoch + t / speed (sim seconds per wall second). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TrackLogic = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const ARRIVING_S = 20;            // this close (s) to landing: "Arriving"
  const ON_TIME_S = 5;              // the same threshold as flight_log.html: within it is on time, beyond it "running late"
  const RESET_S = 30;               // an ETA jump bigger than this (a re-plan) is taken as it is, not smoothed
  const TAU_S = 4;                  // smoothing time constant (s) of the arrival estimate

  // E1's defaults (utm/strategic_planner.py Settings): the last fallback when a scene carries no speeds
  const DEFAULT_SPEEDS = { v_cruise: 1.5, v_climb: 0.5, v_descent: 0.5, v_descent_high: null, slow_band_m: 3.0, start_z: 1.0 };

  const STEPS = ['Booked', 'Approved', 'Taken off', 'On the way', 'Arriving', 'Delivered'];

  /** t (sim s) -> wall clock (epoch s) for a run {start_epoch, speed, clock}: interpolates the knots, and beyond them
      (or without any) goes at `speed`. */
  function clockFn(run) {
    const k = run.clock || [], sp = run.speed || 1;
    if (!k.length) return (t) => run.start_epoch + t / sp;
    return (t) => {
      if (t <= k[0][0]) return k[0][1] - (k[0][0] - t) / sp;
      const n = k.length - 1;
      if (t >= k[n][0]) return k[n][1] + (t - k[n][0]) / sp;
      let lo = 0, hi = n;                                                   // k[lo][0] <= t < k[hi][0]
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (k[m][0] <= t) lo = m; else hi = m; }
      return k[lo][1] + (k[hi][1] - k[lo][1]) * (t - k[lo][0]) / (k[hi][0] - k[lo][0]);
    };
  }

  /** {speeds, approximate}: the scenario's flight profile, or E1's defaults (approximate: true). */
  function resolveSpeeds(s) {
    const ok = s && typeof s.v_cruise === 'number' && typeof s.v_climb === 'number';
    if (!ok) return { speeds: { ...DEFAULT_SPEEDS }, approximate: true };
    const sp = { ...DEFAULT_SPEEDS, ...s };
    if (sp.v_descent == null) sp.v_descent = sp.v_climb;
    return { speeds: sp, approximate: false };
  }

  // ---------------------------------------------------------------- the route and what is left of it
  const hyp = (...v) => Math.hypot(...v);

  /** Seconds to descend from height zf to zt (zt < zf) onto a pad at `ground`: fast above ground + slow_band_m, slow below. */
  function descentTime(zf, zt, sp, ground) {
    const slow = sp.v_descent, fast = sp.v_descent_high == null ? slow : sp.v_descent_high;
    const high = Math.max(0, zf - Math.max(zt, ground + sp.slow_band_m));      // m descended in the fast band
    return high / fast + ((zf - zt) - high) / slow;
  }

  /** Per segment of a 3D route [[x, y, z], ...]: its length (m) and the time E1 books for it. A vertical segment is a
      climb at v_climb or a descent (two speeds); a level one is cruise; a slanted one takes the longer of the two. */
  function profile(route, sp) {
    const ground = route.length ? route[route.length - 1][2] : sp.start_z;
    const segs = [];
    for (let i = 0; i + 1 < route.length; i++) {
      const a = route[i], b = route[i + 1], h = hyp(b[0] - a[0], b[1] - a[1]), dz = b[2] - a[2];
      let time;
      if (h < 0.05) time = dz >= 0 ? dz / sp.v_climb : descentTime(a[2], b[2], sp, ground);
      else time = Math.max(h / sp.v_cruise, dz >= 0 ? dz / sp.v_climb : -dz / sp.v_descent);
      segs.push({ len: hyp(h, dz), time, vertical: h < 0.05, ground });
    }
    return { segs, total_len: segs.reduce((s, x) => s + x.len, 0), total_time: segs.reduce((s, x) => s + x.time, 0) };
  }

  /** The point of the route nearest pos: {seg, u, off, point}. Only segments from minSeg on (a drone does not go back);
      of near-equal candidates the earlier one wins, so a route that crosses itself is not skipped along. */
  function project(route, pos, minSeg) {
    let best = null;
    for (let i = Math.max(0, minSeg || 0); i + 1 < route.length; i++) {
      const a = route[i], b = route[i + 1], d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      const u = l2 < 1e-12 ? 0 : Math.min(1, Math.max(0, ((pos[0] - a[0]) * d[0] + (pos[1] - a[1]) * d[1] + (pos[2] - a[2]) * d[2]) / l2));
      const p = [a[0] + u * d[0], a[1] + u * d[1], a[2] + u * d[2]];
      const off = hyp(pos[0] - p[0], pos[1] - p[1], pos[2] - p[2]);
      if (best === null || off < best.off - 0.3) best = { seg: i, u, off, point: p };
    }
    return best || { seg: 0, u: 0, off: 0, point: route[0] ? route[0].slice() : pos.slice() };
  }

  /** What is left of the route from pos: {seg, u, off, dist_m, time_s, flown_m}. The time follows E1's profile, so it
      counts the climb still to do and the descent over the goal. */
  function remaining(route, pos, sp, minSeg) {
    const pr = profile(route, sp), j = project(route, pos, minSeg);
    let dist = 0, time = 0, flown = 0;
    pr.segs.forEach((s, i) => {
      if (i < j.seg) flown += s.len;
      else if (i === j.seg) {
        dist += s.len * (1 - j.u); flown += s.len * j.u;
        if (s.vertical) {                              // a climb or the descent: from the height it is at (the descent has two speeds)
          const z = j.point[2], zEnd = route[i + 1][2];
          time += zEnd >= z ? (zEnd - z) / sp.v_climb : descentTime(z, zEnd, sp, s.ground);
        } else time += s.time * (1 - j.u);
      } else { dist += s.len; time += s.time; }
    });
    return { seg: j.seg, u: j.u, off: j.off, dist_m: dist, time_s: time, flown_m: flown, total_time_s: pr.total_time, total_m: pr.total_len };
  }

  // ---------------------------------------------------------------- the arrival estimate
  /** A smoothed arrival-time estimate. update(now, waitS, remainingS, booked) -> {arrival, left_s, late_s}: the raw
      estimate is now + the wait for take-off + the time left; it is smoothed over TAU_S (an exponential average), and
      taken as it is when it jumps by more than RESET_S (a re-plan). late_s: against the booked arrival (> 0: late). */
  function makeEta() {
    let est = null, last = null;
    return {
      update(now, waitS, remainingS, booked) {
        const raw = now + Math.max(0, waitS) + remainingS;
        if (est === null || Math.abs(raw - est) > RESET_S) est = raw;
        else if (last !== null && now > last) est += (1 - Math.exp(-(now - last) / TAU_S)) * (raw - est);
        last = now;
        return { arrival: est, left_s: Math.max(0, est - now), late_s: booked == null ? null : est - booked };
      },
      reset() { est = null; last = null; },
    };
  }

  /** "1 min 20 s", "45 s", "2 min", "1 h 05 min". */
  function fmtLeft(sec) {
    const s = Math.max(0, Math.round(sec));
    if (s >= 3600) return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')} min`;
    if (s >= 60) return s % 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${Math.floor(s / 60)} min`;
    return `${s} s`;
  }

  /** HH:MM:SS (24 h) of an epoch in seconds. */
  function fmtClock(epoch) {
    const d = new Date(epoch * 1000), p = (n) => String(n).padStart(2, '0');
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }

  /** The big line. v: {status, left_s, clock_arr (epoch s of the landing, or null), t_arr (sim s)}. */
  function etaLine(v) {
    if (v.status === 'landed') return v.returned ? 'Back at base' : (v.clock_arr != null ? `Delivered at ${fmtClock(v.clock_arr)}` : `Delivered at T+${Math.round(v.t_arr)} s`);
    if (v.status === 'crashed' || v.status === 'cancelled' || v.status === 'rejected' || v.status === 'forced_down') return null;
    if (v.left_s < 3) return 'Arriving now';
    return `Arriving in ${fmtLeft(v.left_s)}`;
  }

  /** The note under the big line: how it compares with the booking. */
  function lateNote(lateS, done) {
    if (lateS == null) return null;
    const r = Math.round(lateS);
    if (Math.abs(lateS) <= ON_TIME_S) return { tone: 'ok', text: done ? 'On time' : 'On time' };
    if (lateS > 0) return { tone: 'warn', text: done ? `Arrived ${r} s late` : `Running late +${r} s` };
    return { tone: 'ok', text: done ? `Arrived ${-r} s early` : `Running ${-r} s early` };
  }

  // ---------------------------------------------------------------- the status stepper
  /** What a flight's events say: re-routed around a zone, sent back to base. events: [{type, text, tag?}]. */
  function classifyEvents(events) {
    const text = (e) => String(e.text || '');
    return {
      rerouted: events.some(e => /no-fly zone/i.test(text(e)) && /re-?rout|avoid|around/i.test(text(e)) && e.type !== 'zone_on'),
      returned: events.some(e => e.tag === 'back to its pad' || /returned to base|back (on|to) its pad/i.test(text(e))),
      forced: events.some(e => e.type === 'forced_down'),
    };
  }

  /** The stepper. i: {status ('waiting' | 'flying' | 'landed' | 'crashed' | 'cancelled' | 'rejected' | 'forced_down'),
      now, t_dep, left_s, late_s, climbing, rerouted, returned, reached (index of the last step done when it ended),
      reason}. -> {steps: [{label, state: 'done' | 'current' | 'todo' | 'failed'}], current, exception, notes}.
      exception: {code, label, tone, detail} when the delivery did not go as planned or is not yet moving. */
  function stage(i) {
    const mark = (cur, failAt) => STEPS.map((label, k) => ({         // steps before the failed one (or the current one) are done
      label, state: failAt === k ? 'failed' : k < (failAt == null ? cur : failAt) ? 'done' : k === cur && failAt == null ? 'current' : 'todo' }));
    let cur, failAt = null, exception = null;
    const notes = [];
    switch (i.status) {
      case 'rejected':
        cur = 0; failAt = 1;
        exception = { code: 'rejected', label: 'Rejected', tone: 'bad', detail: i.reason || 'the Planner found no free slot' };
        break;
      case 'waiting': {
        cur = 1;
        const wait = (i.t_dep == null ? 0 : i.t_dep) - i.now;
        if (wait < -ON_TIME_S) exception = { code: 'delayed', label: 'Delayed', tone: 'warn', detail: `still on its pad ${fmtLeft(-wait)} after its slot` };
        else exception = { code: 'waiting', label: 'Waiting for slot', tone: 'info', detail: wait > 0.5 ? `takes off in ${fmtLeft(wait)}` : 'taking off' };
        break;
      }
      case 'flying':
        cur = i.climbing ? 2 : (i.left_s != null && i.left_s <= ARRIVING_S ? 4 : 3);
        break;
      case 'landed':
        if (i.returned) { cur = 5; failAt = 5; exception = { code: 'returned', label: 'Returned to base', tone: 'warn', detail: 'it could not deliver and flew back to its pad' }; }
        else cur = 6;
        break;
      case 'crashed':
        cur = Math.max(2, i.reached == null ? 3 : i.reached); failAt = cur + 1 > 5 ? 5 : cur + 1;
        exception = { code: 'crashed', label: 'Crashed', tone: 'bad', detail: 'the drone came down' };
        break;
      case 'forced_down':
        cur = Math.max(2, i.reached == null ? 3 : i.reached); failAt = cur + 1 > 5 ? 5 : cur + 1;
        exception = { code: 'forced_down', label: 'Forced down', tone: 'bad', detail: 'landed away from its drop-off' };
        break;
      case 'cancelled':
        cur = 1; failAt = 2;
        exception = { code: 'cancelled', label: 'Cancelled', tone: 'grey', detail: 'the UTM could not keep it safe' };
        break;
      default:
        cur = 0;
    }
    if (i.status === 'flying' || i.status === 'waiting') {
      if (i.late_s != null && i.late_s > ON_TIME_S && !(exception && exception.code === 'delayed'))
        notes.push({ code: 'late', label: `Running late +${Math.round(i.late_s)} s`, tone: 'warn' });
      if (i.rerouted) notes.push({ code: 'reroute', label: 'Re-routed around a zone', tone: 'info' });
    } else if (i.rerouted && i.status === 'landed') notes.push({ code: 'reroute', label: 'Re-routed around a zone', tone: 'info' });
    return { steps: mark(cur, failAt), current: cur, exception, notes };
  }

  // ---------------------------------------------------------------- views: where a flight is at a time
  const CODES = ['waiting', 'flying', 'landed', 'crashed', 'cancelled'];      // utm/flight_log.py status codes

  function lerpAt(T, P, t) {
    if (!T.length) return null;
    let lo = 0, hi = T.length - 1;
    if (t <= T[0]) return { i: 0, pos: P[0] };
    if (t >= T[hi]) return { i: hi, pos: P[hi] };
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
    const u = (t - T[lo]) / (T[hi] - T[lo]);
    return { i: lo, pos: P[lo].map((v, k) => v + u * (P[hi][k] - v)) };
  }

  /** A replay model of one flight from data/<scene>.json (`scene`, written by scripts/export_dashboard.py) and,
      optionally, data/flight_log.json (`log`: the outcome, the full history, the clock). Throws when the flight is not
      in the scene. at(t) -> the flight's view at sim time t. */
  function replayModel(scene, log, flightId, runId) {
    const di = scene.drones.findIndex(d => d.id === flightId);
    if (di < 0) throw new Error(`flight ${flightId} is not in scene ${scene.name}`);
    const d = scene.drones[di];
    let run = null, row = null;
    if (log && log.flights) {
      const runs = (log.runs || []).filter(r => (runId ? r.run_id === runId : r.scenario === scene.name));
      for (const r of runs.slice().reverse()) {                          // the latest run of that scene with this flight
        const f = log.flights.find(x => x.run_id === r.run_id && x.flight_id === flightId);
        if (f) { run = r; row = f; break; }
      }
    }
    const T = scene.frames.t, S = scene.frames.status.map(fr => fr[di]);
    const routes = (scene.routes && scene.routes[flightId]) || [{ t: 0, route: [] }];
    // A drone that has not left (or was removed after a crash) is parked far off the map in the recording: on its pad until it
    // takes off, where it fell afterwards.
    const onMap = (p) => Math.abs(p[0]) < 5000 && Math.abs(p[1]) < 5000;
    let keep = routes[0].route.length ? routes[0].route[0].slice() : (d.start || [0, 0]).concat([1]);
    const P = scene.frames.pos.map(fr => { if (onMap(fr[di])) keep = fr[di]; return keep; });
    const own = scene.events.filter(e => e.flight === flightId);
    const events = row ? row.events.concat((run && run.events) || []).sort((a, b) => a.t - b.t) : own;
    const sp = resolveSpeeds(scene.speeds || (run && run.speeds));
    const firstFly = S.findIndex(c => c === 1);
    const tDep = row && row.sched_dep != null ? row.sched_dep : d.t_dep;
    const arrival = row && row.sched_arr != null ? row.sched_arr : d.arrival;
    const outcome = row ? row.status : null;
    const cls = classifyEvents(events);
    return {
      mode: 'replay', id: flightId, kind: d.kind, operator: row && row.operator != null ? row.operator : d.operator,
      from: (row && row.origin) || d.from || '', to: (row && row.dest) || d.to || '',
      speeds: sp.speeds, approximate: sp.approximate, outcome, events, classes: cls,
      callsign: row ? row.callsign : null, drone: row ? row.drone : null, layer: row ? row.layer : d.layer,
      cruise_z: row ? row.cruise_z : null, distance_m: row ? row.distance_m : null,
      geo: { buildings: scene.map.buildings, bounds: scene.map.bounds, zones: scene.zones || [], places: scene.places || [] },
      tTakeoff: firstFly >= 0 ? T[firstFly] : null, tLand: S.findIndex(c => c === 2) >= 0 ? T[S.findIndex(c => c === 2)] : null,
      t0: Math.max(0, Math.min(tDep == null ? 0 : tDep, firstFly >= 0 ? T[firstFly] : Infinity) - 15),
      t1: (() => { const k = S.findIndex(c => c >= 2); return k < 0 ? T[T.length - 1] : Math.min(T[T.length - 1], T[k] + 8); })(),   // this flight's end + 8 s
      clockOf: run ? clockFn(run) : null,
      at(t) {
        const p = lerpAt(T, P, t), r = routes.filter(x => x.t <= t).pop() || routes[0];
        const code = S[Math.min(T.length - 1, Math.max(0, p.i))];
        let status = CODES[code] || 'waiting';
        if (status === 'landed' && outcome === 'Forced down') status = 'forced_down';
        return { pos: p.pos, status, route: r.route, t_dep: tDep, arrival, replans: routes.length - 1, t };
      },
    };
  }

  /** The replay of a flight the live server logged (GET /api/track: the row, its events and its 1 Hz track). The
      route is the track itself (the plan is not stored): remaining time follows it. scene: GET /api/scene. */
  function trackModel(tr, scene) {
    const f = tr.flight, pts = tr.track, events = tr.events;
    if (!pts.length) {
      if (f.status === 'Rejected') {                       // a refused booking never flew: no track, only the reason
        const text = (events.find(e => e.type === 'refused') || {}).text || '';
        const why = /\((.*)\)\s*$/.exec(text);
        return { mode: 'replay', rejected: true, id: f.flight_id, kind: f.kind, operator: f.operator, from: f.origin || '', to: f.dest || '',
          callsign: f.callsign, events, outcome: 'Rejected', reason: why ? why[1] : '' };
      }
      throw new Error(`no track was logged for ${f.flight_id}`);
    }
    const T = pts.map(p => p[0]), P = pts.map(p => [p[1], p[2], p[3]]);
    const sp = resolveSpeeds((scene && scene.speeds) || tr.run.speeds);
    const tDep = f.sched_dep, arrival = f.sched_arr;
    const outcome = f.status, ground = P[P.length - 1][2];
    const route = [[P[0][0], P[0][1], ground]].concat(P.filter((_, k) => k % 2 === 0), [P[P.length - 1]]);
    return {
      mode: 'replay', id: f.flight_id, kind: f.kind, operator: f.operator, from: f.origin || '', to: f.dest || '',
      speeds: sp.speeds, approximate: sp.approximate, outcome, events, classes: classifyEvents(events),
      callsign: f.callsign, drone: f.drone, layer: f.layer, cruise_z: f.cruise_z, distance_m: f.distance_m,
      geo: { buildings: (scene && scene.buildings) || [], bounds: scene && scene.bounds, zones: [], places: (scene && scene.places) || [] },
      tTakeoff: T[0], tLand: ({ Landed: 1, Returned: 1 }[outcome]) ? T[T.length - 1] : null,
      t0: Math.max(0, Math.min(tDep == null ? T[0] : tDep, T[0]) - 15), t1: T[T.length - 1] + 5,
      clockOf: clockFn(tr.run),
      at(t) {
        const p = lerpAt(T, P, t);
        let status = t < T[0] ? 'waiting' : t >= T[T.length - 1] ? ({ Landed: 'landed', Returned: 'landed', Crashed: 'crashed' }[outcome] || 'flying') : 'flying';
        return { pos: p.pos, status, route, t_dep: tDep, arrival, replans: f.n_replans || 0, t };
      },
    };
  }

  /** Loads what a replay needs. get(url) -> parsed JSON (rejects when missing). params: {scene, flight, run}. The scene
      is required; flight_log.json is optional (without it the outcome comes from the frames' status codes). */
  async function loadReplay(get, base, params) {
    const scene = await get(`${base}data/${params.scene}.json`);
    let log = null;
    try { log = await get(`${base}data/flight_log.json`); } catch (e) { log = null; }
    return replayModel(scene, log, params.flight, params.run);
  }

  /** The flight's short name, as utm/flight_db.py callsign(): operator prefix + the number in its id (OP0-07, HOS-E00, INT-00). */
  function callsign(operator, kind, id) {
    const m = /(\d+)\s*$/.exec(String(id)), num = m ? String(parseInt(m[1], 10)).padStart(2, '0') : String(id);
    const prefix = String(operator || '').replace(/[^A-Za-z0-9]/g, '').toUpperCase().slice(0, 3);
    if (kind === 'intruder') return `INT-${num}`;
    if (kind === 'emergency') return `${prefix || 'EMG'}-E${num}`;
    return `${prefix || 'FLT'}-${num}`;
  }

  return { ARRIVING_S, ON_TIME_S, STEPS, DEFAULT_SPEEDS, callsign, clockFn, resolveSpeeds, descentTime, profile, project, remaining, makeEta,
           fmtLeft, fmtClock, etaLine, lateNote, classifyEvents, stage, lerpAt, replayModel, trackModel, loadReplay };
});
