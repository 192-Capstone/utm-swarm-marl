"""
speeds.py
---------
The flight profile E1 books, as plain numbers for the tracking page's ETA (dashboard/track_logic.js): climb at
v_climb, cruise at v_cruise, descent at v_descent_high above slow_band_m over the pad and v_descent below it
(utm/strategic_planner.py profile()). Built the way scripts/run_scenario.py planner_for() sets E1 up, so the
numbers are the ones the scenario flew with.
"""

from utm.strategic_planner import Settings

KEYS = ('v_cruise', 'v_climb', 'v_descent', 'v_descent_high', 'slow_band_m', 'start_z')
SCENARIO_KEYS = ('v_climb', 'v_descent', 'v_descent_high', 'slow_band_m')       # the ones a scenario's utm block sets


def from_settings(s):
    """{key: number} of an E1 Settings; v_descent is resolved (None: v_climb), v_descent_high stays None when unset."""
    out = {k: getattr(s, k) for k in KEYS}
    if out['v_descent'] is None:
        out['v_descent'] = out['v_climb']
    return {k: None if v is None else float(v) for k, v in out.items()}


def from_scenario(utm):
    """The speeds of a scenario's utm block (null: E1's default)."""
    s = Settings()
    for k in SCENARIO_KEYS:
        if (utm or {}).get(k) is not None:
            setattr(s, k, float(utm[k]))
    return from_settings(s)
