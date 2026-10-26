# D3 — OpenStreetMap City Import and Simulation Integration

## 1. Purpose and status

D3 imports mapped building footprints from OpenStreetMap (OSM) and converts them into the project's box-obstacle YAML schema. The importer is designed to keep the source geography, the simulation-friendly geometry approximation, and the optional test crop as separate artifacts.

The rotated-box importer has now been run against the Electronic City Phase I OSM query in the Windows project environment. It retrieved 116 building features; 115 were represented and one invalid/unsupported geometry was skipped. The corrected full-map output, `bengaluru_ecity_phase1_rotated_v2_*`, contains 311 oriented obstacle boxes. Its mean/max footprint coverage ratios are 1.02932/1.14959, with zero represented buildings exceeding the 1.15 target. Earlier strip-based V3 and first rotated-map outputs are retained as historical artifacts; the corrected V2 map is the current tested output.

## 2. Data pipeline

1. **Retrieve OSM features.** `scripts/import_osm_city.py` calls OSMnx to query features tagged `building` around the configured centre. The default centre is Electronic City Phase I, Bengaluru (`12.84968`, `77.66498`) with a 300 m query distance.
2. **Project to metres.** Footprints are projected to a metre-based CRS (normally UTM zone 43N, EPSG:32643). Local X points east, local Y points north, and the query centre is `(0, 0)`.
3. **Approximate the footprint with oriented boxes.** Near-rectangular footprints use a minimum rotated rectangle. Concave/holey shapes are decomposed into pieces, then adjacent pieces are greedily merged only when the box-union coverage remains close to the source geometry. The exporter checks that the union of boxes covers the source footprint and records coverage-area ratios.
4. **Assign building height.** Use a usable OSM `height` value first (including conversion from explicit feet), otherwise estimate height as `building:levels × floor_height_m`, otherwise use the configured default (12 m by default). Metadata records per-building height provenance and counts of `osm_height`, `building_levels_estimate`, and `default_assumption`.
5. **Export.** Writes `<name>_layout.yaml`, `<name>_preview.png`, and `<name>_metadata.yaml`. The YAML remains compatible with the current simulator schema:

```yaml
- type: box
  center: [x, y]
  half: [hx, hy]      # half-extents along the box's own local axes
  height: h
  yaw: 0.52           # optional radians, counter-clockwise about +Z from +X/east
```

Omitting `yaw` preserves the legacy axis-aligned layout behavior. The crop helper, distance functions, teacher closest-point calculation, and PyBullet box geometry have yaw-aware support in this implementation bundle.

## 3. Generate the rotated full map

Map regeneration needs network access and the optional geospatial dependencies. A simulator run using an already-saved YAML layout does not need to query OSM again.

```powershell
python -m pip install -r requirements-optional-osm.txt
python scripts/import_osm_city.py --center-lat 12.84968 --center-lon 77.66498 --dist 300 --max-box-area-ratio 1.15 --name bengaluru_ecity_phase1_rotated
```

The target box-union/footprint area ratio is 1.15. The measured corrected V2 import contains 311 boxes instead of the earlier 2,641-box strip representation; an 88.2% reduction. All 115 represented buildings remain present; the mean coverage ratio is 1.02932, the maximum is 1.14959, and zero buildings exceed the target.

To generate a separate crop for existing small-world navigation tests:

```powershell
python scripts/crop_osm_city.py --source data/osm_city/bengaluru_ecity_phase1_rotated_v2_layout.yaml --center-x -73 --center-y 230 --width 50 --height 50 --name bengaluru_ecity_phase1_rotated_v2_crop50_cand1
```

The crop clips the full geometry to a selected rectangle, drops obstacles outside it and recentres the crop to `(0, 0)`. Fully internal rotated boxes retain yaw; boxes crossing the crop boundary are conservatively re-boxed inside the crop. The original full layout is unchanged.

## 4. Full map versus crop

**The full OSM map is the primary D3 data output.** The 50 m x 50 m crop is a separate test fixture for the current small-world navigation scripts. The corrected V2 crop contains 23 obstacle boxes. Cropping does not overwrite or limit the full map.

After regenerating the rotated map and crop, run both route scripts against the new crop:

```powershell
python scripts/test_osm_city_navigation.py
python scripts/test_osm_city_swarm_navigation.py
```

These scripts perform baseline geometric/grid-planner navigation checks, not trained MARL-policy evaluation. On the corrected 23-box crop, all 3 individual routes passed with no detected collisions; the minimum obstacle gap was 1.287 m against a 1.25 m planner margin. The simultaneous three-drone test also passed: all drones settled at their goals, no collisions were detected, minimum drone-to-drone distance was 3.500 m against a 3.0 m requirement, and the minimum obstacle gap was 1.293 m. These crop tests do not establish full-city flight performance.

## 5. Load an OSM map as a scenario

`envs/osm_scenario.py` validates saved box-layout YAML, calculates bounds that include rotated box extents, and samples free-space start/goal points inside the imported map's own bounds. `envs/swarm_env.py` supports `scenario_mix: {osm_ecity: 1.0}` and caches the layout rather than rereading YAML for every episode.

A dedicated example configuration is provided at `configs/config_stage3_osm_ecity.yaml`. It activates `osm_ecity` and selects initial curriculum stage 4 without changing the default Stage 2 configuration. Its saved layout path has been updated to `data/osm_city/bengaluru_ecity_phase1_rotated_v2_layout.yaml`. The Windows PyBullet smoke test using that saved configuration loaded all 311 obstacles, selected `osm_ecity` at Stage 4, and sampled start/goal positions for three drones.

The OSM sampler uses bounds derived from map geometry rather than clipping to the generic `world_size`. The corrected map bounds are approximately 764 m ? 698 m, larger than the 100 m ? 100 m Stage 4 world setting; that setting is not a clipping boundary for this OSM scenario.

A coarse full-map connectivity diagnostic used a 2 m grid and a 1.25 m obstacle margin. Across three seeds, all 9 sampled start/goal pairs were connected, none were disconnected, and no endpoints failed to snap to nearby free cells. The grid contained 92,804 free cells, with 92,802 in the largest of three connected components. This is a coarse-grid diagnostic, not proof of full-map PyBullet flight success or trained-policy performance.

Run the diagnostic from the repository root with `python -m scripts.test_osm_city_fullmap_connectivity`.

The curriculum transition method in `training/mappo_trainer.py` remains deliberately disabled in the current source. The new `training.initial_curriculum_stage` option permits an explicitly configured run to start at stage 4, but this should not be mistaken for an end-to-end validated Stage-4 training recipe or a claim that the learned policy has been trained on the OSM map.

## 6. Geometry and sampling tests

The offline tests cover:

- OSM height parsing and provenance priority;
- axis-aligned and rotated rectangles;
- concave footprint decomposition and coverage;
- hole/concavity coverage checks and area-ratio bounds for representative fixtures;
- crop recentering, clipping, rotated-box behavior and keeping boundary fragments inside the crop;
- optional-yaw obstacle distance and rule-teacher closest-point behavior;
- OSM layout validation, rotated map bounds, and valid free-space start/goal sampling;
- reusable grid path-planner primitives used by the D3 navigation scripts.

Run the focused tests:

```powershell
python -m pytest tests/test_trainer_curriculum_stage.py tests/test_osm_city_coverage_ratio.py tests/test_osm_city_import.py tests/test_osm_city_dimensions.py tests/test_osm_city_crop.py tests/test_rotated_obstacle_support.py tests/test_osm_scenario.py tests/test_grid_path_planner.py -q
```

The current Windows `.venv` full project suite passed **104 tests, with 6 warnings in 9.76 seconds**. The warnings are existing `PytestReturnNotNoneWarning` warnings from actor-network and episode-manager tests that return values. The full suite was rerun after the rotated-map implementation and current scenario-path updates.

## 7. Shared planner and integration boundaries

`modules/grid_path_planner.py` centralises the current D3 occupancy-grid construction, connected components, BFS and path reconstruction, and both D3 navigation scripts use it. The D1 generator/planner source was not present in the supplied branch snapshot, so the final call-site migration from D1 into this shared module remains an integration task once that code is available. Do not claim that D1 and D3 have already been merged into one planner until both call sites and tests have been checked together.

The supplied `envs/webots_adapter.py` is empty, and the teammate indicated the live adapter is on the lab machine. Webots yaw-orientation changes and tests must be completed with the lab version. Do not merge a yaw-aware layout into Webots deployment until that adapter has been updated and validated.

A second map centred on PESU EC campus is optional. The importer accepts a different centre via `--center-lat` / `--center-lon`, but a verified campus centre or the teammate's existing campus OSM extract should be used rather than inventing coordinates.

## 8. Dependencies, deliverables and attribution

- Core dependencies remain in `requirements.txt`.
- OSMnx is an optional import-time extra in `requirements-optional-osm.txt`; install it only when generating new OSM footprints. Crop geometry operations also require Shapely, supplied through this geospatial extra.
- Main code: `scripts/import_osm_city.py`, `scripts/crop_osm_city.py`, `envs/osm_scenario.py`, `modules/grid_path_planner.py`.
- Shared optional-yaw integration: `envs/scenario_manager.py`, `envs/pybullet_adapter.py`, `baselines/repulsion_controller.py`, `envs/swarm_env.py`.
- OSM-derived building data is © OpenStreetMap contributors and available under the Open Database License (ODbL). Retain attribution in metadata and downstream reports.

## 9. Remaining work before PR

1. The corrected V2 full map and 23-box crop have been regenerated on Windows; retain the measured coverage results and previews.
2. The full Windows test suite passed 104 tests with 6 existing warnings; both crop navigation tests passed.
3. Coordinate the shared yaw contract with the teammate to avoid conflicting edits; update and test the actual Webots adapter on the lab machine.
4. Integrate the D1 planner call sites when that source is available and test both D1-generated and OSM scenes.
5. Full-map scene loading and sampling have passed a PyBullet smoke test, and 9/9 sampled pairs connected on the coarse grid. Full-map PyBullet route following and trained-policy evaluation remain future validation tasks.
6. Optionally import the campus-centred map once the team supplies verified centre/source data.
7. Review the final diff and decide which map artifacts to stage. No commit or push is part of this validation bundle.
