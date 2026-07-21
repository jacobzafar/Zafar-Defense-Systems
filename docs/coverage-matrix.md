# Data Coverage Matrix

Tracks what real footage/imagery has actually been collected or sourced
across the conditions a fielded anti-drone detector needs to work under.
This is a **template — no real footage has been logged here yet**. Do
not fill in rows with invented or assumed data; only log a scenario once
real footage for it exists and its provenance/license is known (cross-
reference `detector/datasets/manifest.py` if it comes from one of the
named public datasets there).

This complements, and does not replace, `eval/README.md` (the frozen
eval-set policy) and `docs/known-limitations.md`.

## Why track this

A detector's real-world failure modes are usually about *what it never
saw* during training/validation, not the model architecture. Tracking
coverage across drone type, lighting/weather, background, range/angle/
speed, and sensor modality makes those gaps visible and discoverable
before a field test — rather than after.

## Dimensions

- **Drone type**: `fixed-wing`, `multirotor-quad`, `multirotor-hex-oct`,
  `fpv-racing`, `micro-nano`, `other`/`unknown`
- **Lighting / weather**: `daylight-clear`, `daylight-overcast`,
  `dusk-dawn`, `night`, `rain`, `fog-haze`, `snow`
- **Background**: `clean-sky`, `urban-buildings`, `foliage-trees`,
  `terrain-mountains`, `water`, `mixed-cluttered`
- **Range**: `close (<50m)`, `medium (50-200m)`, `far (200-500m)`,
  `very-far (>500m)`
- **Angle**: `ground-looking-up`, `elevated-oblique`, `overhead-down`,
  `side-on`
- **Speed**: `stationary-hover`, `slow (<5 m/s)`, `medium (5-15 m/s)`,
  `fast (>15 m/s)` — note FPV racing drones commonly fall in "fast"
- **Sensor**: `EO` (visible/electro-optical), `IR` (thermal/infrared)

Adjust bucket boundaries if a real dataset's own metadata uses different
cutoffs — record whatever convention was actually used in the "Notes"
column so numbers stay comparable to their source.

## Scenario log

The primary fillable artifact: one row per real, sourced clip/sequence.
Append rows as footage is collected — do not pre-fill placeholder rows
with guessed data.

| ID | Source / dataset | Drone type | Lighting/weather | Background | Range | Angle | Speed | Sensor | # sequences | # frames | License status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _(none logged yet)_ | | | | | | | | | | | | |

Suggested columns to add if useful for your workflow: capture date,
resolution/FPS, whether it's been through `tools/preannotate.py`, whether
it's part of the frozen eval set (`eval/`) or a training set
(`detector/datasets/`).

## Coverage summary (update manually as the log above grows)

Check off a box only once at least one real, license-confirmed scenario
log entry exists for that condition.

**Drone type**
- [ ] Fixed-wing
- [ ] Multirotor (quad)
- [ ] Multirotor (hex/oct)
- [ ] FPV racing
- [ ] Micro/nano
- [ ] Other/unknown

**Lighting / weather**
- [ ] Daylight, clear
- [ ] Daylight, overcast
- [ ] Dusk/dawn
- [ ] Night
- [ ] Rain
- [ ] Fog/haze
- [ ] Snow

**Background**
- [ ] Clean sky
- [ ] Urban/buildings
- [ ] Foliage/trees
- [ ] Terrain/mountains
- [ ] Water
- [ ] Mixed/cluttered

**Range**
- [ ] Close (<50m)
- [ ] Medium (50-200m)
- [ ] Far (200-500m)
- [ ] Very far (>500m)

**Angle**
- [ ] Ground-level looking up
- [ ] Elevated/oblique
- [ ] Overhead looking down
- [ ] Side-on

**Speed**
- [ ] Stationary/hover
- [ ] Slow (<5 m/s)
- [ ] Medium (5-15 m/s)
- [ ] Fast (>15 m/s)

**Sensor**
- [ ] EO (visible)
- [ ] IR (thermal)

## Status

Every box above is currently unchecked and the scenario log is empty —
this document is infrastructure for tracking coverage once real footage
starts being collected, not a record of coverage that already exists. See
`docs/known-limitations.md` for the current state of data/weights in this
repo.
