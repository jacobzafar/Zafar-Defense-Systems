# Third-party licenses

Summary of every dependency's license, split into the core/default install
path and the opt-in extras. Verify with `python scripts/license_audit.py`,
which parses `pyproject.toml` and flags any AGPL/GPL-family package found
in the core dependency list. Keep this file, `scripts/license_audit.py`'s
`_LICENSE_REGISTRY`, and `pyproject.toml` in sync when dependencies change.

## Core dependencies (installed by default)

These are what `pip install -r requirements.txt` or `pip install .`
installs with no extras selected. All permissive, no copyleft.

| Package | License | Notes |
|---|---|---|
| `opencv-python-headless` | MIT (wrapper); bundles Apache-2.0 OpenCV | Headless build — no `libGL` dependency, and nothing in this codebase calls `cv2.imshow`/highgui. |
| `numpy` | BSD-3-Clause | |
| `pyyaml` | MIT | Used by `config/loader.py` to load `config/presets.yaml`. |
| `streamlit` | Apache-2.0 | Operator UI (`ui/app.py`). |
| `pytest` | MIT | Test runner only — not imported by application code. |

## Optional extras (opt-in only, never installed or imported by default)

Selected explicitly via `pip install <package>` or `pip install ".[extra]"`.
None of these are required for the default `motion` detector / `iou`
tracker path, and the corresponding adapter modules guard their import
(`try/except ImportError`) so the rest of the pipeline works with none of
them installed.

| Extra | Package(s) | License | Notes |
|---|---|---|---|
| `ultralytics` | `ultralytics` | **AGPL-3.0** | Requires a commercial license from Ultralytics for closed-source production use. Not installed by default, not imported unless `--detector ultralytics` (or the UI's matching option) is explicitly selected — see `detector/ultralytics_detector.py` and `docs/DECISIONS.md` entry #2. |
| `torchvision` | `torch`, `torchvision` | BSD-3-Clause | The recommended "real detector" path today (`detector/torchvision_detector.py`) — no copyleft, no separate licensing decision needed. See `docs/DECISIONS.md` entry #6. |
| `bytetrack` | `trackers`, `supervision` | Apache-2.0 | Roboflow packages, used by the optional `tracker/bytetrack_adapter.py`. Chosen over the original FoundationVision/ByteTrack repo specifically for license clarity — see `docs/DECISIONS.md` entry #3. |
| `dev` | `pytest` | MIT | Duplicate of the core listing above; kept as an explicit extra for environments that split test tooling out of the runtime install. |

## Policy

- **The core/default install path must never include an AGPL/GPL-family
  package.** This matters both for the observation-only positioning of
  this project and for acquisition/due-diligence review — a buyer's legal
  team should be able to install the default path and know immediately
  that no copyleft obligations attach to it.
- Copyleft packages are acceptable **only** as opt-in extras that are (a)
  never installed by `pip install -r requirements.txt` / a plain
  `pip install .`, and (b) never imported by the codebase unless the
  corresponding backend is explicitly selected at runtime.
- Run `python scripts/license_audit.py` after adding or changing any
  dependency. It exits non-zero if a copyleft package ends up in the core
  dependency list.
- If you add a dependency not yet in `scripts/license_audit.py`'s
  `_LICENSE_REGISTRY`, the audit will flag it as `UNKNOWN` rather than
  silently pass it — add it to the registry and to the tables above.
