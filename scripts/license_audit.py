#!/usr/bin/env python3
"""Static license audit: flag any AGPL/GPL copyleft dependency.

Scans pyproject.toml's core `[project] dependencies` and every
`[project.optional-dependencies]` extra, looks each package up in the
hand-maintained registry below, and flags anything copyleft (the
AGPL/GPL family). This is the automated guard behind the "keep the
shipped/default path free of AGPL" decision — see docs/DECISIONS.md
entry #8 and docs/licenses.md for the full reasoning.

This is deliberately NOT a general-purpose SBOM/license scanner: it only
knows about the packages listed in `_LICENSE_REGISTRY`. Any dependency
not in that registry is reported as UNKNOWN rather than silently assumed
safe — update the registry (and docs/licenses.md alongside it) when
dependencies change.

Usage:
    python scripts/license_audit.py

Exit code: 1 if a copyleft package is found in the CORE dependency list
(that would make it part of the default install), or if the file can't
be parsed. 0 otherwise — copyleft packages are permitted in optional
extras, since those are opt-in and (for detector backends) never
imported unless explicitly selected at runtime; see detector/factory.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# Substrings checked case-insensitively against a license identifier.
# Catches GPL-2.0, GPL-3.0, AGPL-3.0, and LGPL-* (LGPL is more permissive
# than GPL/AGPL but still copyleft enough to warrant a manual look if it
# ever shows up in the core dependency list).
_COPYLEFT_MARKERS = ("AGPL", "GPL")

# Hand-maintained registry: package name as it appears in pyproject.toml
# (lowercased, version specifiers stripped) -> SPDX-ish license identifier.
# Keep this in sync with docs/licenses.md.
_LICENSE_REGISTRY: dict[str, str] = {
    "opencv-python-headless": "MIT",
    "numpy": "BSD-3-Clause",
    "pyyaml": "MIT",
    "streamlit": "Apache-2.0",
    "pytest": "MIT",
    "ultralytics": "AGPL-3.0",
    "torch": "BSD-3-Clause",
    "torchvision": "BSD-3-Clause",
    "trackers": "Apache-2.0",
    "supervision": "Apache-2.0",
}


def _extract_package_name(requirement: str) -> str:
    """'opencv-python-headless>=4.9,<5' -> 'opencv-python-headless'."""
    cleaned = requirement.strip().strip('"').strip("'")
    return re.split(r"[<>=!~\[; ]", cleaned, maxsplit=1)[0].lower()


def _parse_core_dependencies(text: str) -> list[str]:
    match = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    if not match:
        return []
    items = re.findall(r'"([^"]+)"', match.group(1))
    return [_extract_package_name(item) for item in items]


def _parse_optional_dependencies(text: str) -> dict[str, list[str]]:
    section = re.search(r"\[project\.optional-dependencies\](.*?)(\n\[|\Z)", text, re.DOTALL)
    if not section:
        return {}
    body = section.group(1)
    extras: dict[str, list[str]] = {}
    for extra_match in re.finditer(r"^([\w-]+)\s*=\s*\[(.*?)\]", body, re.DOTALL | re.MULTILINE):
        name, items_raw = extra_match.groups()
        items = re.findall(r'"([^"]+)"', items_raw)
        extras[name] = [_extract_package_name(item) for item in items]
    return extras


def _is_copyleft(license_id: str) -> bool:
    return any(marker in license_id.upper() for marker in _COPYLEFT_MARKERS)


def _lookup(name: str) -> str:
    return _LICENSE_REGISTRY.get(name, "UNKNOWN")


def audit(text: str) -> tuple[list[str], dict[str, list[str]], bool, bool]:
    """Parse `text` (a pyproject.toml's contents) and evaluate the audit.

    Returns (core_deps, extras, core_has_copyleft, any_unknown).
    """
    core_deps = _parse_core_dependencies(text)
    extras = _parse_optional_dependencies(text)

    core_has_copyleft = any(_is_copyleft(_lookup(name)) for name in core_deps)
    all_names = core_deps + [name for names in extras.values() for name in names]
    any_unknown = any(_lookup(name) == "UNKNOWN" for name in all_names)

    return core_deps, extras, core_has_copyleft, any_unknown


def main() -> int:
    if not PYPROJECT_PATH.exists():
        print(f"pyproject.toml not found at {PYPROJECT_PATH}", file=sys.stderr)
        return 1

    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    core_deps, extras, core_has_copyleft, any_unknown = audit(text)

    print(f"License audit — {PYPROJECT_PATH.relative_to(REPO_ROOT)}\n")

    print("Core dependencies (installed by default via requirements.txt / pip install .):")
    for name in core_deps:
        license_id = _lookup(name)
        flag = ""
        if license_id == "UNKNOWN":
            flag = "  <-- UNKNOWN LICENSE, VERIFY MANUALLY"
        elif _is_copyleft(license_id):
            flag = "  <-- COPYLEFT IN DEFAULT INSTALL — VIOLATION"
        print(f"  {name:<28} {license_id}{flag}")

    print("\nOptional extras (opt-in only, never installed or imported by default):")
    for extra_name, packages in extras.items():
        print(f"  [{extra_name}]")
        for name in packages:
            license_id = _lookup(name)
            flag = ""
            if license_id == "UNKNOWN":
                flag = "  <-- UNKNOWN LICENSE, VERIFY MANUALLY"
            elif _is_copyleft(license_id):
                flag = "  <-- copyleft, OK here only because this extra is opt-in and not default"
            print(f"    {name:<26} {license_id}{flag}")

    print()
    if core_has_copyleft:
        print("RESULT: FAIL — a copyleft (AGPL/GPL) package is in the CORE dependency list.")
        return 1
    if any_unknown:
        print("RESULT: PASS (with warnings) — no copyleft in core deps, but some packages "
              "have an unknown license in this registry — verify manually and update "
              "_LICENSE_REGISTRY + docs/licenses.md.")
        return 0
    print("RESULT: PASS — no copyleft packages in the core/default install path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
