import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import license_audit  # noqa: E402


def test_extract_package_name_strips_version_specifiers():
    assert license_audit._extract_package_name("opencv-python-headless>=4.9,<5") == "opencv-python-headless"
    assert license_audit._extract_package_name('"numpy>=1.26,<2"') == "numpy"
    assert license_audit._extract_package_name("pytest") == "pytest"


def test_is_copyleft_flags_gpl_family_only():
    assert license_audit._is_copyleft("AGPL-3.0")
    assert license_audit._is_copyleft("GPL-3.0")
    assert not license_audit._is_copyleft("MIT")
    assert not license_audit._is_copyleft("BSD-3-Clause")
    assert not license_audit._is_copyleft("Apache-2.0")


def test_audit_parses_core_and_extras_from_sample_pyproject():
    sample = textwrap.dedent(
        """
        [project]
        name = "sample"
        dependencies = [
            "numpy>=1.26,<2",
            "pyyaml>=6.0,<7",
        ]

        [project.optional-dependencies]
        ultralytics = ["ultralytics>=8.2"]
        torchvision = ["torch>=2.2", "torchvision>=0.17"]
        """
    )
    core_deps, extras, core_has_copyleft, any_unknown = license_audit.audit(sample)

    assert core_deps == ["numpy", "pyyaml"]
    assert extras == {"ultralytics": ["ultralytics"], "torchvision": ["torch", "torchvision"]}
    assert core_has_copyleft is False
    assert any_unknown is False


def test_audit_fails_when_copyleft_package_is_in_core_dependencies():
    sample = textwrap.dedent(
        """
        [project]
        dependencies = [
            "ultralytics>=8.2",
        ]
        """
    )
    _, _, core_has_copyleft, _ = license_audit.audit(sample)
    assert core_has_copyleft is True


def test_audit_flags_unknown_packages_without_failing():
    sample = textwrap.dedent(
        """
        [project]
        dependencies = [
            "some-mystery-package>=1.0",
        ]
        """
    )
    _, _, core_has_copyleft, any_unknown = license_audit.audit(sample)
    assert core_has_copyleft is False
    assert any_unknown is True


def test_real_repo_pyproject_has_no_copyleft_in_core_dependencies():
    """Regression guard: this is the actual invariant Priority 2 cares about."""
    text = license_audit.PYPROJECT_PATH.read_text(encoding="utf-8")
    core_deps, extras, core_has_copyleft, _ = license_audit.audit(text)

    assert core_has_copyleft is False
    assert len(core_deps) > 0
    # The known AGPL package must only ever appear as an opt-in extra.
    assert "ultralytics" not in core_deps
    assert "ultralytics" in extras.get("ultralytics", [])


def test_main_exits_zero_on_real_pyproject(capsys):
    exit_code = license_audit.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "RESULT: PASS" in captured.out
