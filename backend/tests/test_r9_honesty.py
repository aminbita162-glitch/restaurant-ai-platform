"""PHASE R9 tests — Install pack honesty.

Test:
- README.md must not contain: production-ready, enterprise-ready, million-tenant proven.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_README_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "README.md",
)

_FORBIDDEN_PHRASES = [
    "production-ready",
    "enterprise-ready",
    "million-tenant proven",
]


def test_readme_has_no_forbidden_claims():
    """README must not contain production-ready, enterprise-ready, or million-tenant proven."""
    readme_path = os.path.normpath(_README_PATH)
    assert os.path.exists(readme_path), f"README.md not found at {readme_path}"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read().lower()

    for phrase in _FORBIDDEN_PHRASES:
        assert phrase.lower() not in content, (
            f"README.md must not contain '{phrase}' — found it"
        )


def test_readme_has_install_steps():
    """README must contain 30-day install steps that mention demo=true,
    Postgres (DATABASE_URL), and the worker."""
    readme_path = os.path.normpath(_README_PATH)
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "demo=true" in content, "Install steps must mention demo=true"
    assert "DATABASE_URL" in content, "Install steps must mention DATABASE_URL"
    assert "worker" in content.lower(), "Install steps must mention the worker process"


if __name__ == "__main__":
    test_readme_has_no_forbidden_claims()
    print("PASS test_readme_has_no_forbidden_claims")

    test_readme_has_install_steps()
    print("PASS test_readme_has_install_steps")

    print("\nAll PHASE R9 tests passed.")
