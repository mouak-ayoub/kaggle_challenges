"""YAML-backed shared defaults for the Hough boundary-detection notebooks."""

from pathlib import Path

import yaml

# Sections that are prefixed with their section name when flattened.
# Sections not listed here get no prefix (rescaling, enhancement, energy).
_SECTION_PREFIX: dict[str, str] = {
    "hough": "hough_",
    "boundary": "boundary_",
}

# Two keys changed short names inside their section; remap them so that the
# flat output still matches what the notebooks expect.
_KEY_RENAMES: dict[str, str] = {
    "hough_use_edge_values": "hough_opencv_use_edge_values",
    "hough_theta_step_degrees": "theta_step_degrees",
}


def get_hough_notebook_defaults_path() -> Path:
    """Return the shared YAML config path for the Hough notebooks."""

    project_root = Path(__file__).resolve().parents[2]
    return project_root / "config" / "hough_notebooks.yaml"


def _flatten_profile(profile: dict) -> dict[str, object]:
    """Flatten a nested section dict into a single-level key→value dict.

    Rules
    -----
    - Sections listed in ``_SECTION_PREFIX`` get their section name prepended.
    - All other sections (rescaling, enhancement, energy) get no prefix.
    - ``_KEY_RENAMES`` is applied after prefixing to preserve backward-
      compatible flat key names.
    """
    flat: dict[str, object] = {}
    for section, contents in profile.items():
        if not isinstance(contents, dict):
            # Legacy flat key (shouldn't appear after migration, but handle
            # gracefully so old profiles keep working during a transition).
            flat[section] = contents
            continue
        prefix = _SECTION_PREFIX.get(section, "")
        for key, value in contents.items():
            flat_key = f"{prefix}{key}"
            flat_key = _KEY_RENAMES.get(flat_key, flat_key)
            flat[flat_key] = value
    return flat


def load_hough_boundary_notebook_defaults(
    profile_name: str = "shared_baseline",
) -> dict[str, object]:
    """Load one shared Hough notebook baseline from YAML."""

    config_path = get_hough_notebook_defaults_path()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(
            f"Unknown Hough notebook profile '{profile_name}'. Available: {available}"
        )

    profile = dict(profiles[profile_name])
    flat = _flatten_profile(profile)
    return {key.upper(): value for key, value in flat.items()}


def make_hough_boundary_notebook_defaults(
    profile_name: str = "shared_baseline",
) -> dict[str, object]:
    """Backward-compatible alias for older notebook imports."""

    return load_hough_boundary_notebook_defaults(profile_name=profile_name)
