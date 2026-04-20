from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, current.parent, current.parent.parent]
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root from the current working directory.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import EnergyConfig, EnhancementConfig, ResizeConfig, StandardHoughConfig
from src.features import build_energy_image
from src.fitting import line_segment_from_rho_theta, run_standard_hough
from src.io import find_sample_path, load_sample_rgb_image, rgb_to_gray_unit
from src.preprocessing import resize_keep_aspect


def configure_matplotlib_backend(preferred_backend: str | None) -> str:
    import importlib
    import matplotlib
    from matplotlib.backends.registry import backend_registry

    candidates = []
    if preferred_backend:
        candidates.append(preferred_backend)
    candidates.extend(["QtAgg", "TkAgg", "Agg"])

    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            # Validate the backend by actually importing its module
            module_name = backend_registry.backend_for_gui_framework(backend) if hasattr(backend_registry, "backend_for_gui_framework") else None
            if module_name is None:
                # Try resolving via load to catch missing Qt bindings etc.
                backend_registry.load_backend_module(backend)
            return backend
        except Exception:
            continue
    return matplotlib.get_backend()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an external 3D Hough accumulator debug window with green, red, and blue overlays."
    )
    parser.add_argument("--ecg-id", default="11842146")
    parser.add_argument("--scan-type", default="0005")
    parser.add_argument("--max-dim", type=int, default=1000)
    parser.add_argument("--hough-backend", choices=["skimage", "opencv"], default="skimage")
    parser.add_argument("--hough-rho-resolution-pixels", type=float, default=1.0)
    parser.add_argument("--hough-peak-threshold-ratio", type=float, default=0.40)
    parser.add_argument("--hough-min-distance", type=int, default=9)
    parser.add_argument("--hough-min-angle", type=int, default=10)
    parser.add_argument("--hough-opencv-use-edge-values", action="store_true")
    parser.add_argument("--viewer", choices=["plotly", "matplotlib"], default="plotly")
    parser.add_argument("--backend", default=None, help="Preferred Matplotlib GUI backend, for example QtAgg or TkAgg.")
    parser.add_argument(
        "--plotly-renderer",
        default="browser",
        help="Plotly renderer, for example browser, png, or json. Default opens the plot in the browser.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional HTML output path for the Plotly figure.",
    )
    parser.add_argument("--green-lines-per-family", type=int, default=3)
    parser.add_argument("--red-line-count", type=int, default=1)
    parser.add_argument("--green-vertical-target-theta-deg", type=float, default=0.0)
    parser.add_argument("--green-horizontal-target-theta-deg", type=float, default=90.0)
    parser.add_argument("--green-angle-tolerance-deg", type=float, default=None)
    parser.add_argument("--green-rho-min-spacing-bins", type=int, default=2)
    parser.add_argument("--red-target-theta-deg", type=float, default=3.0)
    parser.add_argument("--red-angle-tolerance-deg", type=float, default=0.1)
    parser.add_argument("--red-rho-delta", type=float, default=15.0)
    parser.add_argument("--red-center-rho", type=float, default=None)
    parser.add_argument("--left-search-width-frac", type=float, default=0.12)
    parser.add_argument("--top-accumulator-line-count", type=int, default=5)
    parser.add_argument("--show-top-accumulator-lines", action="store_true")
    return parser.parse_args()


def theta_bin_lookup(result, theta_rad: float) -> tuple[int, float, float]:
    theta_idx = int(np.argmin(np.abs(result.angles - theta_rad)))
    theta_bin = float(result.angles[theta_idx])
    return theta_idx, theta_bin, float(np.rad2deg(theta_bin))


def rho_bin_lookup(result, rho: float) -> tuple[int, float]:
    rho_idx = int(np.argmin(np.abs(result.distances - rho)))
    rho_bin = float(result.distances[rho_idx])
    return rho_idx, rho_bin


def accumulator_lookup(result, rho: float, theta: float) -> dict[str, float | int]:
    theta_idx, theta_bin, _ = theta_bin_lookup(result, theta)
    rho_idx, rho_bin = rho_bin_lookup(result, rho)
    value = float(result.accumulator[rho_idx, theta_idx])
    return {
        "theta_idx": theta_idx,
        "rho_idx": rho_idx,
        "theta_bin": theta_bin,
        "rho_bin": rho_bin,
        "value": value,
        "log_value": float(np.log1p(value)),
    }


def peak_entries_from_result(result, image_shape: tuple[int, int]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for idx, (angle, dist) in enumerate(zip(result.peak_angles, result.peak_distances), start=1):
        segment = line_segment_from_rho_theta(float(dist), float(angle), image_shape)
        if segment is None:
            continue
        stats = accumulator_lookup(result, float(dist), float(angle))
        entries.append(
            {
                "is_peak": True,
                "peak_index": idx,
                "label": f"peak {idx:02d}",
                "segment": segment,
                "rho": float(dist),
                "rho_idx": int(stats["rho_idx"]),
                "theta": float(angle),
                "theta_deg": float(np.rad2deg(angle)),
                "stats": stats,
            }
        )
    return entries


def entries_near_theta(entries: list[dict[str, object]], target_theta_deg: float, angle_tol_deg: float) -> list[dict[str, object]]:
    family = [entry for entry in entries if abs(float(entry["theta_deg"]) - target_theta_deg) <= angle_tol_deg]
    family.sort(key=lambda entry: (-float(entry["stats"]["value"]), float(entry["rho"])))
    return family


def pick_spaced_entries(
    entries: list[dict[str, object]],
    count: int,
    min_rho_bin_gap: int,
) -> list[dict[str, object]]:
    picked: list[dict[str, object]] = []
    for entry in entries:
        if all(abs(int(entry["rho_idx"]) - int(other["rho_idx"])) >= min_rho_bin_gap for other in picked):
            picked.append(entry)
        if len(picked) == count:
            break
    if len(picked) < count:
        for entry in entries:
            if entry not in picked:
                picked.append(entry)
            if len(picked) == count:
                break
    return picked


def build_family_entries(
    entries: list[dict[str, object]],
    target_theta_deg: float,
    angle_tol_deg: float,
    count: int,
    min_rho_bin_gap: int,
) -> list[dict[str, object]]:
    family = entries_near_theta(entries, target_theta_deg, angle_tol_deg)
    if not family:
        family = sorted(
            entries,
            key=lambda entry: (abs(float(entry["theta_deg"]) - target_theta_deg), -float(entry["stats"]["value"])),
        )
    return pick_spaced_entries(family, count=count, min_rho_bin_gap=min_rho_bin_gap)


def top_accumulator_entries(entries: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    ranked = sorted(
        entries,
        key=lambda entry: (-float(entry["stats"]["value"]), abs(float(entry["theta_deg"])), abs(float(entry["rho"]))),
    )
    return ranked[:count]


def red_entries_in_window(
    result: StandardHoughResult,
    peak_bin_set: set[tuple[int, int]],
    image_shape: tuple[int, int],
    center_rho: float,
    rho_delta: float,
    target_theta_deg: float,
    theta_delta_deg: float,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for rho in result.distances:
        rho = float(rho)
        if abs(rho - center_rho) > rho_delta:
            continue
        for theta in result.angles:
            theta = float(theta)
            theta_deg = float(np.rad2deg(theta))
            if abs(theta_deg - target_theta_deg) > theta_delta_deg:
                continue
            segment = line_segment_from_rho_theta(rho, theta, image_shape)
            if segment is None:
                continue
            stats = accumulator_lookup(result, rho, theta)
            entries.append(
                {
                    "is_peak": (int(stats["rho_idx"]), int(stats["theta_idx"])) in peak_bin_set,
                    "segment": segment,
                    "rho": rho,
                    "rho_idx": int(stats["rho_idx"]),
                    "theta": theta,
                    "theta_deg": theta_deg,
                    "stats": stats,
                }
            )
    return entries


def pick_red_entries_from_window(
    entries: list[dict[str, object]],
    center_rho: float,
    target_theta_deg: float,
    count: int,
) -> list[dict[str, object]]:
    ranked = sorted(
        entries,
        key=lambda entry: (
            abs(float(entry["theta_deg"]) - target_theta_deg),
            abs(float(entry["rho"]) - center_rho),
            -float(entry["stats"]["value"]),
            float(entry["rho"]),
            float(entry["theta_deg"]),
        ),
    )
    return ranked[:count]


def red_theta_from_cfg(result, vertical_green_entries: list[dict[str, object]], red_target_theta_deg: float | None) -> tuple[float, float]:
    if red_target_theta_deg is None:
        raw_theta = float(np.median([float(entry["theta"]) for entry in vertical_green_entries]))
    else:
        raw_theta = float(np.deg2rad(red_target_theta_deg))
    _, theta_bin, theta_bin_deg = theta_bin_lookup(result, raw_theta)
    return theta_bin, theta_bin_deg


def print_line_debug(entry: dict[str, object]) -> None:
    segment = tuple((round(x, 2), round(y, 2)) for x, y in entry["segment"])
    stats = entry["stats"]
    print(
        f"{entry['label']} | is_peak={entry['is_peak']} | segment={segment} "
        f"| rho={float(entry['rho']):.2f} | theta_deg={float(entry['theta_deg']):.3f} "
        f"| accumulator={float(stats['value']):.1f} | log1p={float(stats['log_value']):.4f} "
        f"| nearest_bin=(rho={float(stats['rho_bin']):.2f}, theta_deg={np.rad2deg(float(stats['theta_bin'])):.3f})"
    )


def show_plotly_figure(
    theta_deg_3d: np.ndarray,
    rho_3d: np.ndarray,
    accumulator_3d: np.ndarray,
    selected_green_lines: list[dict[str, object]],
    red_debug_lines: list[dict[str, object]],
    blue_debug_lines: list[dict[str, object]],
    plotly_renderer: str,
    output_html: str | None,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=theta_deg_3d,
            y=rho_3d,
            z=accumulator_3d,
            colorscale="magma",
            opacity=0.88,
            colorbar=dict(
                title="log(1+votes)",
                x=0.98,
                y=0.5,
                len=0.88,
                thickness=18,
            ),
            hovertemplate="theta: %{x:.3f} deg<br>rho: %{y:.2f} px<br>log(1+votes): %{z:.4f}<extra></extra>",
        )
    )

    for key, color, label, entries in (
        ("green", "lime", "green selected peaks", selected_green_lines),
        ("red", "red", "red hypotheses", red_debug_lines),
        ("blue", "deepskyblue", "blue top peaks", blue_debug_lines),
    ):
        if not entries:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=[float(entry["theta_deg"]) for entry in entries],
                y=[float(entry["rho"]) for entry in entries],
                z=[float(entry["stats"]["log_value"]) for entry in entries],
                mode="markers+text",
                marker=dict(color=color, size=7, line=dict(color="white", width=1)),
                text=[str(entry["short_label"]) for entry in entries],
                textposition="top center",
                textfont=dict(color=color, size=11),
                name=f"{label} ({len(entries)})",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "theta: %{x:.3f} deg<br>"
                    "rho: %{y:.2f} px<br>"
                    "log(1+votes): %{z:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="3D Hough accumulator — rotate/zoom freely, hover for coordinates",
        scene=dict(
            xaxis_title="theta (degrees)",
            yaxis_title="rho (pixels)",
            zaxis_title="log(1 + votes)",
        ),
        width=1050,
        height=720,
        margin=dict(l=0, r=180, b=0, t=40),
        legend=dict(
            x=1.12,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )

    if output_html:
        fig.write_html(output_html, auto_open=False)
        print(f"Saved Plotly HTML: {output_html}")
    fig.show(renderer=plotly_renderer)


def show_matplotlib_figure(
    theta_deg_3d: np.ndarray,
    rho_3d: np.ndarray,
    accumulator_3d: np.ndarray,
    selected_green_lines: list[dict[str, object]],
    red_debug_lines: list[dict[str, object]],
    blue_debug_lines: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    theta_grid_3d, rho_grid_3d = np.meshgrid(theta_deg_3d, rho_3d)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        theta_grid_3d,
        rho_grid_3d,
        accumulator_3d,
        cmap="magma",
        linewidth=0,
        antialiased=False,
        alpha=0.88,
    )

    for color, entries, size, label in (
        ("lime", selected_green_lines, 42, f"green selected peaks ({len(selected_green_lines)})"),
        ("red", red_debug_lines, 56, f"red hypotheses ({len(red_debug_lines)})"),
        ("deepskyblue", blue_debug_lines, 48, f"blue top peaks ({len(blue_debug_lines)})"),
    ):
        if not entries:
            continue
        ax.scatter(
            [float(entry["theta_deg"]) for entry in entries],
            [float(entry["rho"]) for entry in entries],
            [float(entry["stats"]["log_value"]) for entry in entries],
            color=color,
            s=size,
            depthshade=False,
            label=label,
        )
        for entry in entries:
            ax.text(
                float(entry["theta_deg"]) + 0.15,
                float(entry["rho"]),
                float(entry["stats"]["log_value"]),
                str(entry["short_label"]),
                color=color,
                fontsize=8,
            )

    ax.set_title("3D view of log(1 + Hough accumulator) with debug overlays")
    ax.set_xlabel("theta (degrees)")
    ax.set_ylabel("rho (pixels)")
    ax.set_zlabel("log(1 + votes)")
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08)
    if selected_green_lines or red_debug_lines or blue_debug_lines:
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    backend = None
    if args.viewer == "matplotlib":
        backend = configure_matplotlib_backend(args.backend)

    sample_root = find_sample_path(PROJECT_ROOT)
    resize_cfg = ResizeConfig(max_dim=args.max_dim)
    enhancement_cfg = EnhancementConfig(mode="none", unsharp_radius=2.0, unsharp_amount=1.5)
    energy_cfg = EnergyConfig(
        mode="canny",
        gaussian_sigma=0.0,
        canny_sigma=2.0,
        canny_low_threshold=0.10,
        canny_high_threshold=0.25,
        post_brighten_mode="none",
        post_brighten_gamma=1.0,
        outer_black_border=0,
    )
    theta_step_degrees = round(args.max_dim / 500)
    standard_hough_cfg = StandardHoughConfig(
        backend=args.hough_backend,
        rho_resolution_pixels=args.hough_rho_resolution_pixels,
        theta_step_degrees=theta_step_degrees,
        n_peaks=round(args.max_dim / 25),
        peak_threshold_ratio=args.hough_peak_threshold_ratio,
        min_distance=args.hough_min_distance,
        min_angle=args.hough_min_angle,
        opencv_use_edge_values=args.hough_opencv_use_edge_values,
    )

    rgb_img, image_path = load_sample_rgb_image(sample_root, args.ecg_id, args.scan_type)
    gray_img = rgb_to_gray_unit(rgb_img)
    resized_gray, scale = resize_keep_aspect(gray_img, resize_cfg.max_dim)
    _, _, _, final_energy = build_energy_image(resized_gray, enhancement_cfg, energy_cfg)
    edges = final_energy > 0
    standard_hough_result = run_standard_hough(final_energy, standard_hough_cfg)
    effective_min_accumulator_value = float(args.hough_peak_threshold_ratio) * float(np.max(standard_hough_result.accumulator))

    green_angle_tolerance_deg = (
        args.green_angle_tolerance_deg
        if args.green_angle_tolerance_deg is not None
        else max(float(standard_hough_cfg.theta_step_degrees) * 1.1, 2.5)
    )
    vertical_angle_tolerance_deg = (
        float(args.red_angle_tolerance_deg)
        if args.red_target_theta_deg is not None
        else float(green_angle_tolerance_deg)
    )

    all_peak_entries = peak_entries_from_result(standard_hough_result, resized_gray.shape)
    if not all_peak_entries:
        raise ValueError("No Hough peak entries available for debug plotting.")
    peak_bin_set = {(int(entry["rho_idx"]), int(entry["stats"]["theta_idx"])) for entry in all_peak_entries}

    vertical_seed_theta_deg = (
        float(args.red_target_theta_deg)
        if args.red_target_theta_deg is not None
        else float(
            min(
                all_peak_entries,
                key=lambda entry: abs(float(entry["theta_deg"]) - args.green_vertical_target_theta_deg),
            )["theta_deg"]
        )
    )
    horizontal_seed_theta_deg = float(
        min(
            all_peak_entries,
            key=lambda entry: abs(abs(float(entry["theta_deg"])) - args.green_horizontal_target_theta_deg),
        )["theta_deg"]
    )

    vertical_green_entries = build_family_entries(
        all_peak_entries,
        target_theta_deg=vertical_seed_theta_deg,
        angle_tol_deg=vertical_angle_tolerance_deg,
        count=args.green_lines_per_family,
        min_rho_bin_gap=args.green_rho_min_spacing_bins,
    )
    horizontal_green_entries = build_family_entries(
        all_peak_entries,
        target_theta_deg=horizontal_seed_theta_deg,
        angle_tol_deg=green_angle_tolerance_deg,
        count=args.green_lines_per_family,
        min_rho_bin_gap=args.green_rho_min_spacing_bins,
    )

    selected_green_lines: list[dict[str, object]] = []
    for idx, entry in enumerate(vertical_green_entries, start=1):
        item = dict(entry)
        item["is_peak"] = True
        item["label"] = f"GREEN vertical {idx}"
        item["short_label"] = f"V{idx}"
        selected_green_lines.append(item)
    for idx, entry in enumerate(horizontal_green_entries, start=1):
        item = dict(entry)
        item["is_peak"] = True
        item["label"] = f"GREEN horizontal {idx}"
        item["short_label"] = f"H{idx}"
        selected_green_lines.append(item)

    red_theta, red_theta_deg = red_theta_from_cfg(standard_hough_result, vertical_green_entries, args.red_target_theta_deg)
    distance_step = float(np.median(np.diff(standard_hough_result.distances))) if len(standard_hough_result.distances) > 1 else 1.0
    left_search_width = max(1, int(float(args.left_search_width_frac) * edges.shape[1]))

    if args.red_center_rho is None:
        y_edge, x_edge = np.nonzero(edges[:, :left_search_width])
        if x_edge.size == 0:
            raise ValueError("No edge pixel found in the left search window.")
        rho_votes = x_edge.astype(float) * np.cos(red_theta) + y_edge.astype(float) * np.sin(red_theta)
        left_vote_indices = np.rint((rho_votes - standard_hough_result.distances[0]) / distance_step).astype(int)
        left_vote_indices = np.clip(left_vote_indices, 0, standard_hough_result.distances.size - 1)
        left_vote_hist = np.bincount(left_vote_indices, minlength=standard_hough_result.distances.size)
        best_left_rho_idx = int(np.argmax(left_vote_hist))
    else:
        left_vote_hist = None
        best_left_rho_idx = int(np.argmin(np.abs(standard_hough_result.distances - float(args.red_center_rho))))

    red_center_rho_value = float(standard_hough_result.distances[best_left_rho_idx])
    red_rho_delta = max(0.0, float(args.red_rho_delta))
    red_theta_delta_deg = max(0.0, float(args.red_angle_tolerance_deg))

    red_debug_lines_raw = red_entries_in_window(
        standard_hough_result,
        peak_bin_set,
        resized_gray.shape,
        center_rho=red_center_rho_value,
        rho_delta=red_rho_delta,
        target_theta_deg=red_theta_deg,
        theta_delta_deg=red_theta_delta_deg,
    )

    red_window_lines = [
        entry
        for entry in red_debug_lines_raw
        if float(entry["stats"]["value"]) >= effective_min_accumulator_value
    ]
    red_debug_lines = pick_red_entries_from_window(
        red_window_lines,
        center_rho=red_center_rho_value,
        target_theta_deg=red_theta_deg,
        count=args.red_line_count,
    )
    for idx, entry in enumerate(red_debug_lines, start=1):
        entry["label"] = f"RED window {idx}"
        entry["short_label"] = f"R{idx}"

    blue_debug_lines: list[dict[str, object]] = []
    if args.show_top_accumulator_lines:
        for idx, entry in enumerate(top_accumulator_entries(all_peak_entries, args.top_accumulator_line_count), start=1):
            item = dict(entry)
            item["is_peak"] = True
            item["label"] = f"TOP accumulator {idx}"
            item["short_label"] = f"T{idx}"
            blue_debug_lines.append(item)

    rho_step = max(1, standard_hough_result.accumulator.shape[0] // 300)
    theta_step = max(1, standard_hough_result.accumulator.shape[1] // 180)
    accumulator_3d = np.log1p(standard_hough_result.accumulator[::rho_step, ::theta_step])
    theta_deg_3d = np.rad2deg(standard_hough_result.angles[::theta_step])
    rho_3d = standard_hough_result.distances[::rho_step]

    if backend is not None:
        print(f"Matplotlib backend: {backend}")
    else:
        print(f"Plotly renderer: {args.plotly_renderer}")
    print(f"Image path: {image_path}")
    print(f"Resize scale: {scale:.4f}")
    print(f"Theta step: {theta_step_degrees} deg")
    print()
    for entry in selected_green_lines:
        print_line_debug(entry)
    for entry in red_debug_lines:
        print_line_debug(entry)
    for entry in blue_debug_lines:
        print_line_debug(entry)

    if args.viewer == "plotly":
        show_plotly_figure(
            theta_deg_3d,
            rho_3d,
            accumulator_3d,
            selected_green_lines,
            red_debug_lines,
            blue_debug_lines,
            args.plotly_renderer,
            args.output_html,
        )
    else:
        show_matplotlib_figure(
            theta_deg_3d,
            rho_3d,
            accumulator_3d,
            selected_green_lines,
            red_debug_lines,
            blue_debug_lines,
        )


if __name__ == "__main__":
    main()
