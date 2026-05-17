"""Build a local HTML dashboard from experiment artifacts.

The dashboard reads ignored local artifacts under data/outputs:
- submission metadata folders
- diagnostics zips from older runs
- run bundles from newer Colab notebooks
- full extracted run folders when available

It intentionally does not train or evaluate models. It only summarizes files that
already exist locally.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("data/outputs")
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_ROOT / "reports" / "experiment_dashboard.html"
CHART_COUNTER = 0

TEST_REFERENCE = {
    "00066667": {"family": "bit_manipulation", "gold": "10010111"},
    "000b53cf": {"family": "bit_manipulation", "gold": "01000011"},
    "00189f6a": {"family": "cipher", "gold": "cat imagines book"},
}
FAMILY_ORDER = ["bit_manipulation", "cipher", "equation", "gravity", "numeral", "unit_conversion"]


@dataclass
class Experiment:
    name: str
    source: str
    path: Path
    score_public: float | None = None
    method: str | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    files: set[str] = field(default_factory=set)


def read_json_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip().lower()


def find_zip_member(names: list[str], basename: str) -> str | None:
    matches = [name for name in names if Path(name).name == basename]
    if not matches:
        return None
    return sorted(matches, key=lambda name: (len(name), name))[0]


def read_csv_from_zip(zf: zipfile.ZipFile, names: list[str], basename: str) -> pd.DataFrame:
    member = find_zip_member(names, basename)
    if member is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(zf.read(member)))


def read_json_from_zip(zf: zipfile.ZipFile, names: list[str], basename: str) -> dict[str, Any]:
    member = find_zip_member(names, basename)
    if member is None:
        return {}
    return json.loads(zf.read(member).decode("utf-8"))


def read_artifacts_from_zip(path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame], set[str]]:
    frames: dict[str, pd.DataFrame] = {}
    files: set[str] = set()
    run_config: dict[str, Any] = {}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        basenames = {Path(name).name for name in names}
        files.update(basenames)
        run_config = read_json_from_zip(zf, names, "run_config.json")
        for key, basename in {
            "trainer_log": "trainer_log_history.csv",
            "probe_evolution": "probe_evolution.csv",
            "generated_eval_summary": "generated_eval_summary.csv",
            "checkpoint_eval_summary": "checkpoint_generated_eval_summary.csv",
            "sanity_predictions": "sanity_test_predictions.csv",
            "sanity_predictions_raw": "sanity_test_predictions_raw.csv",
        }.items():
            frame = read_csv_from_zip(zf, names, basename)
            if len(frame):
                frames[key] = frame
    return run_config, frames, files


def read_artifacts_from_dir(path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame], set[str]]:
    frames: dict[str, pd.DataFrame] = {}
    files: set[str] = set()
    for file_path in path.rglob("*"):
        if file_path.is_file():
            files.add(file_path.name)

    run_config = read_json_path(path / "run_config.json")
    candidates = {
        "trainer_log": ["trainer_log_history.csv"],
        "probe_evolution": ["probe_evolution.csv"],
        "generated_eval_summary": ["generated_eval_summary.csv"],
        "sanity_predictions": ["sanity_test_predictions.csv"],
        "sanity_predictions_raw": ["sanity_test_predictions_raw.csv"],
        "checkpoint_eval_summary": [
            "checkpoint_eval" + "/" + "checkpoint_generated_eval_summary.csv",
            "checkpoint_eval" + "/" + "checkpoint_eval_summary_by_family.csv",
            "checkpoint_generated_eval_summary.csv",
            "checkpoint_eval_summary_by_family.csv",
        ],
    }
    for key, rel_paths in candidates.items():
        for rel in rel_paths:
            csv_path = path / rel
            if csv_path.exists():
                frames[key] = pd.read_csv(csv_path)
                break
    return run_config, frames, files


def generated_eval_from_metadata(metadata: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, value in metadata.items():
        if not key.startswith("local_generated_eval_") or not isinstance(value, dict):
            continue
        step_match = re.search(r"_(\d+)$", key)
        step = int(step_match.group(1)) if step_match else None
        phase = key.removeprefix("local_generated_eval_")
        if "accuracy" in value:
            rows.append(
                {
                    "checkpoint": phase,
                    "step": step,
                    "phase": phase,
                    "family": "all",
                    "rows": value.get("rows"),
                    "matches": value.get("matches"),
                    "accuracy": value.get("accuracy"),
                }
            )
        for family, accuracy in value.get("family_accuracy", {}).items():
            rows.append(
                {
                    "checkpoint": phase,
                    "step": step,
                    "phase": phase,
                    "family": family,
                    "accuracy": accuracy,
                }
            )
    return pd.DataFrame(rows)


def probe_summary_from_metadata(metadata: dict[str, Any]) -> pd.DataFrame:
    rows = metadata.get("probe_summary")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def trainer_tail_from_metadata(metadata: dict[str, Any]) -> pd.DataFrame:
    rows = metadata.get("trainer_tail")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_submission_experiment(path: Path) -> Experiment:
    metadata = read_json_path(path / "metadata.json")
    run_config = {}
    for key in ["run_config", "config_summary", "hyperparams"]:
        if isinstance(metadata.get(key), dict):
            run_config = metadata[key]
            break
    frames: dict[str, pd.DataFrame] = {}
    files: set[str] = {item.name for item in path.iterdir() if item.is_file()}

    zip_candidates = sorted(path.glob("*run_bundle.zip")) + sorted(path.glob("*diagnostics.zip"))
    if zip_candidates:
        zip_run_config, zip_frames, zip_files = read_artifacts_from_zip(zip_candidates[0])
        run_config = zip_run_config or run_config
        frames.update(zip_frames)
        files.update(zip_files)
        files.add(zip_candidates[0].name)

    dir_run_config, dir_frames, dir_files = read_artifacts_from_dir(path)
    run_config = dir_run_config or run_config
    frames.update(dir_frames)
    files.update(dir_files)

    if "generated_eval_summary" not in frames:
        generated_eval = generated_eval_from_metadata(metadata)
        if len(generated_eval):
            frames["generated_eval_summary"] = generated_eval
    if "probe_evolution" not in frames:
        probe_summary = probe_summary_from_metadata(metadata)
        if len(probe_summary):
            frames["probe_summary"] = probe_summary
    if "trainer_log" not in frames:
        trainer_tail = trainer_tail_from_metadata(metadata)
        if len(trainer_tail):
            frames["trainer_log"] = trainer_tail

    name = run_config.get("experiment_name") or metadata.get("experiment_name") or metadata.get("id") or path.name
    return Experiment(
        name=str(name),
        source="submission archive",
        path=path,
        score_public=to_number(metadata.get("score_public", metadata.get("kaggle_score"))),
        method=metadata.get("method") or metadata.get("run_id") or metadata.get("source"),
        run_config=run_config,
        metadata=metadata,
        frames=frames,
        files=files,
    )


def load_run_dir_experiment(path: Path) -> Experiment:
    run_config, frames, files = read_artifacts_from_dir(path)
    name = run_config.get("experiment_name") or path.name
    return Experiment(
        name=str(name),
        source="run folder",
        path=path,
        method=run_config.get("prompt_experiment"),
        run_config=run_config,
        frames=frames,
        files=files,
    )


def load_run_bundle_experiment(path: Path) -> Experiment:
    run_config, frames, files = read_artifacts_from_zip(path)
    name = run_config.get("experiment_name") or path.stem.replace("_run_bundle", "")
    return Experiment(
        name=str(name),
        source="run bundle",
        path=path,
        method=run_config.get("prompt_experiment"),
        run_config=run_config,
        frames=frames,
        files=files,
    )


def discover_experiments(output_root: Path) -> list[Experiment]:
    experiments: list[Experiment] = []
    submissions_dir = output_root / "submissions"
    if submissions_dir.exists():
        for path in sorted(p for p in submissions_dir.iterdir() if p.is_dir()):
            experiments.append(load_submission_experiment(path))

    run_roots = [output_root / "experiments", output_root / "runs"]
    for root in run_roots:
        if root.exists():
            for path in sorted(p for p in root.iterdir() if p.is_dir()):
                experiments.append(load_run_dir_experiment(path))

    bundle_roots = [output_root / "run_bundles", output_root]
    for root in bundle_roots:
        if root.exists():
            for path in sorted(root.glob("*run_bundle.zip")):
                experiments.append(load_run_bundle_experiment(path))

    return experiments


def html_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or not len(df):
        return '<p class="muted">No rows.</p>'
    view = df.head(max_rows).copy()
    return view.to_html(index=False, escape=True, classes="dataframe")


def next_chart_id(prefix: str = "chart") -> str:
    global CHART_COUNTER
    CHART_COUNTER += 1
    return f"{prefix}-{CHART_COUNTER}"


def js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def chart_y_bounds(
    values: list[float],
    y_min: float | None,
    y_max: float | None,
) -> tuple[float | None, float | None]:
    if not values:
        return y_min, y_max
    lower_fixed = y_min is not None
    upper_fixed = y_max is not None
    data_min, data_max = min(values), max(values)
    if y_min is None:
        y_min = data_min
    if y_max is None:
        y_max = data_max
    if y_min == y_max:
        delta = max(abs(y_min) * 0.05, 0.05)
        y_min -= delta
        y_max += delta
    pad = (y_max - y_min) * 0.10
    if not lower_fixed:
        y_min -= pad
    if not upper_fixed:
        y_max += pad
    return y_min, y_max


def svg_category_line_chart(
    labels: list[str],
    series: list[tuple[str, dict[str, float]]],
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    compact: bool = False,
) -> str:
    series = [(name, values) for name, values in series if values]
    if not labels or not series:
        return '<p class="muted">No chart data.</p>'

    all_values = [float(value) for _, values in series for value in values.values() if value is not None]
    if not all_values:
        return '<p class="muted">No chart data.</p>'
    y_min, y_max = chart_y_bounds(all_values, y_min, y_max)

    chart_id = next_chart_id()
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    datasets = []
    for idx, (name, values) in enumerate(series):
        color = colors[idx % len(colors)]
        datasets.append(
            {
                "label": name,
                "data": [values.get(label) for label in labels],
                "borderColor": color,
                "backgroundColor": color,
                "pointBackgroundColor": color,
                "pointBorderColor": "#ffffff",
                "pointBorderWidth": 1.5,
                "pointRadius": 4 if not compact else 3,
                "pointHoverRadius": 6,
                "borderWidth": 2.5 if not compact else 2,
                "tension": 0.22,
                "spanGaps": True,
            }
        )
    class_name = "chart-panel small-chart" if compact else "chart-panel wide-chart"
    height = 230 if compact else 340
    return f"""
<div class="{class_name}"><canvas id="{chart_id}"></canvas></div>
<script>
(() => {{
  const ctx = document.getElementById({js_json(chart_id)});
  new Chart(ctx, {{
    type: 'line',
    data: {{ labels: {js_json(labels)}, datasets: {js_json(datasets)} }},
    options: makeLineOptions({{
      yLabel: {js_json(y_label)},
      yMin: {js_json(y_min)},
      yMax: {js_json(y_max)},
      compact: {str(compact).lower()},
      height: {height}
    }})
  }});
}})();
</script>
"""


def svg_bar_chart(rows: pd.DataFrame, label_col: str, value_col: str, y_label: str, max_value: float) -> str:
    if rows is None or not len(rows):
        return '<p class="muted">No chart data.</p>'

    labels = [str(value) for value in rows[label_col].tolist()]
    values = [float(value) for value in rows[value_col].tolist()]
    chart_id = next_chart_id()
    return f"""
<div class="chart-panel wide-chart"><canvas id="{chart_id}"></canvas></div>
<script>
(() => {{
  const ctx = document.getElementById({js_json(chart_id)});
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: {js_json(labels)},
      datasets: [{{
        label: {js_json(y_label)},
        data: {js_json(values)},
        backgroundColor: '#2563eb',
        borderColor: '#1d4ed8',
        borderWidth: 1,
        borderRadius: 4,
        maxBarThickness: 72
      }}]
    }},
    options: makeBarOptions({{
      yLabel: {js_json(y_label)},
      yMax: {js_json(max_value)}
    }})
  }});
}})();
</script>
"""


def probe_summary(exp: Experiment) -> pd.DataFrame:
    if "probe_summary" in exp.frames:
        df = exp.frames["probe_summary"].copy()
        if "rate" in df.columns and "probe_match_rate" not in df.columns:
            df["probe_match_rate"] = df["rate"]
        return df
    df = exp.frames.get("probe_evolution", pd.DataFrame())
    if not len(df) or not {"phase", "step", "match"}.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    work["match_bool"] = work["match"].map(normalize_bool)
    summary = (
        work.groupby(["phase", "step"], as_index=False)
        .agg(probe_matches=("match_bool", "sum"), probe_rows=("match_bool", "count"))
        .sort_values(["step", "phase"])
    )
    summary["probe_match_rate"] = summary["probe_matches"] / summary["probe_rows"]
    return summary


def generated_eval_frame(exp: Experiment) -> pd.DataFrame:
    frames = []
    for key in ["checkpoint_eval_summary", "generated_eval_summary"]:
        df = exp.frames.get(key, pd.DataFrame())
        if len(df):
            frames.append(df.copy())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "family" in combined.columns:
        combined["family"] = combined["family"].fillna("all").astype(str)
    dedupe_cols = [col for col in ["step", "family"] if col in combined.columns]
    if len(dedupe_cols) == 2:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
    return combined


def overview_table(experiments: list[Experiment]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        cfg = exp.run_config
        rows.append(
            {
                "experiment": exp.name,
                "score": exp.score_public,
                "source": exp.source,
                "target": cfg.get("train_target_format"),
                "lora_r": cfg.get("lora_r"),
                "lora_modules": "|".join(cfg.get("lora_target_modules", [])) if isinstance(cfg.get("lora_target_modules"), list) else cfg.get("lora_target_modules"),
                "max_new": cfg.get("max_new_tokens"),
                "train_rows": cfg.get("train_rows_actual", cfg.get("train_rows")),
            }
        )
    return pd.DataFrame(rows)


def test_reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"id": row_id, "family": value["family"], "gold": value["gold"]} for row_id, value in TEST_REFERENCE.items()]
    )


def sanity_predictions(exp: Experiment) -> pd.DataFrame:
    df = exp.frames.get("sanity_predictions_raw")
    if df is None or not len(df):
        df = exp.frames.get("sanity_predictions")
    if df is None or not len(df):
        return pd.DataFrame()

    work = df.copy()
    if "answer" not in work.columns or "id" not in work.columns:
        return pd.DataFrame()
    work["run"] = experiment_code(exp)
    work["experiment"] = exp.name
    work["score"] = exp.score_public
    work["family"] = work["id"].map(lambda row_id: TEST_REFERENCE.get(str(row_id), {}).get("family"))
    work["gold"] = work["id"].map(lambda row_id: TEST_REFERENCE.get(str(row_id), {}).get("gold"))
    work["match"] = work.apply(lambda row: normalize_answer(row.get("answer")) == normalize_answer(row.get("gold")), axis=1)
    cols = [
        "run",
        "experiment",
        "score",
        "id",
        "family",
        "gold",
        "answer",
        "match",
        "seconds",
        "generated_tokens",
        "hit_max_new_tokens",
        "raw_output",
    ]
    return work[[col for col in cols if col in work.columns]]


def sanity_comparison_table(experiments: list[Experiment]) -> pd.DataFrame:
    run_frames: list[tuple[str, pd.DataFrame]] = []
    for exp in experiments:
        frame = sanity_predictions(exp)
        if len(frame):
            run_frames.append((experiment_code(exp), frame))
    if not run_frames:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for row_id, reference in TEST_REFERENCE.items():
        row: dict[str, Any] = {
            "id": row_id,
            "family": reference["family"],
            "gold": reference["gold"],
        }
        match_values: dict[str, Any] = {}
        for label, frame in run_frames:
            subset = frame[frame["id"].astype(str) == row_id]
            if not len(subset):
                row[f"{label} answer"] = ""
                match_values[f"{label} match"] = ""
                continue
            first = subset.iloc[0]
            row[f"{label} answer"] = first.get("answer", "")
            match_values[f"{label} match"] = int(bool(first.get("match")))
        row.update(match_values)
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_code(exp: Experiment) -> str:
    text = f"{exp.name} {exp.method or ''}".lower()
    if "s4" in text or "attention_expand" in text:
        return "04-s4"
    if "raw_full" in text or "active_02" in text:
        return "02-raw-full"
    if "nemotron_lora_score_0_62" in text or exp.score_public == 0.62:
        return "00-raw-1024"
    if "smol" in text:
        return "smol"
    short = re.sub(r"[^a-z0-9]+", "-", exp.name.lower()).strip("-")
    return short[:18] or "run"


def point_label(exp: Experiment, step: Any = None) -> str:
    numeric_step = to_number(step)
    if numeric_step is None:
        return experiment_code(exp)
    return f"{experiment_code(exp)}-step-{int(numeric_step)}"


def run_legend_table(experiments: list[Experiment]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        cfg = exp.run_config
        modules = cfg.get("lora_target_modules")
        if isinstance(modules, list):
            modules_text = "/".join(str(item) for item in modules)
        else:
            modules_text = str(modules or "")
        rows.append(
            {
                "label": experiment_code(exp),
                "score": exp.score_public,
                "main": f"{cfg.get('train_target_format', '')}, r={cfg.get('lora_r', '')}, max_new={cfg.get('max_new_tokens', '')}, {modules_text}",
            }
        )
    return pd.DataFrame(rows)


def add_unique(labels: list[str], label: str) -> None:
    if label not in labels:
        labels.append(label)


def generated_eval_steps(exp: Experiment) -> list[int]:
    df = generated_eval_frame(exp)
    if not len(df) or "step" not in df.columns:
        return []
    if "family" in df.columns:
        df = df[df["family"].astype(str) == "all"].copy()
    steps = {
        int(step)
        for step in (to_number(value) for value in df["step"].tolist())
        if step is not None
    }
    return sorted(steps)


def value_at_step(df: pd.DataFrame, column: str, step: int) -> float | None:
    if column not in df.columns:
        return None
    values: list[float] = []
    for _, row in df.iterrows():
        row_step = to_number(row.get("step"))
        value = to_number(row.get(column))
        if row_step is not None and int(row_step) == step and value is not None:
            values.append(value)
    return values[-1] if values else None


def latest_value(df: pd.DataFrame, columns: list[str]) -> float | None:
    best_step = -1
    best_value: float | None = None
    for _, row in df.iterrows():
        row_step = to_number(row.get("step"))
        if row_step is None:
            continue
        for column in columns:
            value = to_number(row.get(column))
            if value is not None and int(row_step) >= best_step:
                best_step = int(row_step)
                best_value = value
    return best_value


def latest_row_value(df: pd.DataFrame, value_col: str) -> tuple[int, float] | None:
    if "step" not in df.columns or value_col not in df.columns:
        return None
    best: tuple[int, float] | None = None
    for _, row in df.iterrows():
        step = to_number(row.get("step"))
        value = to_number(row.get(value_col))
        if step is None or value is None:
            continue
        candidate = (int(step), value)
        if best is None or candidate[0] >= best[0]:
            best = candidate
    return best


def comparison_loss(experiments: list[Experiment]) -> tuple[list[str], list[tuple[str, dict[str, float]]]]:
    labels: list[str] = []
    train: dict[str, float] = {}
    evals: dict[str, float] = {}
    for exp in experiments:
        df = exp.frames.get("trainer_log", pd.DataFrame())
        if not len(df):
            continue
        checkpoint_steps = generated_eval_steps(exp)
        if len(checkpoint_steps) > 1:
            for step in checkpoint_steps:
                label = point_label(exp, step)
                loss = value_at_step(df, "loss", step)
                if loss is None:
                    loss = value_at_step(df, "train_loss", step)
                eval_loss = value_at_step(df, "eval_loss", step)
                if loss is None and eval_loss is None:
                    continue
                add_unique(labels, label)
                if loss is not None:
                    train[label] = loss
                if eval_loss is not None:
                    evals[label] = eval_loss
            continue

        label = experiment_code(exp)
        loss = latest_value(df, ["train_loss", "loss"])
        eval_loss = latest_value(df, ["eval_loss"])
        if loss is None and eval_loss is None:
            continue
        add_unique(labels, label)
        if loss is not None:
            train[label] = loss
        if eval_loss is not None:
            evals[label] = eval_loss
    return labels, [("train loss", train), ("eval loss", evals)]


def comparison_public_score(experiments: list[Experiment]) -> tuple[list[str], list[tuple[str, dict[str, float]]]]:
    labels: list[str] = []
    values: dict[str, float] = {}
    for exp in experiments:
        score = to_number(exp.score_public)
        if score is None:
            continue
        label = experiment_code(exp)
        add_unique(labels, label)
        values[label] = score
    return labels, [("public score", values)]


def comparison_probe(experiments: list[Experiment]) -> tuple[list[str], list[tuple[str, dict[str, float]]]]:
    labels: list[str] = []
    values: dict[str, float] = {}
    phase_priority = {"probe_final": 0, "after": 0, "final": 0, "eval": 1, "probe_log": 1, "before": 2, "probe_before": 2}
    for exp in experiments:
        summary = probe_summary(exp)
        if not len(summary):
            continue
        y_col = "probe_match_rate" if "probe_match_rate" in summary.columns else "rate"
        work = summary.copy()
        work["phase_priority"] = work["phase"].astype(str).map(phase_priority).fillna(5) if "phase" in work.columns else 5
        work = work.sort_values(["step", "phase_priority"]).drop_duplicates(subset=["step"], keep="first")

        checkpoint_steps = generated_eval_steps(exp)
        if len(checkpoint_steps) > 1:
            wanted_steps = checkpoint_steps
            include_step_in_label = True
        else:
            latest = latest_row_value(work, y_col)
            wanted_steps = [latest[0]] if latest else []
            include_step_in_label = False

        for step in wanted_steps:
            subset = work[pd.to_numeric(work["step"], errors="coerce").astype("Int64") == step]
            if not len(subset):
                continue
            value = to_number(subset.iloc[-1].get(y_col))
            if value is None:
                continue
            label = point_label(exp, step) if include_step_in_label else experiment_code(exp)
            add_unique(labels, label)
            values[label] = value
    return labels, [("probe accuracy", values)]


def comparison_generated_eval(experiments: list[Experiment]) -> tuple[list[str], list[tuple[str, dict[str, float]]], pd.DataFrame]:
    labels: list[str] = []
    values: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str]] = set()
    for exp in experiments:
        df = generated_eval_frame(exp)
        if not len(df):
            continue
        if "family" in df.columns:
            df = df[df["family"].astype(str) == "all"].copy()
        if "step" in df.columns:
            df = df.assign(_step_sort=pd.to_numeric(df["step"], errors="coerce")).sort_values("_step_sort")
        use_step_label = len(generated_eval_steps(exp)) > 1
        for _, row in df.iterrows():
            step = to_number(row.get("step"))
            accuracy = to_number(row.get("accuracy"))
            if step is None or accuracy is None:
                continue
            label = point_label(exp, step) if use_step_label else experiment_code(exp)
            add_unique(labels, label)
            values[label] = accuracy * 100.0
            checkpoint = str(row.get("checkpoint", ""))
            row_key = (label, checkpoint)
            if row_key not in seen_rows:
                rows.append({"run": label, "checkpoint": row.get("checkpoint"), "generated_eval_accuracy_pct": accuracy * 100.0})
                seen_rows.add(row_key)
    return labels, [("generated eval accuracy (%)", values)], pd.DataFrame(rows)


def comparison_generated_eval_by_family(experiments: list[Experiment]) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    labels: list[str] = []
    families: list[str] = []
    family_values: dict[str, dict[str, float]] = {}
    for exp in experiments:
        df = generated_eval_frame(exp)
        if not len(df) or "family" not in df.columns:
            continue
        df = df[df["family"].astype(str) != "all"].copy()
        if "step" in df.columns:
            df = df.assign(_step_sort=pd.to_numeric(df["step"], errors="coerce")).sort_values("_step_sort")
        use_step_label = len(generated_eval_steps(exp)) > 1
        for _, row in df.iterrows():
            step = to_number(row.get("step"))
            accuracy = to_number(row.get("accuracy"))
            family = str(row.get("family", "")).strip()
            if step is None or accuracy is None or not family:
                continue
            label = point_label(exp, step) if use_step_label else experiment_code(exp)
            add_unique(labels, label)
            add_unique(families, family)
            family_values.setdefault(family, {})[label] = accuracy * 100.0
    families = sorted(families, key=lambda family: (FAMILY_ORDER.index(family) if family in FAMILY_ORDER else 999, family))
    return labels, families, family_values


def generated_eval_family_charts(labels: list[str], families: list[str], family_values: dict[str, dict[str, float]]) -> str:
    if not labels or not families:
        return '<p class="muted">No family chart data.</p>'
    parts = ["<div class='small-chart-grid'>"]
    for family in families:
        parts.append("<div class='small-chart-card'>")
        parts.append(f"<h3>{escape(family)}</h3>")
        parts.append(
            svg_category_line_chart(
                labels,
                [("accuracy (%)", family_values.get(family, {}))],
                "accuracy (%)",
                compact=True,
            )
        )
        parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)


def sanity_score_table(experiments: list[Experiment]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        sanity = sanity_predictions(exp)
        if not len(sanity):
            continue
        rows.append(
            {
                "run": experiment_code(exp),
                "score": exp.score_public,
                "correct": int(sanity["match"].sum()),
                "total": int(len(sanity)),
            }
        )
    return pd.DataFrame(rows)


def render_dashboard(experiments: list[Experiment], output_root: Path) -> str:
    overview = overview_table(experiments)
    sanity_scores = sanity_score_table(experiments)
    score_labels, score_series = comparison_public_score(experiments)
    loss_labels, loss_series = comparison_loss(experiments)
    probe_labels, probe_series = comparison_probe(experiments)
    generated_labels, generated_series, generated_table = comparison_generated_eval(experiments)
    family_labels, generated_families, generated_family_values = comparison_generated_eval_by_family(experiments)
    sanity_rows = sanity_comparison_table(experiments)
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Nemotron Experiment Dashboard</title>",
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js'></script>",
        """
<style>
body { font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #172033; background: #f7f8fb; }
h1, h2, h3 { color: #101827; }
section { background: white; border: 1px solid #d9dee8; border-radius: 8px; padding: 18px; margin: 18px 0; }
.muted { color: #667085; }
table.dataframe { border-collapse: collapse; font-size: 13px; width: 100%; margin: 8px 0 16px; }
table.dataframe th, table.dataframe td { border: 1px solid #d9dee8; padding: 6px 8px; text-align: left; vertical-align: top; }
table.dataframe th { background: #eef2f7; }
.chart-panel { position: relative; width: 100%; background: #fbfcfe; border: 1px solid #e5e7eb; border-radius: 6px; margin: 8px 0 16px; padding: 14px 14px 8px; box-sizing: border-box; }
.wide-chart { height: 360px; }
.small-chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 12px; margin-top: 10px; }
.small-chart-card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; background: #fbfcfe; }
.small-chart-card h3 { margin: 0 0 6px; font-size: 14px; }
.small-chart { height: 250px; margin: 0; border: 0; background: transparent; padding: 2px; }
</style>
<script>
function compactNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
  const number = Number(value);
  if (Math.abs(number) >= 100) return number.toFixed(0);
  if (Math.abs(number) >= 10) return number.toFixed(1);
  return number.toFixed(3).replace(/0+$/, '').replace(/[.]$/, '');
}

function commonScales(config) {
  return {
    x: {
      grid: { display: false },
      ticks: {
        color: '#475467',
        maxRotation: config.compact ? 35 : 28,
        minRotation: config.compact ? 35 : 0,
        autoSkip: false,
        font: { size: config.compact ? 10 : 11 }
      }
    },
    y: {
      min: config.yMin,
      max: config.yMax,
      title: { display: true, text: config.yLabel, color: '#475467' },
      grid: { color: '#e5e7eb' },
      border: { color: '#98a2b3' },
      ticks: {
        color: '#475467',
        precision: 0,
        callback: compactNumber
      }
    }
  };
}

function makeLineOptions(config) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: { display: !config.compact, position: 'top', align: 'start', labels: { boxWidth: 12, usePointStyle: true } },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${compactNumber(ctx.parsed.y)}`
        }
      }
    },
    scales: commonScales(config)
  };
}

function makeBarOptions(config) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${config.yLabel}: ${compactNumber(ctx.parsed.y)}`
        }
      }
    },
    scales: commonScales({ yLabel: config.yLabel, yMin: 0, yMax: config.yMax, compact: false })
  };
}
</script>
""",
        "</head><body>",
        "<h1>Nemotron Experiment Dashboard</h1>",
        f"<p class='muted'>Source root: {escape(str(output_root))}</p>",
        "<section><h2>Overview</h2>",
        html_table(overview, max_rows=100),
        "<h3>Run Legend</h3>",
        html_table(run_legend_table(experiments), max_rows=100),
        "</section>",
        "<section><h2>Public Score By Submission</h2>",
        svg_category_line_chart(score_labels, score_series, "public score"),
        "</section>",
        "<section><h2>Public Test Sanity Reference</h2>",
        "<p class='muted'>These are the 3 public sanity prompt answers used in our earlier notebooks. They are for local comparison only; Kaggle scoring still uses the adapter submission flow.</p>",
        html_table(test_reference_table(), max_rows=10),
        "</section>",
        "<section><h2>Saved Test Sanity Responses</h2>",
        svg_bar_chart(sanity_scores, "run", "correct", "correct out of 3", max_value=3),
        html_table(sanity_rows, max_rows=100),
        "</section>",
        "<section><h2>Train And Eval Loss By Run</h2>",
        svg_category_line_chart(loss_labels, loss_series, "loss"),
        "</section>",
        "<section><h2>Probe Accuracy By Run</h2>",
        svg_category_line_chart(probe_labels, probe_series, "probe accuracy", y_min=0, y_max=1),
        "</section>",
        "<section><h2>Generated Eval Accuracy By Run</h2>",
        svg_category_line_chart(generated_labels, generated_series, "generated eval accuracy (%)"),
        generated_eval_family_charts(family_labels, generated_families, generated_family_values),
        html_table(generated_table, max_rows=120),
        "</section>",
    ]
    body.append("</body></html>")
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    experiments = discover_experiments(args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_dashboard(experiments, args.output_root), encoding="utf-8")
    print(f"experiments: {len(experiments)}")
    print(f"wrote: {args.report}")


if __name__ == "__main__":
    main()
