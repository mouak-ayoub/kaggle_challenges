from dataclasses import dataclass
from typing import Callable, Literal

import optuna


@dataclass(frozen=True)
class OptunaRunConfig:
    """Generic Optuna run settings reusable across notebook scenarios."""

    seed: int = 7
    n_trials: int = 20
    timeout_seconds: int | None = 300
    direction: Literal["maximize", "minimize"] = "maximize"


def create_tpe_study(cfg: OptunaRunConfig) -> optuna.Study:
    """Create one TPE-based Optuna study from a small shared config."""

    return optuna.create_study(
        direction=cfg.direction,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
    )


def run_optuna_study(
    objective: Callable[[optuna.trial.Trial], float],
    cfg: OptunaRunConfig,
    baseline_params: dict[str, object] | None = None,
) -> optuna.Study:
    """Run one Optuna study and optionally enqueue a baseline trial first."""

    study = create_tpe_study(cfg)
    if baseline_params:
        study.enqueue_trial(baseline_params)
    study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout_seconds)
    return study


def top_trials(study: optuna.Study, top_k: int = 5) -> list[optuna.trial.FrozenTrial]:
    """Return the strongest finished trials in descending score order."""

    finished = [trial for trial in study.trials if trial.value is not None]
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    return sorted(finished, key=lambda trial: float(trial.value), reverse=reverse)[:top_k]
