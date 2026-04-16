import numpy as np
from skimage.segmentation import active_contour

from ..core.config import SnakeConfig
from ..core.results import SnakeEvolutionResult


def clip_snake_to_bounds(snake: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Keep contour points inside the image bounds."""

    h, w = shape
    snake_clipped = snake.copy().astype(np.float64)
    snake_clipped[:, 0] = np.clip(snake_clipped[:, 0], 0, h - 1)
    snake_clipped[:, 1] = np.clip(snake_clipped[:, 1], 0, w - 1)
    return snake_clipped


def evolve_active_contour(
    image: np.ndarray,
    init_snake: np.ndarray,
    snake_cfg: SnakeConfig,
) -> SnakeEvolutionResult:
    """Run the active contour once or in debug chunks."""

    current_snake = init_snake.copy().astype(np.float64)
    debug_snakes: list[np.ndarray] = []
    debug_iteration_counts: list[int] = []

    if snake_cfg.debug_step is None or snake_cfg.debug_step <= 0:
        final_snake = active_contour(
            image,
            current_snake,
            alpha=snake_cfg.alpha,
            beta=snake_cfg.beta,
            gamma=snake_cfg.gamma,
            w_line=snake_cfg.w_line,
            w_edge=snake_cfg.w_edge,
            max_px_move=snake_cfg.max_px_move,
            max_num_iter=snake_cfg.max_num_iter,
            convergence=snake_cfg.convergence,
            boundary_condition=snake_cfg.boundary_condition,
        )
        final_snake = clip_snake_to_bounds(final_snake, image.shape)
        return SnakeEvolutionResult(
            initial_snake=init_snake,
            final_snake=final_snake,
            debug_snakes=[],
            debug_iteration_counts=[],
        )

    remaining = int(snake_cfg.max_num_iter)
    total_done = 0

    while remaining > 0:
        chunk_iter = int(min(snake_cfg.debug_step, remaining))
        current_snake = active_contour(
            image,
            current_snake,
            alpha=snake_cfg.alpha,
            beta=snake_cfg.beta,
            gamma=snake_cfg.gamma,
            w_line=snake_cfg.w_line,
            w_edge=snake_cfg.w_edge,
            max_px_move=snake_cfg.max_px_move,
            max_num_iter=chunk_iter,
            convergence=snake_cfg.convergence,
            boundary_condition=snake_cfg.boundary_condition,
        )
        current_snake = clip_snake_to_bounds(current_snake, image.shape)
        total_done += chunk_iter
        remaining -= chunk_iter
        debug_snakes.append(current_snake.copy())
        debug_iteration_counts.append(total_done)

    return SnakeEvolutionResult(
        initial_snake=init_snake,
        final_snake=current_snake,
        debug_snakes=debug_snakes,
        debug_iteration_counts=debug_iteration_counts,
    )
