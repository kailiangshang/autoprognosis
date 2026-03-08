# stdlib
from typing import Any, Callable, List, Tuple

# third party
from pydantic import validate_arguments

# autoprognosis absolute
from autoprognosis.explorers.core.optimizers.bayesian import BayesianOptimizer
from autoprognosis.explorers.core.optimizers.hyperband import HyperbandOptimizer


class Optimizer:
    """Wrapper for hyperparameter optimization backends.

    Delegates to the chosen backend for hyperparameter search:
        - "bayesian": Optuna-based TPE (Tree-structured Parzen Estimator) optimization.
        - "hyperband": Hyperband successive halving algorithm (via Optuna).

    Args:
        study_name: str
            Unique study identifier, used as the Optuna study name.
        estimator: Any
            The estimator whose hyperparameters are being optimized.
        evaluation_cbk: Callable
            Callback that evaluates hyperparameters and returns a score.
        optimizer_type: str
            "bayesian" or "hyperband".
        n_trials: int
            Maximum number of Optuna trials (bayesian) or configurations (hyperband).
        timeout: int
            Maximum seconds for the search.
        eta: int
            Hyperband downsampling rate (only used when optimizer_type="hyperband").
        random_state: int
            Random seed for reproducibility.
    """
    def __init__(
        self,
        study_name: str,
        estimator: Any,
        evaluation_cbk: Callable,
        optimizer_type: str = "bayesian",
        n_trials: int = 50,  # bayesian: number of trials
        timeout: int = 60,  # bayesian: timeout per search
        eta: int = 3,  # hyperband: defines configuration downsampling rate (default = 3)
        random_state: int = 0,
    ):
        if optimizer_type not in ["bayesian", "hyperband"]:
            raise RuntimeError(f"Invalid optimizer type {optimizer_type}")

        if optimizer_type == "bayesian":
            self.optimizer = BayesianOptimizer(
                study_name=study_name,
                estimator=estimator,
                evaluation_cbk=evaluation_cbk,
                n_trials=n_trials,
                timeout=timeout,
                random_state=random_state,
            )
        elif optimizer_type == "hyperband":
            self.optimizer = HyperbandOptimizer(
                study_name=study_name,
                estimator=estimator,
                evaluation_cbk=evaluation_cbk,
                max_iter=n_trials,
                eta=eta,
                random_state=random_state,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
    ) -> Tuple[List[float], List[dict]]:
        return self.optimizer.evaluate()


class EnsembleOptimizer:
    """Wrapper for ensemble weight optimization backends.

    Optimizes the blending weights of an ensemble of models using Optuna.
    Supports the same backends as Optimizer ("bayesian" / "hyperband").

    Args:
        study_name: str
            Unique study identifier for the Optuna study.
        ensemble_len: int
            Number of base models in the ensemble.
        evaluation_cbk: Callable
            Callback that takes a weight vector and returns a score.
        optimizer_type: str
            "bayesian" or "hyperband".
        n_trials: int
            Maximum number of Optuna trials.
        timeout: int
            Maximum seconds for the search.
        max_iter: int
            Hyperband max iterations per configuration.
        eta: int
            Hyperband downsampling rate.
        skip_recap: bool
            If True, skip enqueuing initial one-hot weight trials.
        random_state: int
            Random seed for reproducibility.
    """
    def __init__(
        self,
        study_name: str,
        ensemble_len: int,
        evaluation_cbk: Callable,
        optimizer_type: str = "bayesian",
        n_trials: int = 50,  # bayesian: number of trials
        timeout: int = 60,  # bayesian: timeout per search
        max_iter: int = 27,  # hyperband: maximum iterations per configuration
        eta: int = 3,  # hyperband: defines configuration downsampling rate (default = 3)
        skip_recap: bool = False,
        random_state: int = 0,
    ):
        if optimizer_type not in ["bayesian", "hyperband"]:
            raise RuntimeError(f"Invalid optimizer type {optimizer_type}")

        if optimizer_type == "bayesian":
            self.optimizer = BayesianOptimizer(
                study_name=study_name,
                ensemble_len=ensemble_len,
                evaluation_cbk=evaluation_cbk,
                n_trials=n_trials,
                timeout=timeout,
                skip_recap=skip_recap,
                random_state=random_state,
            )
        elif optimizer_type == "hyperband":
            self.optimizer = HyperbandOptimizer(
                study_name=study_name,
                ensemble_len=ensemble_len,
                evaluation_cbk=evaluation_cbk,
                max_iter=max_iter,
                eta=eta,
                random_state=random_state,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
    ) -> Tuple[float, dict]:
        return self.optimizer.evaluate_ensemble()
