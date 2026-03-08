# stdlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
    default_risk_estimation_names,
)
from autoprognosis.explorers.risk_estimation_combos import RiskEnsembleSeeker
from autoprognosis.hooks import DefaultHooks, Hooks
from autoprognosis.studies._base import PATIENCE, SCORE_THRESHOLD, Study
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.serialization import dataframe_hash
from autoprognosis.utils.tester import evaluate_survival_estimator


class RiskEstimationStudy(Study):
    """
    Core logic for risk estimation studies.

    A study automatically handles imputation, preprocessing and model selection for a certain dataset.
    The output is an optimal model architecture, selected by the AutoML logic.

    Args:
        dataset: DataFrame.
            The dataset to analyze.
        target: str.
            The target column in the dataset.
        time_to_event: str.
            The time_to_event column in the dataset.
        num_iter: int.
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "risk_estimators" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
        num_study_iter: int.
            The number of study iterations. This is the limit for the outer optimization loop. After each outer loop, an intermediary model is cached and can be used by another process, while the outer loop continues to improve the result.
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "risk_estimators" list.
        study_name: str.
            The name of the study, to be used in the caches.
        feature_scaling: list.
            Plugin search pool to use in the pipeline for scaling. Defaults to : ['maxabs_scaler', 'scaler', 'feature_normalizer', 'normal_transform', 'uniform_transform', 'nop', 'minmax_scaler']
            Available plugins, retrieved using `Preprocessors(category="feature_scaling").list_available()`:
                - 'maxabs_scaler'
                - 'scaler'
                - 'feature_normalizer'
                - 'normal_transform'
                - 'uniform_transform'
                - 'nop' # empty operation
                - 'minmax_scaler'
        feature_selection: list.
            Plugin search pool to use in the pipeline for feature selection. Defaults ["nop", "variance_threshold", "pca", "fast_ica"]
            Available plugins, retrieved using `Preprocessors(category="dimensionality_reduction").list_available()`:
                - 'feature_agglomeration'
                - 'fast_ica'
                - 'variance_threshold'
                - 'gauss_projection'
                - 'pca'
                - 'nop' # no operation
        imputers: list.
            Plugin search pool to use in the pipeline for imputation. Defaults to ["mean", "ice", "missforest", "hyperimpute"].
            Available plugins, retrieved using `Imputers().list_available()`:
                - 'sinkhorn'
                - 'EM'
                - 'mice'
                - 'ice'
                - 'hyperimpute'
                - 'most_frequent'
                - 'median'
                - 'missforest'
                - 'softimpute'
                - 'nop'
                - 'mean'
                - 'gain'
        risk_estimators: list.
            Plugin search pool to use in the pipeline for risk estimation. Defaults to ["survival_xgboost", "loglogistic_aft", "deephit", "cox_ph", "weibull_aft", "lognormal_aft", "coxnet"]
            Available plugins:
             - 'survival_xgboost'
             - 'loglogistic_aft'
             - 'deephit'
             - 'cox_ph'
             - 'weibull_aft'
             - 'lognormal_aft'
             - 'coxnet'
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
        workspace: Path.
            Where to store the output model.
        score_threshold: float.
            The minimum metric score for a candidate.
        random_state: int
            Random seed
        sample_for_search: bool
            Subsample the evaluation dataset in the search pipeline. Improves the speed of the search.
        max_search_sample_size: int
            Subsample size for the evaluation dataset, if `sample` is True.
    Example:
        >>> import numpy as np
        >>> from pycox import datasets
        >>> from autoprognosis.studies.risk_estimation import RiskEstimationStudy
        >>> from autoprognosis.utils.serialization import load_model_from_file
        >>> from autoprognosis.utils.tester import evaluate_survival_estimator
        >>>
        >>> df = datasets.gbsg.read_df()
        >>> df = df[df["duration"] > 0]
        >>>
        >>> X = df.drop(columns = ["duration"])
        >>> T = df["duration"]
        >>> Y = df["event"]
        >>>
        >>> eval_time_horizons = np.linspace(T.min(), T.max(), 5)[1:-1]
        >>>
        >>> study_name = "example_risks"
        >>> study = RiskEstimationStudy(
        >>>     study_name=study_name,
        >>>     dataset=df,
        >>>     target="event",
        >>>     time_to_event="duration",
        >>>     time_horizons=eval_time_horizons,
        >>> )
        >>>
        >>> model = study.fit()
        >>> # Predict using the model
        >>> model.predict(X, eval_time_horizons)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        time_to_event: str,
        time_horizons: List[int],
        num_iter: int = 20,
        num_study_iter: int = 5,
        num_ensemble_iter: int = 15,
        timeout: int = 360,
        study_name: Optional[str] = None,
        workspace: Path = Path("tmp"),
        risk_estimators: List[str] = default_risk_estimation_names,
        imputers: List[str] = ["ice"],
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        hooks: Hooks = DefaultHooks(),
        score_threshold: float = SCORE_THRESHOLD,
        nan_placeholder: Any = None,
        group_id: Optional[str] = None,
        random_state: int = 0,
        sample_for_search: bool = True,
        max_search_sample_size: int = 10000,
        ensemble_size: int = 3,
        n_folds_cv: int = 5,
    ) -> None:
        enable_reproducible_results(random_state)

        if nan_placeholder is not None:
            dataset = dataset.replace(nan_placeholder, np.nan)

        if dataset.isnull().values.any():
            if len(imputers) == 0:
                raise RuntimeError("Please provide at least one imputation method")
        else:
            imputers = []

        self.time_horizons = time_horizons

        drop_cols = [target, time_to_event]
        self.group_ids = None
        if group_id is not None:
            drop_cols.append(group_id)
            self.group_ids = dataset[group_id]

        self.Y = dataset[target]
        self.T = dataset[time_to_event]
        self.X = dataset.drop(columns=drop_cols)

        if sample_for_search:
            sample_size = min(len(self.Y), max_search_sample_size)
            counts = self.Y.value_counts().to_dict()
            weights = self.Y.apply(lambda s: counts[s])

            self.search_Y = self.Y.sample(
                sample_size, random_state=random_state, weights=weights
            )
            self.search_X = self.X.loc[self.search_Y.index].copy()
            self.search_T = self.T.loc[self.search_Y.index].copy()
            self.search_group_ids = None
            if self.group_ids:
                self.search_group_ids = self.group_ids.loc[self.search_Y.index].copy()
        else:
            self.search_X = self.X
            self.search_Y = self.Y
            self.search_T = self.T
            self.search_group_ids = self.group_ids

        internal_name = dataframe_hash(dataset)
        resolved_name = study_name if study_name is not None else internal_name

        self.random_state = random_state
        self.n_folds_cv = n_folds_cv

        study_config = {
            "num_iter": num_iter,
            "num_study_iter": num_study_iter,
            "num_ensemble_iter": num_ensemble_iter,
            "timeout": timeout,
            "risk_estimators": risk_estimators,
            "feature_scaling": feature_scaling,
            "feature_selection": feature_selection,
            "imputers": imputers,
            "score_threshold": score_threshold,
            "random_state": random_state,
            "sample_for_search": sample_for_search,
            "max_search_sample_size": max_search_sample_size,
            "ensemble_size": ensemble_size,
            "n_folds_cv": n_folds_cv,
            "time_horizons": [float(h) for h in time_horizons],
        }

        super().__init__(
            study_name=resolved_name,
            output_folder=Path(workspace).resolve() / resolved_name,
            score_threshold=score_threshold,
            num_study_iter=num_study_iter,
            metric="c_index - brier_score",
            hooks=hooks,
            study_config=study_config,
        )

        self.seeker = RiskEnsembleSeeker(
            internal_name,
            time_horizons,
            num_iter=num_iter,
            num_ensemble_iter=num_ensemble_iter,
            timeout=timeout,
            estimators=risk_estimators,
            feature_scaling=feature_scaling,
            feature_selection=feature_selection,
            imputers=imputers,
            hooks=hooks,
            random_state=random_state,
            n_folds_cv=n_folds_cv,
        )
        self._last_iteration = 0

    def _search(self, iteration: int) -> Any:
        self._last_iteration = iteration
        return self.seeker.search(
            self.search_X,
            self.search_T,
            self.search_Y,
            skip_recap=(iteration > 0),
            group_ids=self.search_group_ids,
        )

    def _evaluate(self, model: Any) -> Tuple[float, Dict]:
        metrics = evaluate_survival_estimator(
            model,
            self.search_X,
            self.search_T,
            self.search_Y,
            self.time_horizons,
            group_ids=self.search_group_ids,
            n_folds=self.n_folds_cv,
        )
        score = metrics["raw"]["c_index"][0] - metrics["raw"]["brier_score"][0]
        return score, metrics

    def _fit_model(self, model: Any) -> None:
        model.fit(self.X, self.T, self.Y)

    def _get_task_type(self) -> str:
        return "risk_estimation"

    def _get_heartbeat_topic(self) -> str:
        return "risk_estimation_study"
