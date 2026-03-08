# stdlib
import json
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import autoprognosis.logger as log

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.hooks import Hooks
from autoprognosis.report import StudyReport
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file

PATIENCE = 10
SCORE_THRESHOLD = 0.65


class Study(metaclass=ABCMeta):
    """Base class for AutoPrognosis studies using the Template Method pattern.

    Subclasses must implement:
        - _search(iteration) -> model candidate
        - _evaluate(model) -> (score, metrics_dict)
        - _fit_model(model) -> None
        - _get_task_type() -> str
        - _get_heartbeat_topic() -> str

    Common logic (run loop, progress save/load, reporting) lives here.
    """

    PATIENCE: int = PATIENCE
    SCORE_THRESHOLD: float = SCORE_THRESHOLD

    def __init__(
        self,
        study_name: str,
        output_folder: Path,
        score_threshold: float,
        num_study_iter: int,
        metric: str,
        hooks: Hooks,
        study_config: Dict[str, Any],
    ) -> None:
        self.study_name = study_name
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_folder / "model.p"

        self.score_threshold = score_threshold
        self.num_study_iter = num_study_iter
        self.metric = metric
        self.hooks = hooks

        # Report tracking state
        self._search_history: list = []
        self._start_time: float = 0
        self._end_time: float = 0
        self._best_metrics: dict = {}
        self._best_metrics_raw: dict = {}
        self._best_score: float = -1
        self._best_model_name: Optional[str] = None
        self._study_config = study_config

        log.info(f"Study '{self.study_name}' workspace: {self.output_folder}")

    # --- Abstract hooks for subclasses ---

    @abstractmethod
    def _search(self, iteration: int) -> Any:
        """Execute one search iteration and return a candidate model."""
        ...

    @abstractmethod
    def _evaluate(self, model: Any) -> Tuple[float, Dict]:
        """Evaluate a model. Returns (primary_score, full_metrics_dict).

        The metrics dict must have "raw" and "str" keys in the standard format:
            {"raw": {metric_name: (mean, std)}, "str": {metric_name: "mean+/-std"}}
        """
        ...

    @abstractmethod
    def _fit_model(self, model: Any) -> None:
        """Fit the model on the full dataset (task-specific)."""
        ...

    @abstractmethod
    def _get_task_type(self) -> str:
        """Return the task type string for reporting."""
        ...

    @abstractmethod
    def _get_heartbeat_topic(self) -> str:
        """Return the heartbeat topic string."""
        ...

    # --- Common implementations ---

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled(f"{self._get_task_type()} study search cancelled")

    def _load_progress(self) -> Tuple[float, Any]:
        """Load a previously saved model and evaluate it."""
        self._should_continue()

        if not self.output_file.is_file():
            return -1, None

        try:
            log.info("Evaluating previously saved model")
            start = time.time()
            best_model = load_model_from_file(self.output_file)
            score, metrics = self._evaluate(best_model)

            eval_metrics = {}
            for m in metrics["raw"]:
                eval_metrics[m] = metrics["raw"][m][0]
                eval_metrics[f"{m}_str"] = metrics["str"][m]

            self.hooks.heartbeat(
                topic=self._get_heartbeat_topic(),
                subtopic="candidate",
                event_type="candidate",
                name=best_model.name(),
                duration=time.time() - start,
                score=score,
                **eval_metrics,
            )
            log.info(f"Previous best score: {score:.4f}")
            return score, best_model
        except Exception as e:
            log.error(f"Failed to load previous model: {e}")
            return -1, None

    def _save_progress(self, model: Any) -> None:
        """Save model and metadata to workspace."""
        self._should_continue()

        if self.output_file:
            save_model_to_file(self.output_file, model)

        # Save config metadata alongside model
        config_path = self.output_folder / "config.json"
        try:
            meta = {
                "study_name": self.study_name,
                "task_type": self._get_task_type(),
                "best_score": self._best_score,
                **self._study_config,
            }
            config_path.write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass  # metadata save is best-effort

    def _record_iteration(
        self,
        iteration: int,
        model: Any,
        score: float,
        duration: float,
        is_best: bool,
    ) -> None:
        self._search_history.append({
            "iteration": iteration,
            "model_name": model.name(),
            "score": score,
            "duration": duration,
            "is_best": is_best,
        })

    def _send_heartbeat(
        self, model: Any, score: float, metrics: Dict, duration: float
    ) -> None:
        eval_metrics = {}
        for m in metrics["raw"]:
            eval_metrics[m] = metrics["raw"][m][0]
            eval_metrics[f"{m}_str"] = metrics["str"][m]

        self.hooks.heartbeat(
            topic=self._get_heartbeat_topic(),
            subtopic="candidate",
            event_type="candidate",
            name=model.name(),
            duration=duration,
            score=score,
            **eval_metrics,
        )

    def run(self) -> Any:
        """Run the study. Returns the optimal model architecture (not fitted), or None."""
        self._should_continue()
        self._start_time = time.time()
        self._search_history = []

        best_score, best_model = self._load_progress()
        patience = 0

        for it in range(self.num_study_iter):
            self._should_continue()
            start = time.time()

            current_model = self._search(it)
            score, metrics = self._evaluate(current_model)
            iter_duration = time.time() - start

            self._send_heartbeat(current_model, score, metrics, iter_duration)

            if score < self.score_threshold:
                log.info(
                    f"Score {score:.4f} below threshold {self.score_threshold}, keep searching"
                )
                self._record_iteration(it + 1, current_model, score, iter_duration, False)
                continue

            if best_score >= score:
                log.info(
                    f"Score not improved: {score:.4f} <= {best_score:.4f} "
                    f"(patience {patience + 1}/{self.PATIENCE})"
                )
                patience += 1
                self._record_iteration(it + 1, current_model, score, iter_duration, False)

                if patience > self.PATIENCE:
                    log.info(
                        f"Early stopping: no improvement for {self.PATIENCE} iterations"
                    )
                    break
                continue

            patience = 0
            best_score = score
            best_model = current_model

            self._best_metrics = {m: metrics["str"][m] for m in metrics["str"]}
            self._best_metrics_raw = {m: metrics["raw"][m] for m in metrics["raw"]}

            log.info(f"New best model: {best_model.name()} score={best_score:.4f}")

            self._record_iteration(it + 1, current_model, score, iter_duration, True)
            self._save_progress(best_model)

        self._end_time = time.time()
        self._best_score = best_score
        self.hooks.finish()

        if best_score < self.score_threshold:
            log.warning(
                f"No model found above threshold {self.score_threshold}. Returning None"
            )
            self._best_model_name = None
            return None

        self._best_model_name = best_model.name()
        return best_model

    def fit(self) -> Any:
        """Run the study and train the model. Returns the fitted model, or None."""
        model = self.run()
        if model is not None:
            self._fit_model(model)
        return model

    def report(self) -> StudyReport:
        """Generate a study report. Call after fit() or run().

        Prints a console summary and saves an HTML report to the workspace.

        Returns:
            StudyReport with all study results and metadata.
        """
        success = self._best_score >= self.score_threshold

        rpt = StudyReport(
            study_name=self.study_name,
            task_type=self._get_task_type(),
            metric=self.metric,
            best_score=self._best_score,
            best_model_name=self._best_model_name,
            best_metrics=self._best_metrics,
            best_metrics_raw=self._best_metrics_raw,
            search_history=self._search_history,
            study_config=self._study_config,
            workspace=str(self.output_folder),
            total_duration_seconds=(
                self._end_time - self._start_time if self._end_time else 0
            ),
            success=success,
        )

        print(rpt.to_console())
        rpt.save()
        return rpt
