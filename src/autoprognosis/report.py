# stdlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StudyReport:
    """Captures results and metadata from a completed AutoPrognosis study.

    Attributes:
        study_name: Name of the study.
        task_type: "classification", "regression", or "risk_estimation".
        metric: Primary optimization metric name.
        best_score: Best score achieved on the primary metric.
        best_model_name: Name/description of the best model pipeline.
        best_metrics: Formatted metric strings, e.g. {"aucroc": "0.95 +/- 0.01"}.
        best_metrics_raw: Raw metric values, e.g. {"aucroc": (0.95, 0.01)}.
        search_history: List of per-iteration records.
        study_config: Study configuration parameters.
        workspace: Absolute path to the study workspace directory.
        total_duration_seconds: Total wall-clock time for the study.
        success: Whether the study found a model above the score threshold.
    """

    study_name: str
    task_type: str
    metric: str
    best_score: float
    best_model_name: Optional[str] = None
    best_metrics: Dict[str, str] = field(default_factory=dict)
    best_metrics_raw: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    study_config: Dict[str, Any] = field(default_factory=dict)
    workspace: str = ""
    total_duration_seconds: float = 0.0
    success: bool = False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m {secs:.0f}s"
        hours = int(minutes // 60)
        mins = minutes % 60
        return f"{hours}h {mins}m {secs:.0f}s"

    def to_console(self) -> str:
        """Generate a formatted text summary for console output."""
        sep = "=" * 64
        thin = "-" * 64
        lines = [
            sep,
            f"  AutoPrognosis Study Report: {self.study_name}",
            sep,
            f"  Status:     {'SUCCESS' if self.success else 'FAILED'}",
            f"  Task:       {self.task_type}",
            f"  Metric:     {self.metric}",
            f"  Duration:   {self._format_duration(self.total_duration_seconds)}",
            f"  Workspace:  {self.workspace}",
            thin,
        ]

        if self.best_model_name:
            lines.append(f"  Best Model: {self.best_model_name}")
        lines.append(f"  Best Score: {self.best_score:.4f}")
        lines.append(thin)

        if self.best_metrics:
            lines.append("  All Metrics:")
            max_key_len = max(len(k) for k in self.best_metrics)
            for metric_name, metric_str in self.best_metrics.items():
                lines.append(f"    {metric_name:<{max_key_len}}  {metric_str}")
            lines.append(thin)

        if self.search_history:
            lines.append("  Search History:")
            for entry in self.search_history:
                it = entry.get("iteration", "?")
                score = entry.get("score", 0)
                dur = self._format_duration(entry.get("duration", 0))
                name = entry.get("model_name", "unknown")
                marker = "  [best]" if entry.get("is_best", False) else ""
                lines.append(
                    f"    Iter {it}: {score:.4f} ({dur})  {name}{marker}"
                )
            lines.append(thin)

        report_path = Path(self.workspace) / "report.html"
        lines.append(f"  HTML report: {report_path}")
        lines.append(sep)

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate a self-contained HTML report string."""
        # Build metrics table rows
        metrics_rows = ""
        if self.best_metrics:
            for name, val_str in self.best_metrics.items():
                highlight = ' class="highlight"' if name == self.metric else ""
                metrics_rows += (
                    f"<tr{highlight}><td>{name}</td><td>{val_str}</td></tr>\n"
                )

        # Build search history rows
        history_rows = ""
        if self.search_history:
            for entry in self.search_history:
                it = entry.get("iteration", "?")
                score = entry.get("score", 0)
                dur = self._format_duration(entry.get("duration", 0))
                name = entry.get("model_name", "unknown")
                is_best = entry.get("is_best", False)
                row_class = ' class="best-row"' if is_best else ""
                history_rows += (
                    f"<tr{row_class}>"
                    f"<td>{it}</td>"
                    f"<td>{score:.4f}</td>"
                    f"<td>{dur}</td>"
                    f"<td>{name}</td>"
                    f"</tr>\n"
                )

        # Build config JSON
        config_json = json.dumps(self.study_config, indent=2, default=str)

        return _HTML_TEMPLATE.safe_substitute(
            study_name=self.study_name,
            status="SUCCESS" if self.success else "FAILED",
            status_class="success" if self.success else "failure",
            task_type=self.task_type,
            metric=self.metric,
            duration=self._format_duration(self.total_duration_seconds),
            workspace=self.workspace,
            best_model_name=self.best_model_name or "N/A",
            best_score=f"{self.best_score:.4f}",
            metrics_rows=metrics_rows,
            history_rows=history_rows,
            config_json=config_json,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """Save HTML report to the workspace directory.

        Args:
            path: Custom output path. Defaults to workspace/report.html.

        Returns:
            Path to the saved HTML file.
        """
        if path is None:
            path = Path(self.workspace) / "report.html"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def to_dict(self) -> dict:
        """Serialize the report to a JSON-safe dictionary."""
        d = asdict(self)
        # Convert tuple values in best_metrics_raw to lists for JSON
        raw = d.get("best_metrics_raw", {})
        for k, v in raw.items():
            if isinstance(v, tuple):
                raw[k] = list(v)
        return d


_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoPrognosis Report: $study_name</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         "Helvetica Neue", Arial, sans-serif; color: #1a1a2e; background: #f8f9fa;
         line-height: 1.6; padding: 32px; max-width: 960px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin-bottom: 8px; }
  h2 { font-size: 1.15rem; margin: 24px 0 12px; color: #16213e; }
  .header { margin-bottom: 24px; padding-bottom: 16px; border-bottom: 2px solid #e0e0e0; }
  .badge { display: inline-block; padding: 3px 12px; border-radius: 4px; font-size: 0.85rem;
           font-weight: 600; color: #fff; }
  .badge.success { background: #2e7d32; }
  .badge.failure { background: #c62828; }
  .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
             gap: 16px; margin-bottom: 24px; }
  .card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 1.15rem; font-weight: 600; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px;
          overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  th { background: #16213e; color: #fff; text-align: left; padding: 10px 14px; font-size: 0.85rem;
       text-transform: uppercase; letter-spacing: 0.5px; }
  td { padding: 10px 14px; border-bottom: 1px solid #eee; font-size: 0.9rem; }
  tr:last-child td { border-bottom: none; }
  tr.highlight td { background: #e8f5e9; font-weight: 600; }
  tr.best-row td { background: #e3f2fd; font-weight: 600; }
  pre { background: #fff; padding: 16px; border-radius: 8px; overflow-x: auto;
        font-size: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #e0e0e0;
            font-size: 0.8rem; color: #888; }
</style>
</head>
<body>
  <div class="header">
    <h1>AutoPrognosis Study Report: $study_name</h1>
    <span class="badge $status_class">$status</span>
  </div>

  <div class="summary">
    <div class="card">
      <div class="label">Task Type</div>
      <div class="value">$task_type</div>
    </div>
    <div class="card">
      <div class="label">Primary Metric</div>
      <div class="value">$metric</div>
    </div>
    <div class="card">
      <div class="label">Best Score</div>
      <div class="value">$best_score</div>
    </div>
    <div class="card">
      <div class="label">Duration</div>
      <div class="value">$duration</div>
    </div>
  </div>

  <h2>Best Model</h2>
  <div class="card" style="margin-bottom: 24px;">
    <div class="label">Pipeline Architecture</div>
    <div class="value" style="font-size: 0.95rem; word-break: break-all;">$best_model_name</div>
  </div>

  <h2>Evaluation Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Score (mean +/- std)</th></tr></thead>
    <tbody>$metrics_rows</tbody>
  </table>

  <h2>Search History</h2>
  <table>
    <thead><tr><th>Iteration</th><th>Score</th><th>Duration</th><th>Model</th></tr></thead>
    <tbody>$history_rows</tbody>
  </table>

  <h2>Study Configuration</h2>
  <pre>$config_json</pre>

  <div class="footer">
    <p>Workspace: $workspace</p>
    <p>Generated at $generated_at by AutoPrognosis</p>
  </div>
</body>
</html>
""")
