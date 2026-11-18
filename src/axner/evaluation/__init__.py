"""Entity-level evaluation: metrics, bootstrap, aggregation, prediction export."""

from axner.evaluation.entity_metrics import (
    compute_entity_metrics,
    error_summary,
    extract_spans,
    per_type_metrics,
)

__all__ = ["compute_entity_metrics", "error_summary", "extract_spans", "per_type_metrics"]
