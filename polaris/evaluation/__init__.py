"""Evaluation and statistical benchmarking module for POLARIS."""

from polaris.evaluation.stats import (
    StatisticalComparison,
    compare_distributions,
    mann_whitney_u_test,
    vargha_delaney_a12,
    wilcoxon_signed_rank_test,
)

__all__ = [
    "StatisticalComparison",
    "compare_distributions",
    "mann_whitney_u_test",
    "wilcoxon_signed_rank_test",
    "vargha_delaney_a12",
]
