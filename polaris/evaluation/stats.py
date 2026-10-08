"""Statistical Significance & Effect Size Testing for Self-Adaptive Systems.

Implements rigorous empirical software engineering statistical tests:
- Vargha & Delaney A12 non-parametric effect size (Vargha & Delaney, 2000; Arcuri & Briand, 2011)
- Mann-Whitney U test for independent distributions
- Wilcoxon signed-rank test for paired runs
- Automated LaTeX and Markdown table row formatting for conference publications (ICSE, TAAS).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# Defensive safeguard for environments where NumPy is reloaded (e.g. pytest-cov)
_np_dtypes = sys.modules.get("numpy.dtypes")
if _np_dtypes is not None and not hasattr(_np_dtypes, "VoidDType"):
    _np_dtypes.VoidDType = getattr(np, "void", object)  # type: ignore[attr-defined]

from scipy import stats  # noqa: E402


def vargha_delaney_a12(treatment: Sequence[float], control: Sequence[float]) -> Tuple[float, str]:
    """Calculate Vargha and Delaney's A12 non-parametric effect size.

    A12 measures the probability that a random value from the treatment group
    is greater than a random value from the control group.

    Interpretation thresholds (Arcuri & Briand, 2011; Vargha & Delaney, 2000):
        |A12 - 0.5| < 0.06 -> negligible
        |A12 - 0.5| >= 0.06 -> small
        |A12 - 0.5| >= 0.14 -> medium
        |A12 - 0.5| >= 0.21 -> large

    Args:
        treatment: Observed metric values for treatment condition (e.g. POLARIS).
        control: Observed metric values for control condition (e.g. Baseline).

    Returns:
        Tuple of (A12 float value, magnitude classification string).
    """
    m = len(treatment)
    n = len(control)
    if m == 0 or n == 0:
        return 0.5, "negligible"

    # Count wins and ties
    wins = 0.0
    ties = 0.0
    for x in treatment:
        for y in control:
            if x > y:
                wins += 1.0
            elif x == y:
                ties += 1.0

    a12 = (wins + 0.5 * ties) / (m * n)
    diff = abs(a12 - 0.5)

    if diff >= 0.21:
        magnitude = "large"
    elif diff >= 0.14:
        magnitude = "medium"
    elif diff >= 0.06:
        magnitude = "small"
    else:
        magnitude = "negligible"

    return float(a12), magnitude


def mann_whitney_u_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tuple[float, float, bool]:
    """Perform Mann-Whitney U non-parametric test.

    Args:
        sample_a: First distribution sample.
        sample_b: Second distribution sample.
        alpha: Significance threshold (default 0.05).
        alternative: "two-sided", "less", or "greater".

    Returns:
        Tuple of (U statistic, p-value, is_significant boolean).
    """
    if len(sample_a) == 0 or len(sample_b) == 0:
        return 0.0, 1.0, False

    res = stats.mannwhitneyu(sample_a, sample_b, alternative=alternative)
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)
    is_sig = bool(p_val < alpha)
    return u_stat, p_val, is_sig


def wilcoxon_signed_rank_test(
    paired_a: Sequence[float],
    paired_b: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tuple[float, float, bool]:
    """Perform Wilcoxon signed-rank test for paired runs across identical seeds.

    Args:
        paired_a: First paired observation sequence.
        paired_b: Second paired observation sequence.
        alpha: Significance threshold (default 0.05).
        alternative: "two-sided", "less", or "greater".

    Returns:
        Tuple of (W statistic, p-value, is_significant boolean).
    """
    if len(paired_a) != len(paired_b) or len(paired_a) == 0:
        return 0.0, 1.0, False

    diffs = np.array(paired_a) - np.array(paired_b)
    if np.all(diffs == 0):
        return 0.0, 1.0, False

    res = stats.wilcoxon(paired_a, paired_b, alternative=alternative)
    w_stat = float(res.statistic)
    p_val = float(res.pvalue)
    is_sig = bool(p_val < alpha)
    return w_stat, p_val, is_sig


@dataclass(frozen=True)
class StatisticalComparison:
    """Formal statistical comparison between treatment and baseline."""

    metric_name: str
    treatment_label: str
    control_label: str
    treatment_mean: float
    treatment_std: float
    treatment_median: float
    control_mean: float
    control_std: float
    control_median: float
    u_statistic: float
    p_value: float
    is_significant: bool
    a12: float
    effect_magnitude: str

    @property
    def formatted_p_value(self) -> str:
        """Format p-value for scientific publication."""
        if self.p_value < 0.001:
            return "p < 0.001"
        return f"p = {self.p_value:.3f}"

    @property
    def summary_text(self) -> str:
        """Publication-ready reporting string."""
        sig_str = (
            "statistically significant" if self.is_significant else "not statistically significant"
        )
        return (
            f"Comparison of {self.metric_name} between {self.treatment_label} and {self.control_label}: "
            f"{sig_str} ({self.formatted_p_value}), "
            f"Vargha-Delaney A12 = {self.a12:.3f} ({self.effect_magnitude} effect size)."
        )

    def to_markdown_row(self) -> str:
        """Generate formatted Markdown table row."""
        t_str = f"{self.treatment_mean:.3f} ± {self.treatment_std:.3f}"
        c_str = f"{self.control_mean:.3f} ± {self.control_std:.3f}"
        sig_symbol = "✓" if self.is_significant else "✗"
        return (
            f"| {self.metric_name} | {self.treatment_label} ({t_str}) | "
            f"{self.control_label} ({c_str}) | {self.formatted_p_value} | "
            f"{self.a12:.3f} ({self.effect_magnitude}) | {sig_symbol} |"
        )

    def to_latex_row(self) -> str:
        """Generate formatted LaTeX table row for ACM/IEEE paper."""
        t_str = f"${self.treatment_mean:.3f} \\pm {self.treatment_std:.3f}$"
        c_str = f"${self.control_mean:.3f} \\pm {self.control_std:.3f}$"
        p_str = "$p < 0.001$" if self.p_value < 0.001 else f"$p = {self.p_value:.3f}$"
        a_str = f"${self.a12:.3f}$ ({self.effect_magnitude})"
        return f"{self.metric_name} & {t_str} & {c_str} & {p_str} & {a_str} \\\\"


def compare_distributions(
    treatment_values: Sequence[float],
    control_values: Sequence[float],
    metric_name: str,
    treatment_label: str = "POLARIS",
    control_label: str = "Baseline",
    alpha: float = 0.05,
) -> StatisticalComparison:
    """Run full statistical evaluation suite comparing treatment and control.

    Args:
        treatment_values: Observations from treatment system.
        control_values: Observations from control system.
        metric_name: Metric identifier (e.g. "Response Time", "SLA Violation Rate").
        treatment_label: Name of treatment condition.
        control_label: Name of control condition.
        alpha: Statistical significance threshold.

    Returns:
        StatisticalComparison dataclass with complete summary and formatted outputs.
    """
    t_arr = np.array(treatment_values, dtype=float)
    c_arr = np.array(control_values, dtype=float)

    t_mean = float(np.mean(t_arr)) if len(t_arr) > 0 else 0.0
    t_std = float(np.std(t_arr, ddof=1)) if len(t_arr) > 1 else 0.0
    t_med = float(np.median(t_arr)) if len(t_arr) > 0 else 0.0

    c_mean = float(np.mean(c_arr)) if len(c_arr) > 0 else 0.0
    c_std = float(np.std(c_arr, ddof=1)) if len(c_arr) > 1 else 0.0
    c_med = float(np.median(c_arr)) if len(c_arr) > 0 else 0.0

    t_list = [float(x) for x in t_arr]
    c_list = [float(x) for x in c_arr]
    u_stat, p_val, is_sig = mann_whitney_u_test(t_list, c_list, alpha=alpha)
    a12, magnitude = vargha_delaney_a12(t_list, c_list)

    return StatisticalComparison(
        metric_name=metric_name,
        treatment_label=treatment_label,
        control_label=control_label,
        treatment_mean=round(t_mean, 4),
        treatment_std=round(t_std, 4),
        treatment_median=round(t_med, 4),
        control_mean=round(c_mean, 4),
        control_std=round(c_std, 4),
        control_median=round(c_med, 4),
        u_statistic=round(u_stat, 2),
        p_value=float(p_val),
        is_significant=is_sig,
        a12=round(a12, 4),
        effect_magnitude=magnitude,
    )
