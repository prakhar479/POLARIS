"""Tests for empirical software engineering statistical testing engine."""

import pytest

from polaris.evaluation.stats import (
    StatisticalComparison,
    compare_distributions,
    mann_whitney_u_test,
    vargha_delaney_a12,
    wilcoxon_signed_rank_test,
)


def test_vargha_delaney_a12_identical():
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    a12, mag = vargha_delaney_a12(sample, sample)
    assert a12 == 0.5
    assert mag == "negligible"


def test_vargha_delaney_a12_large_effect():
    treatment = [10.0, 11.0, 12.0, 13.0, 14.0]
    control = [1.0, 2.0, 3.0, 4.0, 5.0]
    a12, mag = vargha_delaney_a12(treatment, control)
    assert a12 == 1.0
    assert mag == "large"

    # Inverted
    a12_inv, mag_inv = vargha_delaney_a12(control, treatment)
    assert a12_inv == 0.0
    assert mag_inv == "large"


def test_mann_whitney_u_test():
    treatment = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    control = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    u_stat, p_val, is_sig = mann_whitney_u_test(treatment, control)
    assert is_sig is True
    assert p_val < 0.05
    assert u_stat > 0


def test_wilcoxon_signed_rank_test():
    paired_a = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
    paired_b = [8.0, 10.0, 11.0, 13.0, 15.0, 17.0, 18.0]
    w_stat, p_val, is_sig = wilcoxon_signed_rank_test(paired_a, paired_b)
    assert p_val < 0.05
    assert is_sig is True


def test_compare_distributions_and_formatting():
    treatment = [0.08, 0.09, 0.095, 0.085, 0.092]
    control = [0.15, 0.16, 0.155, 0.148, 0.162]

    comp = compare_distributions(
        treatment_values=treatment,
        control_values=control,
        metric_name="Response Time (s)",
        treatment_label="POLARIS",
        control_label="AdaMLS",
    )

    assert isinstance(comp, StatisticalComparison)
    assert comp.is_significant is True
    assert comp.p_value < 0.05
    assert comp.a12 == 0.0  # treatment strictly lower than control
    assert comp.effect_magnitude == "large"

    md_row = comp.to_markdown_row()
    assert "| Response Time (s) | POLARIS" in md_row
    assert "✓" in md_row

    latex_row = comp.to_latex_row()
    assert "Response Time (s) &" in latex_row
    assert "\\pm" in latex_row

    assert "statistically significant" in comp.summary_text
