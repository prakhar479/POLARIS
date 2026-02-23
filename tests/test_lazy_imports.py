"""Tests for lazy import module attributes."""

import pytest

import polaris
import polaris.meta_learner as meta_learner


def test_polaris_lazy_meta_learner_exports():
    statistical_cls = polaris.__getattr__("StatisticalMetaLearner")
    llm_cls = polaris.__getattr__("LLMMetaLearner")

    assert statistical_cls.__name__ == "StatisticalMetaLearner"
    assert llm_cls.__name__ == "LLMMetaLearner"


def test_polaris_lazy_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="module 'polaris' has no attribute"):
        polaris.__getattr__("DoesNotExist")


def test_meta_learner_lazy_exports():
    statistical_cls = meta_learner.__getattr__("StatisticalMetaLearner")
    llm_cls = meta_learner.__getattr__("LLMMetaLearner")

    assert statistical_cls.__name__ == "StatisticalMetaLearner"
    assert llm_cls.__name__ == "LLMMetaLearner"


def test_meta_learner_lazy_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="module 'polaris.meta_learner' has no attribute"):
        meta_learner.__getattr__("Unknown")
