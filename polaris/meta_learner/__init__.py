"""Meta-learner implementations."""

from typing import Any

__all__ = ["StatisticalMetaLearner", "LLMMetaLearner"]


def __getattr__(name: str) -> Any:
    """Lazy import optional statistical meta-learner dependencies."""
    if name == "StatisticalMetaLearner":
        from polaris.meta_learner.statistical import StatisticalMetaLearner

        return StatisticalMetaLearner
    if name == "LLMMetaLearner":
        from polaris.meta_learner.llm_based import LLMMetaLearner

        return LLMMetaLearner
    raise AttributeError(f"module 'polaris.meta_learner' has no attribute {name!r}")
