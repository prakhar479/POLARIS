"""
Hybrid strategy that delegates to multiple sub-strategies.
"""

from typing import List, Tuple, Optional, Dict, Any
import asyncio

from polaris.abstractions.strategy import AdaptationStrategy, AdaptationContext, ParameterSpec
from polaris.core.models import SystemState, AdaptationAction


class HybridStrategy(AdaptationStrategy):
    """
    Hybrid strategy that combines multiple strategies.

    Can delegate to multiple strategies and select the best action
    based on different selection modes.
    """

    def __init__(
        self,
        # (strategy, priority)
        strategies: List[Tuple[AdaptationStrategy, float]],
        selection_mode: str = 'confidence',
        min_confidence: float = 0.7
    ):
        """
        Initialize hybrid strategy.

        Args:
            strategies: List of (strategy, priority) tuples
            selection_mode: How to select among proposals
                - 'first': Use first strategy that proposes action
                - 'priority': Use highest priority strategy
                - 'confidence': Use highest confidence action
            min_confidence: Minimum confidence threshold
        """
        self.strategies = sorted(strategies, key=lambda x: x[1], reverse=True)
        self.selection_mode = selection_mode
        self.min_confidence = min_confidence
        self._adaptation_count = 0
        self._success_count = 0
        self._strategy_usage = {i: 0 for i in range(len(strategies))}

    async def assess(
        self,
        state: SystemState,
        context: AdaptationContext
    ) -> Optional[AdaptationAction]:
        """Assess using all strategies and select best action."""

        # Query all strategies concurrently
        tasks = []
        for strategy, priority in self.strategies:
            tasks.append(strategy.assess(state, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid proposals
        proposals = []
        for i, (result, (strategy, priority)) in enumerate(zip(results, self.strategies)):
            if isinstance(result, Exception):
                continue
            if result:
                # Estimate confidence (simple heuristic)
                confidence = self._estimate_confidence(strategy, result, state)
                proposals.append((result, confidence, priority, i))

        if not proposals:
            return None

        # Select based on mode
        selected = None
        selected_idx = None

        if self.selection_mode == 'first':
            # Return first proposal (highest priority)
            selected, _, _, selected_idx = proposals[0]

        elif self.selection_mode == 'priority':
            # Use highest priority strategy with valid action
            for action, conf, pri, idx in sorted(proposals, key=lambda x: x[2], reverse=True):
                if conf >= self.min_confidence:
                    selected = action
                    selected_idx = idx
                    break

        elif self.selection_mode == 'confidence':
            # Return highest confidence proposal above threshold
            valid = [(a, c, p, i)
                     for a, c, p, i in proposals if c >= self.min_confidence]
            if valid:
                selected, _, _, selected_idx = max(valid, key=lambda x: x[1])

        # Track which strategy was used
        if selected and selected_idx is not None:
            self._strategy_usage[selected_idx] += 1

        return selected

    def _estimate_confidence(
        self,
        strategy: AdaptationStrategy,
        action: AdaptationAction,
        state: SystemState
    ) -> float:
        """
        Estimate confidence in an action.

        Simple heuristic based on strategy type and action parameters.
        """
        # Base confidence
        confidence = 0.7

        # Note: Cannot call async method from sync context
        # This would require refactoring to make this method async
        # For now, use base confidence
        # TODO: Make this method async and await strategy.get_performance_metrics()

        return confidence

    async def on_action_executed(self, action: AdaptationAction, result) -> None:
        """Track adaptation success."""
        self._adaptation_count += 1
        if hasattr(result, 'status') and result.status.value == 'success':
            self._success_count += 1

        # Propagate to all strategies
        for strategy, _ in self.strategies:
            await strategy.on_action_executed(action, result)

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Aggregate parameters from all sub-strategies."""
        params = {}

        # Add parameters from each strategy
        for i, (strategy, _) in enumerate(self.strategies):
            strategy_params = strategy.get_tunable_parameters()
            for path, spec in strategy_params.items():
                params[f"strategy_{i}.{path}"] = spec

        # Add hybrid-specific parameters
        params["selection_mode"] = ParameterSpec(
            current_value=self.selection_mode,
            type=str,
            allowed_values=['first', 'priority', 'confidence'],
            description="How to select between multiple strategy proposals"
        )
        params["min_confidence"] = ParameterSpec(
            current_value=self.min_confidence,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Minimum confidence threshold for action selection"
        )

        return params

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Route parameter updates to appropriate sub-strategy."""

        if parameter_path.startswith("strategy_"):
            # Parse strategy index and delegate
            parts = parameter_path.split(".", 1)
            strategy_idx = int(parts[0].split("_")[1])
            sub_path = parts[1] if len(parts) > 1 else ""

            if strategy_idx < len(self.strategies):
                return await self.strategies[strategy_idx][0].update_parameter(sub_path, new_value)

        elif parameter_path == "selection_mode":
            if new_value in ['first', 'priority', 'confidence']:
                self.selection_mode = new_value
                return True

        elif parameter_path == "min_confidence":
            self.min_confidence = float(new_value)
            return True

        return False

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy performance metrics."""
        metrics = {}

        if self._adaptation_count > 0:
            metrics['success_rate'] = self._success_count / \
                self._adaptation_count
            metrics['total_adaptations'] = float(self._adaptation_count)

        # Add usage statistics
        for idx, count in self._strategy_usage.items():
            metrics[f'strategy_{idx}_usage'] = float(count)

        return metrics
