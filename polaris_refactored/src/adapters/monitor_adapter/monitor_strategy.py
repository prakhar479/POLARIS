"""
Monitor Adapter Strategy Implementation

This module provides different strategies for collecting metrics from managed systems.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from ...framework.plugin_management import ManagedSystemConnectorFactory

from .monitor_types import MonitoringTarget, CollectionResult

logger = logging.getLogger(__name__)

class MetricCollectionStrategy(ABC):
    """Strategy interface for different metric collection approaches."""
    
    @abstractmethod
    async def collect_metrics(
        self, 
        target: MonitoringTarget, 
        connector_factory: ManagedSystemConnectorFactory
    ) -> CollectionResult:
        """
        Collect metrics using this strategy.
        
        Args:
            target: The monitoring target configuration
            connector_factory: Factory for creating connectors
            
        Returns:
            CollectionResult: Result of the collection operation
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass
    
    @abstractmethod
    def supports_target(self, target: MonitoringTarget) -> bool:
        """Check if this strategy supports the given target."""
        pass


class DirectConnectorStrategy(MetricCollectionStrategy):
    """Strategy that collects metrics directly from managed system connectors."""
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "direct_connector"
    
    def supports_target(self, target: MonitoringTarget) -> bool:
        """Check if this strategy supports the given target."""
        # This strategy supports all targets that have connectors
        return bool(target.connector_type)
    
    async def collect_metrics(
        self, 
        target: MonitoringTarget, 
        connector_factory: ManagedSystemConnectorFactory
    ) -> CollectionResult:
        """Collect metrics directly from the connector."""
        start_time = datetime.utcnow()
        
        try:
            # Create connector for the target system
            connector = connector_factory.create_connector(
                target.connector_type, 
                target.config
            )
            
            if not connector:
                return CollectionResult(
                    system_id=target.system_id,
                    metrics={},
                    timestamp=start_time,
                    success=False,
                    error=f"Failed to create connector for {target.connector_type}"
                )
            
            # Collect metrics from connector
            metrics = await connector.collect_metrics()
            
            end_time = datetime.utcnow()
            
            return CollectionResult(
                system_id=target.system_id,
                metrics=metrics,
                timestamp=end_time,
                success=True,
                collection_duration=end_time - start_time
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            logger.error(f"Failed to collect metrics from {target.system_id}: {e}")
            
            return CollectionResult(
                system_id=target.system_id,
                metrics={},
                timestamp=end_time,
                success=False,
                error=str(e),
                collection_duration=end_time - start_time
            )


class PollingStrategy(MetricCollectionStrategy):
    """Strategy that polls metrics at regular intervals."""
    
    def __init__(self, base_strategy: MetricCollectionStrategy):
        self.base_strategy = base_strategy
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return f"polling_{self.base_strategy.get_strategy_name()}"
    
    def supports_target(self, target: MonitoringTarget) -> bool:
        """Check if this strategy supports the given target."""
        return self.base_strategy.supports_target(target)
    
    async def collect_metrics(
        self, 
        target: MonitoringTarget, 
        connector_factory: ManagedSystemConnectorFactory
    ) -> CollectionResult:
        """Collect metrics using the base strategy with polling behavior."""
        # For now, just delegate to base strategy
        # In a real implementation, this might add polling-specific logic
        return await self.base_strategy.collect_metrics(target, connector_factory)


class BatchCollectionStrategy(MetricCollectionStrategy):
    """Strategy that collects metrics from multiple systems in batches."""
    
    def __init__(self, base_strategy: MetricCollectionStrategy, batch_size: int = 5):
        self.base_strategy = base_strategy
        self.batch_size = batch_size
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return f"batch_{self.base_strategy.get_strategy_name()}"
    
    def supports_target(self, target: MonitoringTarget) -> bool:
        """Check if this strategy supports the given target."""
        return self.base_strategy.supports_target(target)
    
    async def collect_metrics(
        self, 
        target: MonitoringTarget, 
        connector_factory: ManagedSystemConnectorFactory
    ) -> CollectionResult:
        """Collect metrics using batched approach."""
        # For single target, just delegate to base strategy
        # In a real implementation, this would batch multiple targets
        return await self.base_strategy.collect_metrics(target, connector_factory)


class RetryingStrategyDecorator(MetricCollectionStrategy):
    """Decorator strategy that retries on failure with backoff and jitter."""
    
    def __init__(
        self,
        base_strategy: MetricCollectionStrategy,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff: float = 10.0,
        jitter: float = 0.1,
    ):
        self.base_strategy = base_strategy
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.jitter = jitter
    
    def get_strategy_name(self) -> str:
        return f"retrying_{self.base_strategy.get_strategy_name()}"
    
    def supports_target(self, target: MonitoringTarget) -> bool:
        return self.base_strategy.supports_target(target)
    
    async def collect_metrics(
        self,
        target: MonitoringTarget,
        connector_factory: ManagedSystemConnectorFactory
    ) -> CollectionResult:
        attempt = 0
        start_overall = datetime.utcnow()
        last_result: Optional[CollectionResult] = None
        while True:
            attempt += 1
            result = await self.base_strategy.collect_metrics(target, connector_factory)
            last_result = result
            if result.success or attempt > self.max_retries:
                # Set decorator strategy name if not set
                if result and not result.strategy_name:
                    result.strategy_name = self.get_strategy_name()
                # Adjust collection_duration to include total time
                end_overall = datetime.utcnow()
                result.collection_duration = (end_overall - start_overall)
                return result
            # compute backoff with jitter
            backoff = min(self.max_backoff, self.backoff_base * (self.backoff_factor ** (attempt - 1)))
            # simple jitter: +/- jitter * backoff
            jitter_amount = backoff * self.jitter
            sleep_time = max(0.0, backoff - jitter_amount)
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
        # If we broke out due to cancellation, return last known result or a failure
        end_overall = datetime.utcnow()
        if last_result:
            last_result.collection_duration = (end_overall - start_overall)
            if not last_result.strategy_name:
                last_result.strategy_name = self.get_strategy_name()
            return last_result
        return CollectionResult(
            system_id=target.system_id,
            metrics={},
            timestamp=end_overall,
            success=False,
            error="collection cancelled",
            collection_duration=(end_overall - start_overall),
            strategy_name=self.get_strategy_name(),
        )

