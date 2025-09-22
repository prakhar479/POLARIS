"""
Ablation Study Manager

Manages ablation studies for the SWIM POLARIS adaptation system.
Coordinates execution of different configurations, collects results,
and provides analysis capabilities.
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .swim_driver import SwimPolarisDriver, SystemStatus


class AblationStudyStatus(Enum):
    """Status of an ablation study."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AblationConfiguration:
    """Configuration for a single ablation study."""
    name: str
    description: str
    config_file: str
    components: Dict[str, bool]
    expected_impact: List[str]
    duration: int  # seconds
    warmup_period: int = 300  # seconds
    cooldown_period: int = 300  # seconds


@dataclass
class AblationResult:
    """Results from a single ablation study."""
    configuration: AblationConfiguration
    status: AblationStudyStatus
    start_time: datetime
    end_time: Optional[datetime]
    metrics: Dict[str, Any]
    performance_data: Dict[str, List[float]]
    adaptation_count: int
    success_rate: float
    error_log: List[str]
    system_logs: List[str]


class AblationStudyManager:
    """
    Manages execution of ablation studies for SWIM POLARIS system.
    
    Coordinates multiple study configurations, collects results,
    and provides analysis capabilities.
    """
    
    def __init__(self, base_config_path: str, results_dir: str = "results"):
        """Initialize the ablation study manager.
        
        Args:
            base_config_path: Path to base configuration directory
            results_dir: Directory to store results
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_config_path = Path(base_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Study management
        self.studies: Dict[str, AblationConfiguration] = {}
        self.results: Dict[str, AblationResult] = {}
        self.current_study: Optional[str] = None
        self.current_driver: Optional[SwimPolarisDriver] = None
        
        # Load available study configurations
        self._load_study_configurations()
    
    def _load_study_configurations(self) -> None:
        """Load all available ablation study configurations."""
        ablation_configs_dir = self.base_config_path / "ablation_configs"
        
        if not ablation_configs_dir.exists():
            self.logger.warning(f"Ablation configs directory not found: {ablation_configs_dir}")
            return
        
        for config_file in ablation_configs_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract ablation configuration
                ablation_config = config_data.get('ablation', {})
                
                study_config = AblationConfiguration(
                    name=config_file.stem,
                    description=ablation_config.get('description', f"Ablation study: {config_file.stem}"),
                    config_file=str(config_file),
                    components=ablation_config.get('components', {}),
                    expected_impact=ablation_config.get('expected_impact', []),
                    duration=ablation_config.get('study_parameters', {}).get('duration', 3600),
                    warmup_period=ablation_config.get('study_parameters', {}).get('warmup_period', 300),
                    cooldown_period=ablation_config.get('study_parameters', {}).get('cooldown_period', 300)
                )
                
                self.studies[study_config.name] = study_config
                self.logger.info(f"Loaded ablation study configuration: {study_config.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load ablation config {config_file}: {e}")
    
    def list_studies(self) -> List[str]:
        """Get list of available study names."""
        return list(self.studies.keys())
    
    def get_study_info(self, study_name: str) -> Optional[AblationConfiguration]:
        """Get information about a specific study."""
        return self.studies.get(study_name)
    
    async def run_study(self, study_name: str) -> AblationResult:
        """
        Run a single ablation study.
        
        Args:
            study_name: Name of the study to run
            
        Returns:
            AblationResult with study results
        """
        if study_name not in self.studies:
            raise ValueError(f"Unknown study: {study_name}")
        
        study_config = self.studies[study_name]
        self.current_study = study_name
        
        self.logger.info(f"Starting ablation study: {study_name}")
        
        # Initialize result tracking
        result = AblationResult(
            configuration=study_config,
            status=AblationStudyStatus.RUNNING,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            metrics={},
            performance_data={},
            adaptation_count=0,
            success_rate=0.0,
            error_log=[],
            system_logs=[]
        )
        
        try:
            # Initialize driver with study configuration
            self.current_driver = SwimPolarisDriver(study_config.config_file)
            
            # Start the system
            await self.current_driver.start()
            
            # Wait for warmup period
            if study_config.warmup_period > 0:
                self.logger.info(f"Warmup period: {study_config.warmup_period} seconds")
                await asyncio.sleep(study_config.warmup_period)
            
            # Run the study
            await self._execute_study(result, study_config)
            
            # Cooldown period
            if study_config.cooldown_period > 0:
                self.logger.info(f"Cooldown period: {study_config.cooldown_period} seconds")
                await asyncio.sleep(study_config.cooldown_period)
            
            result.status = AblationStudyStatus.COMPLETED
            result.end_time = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Study {study_name} failed: {e}")
            result.status = AblationStudyStatus.FAILED
            result.error_log.append(str(e))
            result.end_time = datetime.now(timezone.utc)
        
        finally:
            # Clean up
            if self.current_driver:
                await self.current_driver.stop()
                self.current_driver = None
            
            self.current_study = None
        
        # Store results
        self.results[study_name] = result
        await self._save_results(study_name, result)
        
        self.logger.info(f"Completed ablation study: {study_name}")
        return result
    
    async def _execute_study(self, result: AblationResult, config: AblationConfiguration) -> None:
        """Execute the main study period with data collection."""
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=config.duration)
        
        # Initialize performance tracking
        performance_data = {
            'timestamps': [],
            'response_times': [],
            'server_utilization': [],
            'adaptation_actions': [],
            'system_health': []
        }
        
        adaptation_count = 0
        successful_adaptations = 0
        
        self.logger.info(f"Executing study for {config.duration} seconds")
        
        while datetime.now(timezone.utc) < end_time:
            try:
                # Collect current metrics
                if self.current_driver:
                    status = await self.current_driver.get_status()
                    
                    # Record performance data
                    current_time = datetime.now(timezone.utc)
                    performance_data['timestamps'].append(current_time.isoformat())
                    
                    if status.system_metrics:
                        performance_data['response_times'].append(
                            status.system_metrics.get('basic_response_time', 0.0)
                        )
                        performance_data['server_utilization'].append(
                            status.system_metrics.get('server_utilization', 0.0)
                        )
                    
                    performance_data['system_health'].append(status.health_status.value)
                    
                    # Track adaptations
                    if hasattr(status, 'recent_adaptations'):
                        new_adaptations = len(status.recent_adaptations)
                        if new_adaptations > adaptation_count:
                            adaptation_count = new_adaptations
                            # Count successful adaptations
                            for adaptation in status.recent_adaptations[adaptation_count:]:
                                if adaptation.get('status') == 'SUCCESS':
                                    successful_adaptations += 1
                                performance_data['adaptation_actions'].append({
                                    'timestamp': current_time.isoformat(),
                                    'action': adaptation.get('action_type', 'unknown'),
                                    'status': adaptation.get('status', 'unknown')
                                })
                
                # Wait before next collection
                await asyncio.sleep(10)  # Collect data every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error during study execution: {e}")
                result.error_log.append(f"Execution error: {str(e)}")
        
        # Calculate final metrics
        result.performance_data = performance_data
        result.adaptation_count = adaptation_count
        result.success_rate = successful_adaptations / max(adaptation_count, 1)
        
        # Calculate summary metrics
        if performance_data['response_times']:
            result.metrics['avg_response_time'] = sum(performance_data['response_times']) / len(performance_data['response_times'])
            result.metrics['max_response_time'] = max(performance_data['response_times'])
            result.metrics['min_response_time'] = min(performance_data['response_times'])
        
        if performance_data['server_utilization']:
            result.metrics['avg_utilization'] = sum(performance_data['server_utilization']) / len(performance_data['server_utilization'])
            result.metrics['max_utilization'] = max(performance_data['server_utilization'])
            result.metrics['min_utilization'] = min(performance_data['server_utilization'])
        
        result.metrics['total_adaptations'] = adaptation_count
        result.metrics['successful_adaptations'] = successful_adaptations
        result.metrics['adaptation_success_rate'] = result.success_rate
    
    async def run_multiple_studies(self, study_names: List[str]) -> Dict[str, AblationResult]:
        """
        Run multiple ablation studies sequentially.
        
        Args:
            study_names: List of study names to run
            
        Returns:
            Dictionary mapping study names to results
        """
        results = {}
        
        for study_name in study_names:
            self.logger.info(f"Running study {study_name} ({study_names.index(study_name) + 1}/{len(study_names)})")
            
            try:
                result = await self.run_study(study_name)
                results[study_name] = result
                
                # Brief pause between studies
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Failed to run study {study_name}: {e}")
                # Create failed result
                results[study_name] = AblationResult(
                    configuration=self.studies[study_name],
                    status=AblationStudyStatus.FAILED,
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    metrics={},
                    performance_data={},
                    adaptation_count=0,
                    success_rate=0.0,
                    error_log=[str(e)],
                    system_logs=[]
                )
        
        return results
    
    async def _save_results(self, study_name: str, result: AblationResult) -> None:
        """Save study results to file."""
        result_file = self.results_dir / f"{study_name}_{result.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert result to serializable format
        result_data = {
            'configuration': asdict(result.configuration),
            'status': result.status.value,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'metrics': result.metrics,
            'performance_data': result.performance_data,
            'adaptation_count': result.adaptation_count,
            'success_rate': result.success_rate,
            'error_log': result.error_log,
            'system_logs': result.system_logs
        }
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            self.logger.info(f"Saved results to {result_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def generate_comparison_report(self, study_names: List[str]) -> Dict[str, Any]:
        """
        Generate a comparison report for multiple studies.
        
        Args:
            study_names: List of study names to compare
            
        Returns:
            Comparison report data
        """
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'studies': study_names,
            'comparison': {},
            'summary': {}
        }
        
        # Collect metrics for comparison
        metrics_data = {}
        for study_name in study_names:
            if study_name in self.results:
                result = self.results[study_name]
                metrics_data[study_name] = {
                    'status': result.status.value,
                    'duration': (result.end_time - result.start_time).total_seconds() if result.end_time else 0,
                    'metrics': result.metrics,
                    'components': result.configuration.components
                }
        
        report['comparison'] = metrics_data
        
        # Generate summary statistics
        if metrics_data:
            # Compare key metrics
            for metric in ['avg_response_time', 'avg_utilization', 'adaptation_success_rate']:
                values = []
                for study_data in metrics_data.values():
                    if metric in study_data['metrics']:
                        values.append(study_data['metrics'][metric])
                
                if values:
                    report['summary'][metric] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'range': max(values) - min(values)
                    }
        
        return report
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.current_driver:
            await self.current_driver.stop()
            self.current_driver = None
        
        self.current_study = None