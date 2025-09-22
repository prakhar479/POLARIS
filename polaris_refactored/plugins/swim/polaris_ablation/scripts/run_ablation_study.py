#!/usr/bin/env python3
"""
Ablation Study Runner for SWIM POLARIS Adaptation System

Provides automated execution of ablation studies with different configurations,
result collection, analysis, and report generation.
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ablation_manager import AblationStudyManager, AblationStudyStatus
from config_manager import HierarchicalConfigurationManager


class AblationStudyRunner:
    """Manages execution of ablation studies."""
    
    def __init__(self, config_dir: str, results_dir: str = "results"):
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize managers
        self.ablation_manager = AblationStudyManager(str(self.config_dir), str(self.results_dir))
        self.config_manager = HierarchicalConfigurationManager(str(self.config_dir))
    
    async def list_available_studies(self) -> List[str]:
        """List all available ablation studies."""
        return self.ablation_manager.list_studies()
    
    def print_study_info(self, study_name: str) -> None:
        """Print information about a specific study."""
        study_info = self.ablation_manager.get_study_info(study_name)
        
        if not study_info:
            print(f"Study '{study_name}' not found")
            return
        
        print(f"\nStudy: {study_name}")
        print(f"Description: {study_info.description}")
        print(f"Duration: {study_info.duration} seconds")
        print(f"Warmup: {study_info.warmup_period} seconds")
        print(f"Cooldown: {study_info.cooldown_period} seconds")
        print(f"Config File: {study_info.config_file}")
        
        print("\nComponents:")
        for component, enabled in study_info.components.items():
            status = "enabled" if enabled else "disabled"
            print(f"  {component}: {status}")
        
        if study_info.expected_impact:
            print("\nExpected Impact:")
            for impact in study_info.expected_impact:
                print(f"  - {impact}")
    
    async def run_single_study(self, study_name: str, verbose: bool = False) -> bool:
        """Run a single ablation study.
        
        Args:
            study_name: Name of the study to run
            verbose: Print verbose output
            
        Returns:
            True if study completed successfully
        """
        if study_name not in await self.list_available_studies():
            print(f"Error: Study '{study_name}' not found")
            return False
        
        print(f"Starting ablation study: {study_name}")
        
        if verbose:
            self.print_study_info(study_name)
        
        try:
            # Run the study
            result = await self.ablation_manager.run_study(study_name)
            
            # Print results
            print(f"\nStudy '{study_name}' completed with status: {result.status.value}")
            
            if result.status == AblationStudyStatus.COMPLETED:
                print(f"Duration: {(result.end_time - result.start_time).total_seconds():.1f} seconds")
                print(f"Adaptations: {result.adaptation_count}")
                print(f"Success Rate: {result.success_rate:.2%}")
                
                # Print key metrics
                if result.metrics:
                    print("\nKey Metrics:")
                    for metric, value in result.metrics.items():
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.3f}")
                        else:
                            print(f"  {metric}: {value}")
                
                return True
            
            else:
                print(f"Study failed: {result.error_log}")
                return False
        
        except Exception as e:
            print(f"Error running study: {e}")
            return False
    
    async def run_multiple_studies(self, 
                                 study_names: List[str], 
                                 sequential: bool = True,
                                 verbose: bool = False) -> Dict[str, bool]:
        """Run multiple ablation studies.
        
        Args:
            study_names: List of study names to run
            sequential: Run studies sequentially (vs parallel)
            verbose: Print verbose output
            
        Returns:
            Dictionary mapping study names to success status
        """
        results = {}
        
        if sequential:
            for study_name in study_names:
                print(f"\n{'='*60}")
                print(f"Running study {study_names.index(study_name) + 1}/{len(study_names)}: {study_name}")
                print('='*60)
                
                success = await self.run_single_study(study_name, verbose)
                results[study_name] = success
                
                if not success:
                    print(f"Study {study_name} failed, continuing with next study...")
                
                # Brief pause between studies
                if study_names.index(study_name) < len(study_names) - 1:
                    print("Waiting 30 seconds before next study...")
                    await asyncio.sleep(30)
        
        else:
            # Parallel execution (not recommended for resource-intensive studies)
            print("Running studies in parallel...")
            tasks = [self.run_single_study(name, verbose) for name in study_names]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(task_results):
                study_name = study_names[i]
                if isinstance(result, Exception):
                    print(f"Study {study_name} failed with exception: {result}")
                    results[study_name] = False
                else:
                    results[study_name] = result
        
        return results
    
    async def run_all_studies(self, verbose: bool = False) -> Dict[str, bool]:
        """Run all available ablation studies."""
        study_names = await self.list_available_studies()
        print(f"Running all {len(study_names)} available studies...")
        
        return await self.run_multiple_studies(study_names, sequential=True, verbose=verbose)
    
    def generate_comparison_report(self, study_names: List[str], output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comparison report for studies.
        
        Args:
            study_names: List of study names to compare
            output_file: Optional output file path
            
        Returns:
            Comparison report data
        """
        report = self.ablation_manager.generate_comparison_report(study_names)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Comparison report saved to: {output_path}")
        
        return report
    
    def print_comparison_report(self, study_names: List[str]) -> None:
        """Print comparison report to console."""
        report = self.generate_comparison_report(study_names)
        
        print("\n" + "="*80)
        print("ABLATION STUDY COMPARISON REPORT")
        print("="*80)
        
        print(f"Generated: {report['timestamp']}")
        print(f"Studies: {', '.join(report['studies'])}")
        
        # Print comparison data
        comparison = report.get('comparison', {})
        if comparison:
            print(f"\nStudy Results:")
            print(f"{'Study':<20} {'Status':<12} {'Duration':<10} {'Success Rate':<12} {'Avg Response':<15}")
            print("-" * 80)
            
            for study_name, data in comparison.items():
                status = data.get('status', 'unknown')
                duration = data.get('duration', 0)
                metrics = data.get('metrics', {})
                success_rate = metrics.get('adaptation_success_rate', 0)
                avg_response = metrics.get('avg_response_time', 0)
                
                print(f"{study_name:<20} {status:<12} {duration:<10.1f} "
                      f"{success_rate:<12.2%} {avg_response:<15.3f}")
        
        # Print summary statistics
        summary = report.get('summary', {})
        if summary:
            print(f"\nSummary Statistics:")
            for metric, stats in summary.items():
                print(f"  {metric}:")
                print(f"    Min: {stats.get('min', 0):.3f}")
                print(f"    Max: {stats.get('max', 0):.3f}")
                print(f"    Avg: {stats.get('avg', 0):.3f}")
                print(f"    Range: {stats.get('range', 0):.3f}")
        
        print("="*80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SWIM POLARIS Ablation Studies")
    parser.add_argument("--config-dir", "-c",
                       default="config",
                       help="Configuration directory path")
    parser.add_argument("--results-dir", "-r",
                       default="results",
                       help="Results directory path")
    parser.add_argument("--study", "-s",
                       help="Run specific study by name")
    parser.add_argument("--studies",
                       nargs="+",
                       help="Run multiple specific studies")
    parser.add_argument("--all", "-a",
                       action="store_true",
                       help="Run all available studies")
    parser.add_argument("--list", "-l",
                       action="store_true",
                       help="List available studies")
    parser.add_argument("--info", "-i",
                       help="Show information about a specific study")
    parser.add_argument("--compare",
                       nargs="+",
                       help="Generate comparison report for studies")
    parser.add_argument("--report-file",
                       help="Output file for comparison report")
    parser.add_argument("--parallel", "-p",
                       action="store_true",
                       help="Run studies in parallel (not recommended)")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create runner
    runner = AblationStudyRunner(args.config_dir, args.results_dir)
    
    try:
        # Handle different modes
        if args.list:
            studies = await runner.list_available_studies()
            print(f"Available ablation studies ({len(studies)}):")
            for study in studies:
                print(f"  - {study}")
        
        elif args.info:
            runner.print_study_info(args.info)
        
        elif args.compare:
            runner.print_comparison_report(args.compare)
            if args.report_file:
                runner.generate_comparison_report(args.compare, args.report_file)
        
        elif args.study:
            success = await runner.run_single_study(args.study, args.verbose)
            sys.exit(0 if success else 1)
        
        elif args.studies:
            results = await runner.run_multiple_studies(
                args.studies, 
                sequential=not args.parallel, 
                verbose=args.verbose
            )
            
            # Print summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            print(f"\nStudy Summary: {successful}/{total} studies completed successfully")
            
            for study, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {study}")
            
            sys.exit(0 if successful == total else 1)
        
        elif args.all:
            results = await runner.run_all_studies(args.verbose)
            
            # Print summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            print(f"\nAll Studies Summary: {successful}/{total} studies completed successfully")
            
            sys.exit(0 if successful == total else 1)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nAblation study execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())