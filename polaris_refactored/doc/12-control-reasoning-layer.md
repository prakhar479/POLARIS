# Control & Reasoning Layer

## Overview

The Control & Reasoning Layer implements the MAPE-K (Monitor-Analyze-Plan-Execute-Knowledge) loop for adaptive system control. This layer combines intelligent reasoning with multiple control strategies to make optimal adaptation decisions based on system state, learned patterns, and predictive analytics.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Control & Reasoning Layer                    │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Adaptive Controller │  │     Reasoning Engine           │   │
│  │ - MAPE-K Loop      │  │ - Statistical Reasoning        │   │
│  │ - Control Strategies│  │ - Causal Reasoning            │   │
│  │ - Strategy Selection│  │ - Experience-based Reasoning   │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Control Strategies                       │   │
│  │ - Reactive Control (Threshold-based)                   │   │
│  │ - Predictive Control (Model-based)                     │   │
│  │ - Learning Control (Experience-based)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Adaptive Controller (PolarisAdaptiveController)

### Purpose
Orchestrates the complete MAPE-K loop, coordinating monitoring, analysis, planning, and execution phases while managing knowledge feedback.

### Key Features
- **Complete MAPE-K Implementation**: Full Monitor-Analyze-Plan-Execute-Knowledge cycle
- **Dynamic Strategy Selection**: Chooses optimal control strategy based on context
- **Multi-objective Optimization**: Balances performance, cost, and reliability goals
- **Continuous Learning Integration**: Incorporates learned knowledge into decisions
- **Adaptation Coordination**: Manages concurrent adaptations across systems

### Implementation

#### Adaptive Controller Core
```python
class PolarisAdaptiveController(Injectable):
    """Main adaptive controller implementing the MAPE-K loop."""
    
    def __init__(self, world_model: PolarisWorldModel, 
                 knowledge_base: PolarisKnowledgeBase,
                 reasoning_engine: PolarisReasoningEngine,
                 event_bus: PolarisEventBus,
                 config: AdaptiveControllerConfig):
        self._world_model = world_model
        self._knowledge_base = knowledge_base
        self._reasoning_engine = reasoning_engine
        self._event_bus = event_bus
        self._config = config
        
        # Control strategies
        self._control_strategies: Dict[str, ControlStrategy] = {}
        self._strategy_selector = ControlStrategySelector(config.strategy_selection)
        
        # MAPE-K state
        self._active_adaptations: Dict[str, AdaptationContext] = {}
        self._adaptation_queue: asyncio.Queue = asyncio.Queue()
        self._mape_k_task: Optional[asyncio.Task] = None
        
        # Metrics and monitoring
        self._controller_metrics = ControllerMetrics()
        
        # Initialize control strategies
        self._initialize_control_strategies()
    
    async def start(self) -> None:
        """Start the adaptive controller."""
        logger.info("Starting adaptive controller")
        
        # Subscribe to telemetry events
        await self._event_bus.subscribe(TelemetryEvent, self._handle_telemetry_event)
        
        # Start MAPE-K loop
        self._mape_k_task = asyncio.create_task(self._mape_k_loop())
        
        logger.info("Adaptive controller started")
    
    async def stop(self) -> None:
        """Stop the adaptive controller."""
        logger.info("Stopping adaptive controller")
        
        # Cancel MAPE-K loop
        if self._mape_k_task:
            self._mape_k_task.cancel()
            try:
                await self._mape_k_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active adaptations to complete
        if self._active_adaptations:
            logger.info(f"Waiting for {len(self._active_adaptations)} active adaptations")
            await asyncio.gather(
                *[ctx.completion_event.wait() for ctx in self._active_adaptations.values()],
                return_exceptions=True
            )
        
        logger.info("Adaptive controller stopped")
    
    async def _handle_telemetry_event(self, event: TelemetryEvent) -> None:
        """Handle incoming telemetry event (Monitor phase)."""
        system_state = event.system_state
        
        # Update world model
        await self._world_model.update_system_state(system_state)
        
        # Queue for analysis
        await self._adaptation_queue.put(system_state)
        
        # Record monitoring metrics
        self._controller_metrics.telemetry_processed.labels(
            system_id=system_state.system_id
        ).inc()
    
    async def _mape_k_loop(self) -> None:
        """Main MAPE-K control loop."""
        while True:
            try:
                # Get next system state to analyze
                system_state = await asyncio.wait_for(
                    self._adaptation_queue.get(),
                    timeout=1.0
                )
                
                # Execute MAPE-K phases
                await self._execute_mape_k_cycle(system_state)
                
            except asyncio.TimeoutError:
                # No new states to process, continue loop
                continue
            except Exception as e:
                logger.error(f"MAPE-K loop error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_mape_k_cycle(self, system_state: SystemState) -> None:
        """Execute complete MAPE-K cycle for a system state."""
        cycle_start = time.time()
        system_id = system_state.system_id
        
        try:
            # ANALYZE: Assess adaptation needs
            adaptation_need = await self._analyze_adaptation_need(system_state)
            
            if adaptation_need.requires_adaptation:
                logger.info(
                    f"Adaptation needed for {system_id}: {adaptation_need.reason} "
                    f"(urgency: {adaptation_need.urgency.value})"
                )
                
                # PLAN: Generate adaptation plan
                adaptation_plan = await self._plan_adaptation(system_state, adaptation_need)
                
                if adaptation_plan:
                    # EXECUTE: Execute adaptation actions
                    await self._execute_adaptation(adaptation_plan)
                    
                    # KNOWLEDGE: Update knowledge base
                    await self._update_knowledge(system_state, adaptation_need, adaptation_plan)
            
            # Record cycle metrics
            cycle_duration = time.time() - cycle_start
            self._controller_metrics.mape_k_cycle_duration.labels(
                system_id=system_id
            ).observe(cycle_duration)
            
        except Exception as e:
            logger.error(f"MAPE-K cycle failed for {system_id}: {e}")
            
            self._controller_metrics.mape_k_cycle_errors.labels(
                system_id=system_id
            ).inc()
    
    async def _analyze_adaptation_need(self, system_state: SystemState) -> AdaptationNeed:
        """Analyze system state to determine adaptation needs."""
        analysis_start = time.time()
        
        try:
            # Get reasoning context
            reasoning_context = ReasoningContext(
                system_state=system_state,
                historical_states=await self._get_historical_context(system_state.system_id),
                learned_patterns=await self._get_relevant_patterns(system_state.system_id),
                system_dependencies=await self._world_model.get_system_dependencies(system_state.system_id)
            )
            
            # Use reasoning engine to analyze situation
            reasoning_result = await self._reasoning_engine.analyze_situation(reasoning_context)
            
            # Determine adaptation need based on reasoning
            adaptation_need = AdaptationNeed(
                system_id=system_state.system_id,
                requires_adaptation=reasoning_result.requires_action,
                reason=reasoning_result.primary_reason,
                urgency=self._determine_urgency(reasoning_result),
                confidence=reasoning_result.confidence,
                analysis_timestamp=datetime.utcnow(),
                reasoning_details=reasoning_result.details
            )
            
            # Record analysis metrics
            analysis_duration = time.time() - analysis_start
            self._controller_metrics.analysis_duration.labels(
                system_id=system_state.system_id
            ).observe(analysis_duration)
            
            return adaptation_need
            
        except Exception as e:
            logger.error(f"Analysis failed for {system_state.system_id}: {e}")
            
            # Return no adaptation needed on analysis failure
            return AdaptationNeed(
                system_id=system_state.system_id,
                requires_adaptation=False,
                reason="Analysis failed",
                urgency=ActionPriority.LOW,
                confidence=0.0,
                analysis_timestamp=datetime.utcnow()
            )
    
    async def _plan_adaptation(self, system_state: SystemState, 
                             adaptation_need: AdaptationNeed) -> Optional[AdaptationPlan]:
        """Plan adaptation actions based on analysis results."""
        planning_start = time.time()
        
        try:
            # Select appropriate control strategy
            strategy = await self._strategy_selector.select_strategy(
                system_state, adaptation_need, self._control_strategies
            )
            
            logger.info(f"Selected strategy {strategy.name} for {system_state.system_id}")
            
            # Generate adaptation actions using selected strategy
            actions = await strategy.generate_actions(
                system_state.system_id, 
                system_state.to_dict(),
                adaptation_need
            )
            
            if not actions:
                logger.warning(f"No actions generated by strategy {strategy.name}")
                return None
            
            # Create adaptation plan
            adaptation_plan = AdaptationPlan(
                plan_id=str(uuid.uuid4()),
                system_id=system_state.system_id,
                actions=actions,
                strategy_name=strategy.name,
                adaptation_need=adaptation_need,
                estimated_impact=await self._estimate_plan_impact(actions),
                plan_timestamp=datetime.utcnow()
            )
            
            # Validate plan
            validation_result = await self._validate_adaptation_plan(adaptation_plan)
            
            if not validation_result.is_valid:
                logger.warning(
                    f"Adaptation plan validation failed: {validation_result.errors}"
                )
                return None
            
            # Record planning metrics
            planning_duration = time.time() - planning_start
            self._controller_metrics.planning_duration.labels(
                system_id=system_state.system_id,
                strategy=strategy.name
            ).observe(planning_duration)
            
            return adaptation_plan
            
        except Exception as e:
            logger.error(f"Planning failed for {system_state.system_id}: {e}")
            return None
    
    async def _execute_adaptation(self, adaptation_plan: AdaptationPlan) -> None:
        """Execute adaptation plan."""
        execution_start = time.time()
        system_id = adaptation_plan.system_id
        
        try:
            # Create adaptation context
            adaptation_context = AdaptationContext(
                plan=adaptation_plan,
                start_time=datetime.utcnow(),
                completion_event=asyncio.Event()
            )
            
            # Track active adaptation
            self._active_adaptations[adaptation_plan.plan_id] = adaptation_context
            
            # Execute actions sequentially or in parallel based on dependencies
            execution_results = []
            
            for action in adaptation_plan.actions:
                # Publish adaptation action event
                await self._event_bus.publish(AdaptationActionEvent(
                    system_id=system_id,
                    action=action,
                    plan_id=adaptation_plan.plan_id
                ))
                
                # Wait for execution result (with timeout)
                result = await self._wait_for_execution_result(
                    action.action_id, 
                    timeout=action.timeout_seconds
                )
                
                execution_results.append(result)
                
                # Stop execution if action failed and plan requires all actions to succeed
                if result.is_failed and adaptation_plan.requires_all_success:
                    logger.error(f"Action {action.action_id} failed, stopping plan execution")
                    break
            
            # Update adaptation context
            adaptation_context.results = execution_results
            adaptation_context.end_time = datetime.utcnow()
            adaptation_context.completion_event.set()
            
            # Record execution metrics
            execution_duration = time.time() - execution_start
            successful_actions = sum(1 for r in execution_results if r.is_successful)
            
            self._controller_metrics.execution_duration.labels(
                system_id=system_id
            ).observe(execution_duration)
            
            self._controller_metrics.adaptation_success_rate.labels(
                system_id=system_id
            ).observe(successful_actions / len(execution_results))
            
            logger.info(
                f"Adaptation plan {adaptation_plan.plan_id} completed: "
                f"{successful_actions}/{len(execution_results)} actions successful"
            )
            
        except Exception as e:
            logger.error(f"Execution failed for plan {adaptation_plan.plan_id}: {e}")
        
        finally:
            # Clean up adaptation context
            if adaptation_plan.plan_id in self._active_adaptations:
                del self._active_adaptations[adaptation_plan.plan_id]
    
    async def _update_knowledge(self, system_state: SystemState, 
                              adaptation_need: AdaptationNeed,
                              adaptation_plan: AdaptationPlan) -> None:
        """Update knowledge base with adaptation experience."""
        try:
            # Create adaptation experience record
            experience = AdaptationExperience(
                experience_id=str(uuid.uuid4()),
                system_id=system_state.system_id,
                initial_state=system_state,
                adaptation_need=adaptation_need,
                adaptation_plan=adaptation_plan,
                results=self._active_adaptations.get(adaptation_plan.plan_id, {}).get('results', []),
                timestamp=datetime.utcnow()
            )
            
            # Store in knowledge base
            await self._knowledge_base.store_adaptation_experience(experience)
            
            # Trigger learning from this experience
            await self._event_bus.publish(AdaptationExperienceEvent(experience))
            
        except Exception as e:
            logger.error(f"Knowledge update failed: {e}")
    
    def _initialize_control_strategies(self) -> None:
        """Initialize available control strategies."""
        self._control_strategies = {
            "reactive": ReactiveControlStrategy(
                config=self._config.reactive_strategy
            ),
            "predictive": PredictiveControlStrategy(
                world_model=self._world_model,
                config=self._config.predictive_strategy
            ),
            "learning": LearningControlStrategy(
                knowledge_base=self._knowledge_base,
                config=self._config.learning_strategy
            )
        }
```

#### Control Strategy Selector
```python
class ControlStrategySelector:
    """Selects optimal control strategy based on system context."""
    
    def __init__(self, config: StrategySelectionConfig):
        self._config = config
        self._strategy_performance: Dict[str, StrategyPerformanceTracker] = {}
    
    async def select_strategy(self, system_state: SystemState,
                            adaptation_need: AdaptationNeed,
                            available_strategies: Dict[str, ControlStrategy]) -> ControlStrategy:
        """Select the best control strategy for the current situation."""
        
        # Calculate strategy scores
        strategy_scores = {}
        
        for strategy_name, strategy in available_strategies.items():
            score = await self._calculate_strategy_score(
                strategy_name, strategy, system_state, adaptation_need
            )
            strategy_scores[strategy_name] = score
        
        # Select strategy with highest score
        best_strategy_name = max(strategy_scores, key=strategy_scores.get)
        best_strategy = available_strategies[best_strategy_name]
        
        logger.info(
            f"Strategy selection for {system_state.system_id}: "
            f"{best_strategy_name} (score: {strategy_scores[best_strategy_name]:.3f})"
        )
        
        return best_strategy
    
    async def _calculate_strategy_score(self, strategy_name: str, 
                                      strategy: ControlStrategy,
                                      system_state: SystemState,
                                      adaptation_need: AdaptationNeed) -> float:
        """Calculate score for a strategy based on multiple factors."""
        
        # Base suitability score
        suitability_score = await strategy.calculate_suitability(system_state, adaptation_need)
        
        # Historical performance score
        performance_score = self._get_strategy_performance_score(strategy_name, system_state.system_id)
        
        # Urgency factor
        urgency_factor = self._get_urgency_factor(adaptation_need.urgency)
        
        # Confidence factor
        confidence_factor = adaptation_need.confidence
        
        # Combine scores with weights
        total_score = (
            suitability_score * self._config.suitability_weight +
            performance_score * self._config.performance_weight +
            urgency_factor * self._config.urgency_weight +
            confidence_factor * self._config.confidence_weight
        )
        
        return total_score
    
    def _get_strategy_performance_score(self, strategy_name: str, system_id: str) -> float:
        """Get historical performance score for strategy."""
        if strategy_name not in self._strategy_performance:
            return 0.5  # Neutral score for new strategies
        
        tracker = self._strategy_performance[strategy_name]
        return tracker.get_performance_score(system_id)
```

## Reasoning Engine (PolarisReasoningEngine)

### Purpose
Provides intelligent reasoning capabilities using multiple reasoning strategies to analyze system situations and recommend actions.

### Key Features
- **Multi-strategy Reasoning**: Combines statistical, causal, and experience-based reasoning
- **Situation Analysis**: Comprehensive analysis of system state and context
- **Confidence Scoring**: Provides confidence levels for reasoning results
- **Result Fusion**: Combines insights from multiple reasoning approaches
- **Contextual Reasoning**: Considers system dependencies and historical patterns

### Implementation

#### Reasoning Engine Core
```python
class PolarisReasoningEngine(Injectable):
    """Multi-strategy reasoning engine for system analysis and decision support."""
    
    def __init__(self, world_model: PolarisWorldModel,
                 knowledge_base: PolarisKnowledgeBase,
                 config: ReasoningEngineConfig):
        self._world_model = world_model
        self._knowledge_base = knowledge_base
        self._config = config
        
        # Reasoning strategies
        self._reasoning_strategies: List[ReasoningStrategy] = []
        self._initialize_reasoning_strategies()
        
        # Result fusion
        self._result_fusion = ReasoningResultFusion(config.fusion)
        
        # Reasoning metrics
        self._reasoning_metrics = ReasoningMetrics()
    
    async def analyze_situation(self, context: ReasoningContext) -> ReasoningResult:
        """Analyze system situation using multiple reasoning strategies."""
        analysis_start = time.time()
        
        try:
            # Execute all reasoning strategies
            strategy_results = []
            
            for strategy in self._reasoning_strategies:
                try:
                    result = await strategy.reason(context)
                    strategy_results.append((strategy.name, result))
                except Exception as e:
                    logger.error(f"Reasoning strategy {strategy.name} failed: {e}")
            
            if not strategy_results:
                raise ReasoningException("No reasoning strategies produced results")
            
            # Fuse results from multiple strategies
            fused_result = self._result_fusion.fuse_results(strategy_results)
            
            # Record reasoning metrics
            analysis_duration = time.time() - analysis_start
            self._reasoning_metrics.reasoning_duration.labels(
                system_id=context.system_state.system_id
            ).observe(analysis_duration)
            
            self._reasoning_metrics.reasoning_confidence.labels(
                system_id=context.system_state.system_id
            ).observe(fused_result.confidence)
            
            return fused_result
            
        except Exception as e:
            logger.error(f"Situation analysis failed: {e}")
            
            # Return default result on failure
            return ReasoningResult(
                requires_action=False,
                primary_reason="Analysis failed",
                confidence=0.0,
                details={"error": str(e)},
                reasoning_timestamp=datetime.utcnow()
            )
    
    async def recommend_actions(self, context: ReasoningContext) -> List[ActionRecommendation]:
        """Recommend specific actions based on situation analysis."""
        try:
            # First analyze the situation
            situation_analysis = await self.analyze_situation(context)
            
            if not situation_analysis.requires_action:
                return []
            
            # Get action recommendations from strategies
            all_recommendations = []
            
            for strategy in self._reasoning_strategies:
                if hasattr(strategy, 'recommend_actions'):
                    try:
                        recommendations = await strategy.recommend_actions(context)
                        all_recommendations.extend(recommendations)
                    except Exception as e:
                        logger.error(f"Action recommendation failed for {strategy.name}: {e}")
            
            # Deduplicate and rank recommendations
            ranked_recommendations = self._rank_action_recommendations(all_recommendations)
            
            return ranked_recommendations[:self._config.max_recommendations]
            
        except Exception as e:
            logger.error(f"Action recommendation failed: {e}")
            return []
    
    def _initialize_reasoning_strategies(self) -> None:
        """Initialize reasoning strategies."""
        self._reasoning_strategies = [
            StatisticalReasoningStrategy(
                world_model=self._world_model,
                config=self._config.statistical_reasoning
            ),
            CausalReasoningStrategy(
                knowledge_base=self._knowledge_base,
                config=self._config.causal_reasoning
            ),
            ExperienceBasedReasoningStrategy(
                knowledge_base=self._knowledge_base,
                config=self._config.experience_reasoning
            )
        ]

class StatisticalReasoningStrategy(ReasoningStrategy):
    """Statistical reasoning based on metrics analysis and thresholds."""
    
    def __init__(self, world_model: PolarisWorldModel, config: StatisticalReasoningConfig):
        super().__init__("statistical")
        self._world_model = world_model
        self._config = config
        self._threshold_analyzer = ThresholdAnalyzer(config)
        self._trend_analyzer = TrendAnalyzer(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform statistical reasoning on system state."""
        system_state = context.system_state
        
        # Analyze current metrics against thresholds
        threshold_analysis = self._threshold_analyzer.analyze_thresholds(system_state)
        
        # Analyze trends in historical data
        trend_analysis = await self._trend_analyzer.analyze_trends(
            context.historical_states
        )
        
        # Predict future state
        try:
            prediction = await self._world_model.predict_system_behavior(
                system_state.system_id, 
                time_horizon=self._config.prediction_horizon
            )
            prediction_analysis = self._analyze_prediction(prediction)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            prediction_analysis = {"prediction_available": False}
        
        # Determine if action is required
        requires_action = (
            threshold_analysis.has_violations or
            trend_analysis.has_concerning_trends or
            prediction_analysis.get("requires_action", False)
        )
        
        # Build reasoning details
        details = {
            "threshold_analysis": threshold_analysis.to_dict(),
            "trend_analysis": trend_analysis.to_dict(),
            "prediction_analysis": prediction_analysis,
            "statistical_indicators": self._calculate_statistical_indicators(system_state)
        }
        
        # Calculate confidence based on data quality and consistency
        confidence = self._calculate_statistical_confidence(
            threshold_analysis, trend_analysis, prediction_analysis
        )
        
        # Determine primary reason
        primary_reason = self._determine_primary_reason(
            threshold_analysis, trend_analysis, prediction_analysis
        )
        
        return ReasoningResult(
            requires_action=requires_action,
            primary_reason=primary_reason,
            confidence=confidence,
            details=details,
            reasoning_timestamp=datetime.utcnow(),
            strategy_name=self.name
        )

class CausalReasoningStrategy(ReasoningStrategy):
    """Causal reasoning based on system dependencies and cause-effect relationships."""
    
    def __init__(self, knowledge_base: PolarisKnowledgeBase, config: CausalReasoningConfig):
        super().__init__("causal")
        self._knowledge_base = knowledge_base
        self._config = config
        self._causal_analyzer = CausalAnalyzer(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform causal reasoning on system state and dependencies."""
        system_state = context.system_state
        
        # Analyze system dependencies
        dependencies = context.system_dependencies
        dependency_analysis = await self._analyze_dependency_impact(
            system_state, dependencies
        )
        
        # Look for causal patterns in learned knowledge
        causal_patterns = await self._find_causal_patterns(
            system_state.system_id, context.learned_patterns
        )
        
        # Analyze root causes of current issues
        root_cause_analysis = await self._analyze_root_causes(
            system_state, context.historical_states
        )
        
        # Determine if action is required based on causal analysis
        requires_action = (
            dependency_analysis.has_cascading_risks or
            len(causal_patterns) > 0 or
            root_cause_analysis.has_actionable_causes
        )
        
        # Build reasoning details
        details = {
            "dependency_analysis": dependency_analysis.to_dict(),
            "causal_patterns": [p.to_dict() for p in causal_patterns],
            "root_cause_analysis": root_cause_analysis.to_dict(),
            "causal_chains": self._identify_causal_chains(system_state, dependencies)
        }
        
        # Calculate confidence based on causal evidence strength
        confidence = self._calculate_causal_confidence(
            dependency_analysis, causal_patterns, root_cause_analysis
        )
        
        # Determine primary reason from causal analysis
        primary_reason = self._determine_causal_reason(
            dependency_analysis, causal_patterns, root_cause_analysis
        )
        
        return ReasoningResult(
            requires_action=requires_action,
            primary_reason=primary_reason,
            confidence=confidence,
            details=details,
            reasoning_timestamp=datetime.utcnow(),
            strategy_name=self.name
        )

class ExperienceBasedReasoningStrategy(ReasoningStrategy):
    """Experience-based reasoning using learned patterns and historical outcomes."""
    
    def __init__(self, knowledge_base: PolarisKnowledgeBase, config: ExperienceReasoningConfig):
        super().__init__("experience")
        self._knowledge_base = knowledge_base
        self._config = config
        self._pattern_matcher = PatternMatcher(config)
        self._outcome_analyzer = OutcomeAnalyzer(config)
    
    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform experience-based reasoning using learned patterns."""
        system_state = context.system_state
        
        # Find similar historical situations
        similar_situations = await self._find_similar_situations(
            system_state, context.learned_patterns
        )
        
        # Analyze outcomes of similar situations
        outcome_analysis = await self._analyze_historical_outcomes(
            similar_situations
        )
        
        # Match current situation to learned patterns
        pattern_matches = self._pattern_matcher.find_matching_patterns(
            system_state, context.learned_patterns
        )
        
        # Determine if action is required based on experience
        requires_action = (
            outcome_analysis.suggests_action or
            any(p.confidence > self._config.pattern_confidence_threshold 
                for p in pattern_matches)
        )
        
        # Build reasoning details
        details = {
            "similar_situations": [s.to_dict() for s in similar_situations],
            "outcome_analysis": outcome_analysis.to_dict(),
            "pattern_matches": [p.to_dict() for p in pattern_matches],
            "experience_summary": self._summarize_experience(similar_situations, outcome_analysis)
        }
        
        # Calculate confidence based on experience quality and quantity
        confidence = self._calculate_experience_confidence(
            similar_situations, outcome_analysis, pattern_matches
        )
        
        # Determine primary reason from experience
        primary_reason = self._determine_experience_reason(
            outcome_analysis, pattern_matches
        )
        
        return ReasoningResult(
            requires_action=requires_action,
            primary_reason=primary_reason,
            confidence=confidence,
            details=details,
            reasoning_timestamp=datetime.utcnow(),
            strategy_name=self.name
        )
```

## Control Strategies

### Reactive Control Strategy

**Purpose**: Implements threshold-based reactive control for immediate response to system issues.

```python
class ReactiveControlStrategy(ControlStrategy):
    """Reactive control strategy based on thresholds and rules."""
    
    def __init__(self, config: ReactiveControlConfig):
        super().__init__("reactive")
        self._config = config
        self._rule_engine = RuleEngine(config.rules)
        self._threshold_manager = ThresholdManager(config.thresholds)
    
    async def generate_actions(self, system_id: str, current_state: Dict[str, Any],
                             adaptation_need: AdaptationNeed) -> List[AdaptationAction]:
        """Generate reactive actions based on rules and thresholds."""
        
        # Convert state dict to structured format
        system_state = SystemState.from_dict(current_state)
        
        # Evaluate rules against current state
        triggered_rules = self._rule_engine.evaluate_rules(system_state)
        
        # Generate actions from triggered rules
        actions = []
        
        for rule in triggered_rules:
            rule_actions = await self._generate_actions_from_rule(
                rule, system_state, adaptation_need
            )
            actions.extend(rule_actions)
        
        # Prioritize actions based on urgency and impact
        prioritized_actions = self._prioritize_actions(actions, adaptation_need)
        
        return prioritized_actions[:self._config.max_actions_per_cycle]
    
    async def calculate_suitability(self, system_state: SystemState, 
                                  adaptation_need: AdaptationNeed) -> float:
        """Calculate suitability of reactive strategy for current situation."""
        
        # Reactive strategy is most suitable for:
        # 1. High urgency situations
        # 2. Clear threshold violations
        # 3. Simple, well-defined problems
        
        urgency_score = adaptation_need.urgency.numeric_value() / 4.0
        
        # Check for clear threshold violations
        threshold_violations = self._threshold_manager.check_violations(system_state)
        violation_score = min(len(threshold_violations) / 3.0, 1.0)
        
        # Check for rule matches
        rule_matches = len(self._rule_engine.evaluate_rules(system_state))
        rule_score = min(rule_matches / 2.0, 1.0)
        
        # Combine scores
        suitability = (urgency_score * 0.4 + violation_score * 0.4 + rule_score * 0.2)
        
        return min(suitability, 1.0)
```

### Predictive Control Strategy

**Purpose**: Uses predictive models to anticipate issues and take proactive actions.

```python
class PredictiveControlStrategy(ControlStrategy):
    """Predictive control strategy using world model forecasts."""
    
    def __init__(self, world_model: PolarisWorldModel, config: PredictiveControlConfig):
        super().__init__("predictive")
        self._world_model = world_model
        self._config = config
        self._mpc_controller = ModelPredictiveController(config.mpc)
        self._scenario_planner = ScenarioPlanner(config.scenarios)
    
    async def generate_actions(self, system_id: str, current_state: Dict[str, Any],
                             adaptation_need: AdaptationNeed) -> List[AdaptationAction]:
        """Generate predictive actions based on forecasts and scenarios."""
        
        system_state = SystemState.from_dict(current_state)
        
        # Generate predictions for multiple time horizons
        predictions = await self._generate_predictions(system_id)
        
        # Identify potential future problems
        future_issues = self._identify_future_issues(predictions)
        
        if not future_issues:
            return []
        
        # Generate scenarios for different action alternatives
        scenarios = await self._scenario_planner.generate_scenarios(
            system_state, future_issues
        )
        
        # Use MPC to find optimal action sequence
        optimal_actions = await self._mpc_controller.optimize_actions(
            system_state, scenarios, self._config.optimization_horizon
        )
        
        return optimal_actions
    
    async def calculate_suitability(self, system_state: SystemState, 
                                  adaptation_need: AdaptationNeed) -> float:
        """Calculate suitability of predictive strategy."""
        
        # Predictive strategy is most suitable for:
        # 1. Medium urgency situations (time for prediction)
        # 2. Systems with good predictive models
        # 3. Situations where proactive action is beneficial
        
        # Check if we have good prediction capability
        try:
            prediction = await self._world_model.predict_system_behavior(
                system_state.system_id, time_horizon=5
            )
            prediction_quality = prediction.confidence
        except Exception:
            prediction_quality = 0.0
        
        # Medium urgency is optimal for predictive control
        urgency_score = 1.0 - abs(adaptation_need.urgency.numeric_value() - 2.5) / 2.5
        
        # Consider system stability (predictive works better on stable systems)
        stability_score = self._assess_system_stability(system_state)
        
        suitability = (
            prediction_quality * 0.5 + 
            urgency_score * 0.3 + 
            stability_score * 0.2
        )
        
        return min(suitability, 1.0)
```

### Learning Control Strategy

**Purpose**: Uses learned patterns and experiences to make intelligent adaptation decisions.

```python
class LearningControlStrategy(ControlStrategy):
    """Learning-based control strategy using historical patterns and outcomes."""
    
    def __init__(self, knowledge_base: PolarisKnowledgeBase, config: LearningControlConfig):
        super().__init__("learning")
        self._knowledge_base = knowledge_base
        self._config = config
        self._pattern_matcher = PatternMatcher(config.pattern_matching)
        self._case_based_reasoner = CaseBasedReasoner(config.case_based_reasoning)
    
    async def generate_actions(self, system_id: str, current_state: Dict[str, Any],
                             adaptation_need: AdaptationNeed) -> List[AdaptationAction]:
        """Generate actions based on learned patterns and similar cases."""
        
        system_state = SystemState.from_dict(current_state)
        
        # Find similar historical cases
        similar_cases = await self._find_similar_cases(system_state, adaptation_need)
        
        if not similar_cases:
            logger.info(f"No similar cases found for {system_id}")
            return []
        
        # Use case-based reasoning to adapt solutions
        adapted_actions = await self._case_based_reasoner.adapt_solutions(
            system_state, adaptation_need, similar_cases
        )
        
        # Find matching learned patterns
        matching_patterns = await self._find_matching_patterns(system_state)
        
        # Generate actions from patterns
        pattern_actions = await self._generate_actions_from_patterns(
            matching_patterns, system_state, adaptation_need
        )
        
        # Combine and rank all actions
        all_actions = adapted_actions + pattern_actions
        ranked_actions = self._rank_actions_by_learning_confidence(all_actions)
        
        return ranked_actions[:self._config.max_actions_per_cycle]
    
    async def calculate_suitability(self, system_state: SystemState, 
                                  adaptation_need: AdaptationNeed) -> float:
        """Calculate suitability of learning strategy."""
        
        # Learning strategy is most suitable for:
        # 1. Systems with rich historical data
        # 2. Complex situations that benefit from experience
        # 3. Lower urgency situations (time for analysis)
        
        # Check availability of historical data
        historical_data_score = await self._assess_historical_data_availability(
            system_state.system_id
        )
        
        # Check for pattern matches
        patterns = await self._knowledge_base.query_patterns(
            "behavior_pattern", {"system_id": system_state.system_id}
        )
        pattern_score = min(len(patterns) / 10.0, 1.0)
        
        # Lower urgency is better for learning-based approach
        urgency_score = 1.0 - (adaptation_need.urgency.numeric_value() / 4.0)
        
        suitability = (
            historical_data_score * 0.4 + 
            pattern_score * 0.4 + 
            urgency_score * 0.2
        )
        
        return min(suitability, 1.0)
```

## Configuration Examples

### Control & Reasoning Configuration
```yaml
control_reasoning:
  adaptive_controller:
    mape_k_cycle_interval: 5  # seconds
    max_concurrent_adaptations: 10
    adaptation_timeout: 300  # seconds
    
    strategy_selection:
      suitability_weight: 0.4
      performance_weight: 0.3
      urgency_weight: 0.2
      confidence_weight: 0.1
  
  reasoning_engine:
    max_recommendations: 5
    
    statistical_reasoning:
      enabled: true
      prediction_horizon: 10  # minutes
      threshold_sensitivity: 0.8
      trend_analysis_window: 50
    
    causal_reasoning:
      enabled: true
      dependency_analysis_depth: 3
      causal_confidence_threshold: 0.7
      root_cause_analysis_window: 100
    
    experience_reasoning:
      enabled: true
      pattern_confidence_threshold: 0.6
      similarity_threshold: 0.8
      max_similar_cases: 10
    
    fusion:
      confidence_weighting: true
      consensus_threshold: 0.6
      conflict_resolution: "highest_confidence"
  
  control_strategies:
    reactive:
      enabled: true
      max_actions_per_cycle: 3
      
      thresholds:
        cpu_usage:
          warning: 0.8
          critical: 0.95
        memory_usage:
          warning: 0.85
          critical: 0.95
        error_rate:
          warning: 0.01
          critical: 0.05
      
      rules:
        - name: "high_cpu_scale_out"
          condition: "cpu_usage > 0.9"
          actions: ["scale_out"]
          priority: "high"
        
        - name: "high_error_rate_restart"
          condition: "error_rate > 0.05"
          actions: ["restart_service"]
          priority: "critical"
    
    predictive:
      enabled: true
      optimization_horizon: 30  # minutes
      
      mpc:
        prediction_horizon: 10
        control_horizon: 5
        optimization_method: "gradient_descent"
        constraints:
          - "cpu_usage <= 0.95"
          - "memory_usage <= 0.9"
          - "cost_increase <= 0.2"
      
      scenarios:
        generate_count: 5
        probability_threshold: 0.1
        impact_threshold: 0.05
    
    learning:
      enabled: true
      max_actions_per_cycle: 2
      
      pattern_matching:
        similarity_algorithm: "cosine"
        feature_weights:
          cpu_usage: 0.3
          memory_usage: 0.3
          response_time: 0.2
          error_rate: 0.2
      
      case_based_reasoning:
        adaptation_strategy: "weighted_combination"
        case_retention_limit: 1000
        case_similarity_threshold: 0.7
```

---

*Continue to [Adaptive Control](./14-adaptive-control.md) →*