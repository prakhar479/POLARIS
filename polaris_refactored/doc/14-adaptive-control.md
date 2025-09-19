# Adaptive Control

## Overview

Adaptive Control in POLARIS implements sophisticated control algorithms and strategies that enable systems to automatically adjust their behavior in response to changing conditions. This document details the control mechanisms, algorithms, and strategies that make POLARIS truly adaptive and intelligent.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Adaptive Control                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Control Loops   │ │ Control Algos   │ │ Adaptation Mgmt │   │
│  │ - MAPE-K Loop   │ │ - PID Control   │ │ - Action Queue  │   │
│  │ - Feedback Loop │ │ - Fuzzy Logic   │ │ - Coordination  │   │
│  │ - Feedforward   │ │ - Neural Nets   │ │ - Conflict Res  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Multi-Objective │ │ Learning & Adapt│ │ Safety & Bounds │   │
│  │ - Pareto Optimal│ │ - Online Learn  │ │ - Constraints   │   │
│  │ - Trade-offs    │ │ - Parameter Tune│ │ - Fail-safes   │   │
│  │ - Optimization  │ │ - Model Update  │ │ - Validation    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Control Loop Implementation

### MAPE-K Control Loop

**Purpose**: Implements the complete Monitor-Analyze-Plan-Execute-Knowledge feedback control loop with adaptive parameters.

```python
class MAPEKControlLoop:
    """Complete MAPE-K control loop with adaptive behavior."""
    
    def __init__(self, system_id: str, config: MAPEKConfig,
                 world_model: PolarisWorldModel,
                 knowledge_base: PolarisKnowledgeBase):
        self.system_id = system_id
        self._config = config
        self._world_model = world_model
        self._knowledge_base = knowledge_base
        
        # Control loop state
        self._loop_state = LoopState.IDLE
        self._current_cycle = 0
        self._adaptation_history: deque = deque(maxlen=config.history_size)
        
        # Adaptive parameters
        self._loop_parameters = AdaptiveLoopParameters(config)
        self._performance_tracker = LoopPerformanceTracker()
        
        # Control components
        self._monitor = AdaptiveMonitor(config.monitor)
        self._analyzer = AdaptiveAnalyzer(config.analyzer)
        self._planner = AdaptivePlanner(config.planner)
        self._executor = AdaptiveExecutor(config.executor)
        self._knowledge_manager = AdaptiveKnowledgeManager(config.knowledge)
    
    async def run_control_cycle(self) -> ControlCycleResult:
        """Execute one complete MAPE-K control cycle."""
        cycle_start = time.time()
        self._current_cycle += 1
        
        try:
            self._loop_state = LoopState.RUNNING
            
            # MONITOR: Collect and process system data
            monitoring_result = await self._monitor.collect_system_data(
                self.system_id, self._loop_parameters.monitoring_interval
            )
            
            # ANALYZE: Assess system state and identify adaptation needs
            analysis_result = await self._analyzer.analyze_system_state(
                monitoring_result, self._adaptation_history
            )
            
            # PLAN: Generate adaptation plan if needed
            planning_result = None
            if analysis_result.requires_adaptation:
                planning_result = await self._planner.generate_adaptation_plan(
                    analysis_result, self._world_model
                )
            
            # EXECUTE: Implement adaptation plan
            execution_result = None
            if planning_result and planning_result.has_valid_plan:
                execution_result = await self._executor.execute_adaptation_plan(
                    planning_result.plan
                )
            
            # KNOWLEDGE: Update knowledge base and learn from experience
            knowledge_result = await self._knowledge_manager.update_knowledge(
                monitoring_result, analysis_result, planning_result, execution_result
            )
            
            # Create cycle result
            cycle_result = ControlCycleResult(
                cycle_number=self._current_cycle,
                system_id=self.system_id,
                monitoring_result=monitoring_result,
                analysis_result=analysis_result,
                planning_result=planning_result,
                execution_result=execution_result,
                knowledge_result=knowledge_result,
                cycle_duration=time.time() - cycle_start,
                timestamp=datetime.utcnow()
            )
            
            # Update adaptation history
            self._adaptation_history.append(cycle_result)
            
            # Adapt loop parameters based on performance
            await self._adapt_loop_parameters(cycle_result)
            
            # Update performance metrics
            self._performance_tracker.record_cycle(cycle_result)
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"MAPE-K cycle failed for {self.system_id}: {e}")
            
            return ControlCycleResult(
                cycle_number=self._current_cycle,
                system_id=self.system_id,
                error=str(e),
                cycle_duration=time.time() - cycle_start,
                timestamp=datetime.utcnow()
            )
        
        finally:
            self._loop_state = LoopState.IDLE
    
    async def _adapt_loop_parameters(self, cycle_result: ControlCycleResult) -> None:
        """Adapt loop parameters based on cycle performance."""
        
        # Analyze cycle performance
        performance_metrics = self._performance_tracker.get_recent_performance()
        
        # Adapt monitoring interval based on system volatility
        if performance_metrics.system_volatility > self._config.high_volatility_threshold:
            # Increase monitoring frequency for volatile systems
            new_interval = max(
                self._loop_parameters.monitoring_interval * 0.8,
                self._config.min_monitoring_interval
            )
            self._loop_parameters.monitoring_interval = new_interval
        elif performance_metrics.system_volatility < self._config.low_volatility_threshold:
            # Decrease monitoring frequency for stable systems
            new_interval = min(
                self._loop_parameters.monitoring_interval * 1.2,
                self._config.max_monitoring_interval
            )
            self._loop_parameters.monitoring_interval = new_interval
        
        # Adapt analysis sensitivity based on false positive rate
        if performance_metrics.false_positive_rate > self._config.max_false_positive_rate:
            # Reduce sensitivity to decrease false positives
            self._loop_parameters.analysis_sensitivity *= 0.95
        elif performance_metrics.false_negative_rate > self._config.max_false_negative_rate:
            # Increase sensitivity to catch more issues
            self._loop_parameters.analysis_sensitivity *= 1.05
        
        # Adapt planning horizon based on prediction accuracy
        if performance_metrics.prediction_accuracy > self._config.high_accuracy_threshold:
            # Extend planning horizon for accurate predictions
            self._loop_parameters.planning_horizon = min(
                self._loop_parameters.planning_horizon * 1.1,
                self._config.max_planning_horizon
            )
        elif performance_metrics.prediction_accuracy < self._config.low_accuracy_threshold:
            # Shorten planning horizon for inaccurate predictions
            self._loop_parameters.planning_horizon = max(
                self._loop_parameters.planning_horizon * 0.9,
                self._config.min_planning_horizon
            )
        
        logger.debug(
            f"Adapted loop parameters for {self.system_id}: "
            f"monitoring_interval={self._loop_parameters.monitoring_interval:.1f}s, "
            f"analysis_sensitivity={self._loop_parameters.analysis_sensitivity:.3f}, "
            f"planning_horizon={self._loop_parameters.planning_horizon:.1f}min"
        )
```

### Feedback Control Systems

**Purpose**: Implements classical feedback control algorithms adapted for system management.

```python
class PIDController:
    """PID controller adapted for system resource management."""
    
    def __init__(self, config: PIDConfig):
        self._config = config
        
        # PID parameters
        self.kp = config.proportional_gain
        self.ki = config.integral_gain
        self.kd = config.derivative_gain
        
        # Control state
        self._previous_error = 0.0
        self._integral = 0.0
        self._previous_time = None
        
        # Adaptive parameters
        self._adaptive_gains = config.adaptive_gains
        self._gain_scheduler = GainScheduler(config.gain_scheduling) if config.gain_scheduling else None
        
        # Anti-windup and saturation
        self._integral_limit = config.integral_limit
        self._output_limits = config.output_limits
    
    def compute_control_output(self, setpoint: float, measured_value: float, 
                             current_time: datetime) -> ControlOutput:
        """Compute PID control output."""
        
        # Calculate error
        error = setpoint - measured_value
        
        # Calculate time delta
        if self._previous_time is None:
            dt = self._config.default_dt
        else:
            dt = (current_time - self._previous_time).total_seconds()
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with anti-windup
        self._integral += error * dt
        if self._integral_limit:
            self._integral = max(min(self._integral, self._integral_limit), -self._integral_limit)
        integral = self.ki * self._integral
        
        # Derivative term
        if dt > 0:
            derivative = self.kd * (error - self._previous_error) / dt
        else:
            derivative = 0.0
        
        # Compute total output
        output = proportional + integral + derivative
        
        # Apply output limits
        if self._output_limits:
            output = max(min(output, self._output_limits.max), self._output_limits.min)
        
        # Adapt gains if configured
        if self._gain_scheduler:
            self._adapt_gains(measured_value, error)
        
        # Update state
        self._previous_error = error
        self._previous_time = current_time
        
        return ControlOutput(
            output=output,
            proportional_term=proportional,
            integral_term=integral,
            derivative_term=derivative,
            error=error,
            timestamp=current_time
        )
    
    def _adapt_gains(self, measured_value: float, error: float) -> None:
        """Adapt PID gains based on system behavior."""
        if self._gain_scheduler:
            new_gains = self._gain_scheduler.schedule_gains(measured_value, error)
            self.kp = new_gains.kp
            self.ki = new_gains.ki
            self.kd = new_gains.kd

class FuzzyController:
    """Fuzzy logic controller for handling uncertainty and non-linear behavior."""
    
    def __init__(self, config: FuzzyControlConfig):
        self._config = config
        
        # Fuzzy sets and rules
        self._input_variables = self._create_input_variables(config.inputs)
        self._output_variables = self._create_output_variables(config.outputs)
        self._rules = self._create_fuzzy_rules(config.rules)
        
        # Inference engine
        self._inference_engine = FuzzyInferenceEngine(config.inference)
    
    def compute_fuzzy_output(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Compute fuzzy control output."""
        
        # Fuzzification: Convert crisp inputs to fuzzy sets
        fuzzified_inputs = {}
        for var_name, crisp_value in inputs.items():
            if var_name in self._input_variables:
                fuzzified_inputs[var_name] = self._fuzzify_input(
                    var_name, crisp_value
                )
        
        # Rule evaluation: Apply fuzzy rules
        rule_outputs = []
        for rule in self._rules:
            rule_strength = self._evaluate_rule(rule, fuzzified_inputs)
            if rule_strength > 0:
                rule_outputs.append((rule, rule_strength))
        
        # Aggregation: Combine rule outputs
        aggregated_outputs = self._aggregate_rule_outputs(rule_outputs)
        
        # Defuzzification: Convert fuzzy outputs to crisp values
        crisp_outputs = {}
        for var_name, fuzzy_output in aggregated_outputs.items():
            crisp_outputs[var_name] = self._defuzzify_output(var_name, fuzzy_output)
        
        return crisp_outputs
    
    def _fuzzify_input(self, variable_name: str, crisp_value: float) -> Dict[str, float]:
        """Convert crisp input to fuzzy membership values."""
        variable = self._input_variables[variable_name]
        memberships = {}
        
        for set_name, membership_function in variable.fuzzy_sets.items():
            membership_degree = membership_function.compute_membership(crisp_value)
            if membership_degree > 0:
                memberships[set_name] = membership_degree
        
        return memberships
    
    def _evaluate_rule(self, rule: FuzzyRule, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """Evaluate a fuzzy rule and return its strength."""
        
        # Evaluate antecedent (IF part)
        antecedent_strength = 1.0
        
        for condition in rule.antecedent:
            var_name = condition.variable
            set_name = condition.fuzzy_set
            
            if var_name in fuzzified_inputs and set_name in fuzzified_inputs[var_name]:
                membership = fuzzified_inputs[var_name][set_name]
                
                # Apply fuzzy operator (AND = min, OR = max)
                if condition.operator == FuzzyOperator.AND:
                    antecedent_strength = min(antecedent_strength, membership)
                elif condition.operator == FuzzyOperator.OR:
                    antecedent_strength = max(antecedent_strength, membership)
            else:
                # Variable or set not found, rule doesn't apply
                return 0.0
        
        return antecedent_strength
```

## Multi-Objective Optimization

### Pareto Optimization

**Purpose**: Handles trade-offs between conflicting objectives like performance, cost, and reliability.

```python
class ParetoOptimizer:
    """Multi-objective optimizer using Pareto efficiency principles."""
    
    def __init__(self, config: ParetoOptimizerConfig):
        self._config = config
        self._objectives = config.objectives
        self._constraints = config.constraints
        
        # Optimization algorithms
        self._nsga2 = NSGA2Algorithm(config.nsga2) if config.use_nsga2 else None
        self._moea_d = MOEADAlgorithm(config.moea_d) if config.use_moea_d else None
        
        # Solution archive
        self._pareto_archive = ParetoArchive(config.archive_size)
    
    async def optimize_adaptations(self, system_state: SystemState,
                                 candidate_actions: List[AdaptationAction],
                                 world_model: PolarisWorldModel) -> OptimizationResult:
        """Find Pareto-optimal adaptation solutions."""
        
        # Evaluate all candidate actions
        evaluated_solutions = []
        
        for action in candidate_actions:
            # Simulate action impact
            simulation_result = await world_model.simulate_adaptation_impact(
                system_state.system_id, action
            )
            
            # Calculate objective values
            objective_values = await self._calculate_objective_values(
                system_state, action, simulation_result
            )
            
            # Check constraints
            constraint_violations = self._check_constraints(
                system_state, action, simulation_result
            )
            
            solution = OptimizationSolution(
                action=action,
                objective_values=objective_values,
                constraint_violations=constraint_violations,
                simulation_result=simulation_result
            )
            
            evaluated_solutions.append(solution)
        
        # Find Pareto front
        pareto_front = self._find_pareto_front(evaluated_solutions)
        
        # Select best solution from Pareto front
        selected_solution = self._select_solution_from_pareto_front(
            pareto_front, system_state
        )
        
        # Update archive
        self._pareto_archive.update(pareto_front)
        
        return OptimizationResult(
            selected_solution=selected_solution,
            pareto_front=pareto_front,
            all_solutions=evaluated_solutions,
            optimization_timestamp=datetime.utcnow()
        )
    
    def _find_pareto_front(self, solutions: List[OptimizationSolution]) -> List[OptimizationSolution]:
        """Find Pareto-optimal solutions."""
        pareto_front = []
        
        for solution in solutions:
            is_dominated = False
            
            # Check if solution is dominated by any other solution
            for other_solution in solutions:
                if solution != other_solution:
                    if self._dominates(other_solution, solution):
                        is_dominated = True
                        break
            
            # If not dominated, add to Pareto front
            if not is_dominated:
                pareto_front.append(solution)
        
        return pareto_front
    
    def _dominates(self, solution_a: OptimizationSolution, 
                  solution_b: OptimizationSolution) -> bool:
        """Check if solution A dominates solution B."""
        
        # Solution A dominates B if:
        # 1. A is at least as good as B in all objectives
        # 2. A is strictly better than B in at least one objective
        
        at_least_as_good = True
        strictly_better = False
        
        for obj_name in self._objectives:
            value_a = solution_a.objective_values.get(obj_name, 0.0)
            value_b = solution_b.objective_values.get(obj_name, 0.0)
            
            # Determine if objective should be maximized or minimized
            objective_config = self._objectives[obj_name]
            
            if objective_config.direction == ObjectiveDirection.MAXIMIZE:
                if value_a < value_b:
                    at_least_as_good = False
                    break
                elif value_a > value_b:
                    strictly_better = True
            else:  # MINIMIZE
                if value_a > value_b:
                    at_least_as_good = False
                    break
                elif value_a < value_b:
                    strictly_better = True
        
        return at_least_as_good and strictly_better
    
    async def _calculate_objective_values(self, system_state: SystemState,
                                        action: AdaptationAction,
                                        simulation_result: SimulationResult) -> Dict[str, float]:
        """Calculate objective function values for a solution."""
        objective_values = {}
        
        for obj_name, objective_config in self._objectives.items():
            if obj_name == "performance":
                # Performance objective based on response time and throughput
                current_response_time = system_state.get_metric_value("response_time") or 0.0
                predicted_response_time = simulation_result.expected_changes.get("response_time", 0.0)
                new_response_time = current_response_time + predicted_response_time
                
                # Lower response time is better (minimize)
                objective_values[obj_name] = -new_response_time
                
            elif obj_name == "cost":
                # Cost objective based on resource usage
                current_cost = self._calculate_current_cost(system_state)
                cost_change = simulation_result.expected_changes.get("cost", 0.0)
                new_cost = current_cost + cost_change
                
                # Lower cost is better (minimize)
                objective_values[obj_name] = -new_cost
                
            elif obj_name == "reliability":
                # Reliability objective based on error rate and availability
                current_error_rate = system_state.get_metric_value("error_rate") or 0.0
                predicted_error_rate = simulation_result.expected_changes.get("error_rate", 0.0)
                new_error_rate = current_error_rate + predicted_error_rate
                
                # Lower error rate means higher reliability (maximize)
                reliability_score = 1.0 - new_error_rate
                objective_values[obj_name] = reliability_score
        
        return objective_values
```

### Utility Function Optimization

**Purpose**: Optimizes a weighted utility function combining multiple objectives.

```python
class UtilityFunctionOptimizer:
    """Optimizer using weighted utility functions for multi-objective problems."""
    
    def __init__(self, config: UtilityOptimizerConfig):
        self._config = config
        self._utility_function = UtilityFunction(config.utility_function)
        self._weight_adapter = WeightAdapter(config.weight_adaptation)
    
    async def optimize_utility(self, system_state: SystemState,
                             candidate_actions: List[AdaptationAction],
                             world_model: PolarisWorldModel) -> UtilityOptimizationResult:
        """Find action that maximizes utility function."""
        
        best_action = None
        best_utility = float('-inf')
        action_utilities = []
        
        for action in candidate_actions:
            # Simulate action impact
            simulation_result = await world_model.simulate_adaptation_impact(
                system_state.system_id, action
            )
            
            # Calculate utility
            utility_score = await self._calculate_utility_score(
                system_state, action, simulation_result
            )
            
            action_utilities.append((action, utility_score))
            
            if utility_score > best_utility:
                best_utility = utility_score
                best_action = action
        
        # Adapt weights based on recent performance
        await self._weight_adapter.adapt_weights(
            system_state.system_id, action_utilities
        )
        
        return UtilityOptimizationResult(
            best_action=best_action,
            best_utility=best_utility,
            all_utilities=action_utilities,
            current_weights=self._utility_function.get_weights(),
            optimization_timestamp=datetime.utcnow()
        )
    
    async def _calculate_utility_score(self, system_state: SystemState,
                                     action: AdaptationAction,
                                     simulation_result: SimulationResult) -> float:
        """Calculate utility score for an action."""
        
        # Get current objective values
        current_objectives = self._extract_current_objectives(system_state)
        
        # Calculate predicted objective values after action
        predicted_objectives = self._calculate_predicted_objectives(
            current_objectives, simulation_result
        )
        
        # Calculate utility using weighted sum
        utility_score = self._utility_function.calculate_utility(predicted_objectives)
        
        # Apply risk adjustment based on uncertainty
        risk_adjustment = self._calculate_risk_adjustment(
            simulation_result.confidence, action.estimated_impact
        )
        
        adjusted_utility = utility_score * risk_adjustment
        
        return adjusted_utility
    
    def _calculate_risk_adjustment(self, confidence: float, 
                                 estimated_impact: Optional[Dict[str, float]]) -> float:
        """Calculate risk adjustment factor based on uncertainty."""
        
        # Base risk adjustment on confidence
        confidence_factor = confidence
        
        # Adjust for impact magnitude (higher impact = higher risk)
        if estimated_impact:
            impact_magnitude = sum(abs(v) for v in estimated_impact.values())
            impact_factor = 1.0 / (1.0 + impact_magnitude * self._config.risk_aversion)
        else:
            impact_factor = 1.0
        
        # Combine factors
        risk_adjustment = confidence_factor * impact_factor
        
        return max(risk_adjustment, self._config.min_risk_adjustment)
```

## Learning and Adaptation

### Online Learning

**Purpose**: Continuously learns and adapts control parameters based on system behavior and outcomes.

```python
class OnlineLearningController:
    """Controller that learns and adapts its parameters online."""
    
    def __init__(self, config: OnlineLearningConfig):
        self._config = config
        
        # Learning algorithms
        self._parameter_learner = ParameterLearner(config.parameter_learning)
        self._model_learner = ModelLearner(config.model_learning)
        self._strategy_learner = StrategyLearner(config.strategy_learning)
        
        # Learning state
        self._learning_history: deque = deque(maxlen=config.history_size)
        self._performance_tracker = PerformanceTracker()
        
        # Adaptive parameters
        self._current_parameters = ControlParameters(config.initial_parameters)
        self._parameter_bounds = config.parameter_bounds
    
    async def learn_from_outcome(self, control_action: AdaptationAction,
                               outcome: ExecutionResult,
                               system_state_before: SystemState,
                               system_state_after: Optional[SystemState]) -> None:
        """Learn from the outcome of a control action."""
        
        # Create learning sample
        learning_sample = LearningSample(
            action=control_action,
            outcome=outcome,
            state_before=system_state_before,
            state_after=system_state_after,
            timestamp=datetime.utcnow()
        )
        
        # Add to learning history
        self._learning_history.append(learning_sample)
        
        # Update performance tracking
        self._performance_tracker.record_outcome(learning_sample)
        
        # Learn parameter adjustments
        if len(self._learning_history) >= self._config.min_learning_samples:
            await self._learn_parameter_adjustments()
        
        # Learn model improvements
        await self._learn_model_improvements(learning_sample)
        
        # Learn strategy effectiveness
        await self._learn_strategy_effectiveness(learning_sample)
    
    async def _learn_parameter_adjustments(self) -> None:
        """Learn adjustments to control parameters."""
        
        # Analyze recent performance
        recent_performance = self._performance_tracker.get_recent_performance()
        
        # Use reinforcement learning to adjust parameters
        parameter_adjustments = await self._parameter_learner.learn_adjustments(
            self._current_parameters,
            recent_performance,
            list(self._learning_history)
        )
        
        # Apply parameter adjustments with bounds checking
        for param_name, adjustment in parameter_adjustments.items():
            if param_name in self._current_parameters:
                current_value = getattr(self._current_parameters, param_name)
                new_value = current_value + adjustment
                
                # Apply bounds
                if param_name in self._parameter_bounds:
                    bounds = self._parameter_bounds[param_name]
                    new_value = max(min(new_value, bounds.max), bounds.min)
                
                setattr(self._current_parameters, param_name, new_value)
        
        logger.info(f"Updated control parameters: {parameter_adjustments}")
    
    async def _learn_model_improvements(self, learning_sample: LearningSample) -> None:
        """Learn improvements to internal models."""
        
        # Update prediction models based on actual outcomes
        if learning_sample.state_after:
            prediction_error = self._calculate_prediction_error(learning_sample)
            
            if prediction_error > self._config.model_update_threshold:
                await self._model_learner.update_models(
                    learning_sample, prediction_error
                )
        
        # Update cost models
        actual_cost = self._calculate_actual_cost(learning_sample)
        estimated_cost = learning_sample.action.estimated_impact.get("cost", 0.0)
        
        cost_error = abs(actual_cost - estimated_cost)
        if cost_error > self._config.cost_model_threshold:
            await self._model_learner.update_cost_model(
                learning_sample, cost_error
            )
    
    def get_adapted_parameters(self) -> ControlParameters:
        """Get current adapted control parameters."""
        return self._current_parameters.copy()

class ReinforcementLearningController:
    """Controller using reinforcement learning for action selection."""
    
    def __init__(self, config: RLControllerConfig):
        self._config = config
        
        # RL components
        self._state_encoder = StateEncoder(config.state_encoding)
        self._action_encoder = ActionEncoder(config.action_encoding)
        self._q_network = QNetwork(config.q_network)
        self._experience_replay = ExperienceReplay(config.experience_replay)
        
        # Learning parameters
        self._epsilon = config.initial_epsilon
        self._epsilon_decay = config.epsilon_decay
        self._learning_rate = config.learning_rate
        
        # Training state
        self._training_step = 0
        self._last_state = None
        self._last_action = None
    
    async def select_action(self, system_state: SystemState,
                          available_actions: List[AdaptationAction]) -> AdaptationAction:
        """Select action using epsilon-greedy policy."""
        
        # Encode current state
        state_vector = self._state_encoder.encode_state(system_state)
        
        # Epsilon-greedy action selection
        if random.random() < self._epsilon:
            # Explore: random action
            selected_action = random.choice(available_actions)
        else:
            # Exploit: best action according to Q-network
            action_values = {}
            
            for action in available_actions:
                action_vector = self._action_encoder.encode_action(action)
                q_value = self._q_network.predict_q_value(state_vector, action_vector)
                action_values[action.action_id] = q_value
            
            # Select action with highest Q-value
            best_action_id = max(action_values, key=action_values.get)
            selected_action = next(a for a in available_actions if a.action_id == best_action_id)
        
        # Store for learning
        self._last_state = state_vector
        self._last_action = self._action_encoder.encode_action(selected_action)
        
        return selected_action
    
    async def learn_from_reward(self, reward: float, next_state: SystemState,
                              done: bool = False) -> None:
        """Learn from reward signal."""
        
        if self._last_state is not None and self._last_action is not None:
            # Encode next state
            next_state_vector = self._state_encoder.encode_state(next_state)
            
            # Create experience tuple
            experience = Experience(
                state=self._last_state,
                action=self._last_action,
                reward=reward,
                next_state=next_state_vector,
                done=done
            )
            
            # Add to experience replay buffer
            self._experience_replay.add_experience(experience)
            
            # Train Q-network if enough experiences
            if len(self._experience_replay) >= self._config.min_replay_size:
                await self._train_q_network()
            
            # Decay epsilon
            self._epsilon = max(
                self._epsilon * self._epsilon_decay,
                self._config.min_epsilon
            )
            
            self._training_step += 1
    
    async def _train_q_network(self) -> None:
        """Train Q-network using experience replay."""
        
        # Sample batch of experiences
        batch = self._experience_replay.sample_batch(self._config.batch_size)
        
        # Prepare training data
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        # Calculate target Q-values
        next_q_values = self._q_network.predict_batch(next_states, actions)
        target_q_values = rewards + (1 - dones) * self._config.gamma * next_q_values
        
        # Train network
        loss = await self._q_network.train_batch(states, actions, target_q_values)
        
        if self._training_step % self._config.log_interval == 0:
            logger.info(f"RL training step {self._training_step}, loss: {loss:.4f}, epsilon: {self._epsilon:.3f}")
```

## Safety and Constraints

### Safety Constraints

**Purpose**: Ensures that adaptive control actions never violate safety constraints or cause system instability.

```python
class SafetyConstraintManager:
    """Manages safety constraints and validates control actions."""
    
    def __init__(self, config: SafetyConstraintConfig):
        self._config = config
        
        # Constraint types
        self._hard_constraints = HardConstraintSet(config.hard_constraints)
        self._soft_constraints = SoftConstraintSet(config.soft_constraints)
        self._temporal_constraints = TemporalConstraintSet(config.temporal_constraints)
        
        # Safety monitors
        self._safety_monitors = [
            ResourceSafetyMonitor(config.resource_safety),
            PerformanceSafetyMonitor(config.performance_safety),
            StabilitySafetyMonitor(config.stability_safety)
        ]
        
        # Constraint violation history
        self._violation_history: deque = deque(maxlen=config.history_size)
    
    async def validate_action_safety(self, action: AdaptationAction,
                                   current_state: SystemState,
                                   predicted_state: Optional[SystemState] = None) -> SafetyValidationResult:
        """Validate that an action is safe to execute."""
        
        validation_start = time.time()
        violations = []
        warnings = []
        
        # Check hard constraints (must not be violated)
        hard_violations = await self._hard_constraints.check_violations(
            action, current_state, predicted_state
        )
        violations.extend(hard_violations)
        
        # Check soft constraints (should not be violated)
        soft_violations = await self._soft_constraints.check_violations(
            action, current_state, predicted_state
        )
        warnings.extend(soft_violations)
        
        # Check temporal constraints (rate limits, cooldowns)
        temporal_violations = await self._temporal_constraints.check_violations(
            action, self._violation_history
        )
        violations.extend(temporal_violations)
        
        # Run safety monitors
        for monitor in self._safety_monitors:
            monitor_result = await monitor.check_safety(
                action, current_state, predicted_state
            )
            
            violations.extend(monitor_result.violations)
            warnings.extend(monitor_result.warnings)
        
        # Determine overall safety
        is_safe = len(violations) == 0
        
        # Create validation result
        result = SafetyValidationResult(
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            validation_duration=time.time() - validation_start,
            timestamp=datetime.utcnow()
        )
        
        # Record violations for temporal constraint checking
        if violations:
            self._violation_history.append(ConstraintViolationRecord(
                action=action,
                violations=violations,
                timestamp=datetime.utcnow()
            ))
        
        return result
    
    async def get_safe_action_bounds(self, action_type: str,
                                   current_state: SystemState) -> ActionBounds:
        """Get safe parameter bounds for an action type."""
        
        bounds = ActionBounds()
        
        # Get bounds from hard constraints
        hard_bounds = await self._hard_constraints.get_action_bounds(
            action_type, current_state
        )
        bounds.merge_bounds(hard_bounds)
        
        # Get bounds from safety monitors
        for monitor in self._safety_monitors:
            monitor_bounds = await monitor.get_safe_bounds(
                action_type, current_state
            )
            bounds.merge_bounds(monitor_bounds)
        
        return bounds

class StabilitySafetyMonitor:
    """Monitors system stability and prevents destabilizing actions."""
    
    def __init__(self, config: StabilityMonitorConfig):
        self._config = config
        self._stability_analyzer = StabilityAnalyzer(config)
        self._oscillation_detector = OscillationDetector(config)
    
    async def check_safety(self, action: AdaptationAction,
                         current_state: SystemState,
                         predicted_state: Optional[SystemState]) -> SafetyCheckResult:
        """Check if action could destabilize the system."""
        
        violations = []
        warnings = []
        
        # Check for potential oscillations
        oscillation_risk = await self._oscillation_detector.assess_oscillation_risk(
            action, current_state
        )
        
        if oscillation_risk.risk_level > self._config.max_oscillation_risk:
            violations.append(SafetyViolation(
                type="oscillation_risk",
                severity="high",
                message=f"Action may cause oscillations: {oscillation_risk.reason}",
                risk_level=oscillation_risk.risk_level
            ))
        
        # Check system stability margins
        stability_margins = await self._stability_analyzer.calculate_stability_margins(
            current_state, predicted_state
        )
        
        for margin_name, margin_value in stability_margins.items():
            if margin_value < self._config.min_stability_margins.get(margin_name, 0.0):
                violations.append(SafetyViolation(
                    type="stability_margin",
                    severity="medium",
                    message=f"Insufficient {margin_name} stability margin: {margin_value:.3f}",
                    margin_name=margin_name,
                    margin_value=margin_value
                ))
        
        # Check for rapid parameter changes
        if self._is_rapid_change(action, current_state):
            warnings.append(SafetyWarning(
                type="rapid_change",
                message="Action involves rapid parameter changes",
                recommendation="Consider gradual adjustment"
            ))
        
        return SafetyCheckResult(
            violations=violations,
            warnings=warnings,
            stability_assessment=stability_margins
        )
    
    def _is_rapid_change(self, action: AdaptationAction, 
                        current_state: SystemState) -> bool:
        """Check if action involves rapid parameter changes."""
        
        if action.action_type == "scale_out":
            current_replicas = current_state.get_metric_value("replicas") or 1
            target_replicas = action.get_parameter("replicas", current_replicas)
            
            change_rate = abs(target_replicas - current_replicas) / current_replicas
            return change_rate > self._config.max_change_rate
        
        return False
```

## Configuration Examples

### Adaptive Control Configuration
```yaml
adaptive_control:
  mape_k_loop:
    history_size: 100
    high_volatility_threshold: 0.8
    low_volatility_threshold: 0.2
    max_false_positive_rate: 0.1
    max_false_negative_rate: 0.05
    
    monitor:
      default_interval: 30  # seconds
      min_interval: 5
      max_interval: 300
    
    analyzer:
      sensitivity: 0.7
      trend_window: 20
      anomaly_threshold: 2.0
    
    planner:
      horizon: 10  # minutes
      min_horizon: 2
      max_horizon: 60
    
    executor:
      timeout: 300  # seconds
      retry_attempts: 3
      parallel_execution: false
  
  pid_controller:
    proportional_gain: 1.0
    integral_gain: 0.1
    derivative_gain: 0.05
    
    adaptive_gains: true
    integral_limit: 10.0
    
    output_limits:
      min: -1.0
      max: 1.0
    
    gain_scheduling:
      enabled: true
      operating_points:
        - range: [0.0, 0.5]
          gains: {kp: 0.8, ki: 0.05, kd: 0.02}
        - range: [0.5, 0.8]
          gains: {kp: 1.0, ki: 0.1, kd: 0.05}
        - range: [0.8, 1.0]
          gains: {kp: 1.2, ki: 0.15, kd: 0.08}
  
  multi_objective:
    pareto_optimization:
      enabled: true
      archive_size: 100
      
      objectives:
        performance:
          direction: "maximize"
          weight: 0.4
        cost:
          direction: "minimize"
          weight: 0.3
        reliability:
          direction: "maximize"
          weight: 0.3
      
      constraints:
        - "cpu_usage <= 0.95"
        - "memory_usage <= 0.9"
        - "cost_increase <= 0.2"
    
    utility_function:
      weights:
        performance: 0.4
        cost: 0.3
        reliability: 0.3
      
      weight_adaptation:
        enabled: true
        adaptation_rate: 0.1
        performance_window: 50
  
  online_learning:
    enabled: true
    min_learning_samples: 20
    history_size: 500
    
    parameter_learning:
      algorithm: "gradient_descent"
      learning_rate: 0.01
      momentum: 0.9
    
    model_learning:
      update_threshold: 0.1
      cost_model_threshold: 0.05
    
    reinforcement_learning:
      enabled: false
      initial_epsilon: 0.1
      epsilon_decay: 0.995
      min_epsilon: 0.01
      learning_rate: 0.001
      gamma: 0.95
      batch_size: 32
      min_replay_size: 1000
  
  safety_constraints:
    hard_constraints:
      - "cpu_usage <= 1.0"
      - "memory_usage <= 1.0"
      - "replicas >= 1"
      - "replicas <= 100"
    
    soft_constraints:
      - "response_time <= 1000"  # ms
      - "error_rate <= 0.01"
    
    temporal_constraints:
      - action: "scale_out"
        cooldown: 300  # seconds
        max_frequency: 5  # per hour
      
      - action: "restart"
        cooldown: 600
        max_frequency: 2  # per hour
    
    stability_monitoring:
      max_oscillation_risk: 0.7
      min_stability_margins:
        phase_margin: 45  # degrees
        gain_margin: 6    # dB
      max_change_rate: 0.5  # 50% change limit
```

---

*This completes the documentation for documents 10-14. The documentation suite now provides comprehensive coverage of the Adapter Layer, Digital Twin Layer, Control & Reasoning Layer, and Adaptive Control mechanisms in POLARIS.*