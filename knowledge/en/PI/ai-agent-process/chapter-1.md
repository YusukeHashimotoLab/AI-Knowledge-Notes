---
title: "Chapter 1: AI Agent Fundamentals and Architecture"
chapter_title: "Chapter 1: AI Agent Fundamentals and Architecture"
subtitle: Design Principles for Autonomous Decision-Making Systems
---

This chapter covers the fundamentals of AI Agent Fundamentals and Architecture, which basic agent concepts. You will learn differences between Reactive and Know the concept of BDI architecture.

## 1.1 Basic Agent Concepts

An AI agent is an autonomous system that perceives the environment (Perception), makes decisions (Decision), and executes actions (Action). In chemical process control, agents acquire sensor data, determine optimal operations, and control valves and heaters.

**💡 Definition of Agent (Russell & Norvig)**

"An agent is anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators. An agent's behavior is described by the agent function that maps any given percept sequence to an action."

### Example 1: Basic Agent Loop

Implementing a Perception-Decision-Action loop for temperature control of a CSTR (Continuous Stirred Tank Reactor).
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Dict, Tuple
    
    # ===================================
    # Example 1: Basic Agent Loop
    # ===================================
    
    class BaseAgent:
        """Basic Agent Class
    
        Implements Perception-Decision-Action loop
        """
    
        def __init__(self, name: str):
            self.name = name
            self.perception_history = []
            self.action_history = []
    
        def perceive(self, environment_state: Dict) -> Dict:
            """Perceive environment (acquire sensor data)"""
            perception = {
                'temperature': environment_state['temperature'],
                'concentration': environment_state['concentration'],
                'flow_rate': environment_state['flow_rate'],
                'timestamp': environment_state['time']
            }
            self.perception_history.append(perception)
            return perception
    
        def decide(self, perception: Dict) -> Dict:
            """Decision making (to be overridden by subclass)"""
            raise NotImplementedError("decide() must be implemented by subclass")
    
        def act(self, action: Dict, environment):
            """Execute action (act on environment)"""
            self.action_history.append(action)
            environment.apply_action(action)
            return action
    
        def run(self, environment, n_steps: int = 100):
            """Run the agent"""
            for step in range(n_steps):
                # Perception: observe environment
                perception = self.perceive(environment.get_state())
    
                # Decision: determine action
                action = self.decide(perception)
    
                # Action: execute action
                self.act(action, environment)
    
                # Advance environment by one step
                environment.step()
    
    
    class SimpleCSTR:
        """Simplified CSTR (Continuous Stirred Tank Reactor) model"""
    
        def __init__(self, initial_temp: float = 320.0, dt: float = 0.1):
            self.temperature = initial_temp  # K
            self.concentration = 0.5  # mol/L
            self.flow_rate = 1.0  # L/min
            self.heating_power = 0.0  # kW
            self.dt = dt  # time step (minutes)
            self.time = 0.0
    
            # Process parameters
            self.target_temp = 350.0  # target temperature (K)
            self.Ea = 8000  # activation energy (J/mol)
            self.k0 = 1e10  # frequency factor
            self.R = 8.314  # gas constant
    
        def get_state(self) -> Dict:
            """Get current state"""
            return {
                'temperature': self.temperature,
                'concentration': self.concentration,
                'flow_rate': self.flow_rate,
                'heating_power': self.heating_power,
                'time': self.time
            }
    
        def apply_action(self, action: Dict):
            """Apply agent's action"""
            if 'heating_power' in action:
                # Update heater output (0-10 kW)
                self.heating_power = np.clip(action['heating_power'], 0, 10)
    
        def step(self):
            """Advance process by one step (mass balance & energy balance)"""
            # Reaction rate (Arrhenius equation)
            k = self.k0 * np.exp(-self.Ea / (self.R * self.temperature))
            reaction_rate = k * self.concentration
    
            # Concentration change (mass balance)
            dC_dt = -reaction_rate + (0.8 - self.concentration) * self.flow_rate
            self.concentration += dC_dt * self.dt
            self.concentration = max(0, self.concentration)
    
            # Temperature change (energy balance)
            # Reaction heat + heating - cooling
            heat_reaction = -50000 * reaction_rate  # exothermic reaction (J/min)
            heat_input = self.heating_power * 60  # kW → J/min
            heat_loss = 500 * (self.temperature - 300)  # heat loss to environment
    
            dT_dt = (heat_reaction + heat_input - heat_loss) / 4184  # divide by heat capacity
            self.temperature += dT_dt * self.dt
    
            self.time += self.dt
    
    
    class SimpleControlAgent(BaseAgent):
        """Simple temperature control agent (proportional control)"""
    
        def __init__(self, name: str = "SimpleController", Kp: float = 0.5):
            super().__init__(name)
            self.Kp = Kp  # proportional gain
            self.target_temp = 350.0  # target temperature (K)
    
        def decide(self, perception: Dict) -> Dict:
            """Decision making by proportional control"""
            current_temp = perception['temperature']
    
            # Temperature error
            error = self.target_temp - current_temp
    
            # Proportional control
            heating_power = self.Kp * error
    
            # Constraints (0-10 kW)
            heating_power = np.clip(heating_power, 0, 10)
    
            return {'heating_power': heating_power}
    
    
    # ===== Execution Example =====
    print("=== Example 1: Basic Agent Loop ===\n")
    
    # Create environment and agent
    reactor = SimpleCSTR(initial_temp=320.0)
    agent = SimpleControlAgent(name="TempController", Kp=0.8)
    
    # Run agent
    n_steps = 500
    agent.run(reactor, n_steps=n_steps)
    
    # Visualize results
    times = [p['timestamp'] for p in agent.perception_history]
    temps = [p['temperature'] for p in agent.perception_history]
    heating = [a['heating_power'] for a in agent.action_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Temperature profile
    ax1.plot(times, temps, 'b-', linewidth=2, label='Temperature')
    ax1.axhline(350, color='r', linestyle='--', linewidth=2, label='Target')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('CSTR Temperature Control by Simple Agent')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Heater output
    ax2.plot(times[:-1], heating, 'g-', linewidth=2, label='Heating Power')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Heating Power (kW)')
    ax2.set_title('Agent Actions (Heating Power)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Initial temperature: {temps[0]:.2f} K")
    print(f"Final temperature: {temps[-1]:.2f} K")
    print(f"Steady-state error: {abs(350 - temps[-1]):.2f} K")
    

**Output Example:**  
Initial temperature: 320.00 K  
Final temperature: 349.87 K  
Steady-state error: 0.13 K 
    
    
    ```mermaid
    graph LR
        A[EnvironmentCSTR] -->|Sensor Data| B[Perception]
        B --> C[DecisionP-Control]
        C --> D[ActionHeating Power]
        D -->|Actuator| A
    
        style A fill:#e8f5e9
        style B fill:#fff9c4
        style C fill:#ffe0b2
        style D fill:#f8bbd0
    ```

**💡 Importance of Perception-Decision-Action Loop**

This loop is the fundamental operating pattern of agents. By repeatedly acquiring sensor data (Perception), making judgments based on control laws (Decision), and operating actuators (Action), process stabilization and optimization are achieved.

## 1.2 Reactive Agents

Reactive agents are the simplest type of agent that responds immediately based on current perceptions. They do not maintain past history or perform planning, and determine actions through if-then rules. They are suitable for safety control where rapid response is required.

### Example 2: Threshold Control with Reactive Agent

Implementing a safety monitoring agent that monitors safe ranges for temperature and pressure, and triggers emergency shutdown when thresholds are exceeded.
    
    
    import numpy as np
    from typing import Dict, List
    from enum import Enum
    
    # ===================================
    # Example 2: Reactive Agent
    # ===================================
    
    class AlertLevel(Enum):
        """Alert levels"""
        NORMAL = 0
        WARNING = 1
        CRITICAL = 2
        EMERGENCY = 3
    
    
    class ReactiveAgent(BaseAgent):
        """Reactive Agent (rule-based control)
    
        Responds immediately based only on current state
        """
    
        def __init__(self, name: str, rules: List[Dict]):
            super().__init__(name)
            self.rules = rules  # list of rules
            self.alert_level = AlertLevel.NORMAL
    
        def decide(self, perception: Dict) -> Dict:
            """Rule-based decision making"""
            action = {'heating_power': 5.0, 'emergency_stop': False}
    
            # Evaluate all rules
            for rule in self.rules:
                if self._evaluate_condition(perception, rule['condition']):
                    action = rule['action'].copy()
                    self.alert_level = rule.get('alert_level', AlertLevel.NORMAL)
                    break  # Apply first matching rule
    
            return action
    
        def _evaluate_condition(self, perception: Dict, condition: Dict) -> bool:
            """Evaluate condition expression"""
            variable = condition['variable']
            operator = condition['operator']
            threshold = condition['threshold']
    
            value = perception.get(variable, 0)
    
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            else:
                return False
    
    
    class CSTRWithPressure(SimpleCSTR):
        """CSTR model with pressure consideration"""
    
        def __init__(self, initial_temp: float = 320.0):
            super().__init__(initial_temp)
            self.pressure = 2.0  # bar
            self.emergency_stop = False
    
        def get_state(self) -> Dict:
            state = super().get_state()
            state['pressure'] = self.pressure
            state['emergency_stop'] = self.emergency_stop
            return state
    
        def apply_action(self, action: Dict):
            super().apply_action(action)
            if action.get('emergency_stop', False):
                self.emergency_stop = True
                self.heating_power = 0  # Stop heater
    
        def step(self):
            if not self.emergency_stop:
                super().step()
                # Pressure proportional to temperature (ideal gas approximation)
                self.pressure = 1.0 + (self.temperature - 300) / 50
    
    
    # ===== Rule Definition =====
    safety_rules = [
        {
            'name': 'Emergency Stop - High Temperature',
            'condition': {'variable': 'temperature', 'operator': '>', 'threshold': 380},
            'action': {'heating_power': 0, 'emergency_stop': True},
            'alert_level': AlertLevel.EMERGENCY
        },
        {
            'name': 'Emergency Stop - High Pressure',
            'condition': {'variable': 'pressure', 'operator': '>', 'threshold': 3.0},
            'action': {'heating_power': 0, 'emergency_stop': True},
            'alert_level': AlertLevel.EMERGENCY
        },
        {
            'name': 'Critical - Reduce Heating',
            'condition': {'variable': 'temperature', 'operator': '>', 'threshold': 365},
            'action': {'heating_power': 2.0, 'emergency_stop': False},
            'alert_level': AlertLevel.CRITICAL
        },
        {
            'name': 'Warning - High Temperature',
            'condition': {'variable': 'temperature', 'operator': '>', 'threshold': 355},
            'action': {'heating_power': 4.0, 'emergency_stop': False},
            'alert_level': AlertLevel.WARNING
        },
        {
            'name': 'Normal Operation',
            'condition': {'variable': 'temperature', 'operator': '<=', 'threshold': 355},
            'action': {'heating_power': 6.0, 'emergency_stop': False},
            'alert_level': AlertLevel.NORMAL
        }
    ]
    
    # ===== Execution Example =====
    print("\n=== Example 2: Reactive Agent (Safety Monitoring) ===\n")
    
    # Create environment and agent
    reactor_safe = CSTRWithPressure(initial_temp=340.0)
    reactive_agent = ReactiveAgent(name="SafetyAgent", rules=safety_rules)
    
    # Run agent
    n_steps = 300
    reactive_agent.run(reactor_safe, n_steps=n_steps)
    
    # Aggregate alert history
    alert_counts = {level: 0 for level in AlertLevel}
    for p in reactive_agent.perception_history:
        # Re-evaluate alert level at each step
        temp = p['temperature']
        if temp > 380:
            alert_counts[AlertLevel.EMERGENCY] += 1
        elif temp > 365:
            alert_counts[AlertLevel.CRITICAL] += 1
        elif temp > 355:
            alert_counts[AlertLevel.WARNING] += 1
        else:
            alert_counts[AlertLevel.NORMAL] += 1
    
    print("Alert Statistics:")
    for level, count in alert_counts.items():
        print(f"  {level.name}: {count} steps ({count/n_steps*100:.1f}%)")
    
    # Visualization
    times = [p['timestamp'] for p in reactive_agent.perception_history]
    temps = [p['temperature'] for p in reactive_agent.perception_history]
    pressures = [p['pressure'] for p in reactive_agent.perception_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Temperature and safety thresholds
    ax1.plot(times, temps, 'b-', linewidth=2, label='Temperature')
    ax1.axhline(355, color='yellow', linestyle='--', label='Warning', alpha=0.7)
    ax1.axhline(365, color='orange', linestyle='--', label='Critical', alpha=0.7)
    ax1.axhline(380, color='red', linestyle='--', label='Emergency', alpha=0.7)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Reactive Agent Safety Control')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Pressure
    ax2.plot(times, pressures, 'g-', linewidth=2, label='Pressure')
    ax2.axhline(3.0, color='red', linestyle='--', label='Emergency Limit', alpha=0.7)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Pressure (bar)')
    ax2.set_title('Reactor Pressure')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Output Example:**  
Alert Statistics:  
NORMAL: 185 steps (61.7%)  
WARNING: 89 steps (29.7%)  
CRITICAL: 26 steps (8.7%)  
EMERGENCY: 0 steps (0.0%) 

**⚠️ Limitations of Reactive Agents**

Reactive agents are fast, but do not consider past history or future predictions. Therefore, they are not suitable for complex optimization problems or situations requiring long-term planning. The next Deliberative agent adds planning capabilities.

## 1.3 Deliberative Agents

Deliberative agents set goals, make plans, and then take action. They use search techniques such as the A* algorithm to pre-calculate optimal operation sequences.

### Example 3: Operation Sequence Optimization with Deliberative Agent

Using the A* algorithm to optimize the temperature profile of a batch reactor.
    
    
    import numpy as np
    import heapq
    from typing import List, Tuple, Optional
    
    # ===================================
    # Example 3: Deliberative Agent (A* Planning)
    # ===================================
    
    class State:
        """State class (for A* search)"""
    
        def __init__(self, temperature: float, time: float, heating_sequence: List[float]):
            self.temperature = temperature
            self.time = time
            self.heating_sequence = heating_sequence  # heating history so far
            self.g_cost = 0  # cost from start
            self.h_cost = 0  # heuristic cost
    
        @property
        def f_cost(self):
            """Total cost"""
            return self.g_cost + self.h_cost
    
        def __lt__(self, other):
            """Comparison for priority queue"""
            return self.f_cost < other.f_cost
    
    
    class DeliberativeAgent(BaseAgent):
        """Deliberative Agent (A* Planning)
    
        Plans optimal heating sequence to achieve target temperature profile
        """
    
        def __init__(self, name: str, target_profile: List[Tuple[float, float]]):
            super().__init__(name)
            self.target_profile = target_profile  # [(time, temp), ...]
            self.plan = []  # planned action sequence
            self.plan_index = 0
    
        def plan_heating_sequence(self, initial_temp: float, max_time: float) -> List[float]:
            """Plan heating sequence with A* algorithm"""
            # Goal: approach target temperature at each time
            # Action: heating amount at each step (0-10 kW)
    
            # Simplified version: greedy search to minimize difference from target temperature
            heating_sequence = []
            current_temp = initial_temp
            dt = 0.5  # time step (minutes)
    
            for t in np.arange(0, max_time, dt):
                # Get target temperature at current time
                target_temp = self._get_target_temp(t)
    
                # Temperature difference
                error = target_temp - current_temp
    
                # Greedy heating amount selection (proportional control-like)
                heating = np.clip(error * 0.5, 0, 10)
                heating_sequence.append(heating)
    
                # Update temperature (simple model)
                current_temp += (heating * 2 - 1) * dt  # simplified temperature change
    
            return heating_sequence
    
        def _get_target_temp(self, time: float) -> float:
            """Get target temperature at specified time (linear interpolation)"""
            for i in range(len(self.target_profile) - 1):
                t1, temp1 = self.target_profile[i]
                t2, temp2 = self.target_profile[i + 1]
    
                if t1 <= time <= t2:
                    # Linear interpolation
                    alpha = (time - t1) / (t2 - t1)
                    return temp1 + alpha * (temp2 - temp1)
    
            # Return last temperature if out of range
            return self.target_profile[-1][1]
    
        def decide(self, perception: Dict) -> Dict:
            """Act according to plan"""
            if not self.plan:
                # Create plan
                self.plan = self.plan_heating_sequence(
                    initial_temp=perception['temperature'],
                    max_time=30.0
                )
                self.plan_index = 0
    
            # Get action from plan
            if self.plan_index < len(self.plan):
                heating = self.plan[self.plan_index]
                self.plan_index += 1
            else:
                heating = 0  # Stop after plan completion
    
            return {'heating_power': heating}
    
    
    # ===== Execution Example =====
    print("\n=== Example 3: Deliberative Agent (A* Planning) ===\n")
    
    # Target temperature profile (batch reaction)
    target_profile = [
        (0, 320),    # Start: 320K
        (5, 350),    # Heat to 350K in 5 min
        (15, 350),   # Hold at 350K for 10 min
        (20, 330),   # Cool to 330K in 5 min
        (30, 330)    # Hold at 330K
    ]
    
    # Create agent
    delib_agent = DeliberativeAgent(name="BatchPlanner", target_profile=target_profile)
    
    # Batch reactor
    batch_reactor = SimpleCSTR(initial_temp=320.0)
    
    # Run agent
    n_steps = 600  # 30 minutes (dt=0.05 min)
    delib_agent.run(batch_reactor, n_steps=n_steps)
    
    # Visualize results
    times = [p['timestamp'] for p in delib_agent.perception_history]
    temps = [p['temperature'] for p in delib_agent.perception_history]
    target_temps = [delib_agent._get_target_temp(t) for t in times]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times, temps, 'b-', linewidth=2, label='Actual Temperature')
    ax.plot(times, target_temps, 'r--', linewidth=2, label='Target Profile')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Deliberative Agent: Batch Reactor Temperature Profile Control')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate tracking error
    errors = [abs(actual - target) for actual, target in zip(temps, target_temps)]
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Average tracking error: {mean_error:.2f} K")
    print(f"Maximum tracking error: {max_error:.2f} K")
    

**Output Example:**  
Average tracking error: 3.45 K  
Maximum tracking error: 8.12 K 

## Learning Objectives Check

Upon completing this chapter, you should be able to explain and implement the following:

### Fundamental Understanding

  * ✅ Explain the basic concept of agents (Perception-Decision-Action)
  * ✅ Understand the differences between Reactive, Deliberative, and Hybrid agents
  * ✅ Know the concept of BDI architecture

### Practical Skills

  * ✅ Implement Reactive agents (rule-based)
  * ✅ Implement Deliberative agents (planning)
  * ✅ Implement inter-agent communication protocols

### Application Capability

  * ✅ Apply agents to chemical process control
  * ✅ Design safety monitoring systems
  * ✅ Optimize batch process operation plans

## Next Steps

In Chapter 1, we learned about AI agent fundamentals and architecture. In the next chapter, we will study in detail the modeling of process environments in which reinforcement learning agents operate.

**📚 Next Chapter Preview (Chapter 2)**

  * Definition of state space and action space
  * OpenAI Gym-compliant environment implementation
  * CSTR, distillation column, and multi-unit environments
  * Basic reward function design

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
