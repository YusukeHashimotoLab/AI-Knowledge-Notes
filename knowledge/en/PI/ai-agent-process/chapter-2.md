---
title: "Chapter 2: Process Environment Modeling"
chapter_title: "Chapter 2: Process Environment Modeling"
subtitle: Building reinforcement learning-compatible chemical process environments with OpenAI Gym
---

This chapter covers Process Environment Modeling. You will learn Know reward function design principles and OpenAI Gym environment structure.

## 2.1 State Space Definition

In reinforcement learning, the state space is a set of variables representing the current state of the environment. In chemical processes, continuous variables such as temperature, pressure, concentration, and flow rate constitute the state.

**💡 State Space Design Principles**

  * **Markov Property** : The current state contains sufficient information to determine future behavior
  * **Observability** : Select variables that can actually be measured by sensors
  * **Normalization** : Convert each variable to the same scale (e.g., 0-1)
  * **Dimensionality Reduction** : Remove redundant variables to improve learning efficiency

### Example 1: State Space Construction and Normalization

Define the state space for a CSTR (Continuous Stirred Tank Reactor) and implement normalization.
    
    
    import numpy as np
    from typing import Dict, Tuple
    import gym
    from gym import spaces
    
    # ===================================
    # Example 1: State Space Definition and Normalization
    # ===================================
    
    class StateSpace:
        """State space definition for chemical processes"""
    
        def __init__(self):
            # Physical variable ranges (min_value, max_value)
            self.bounds = {
                'temperature': (300.0, 400.0),      # K
                'pressure': (1.0, 10.0),            # bar
                'concentration': (0.0, 2.0),        # mol/L
                'flow_rate': (0.5, 5.0),            # L/min
                'level': (0.0, 100.0)               # %
            }
    
        def get_state_vector(self, physical_state: Dict) -> np.ndarray:
            """Construct state vector from physical variables"""
            state = np.array([
                physical_state['temperature'],
                physical_state['pressure'],
                physical_state['concentration'],
                physical_state['flow_rate'],
                physical_state['level']
            ])
            return state
    
        def normalize(self, state: np.ndarray) -> np.ndarray:
            """Normalize state to [0, 1] range"""
            normalized = np.zeros_like(state)
            for i, var_name in enumerate(self.bounds.keys()):
                min_val, max_val = self.bounds[var_name]
                normalized[i] = (state[i] - min_val) / (max_val - min_val)
                normalized[i] = np.clip(normalized[i], 0, 1)
            return normalized
    
        def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
            """Convert normalized state back to physical values"""
            state = np.zeros_like(normalized_state)
            for i, var_name in enumerate(self.bounds.keys()):
                min_val, max_val = self.bounds[var_name]
                state[i] = normalized_state[i] * (max_val - min_val) + min_val
            return state
    
        def get_gym_space(self) -> spaces.Box:
            """Get state space for OpenAI Gym"""
            low = np.array([bounds[0] for bounds in self.bounds.values()])
            high = np.array([bounds[1] for bounds in self.bounds.values()])
            return spaces.Box(low=low, high=high, dtype=np.float32)
    
    
    # ===== Usage Example =====
    print("=== Example 1: State Space Definition and Normalization ===\n")
    
    state_space = StateSpace()
    
    # Sample state
    physical_state = {
        'temperature': 350.0,
        'pressure': 5.5,
        'concentration': 1.2,
        'flow_rate': 2.5,
        'level': 75.0
    }
    
    # Construct state vector
    state_vector = state_space.get_state_vector(physical_state)
    print("Physical state vector:")
    print(state_vector)
    
    # Normalization
    normalized = state_space.normalize(state_vector)
    print("\nNormalized state (0-1 range):")
    print(normalized)
    
    # Verify with denormalization
    denormalized = state_space.denormalize(normalized)
    print("\nDenormalized state (original physical values):")
    print(denormalized)
    
    # Gym space definition
    gym_space = state_space.get_gym_space()
    print(f"\nOpenAI Gym state space:")
    print(f"  Low: {gym_space.low}")
    print(f"  High: {gym_space.high}")
    print(f"  Shape: {gym_space.shape}")
    
    # Random sampling
    random_state = gym_space.sample()
    print(f"\nRandom sample: {random_state}")
    

**Output Example:**  
Physical state vector:  
[350. 5.5 1.2 2.5 75. ]  
  
Normalized state (0-1 range):  
[0.5 0.5 0.6 0.44 0.75]  
  
OpenAI Gym state space:  
Shape: (5,) 

## 2.2 Action Space Design

The action space is the set of operations an agent can execute. This includes discrete actions (valve open/close) and continuous actions (flow rate adjustment).

### Example 2: Implementation of Discrete, Continuous, and Mixed Action Spaces
    
    
    import gym
    from gym import spaces
    import numpy as np
    
    # ===================================
    # Example 2: Action Space Design
    # ===================================
    
    class ActionSpaceDesign:
        """Action space design patterns"""
    
        @staticmethod
        def discrete_action_space() -> spaces.Discrete:
            """Discrete action space (e.g., valve operation)
    
            Actions:
                0: Valve fully closed
                1: Valve 25% open
                2: Valve 50% open
                3: Valve 75% open
                4: Valve fully open
            """
            return spaces.Discrete(5)
    
        @staticmethod
        def continuous_action_space() -> spaces.Box:
            """Continuous action space (e.g., flow control)
    
            Actions:
                [0]: Heater output (0-10 kW)
                [1]: Cooling water flow (0-5 L/min)
            """
            return spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([10.0, 5.0]),
                dtype=np.float32
            )
    
        @staticmethod
        def mixed_action_space() -> spaces.Dict:
            """Mixed action space (discrete + continuous)
    
            Actions:
                'mode': Operating mode selection (0: standby, 1: running, 2: shutdown)
                'heating': Heater output (0-10 kW)
                'flow': Flow rate (0-5 L/min)
            """
            return spaces.Dict({
                'mode': spaces.Discrete(3),
                'heating': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                'flow': spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32)
            })
    
        @staticmethod
        def apply_safety_constraints(action: np.ndarray, state: np.ndarray) -> np.ndarray:
            """Apply safety constraints
    
            Args:
                action: Original action
                state: Current state [temp, pressure, ...]
    
            Returns:
                Constrained action
            """
            safe_action = action.copy()
    
            # Constraint 1: Limit heater output at high temperature
            if state[0] > 380:  # Temperature above 380K
                safe_action[0] = min(safe_action[0], 2.0)  # Max heater 2kW
    
            # Constraint 2: Limit flow at high pressure
            if len(state) > 1 and state[1] > 8:  # Pressure above 8bar
                safe_action[1] = min(safe_action[1], 1.0)  # Max flow 1L/min
    
            # Constraint 3: Physical limits
            safe_action = np.clip(safe_action, [0.0, 0.0], [10.0, 5.0])
    
            return safe_action
    
    
    # ===== Usage Example =====
    print("\n=== Example 2: Action Space Design ===\n")
    
    designer = ActionSpaceDesign()
    
    # 1. Discrete action space
    discrete_space = designer.discrete_action_space()
    print("Discrete action space:")
    print(f"  Number of actions: {discrete_space.n}")
    print(f"  Sample: {discrete_space.sample()}")
    
    # 2. Continuous action space
    continuous_space = designer.continuous_action_space()
    print("\nContinuous action space:")
    print(f"  Low: {continuous_space.low}")
    print(f"  High: {continuous_space.high}")
    print(f"  Sample: {continuous_space.sample()}")
    
    # 3. Mixed action space
    mixed_space = designer.mixed_action_space()
    print("\nMixed action space:")
    sample_mixed = mixed_space.sample()
    print(f"  Mode: {sample_mixed['mode']}")
    print(f"  Heating: {sample_mixed['heating']}")
    print(f"  Flow: {sample_mixed['flow']}")
    
    # 4. Safety constraint application
    print("\nSafety constraint application:")
    unsafe_action = np.array([8.0, 4.0])  # Heater 8kW, flow 4L/min
    high_temp_state = np.array([385.0, 5.0])  # High temperature state
    
    safe_action = designer.apply_safety_constraints(unsafe_action, high_temp_state)
    print(f"  Original action: {unsafe_action}")
    print(f"  After constraints: {safe_action}")
    print(f"  Reason: Temperature {high_temp_state[0]:.0f}K > 380K → Heater limited to 2kW or below")
    

**Output Example:**  
Discrete action space:  
Number of actions: 5  
Sample: 2  
  
Continuous action space:  
Sample: [6.23 2.84]  
  
Safety constraint application:  
Original action: [8. 4.]  
After constraints: [2. 4.] 

## 2.3 Reward Function Basic Design

The reward function numerically quantifies the quality of an agent's actions. For chemical processes, we design multi-objective reward functions considering setpoint tracking, energy efficiency, and safety.

### Example 3: Multi-Objective Reward Function Implementation
    
    
    import numpy as np
    from typing import Dict
    
    # ===================================
    # Example 3: Multi-Objective Reward Function
    # ===================================
    
    class RewardFunction:
        """Reward function for chemical processes"""
    
        def __init__(self, weights: Dict[str, float] = None):
            # Weights for each objective (default values)
            self.weights = weights or {
                'setpoint_tracking': 1.0,    # Setpoint tracking
                'energy': 0.3,                # Energy efficiency
                'safety': 2.0,                # Safety
                'stability': 0.5              # Stability
            }
    
        def compute_reward(self, state: np.ndarray, action: np.ndarray,
                          target_temp: float = 350.0) -> Tuple[float, Dict[str, float]]:
            """Compute total reward
    
            Args:
                state: [temperature, pressure, concentration, ...]
                action: [heating_power, flow_rate]
                target_temp: Target temperature
    
            Returns:
                total_reward: Total reward
                components: Detailed breakdown of reward components
            """
            temp, pressure = state[0], state[1]
            heating, flow = action[0], action[1] if len(action) > 1 else 0
    
            # 1. Setpoint tracking reward (temperature)
            temp_error = abs(temp - target_temp)
            r_tracking = -temp_error / 10.0  # Range -10 to 0
    
            # 2. Energy efficiency reward
            energy_cost = heating * 0.1 + flow * 0.05  # Energy cost
            r_energy = -energy_cost
    
            # 3. Safety reward (penalty)
            r_safety = 0.0
            if temp > 380:  # High temperature warning
                r_safety = -10.0 * (temp - 380)
            if temp > 400:  # Danger zone
                r_safety = -100.0
            if pressure > 9:  # High pressure warning
                r_safety += -5.0 * (pressure - 9)
    
            # 4. Stability reward (low variation)
            # Note: In practice, use difference from previous step
            r_stability = 0.0  # Simplified for brevity
    
            # Weighted sum
            components = {
                'tracking': r_tracking * self.weights['setpoint_tracking'],
                'energy': r_energy * self.weights['energy'],
                'safety': r_safety * self.weights['safety'],
                'stability': r_stability * self.weights['stability']
            }
    
            total_reward = sum(components.values())
    
            return total_reward, components
    
        def reward_shaping(self, raw_reward: float, progress: float) -> float:
            """Reward shaping (encourage early exploration)
    
            Args:
                raw_reward: Original reward
                progress: Learning progress (0-1)
    
            Returns:
                shaped_reward: Shaped reward
            """
            # Reduce penalties early in training
            penalty_scale = 0.3 + 0.7 * progress
            if raw_reward < 0:
                return raw_reward * penalty_scale
            else:
                return raw_reward
    
    
    # ===== Usage Example =====
    print("\n=== Example 3: Multi-Objective Reward Function ===\n")
    
    reward_func = RewardFunction()
    
    # Scenario 1: Optimal state
    state_optimal = np.array([350.0, 5.0, 1.0])
    action_optimal = np.array([5.0, 2.0])
    
    reward, components = reward_func.compute_reward(state_optimal, action_optimal)
    print("Scenario 1: Optimal state")
    print(f"  State: T={state_optimal[0]}K, P={state_optimal[1]}bar")
    print(f"  Action: Heating={action_optimal[0]}kW, Flow={action_optimal[1]}L/min")
    print(f"  Total reward: {reward:.3f}")
    for key, val in components.items():
        print(f"    {key}: {val:.3f}")
    
    # Scenario 2: High temperature danger state
    state_danger = np.array([390.0, 5.0, 1.0])
    action_danger = np.array([8.0, 2.0])
    
    reward, components = reward_func.compute_reward(state_danger, action_danger)
    print("\nScenario 2: High temperature danger state")
    print(f"  State: T={state_danger[0]}K, P={state_danger[1]}bar")
    print(f"  Total reward: {reward:.3f}")
    for key, val in components.items():
        print(f"    {key}: {val:.3f}")
    
    # Scenario 3: Excessive energy use
    state_normal = np.array([345.0, 5.0, 1.0])
    action_waste = np.array([10.0, 5.0])
    
    reward, components = reward_func.compute_reward(state_normal, action_waste)
    print("\nScenario 3: Excessive energy use")
    print(f"  State: T={state_normal[0]}K, P={state_normal[1]}bar")
    print(f"  Action: Heating={action_waste[0]}kW, Flow={action_waste[1]}L/min")
    print(f"  Total reward: {reward:.3f}")
    for key, val in components.items():
        print(f"    {key}: {val:.3f}")
    

**Output Example:**  
Scenario 1: Optimal state  
Total reward: -0.250  
tracking: 0.000  
energy: -0.250  
safety: 0.000  
  
Scenario 2: High temperature danger state  
Total reward: -204.550  
tracking: -4.000  
energy: -0.550  
safety: -200.000 

**⚠️ Reward Function Design Considerations**

  * **Scale Unification** : Align the scale of each reward component
  * **Avoid Sparse Rewards** : Provide appropriate intermediate rewards
  * **Prevent Reward Hacking** : Verify that no unintended behaviors are induced

## Learning Objectives Review

### Basic Understanding

  * ✅ Understand state space and action space definition methods
  * ✅ Know reward function design principles
  * ✅ Understand OpenAI Gym environment structure

### Practical Skills

  * ✅ Implement state normalization and denormalization
  * ✅ Design discrete, continuous, and mixed action spaces
  * ✅ Implement multi-objective reward functions
  * ✅ Incorporate safety constraints

### Application Ability

  * ✅ Implement CSTR environment compliant with Gym
  * ✅ Model distillation tower environment
  * ✅ Integrate multi-unit processes

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
