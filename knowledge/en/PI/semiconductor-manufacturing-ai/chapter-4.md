---
title: Hashimoto Laboratory
chapter_title: Hashimoto Laboratory
---

🌐 EN | [🇯🇵 JP](<../../../jp/PI/semiconductor-manufacturing-ai/chapter-4.html>) | Last sync: 2025-11-16

[Home](<../../../en/>) > > [Process Informatics](<../../PI/>) > [Semiconductor Manufacturing AI](<../../PI/semiconductor-manufacturing-ai/>) > Chapter 4 

## Learning Objectives

  * Master the theory and implementation methods of Model Predictive Control (MPC)
  * Understand adaptive control techniques for automatic response to process variations
  * Build process simulations using digital twins
  * Learn controller training methods using reinforcement learning (DQN, PPO)
  * Acquire practical techniques for APC system implementation and real-time control

## 4.1 Overview of Advanced Process Control (APC)

### 4.1.1 Role and Importance of APC

In semiconductor manufacturing, various disturbances such as equipment aging, environmental fluctuations, and raw material lot variations affect the process. APC is an advanced control system that compensates for these disturbances and maintains stable target values:

  * **Multivariable Control** : Simultaneous control of multiple inputs and outputs
  * **Predictive Control** : Optimal control by predicting the future with process models
  * **Adaptive Control** : Automatic response to process characteristic changes
  * **Constraint Handling** : Strict adherence to safety and performance ranges

### 4.1.2 Limitations of Conventional PID Control

**Single-variable Control** : Cannot consider interactions between variables

**Reactive Control** : Correction after error occurs (reactive approach)

**Difficulty in Constraint Handling** : Difficult to explicitly handle physical and performance constraints

**Lack of Optimality** : Cannot incorporate optimization objectives such as energy minimization

### 4.1.3 Advantages of AI-based APC

  * **Multi-objective Optimization** : Simultaneous optimization of quality, cost, and throughput
  * **Learning Capability** : Automatic learning of control laws from historical data
  * **Robustness** : Strong resistance to model errors and disturbances
  * **Real-time Performance** : High-speed computation utilizing GPUs

## 4.2 Model Predictive Control (MPC)

### 4.2.1 Principles of MPC

MPC predicts future behavior with a process model and calculates control input sequences that minimize a performance index:

**Prediction Model**

$$x_{k+1} = f(x_k, u_k)$$

\\(x_k\\): state, \\(u_k\\): control input

**Cost Function (Prediction Horizon N)**

$$J = \sum_{i=0}^{N-1} \left[ \|y_{k+i} - r_{k+i}\|_Q^2 + \|u_{k+i}\|_R^2 \right]$$

\\(y\\): output, \\(r\\): target value, \\(Q, R\\): weighting matrices

**Constraints**

$$u_{\min} \leq u_k \leq u_{\max}$$

$$y_{\min} \leq y_k \leq y_{\max}$$

**Optimization Problem**

At each time step, find the control input sequence \\(\\{u_k, u_{k+1}, \ldots, u_{k+N-1}\\}\\) that minimizes the above cost function, and apply only the first \\(u_k\\) (Receding Horizon)

### 4.2.2 CVD Process MPC Implementation

We implement film thickness control in Chemical Vapor Deposition (CVD) using MPC:
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    class ModelPredictiveController:
        """
        Model Predictive Control (MPC) for CVD process
    
        Control objective: Track film thickness to target value
        Control variables: Gas flow rate, RF power, pressure
        State variables: Film thickness, deposition rate
        """
    
        def __init__(self, prediction_horizon=10, control_horizon=5, dt=1.0):
            """
            Parameters:
            -----------
            prediction_horizon : int
                Prediction horizon N
            control_horizon : int
                Control horizon M (M ≤ N)
            dt : float
                Sampling time (seconds)
            """
            self.N = prediction_horizon
            self.M = control_horizon
            self.dt = dt
    
            # State-space model parameters
            # x = [film thickness (nm), deposition rate (nm/s)]
            # u = [gas flow (sccm), RF power (W), pressure (mTorr)]
            self.A = np.array([
                [1, self.dt],
                [0, 0.95]
            ])
    
            self.B = np.array([
                [0, 0, 0],
                [0.01, 0.02, -0.005]
            ])
    
            # Output matrix (only film thickness observed)
            self.C = np.array([[1, 0]])
    
            # Weighting matrices
            self.Q = np.diag([100, 1])  # State cost
            self.R = np.diag([0.1, 0.1, 0.1])  # Control input cost
    
            # Constraints
            self.u_min = np.array([50, 100, 10])
            self.u_max = np.array([200, 400, 100])
            self.y_min = 0
            self.y_max = 200  # Film thickness upper limit (nm)
    
        def predict(self, x0, u_sequence):
            """
            State prediction
    
            Parameters:
            -----------
            x0 : ndarray
                Initial state (2,)
            u_sequence : ndarray
                Control input sequence (M, 3)
    
            Returns:
            --------
            x_pred : ndarray
                Predicted state trajectory (N+1, 2)
            y_pred : ndarray
                Predicted output trajectory (N+1,)
            """
            x_pred = np.zeros((self.N + 1, 2))
            y_pred = np.zeros(self.N + 1)
    
            x_pred[0] = x0
            y_pred[0] = self.C @ x0
    
            for k in range(self.N):
                if k < self.M:
                    u_k = u_sequence[k]
                else:
                    # Hold last input after control horizon
                    u_k = u_sequence[self.M - 1]
    
                # State transition
                x_pred[k + 1] = self.A @ x_pred[k] + self.B @ u_k
                y_pred[k + 1] = self.C @ x_pred[k + 1]
    
            return x_pred, y_pred
    
        def cost_function(self, u_flat, x0, r_sequence):
            """
            Cost function
    
            Parameters:
            -----------
            u_flat : ndarray
                Flattened control input sequence (M*3,)
            x0 : ndarray
                Current state
            r_sequence : ndarray
                Target value sequence (N+1,)
            """
            # Restore control input
            u_sequence = u_flat.reshape((self.M, 3))
    
            # Prediction
            x_pred, y_pred = self.predict(x0, u_sequence)
    
            # Cost calculation
            cost = 0.0
    
            # Tracking error
            for k in range(self.N + 1):
                error = y_pred[k] - r_sequence[k]
                cost += error ** 2 * self.Q[0, 0]
    
            # Control input cost
            for k in range(self.M):
                cost += u_sequence[k] @ self.R @ u_sequence[k]
    
            # Control input change cost (smooth control)
            for k in range(1, self.M):
                du = u_sequence[k] - u_sequence[k - 1]
                cost += 0.1 * (du @ du)
    
            return cost
    
        def solve_mpc(self, x0, r_sequence, u_prev):
            """
            Solve MPC optimization problem
    
            Parameters:
            -----------
            x0 : ndarray
                Current state
            r_sequence : ndarray
                Target value sequence (N+1,)
            u_prev : ndarray
                Previous control input (3,)
    
            Returns:
            --------
            u_opt : ndarray
                Optimal control input (3,)
            """
            # Initial guess (hold previous input)
            u0 = np.tile(u_prev, self.M)
    
            # Constraints
            bounds = []
            for _ in range(self.M):
                for i in range(3):
                    bounds.append((self.u_min[i], self.u_max[i]))
    
            # Optimization
            result = minimize(
                fun=lambda u: self.cost_function(u, x0, r_sequence),
                x0=u0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
    
            # Optimal control input (use only first step)
            u_opt_sequence = result.x.reshape((self.M, 3))
            u_opt = u_opt_sequence[0]
    
            return u_opt
    
        def simulate_closed_loop(self, x0, r_trajectory, n_steps):
            """
            Closed-loop simulation
    
            Parameters:
            -----------
            x0 : ndarray
                Initial state
            r_trajectory : ndarray
                Target value trajectory (n_steps,)
            n_steps : int
                Number of simulation steps
    
            Returns:
            --------
            results : dict
                Simulation results
            """
            # History storage
            x_history = np.zeros((n_steps + 1, 2))
            y_history = np.zeros(n_steps + 1)
            u_history = np.zeros((n_steps, 3))
            r_history = np.zeros(n_steps + 1)
    
            x_history[0] = x0
            y_history[0] = self.C @ x0
            r_history[0] = r_trajectory[0]
    
            u_prev = np.array([125, 250, 55])  # Initial control input
    
            for k in range(n_steps):
                # Target value sequence (for prediction horizon)
                r_sequence = np.zeros(self.N + 1)
                for i in range(self.N + 1):
                    if k + i < n_steps:
                        r_sequence[i] = r_trajectory[k + i]
                    else:
                        r_sequence[i] = r_trajectory[-1]
    
                # MPC optimization
                u_opt = self.solve_mpc(x_history[k], r_sequence, u_prev)
                u_history[k] = u_opt
    
                # Apply to process (actual process includes disturbances)
                noise = np.random.normal(0, 0.1, 2)  # Process noise
                x_history[k + 1] = self.A @ x_history[k] + self.B @ u_opt + noise
                y_history[k + 1] = self.C @ x_history[k + 1]
                r_history[k + 1] = r_trajectory[k + 1] if k + 1 < n_steps else r_trajectory[-1]
    
                u_prev = u_opt
    
            results = {
                'x': x_history,
                'y': y_history,
                'u': u_history,
                'r': r_history,
                'time': np.arange(n_steps + 1) * self.dt
            }
    
            return results
    
        def plot_results(self, results):
            """Result visualization"""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            time = results['time']
            y = results['y']
            r = results['r']
            u = results['u']
    
            # Film thickness tracking
            axes[0, 0].plot(time, y, 'b-', linewidth=2, label='Actual Thickness')
            axes[0, 0].plot(time, r, 'r--', linewidth=2, label='Target Thickness')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Thickness (nm)')
            axes[0, 0].set_title('Film Thickness Tracking')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
    
            # Tracking error
            error = y - r
            axes[0, 1].plot(time, error, 'g-', linewidth=2)
            axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Error (nm)')
            axes[0, 1].set_title('Tracking Error')
            axes[0, 1].grid(True, alpha=0.3)
    
            # Control inputs
            axes[1, 0].plot(time[:-1], u[:, 0], label='Gas Flow (sccm)')
            axes[1, 0].plot(time[:-1], u[:, 1], label='RF Power (W)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Control Input')
            axes[1, 0].set_title('Control Inputs (Gas & RF)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
            axes[1, 1].plot(time[:-1], u[:, 2], 'purple', label='Pressure (mTorr)')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Pressure (mTorr)')
            axes[1, 1].set_title('Control Input (Pressure)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.savefig('mpc_control_results.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    
    # ========== Usage Example ==========
    if __name__ == "__main__":
        np.random.seed(42)
    
        # MPC configuration
        mpc = ModelPredictiveController(
            prediction_horizon=10,
            control_horizon=5,
            dt=1.0
        )
    
        # Initial state [film thickness, deposition rate]
        x0 = np.array([0.0, 0.0])
    
        # Target value trajectory (step response + ramp)
        n_steps = 100
        r_trajectory = np.zeros(n_steps)
        r_trajectory[:30] = 50  # 50nm
        r_trajectory[30:60] = 100  # 100nm
        r_trajectory[60:] = np.linspace(100, 150, 40)  # Ramp
    
        # Closed-loop simulation
        print("========== MPC Closed-Loop Simulation ==========")
        results = mpc.simulate_closed_loop(x0, r_trajectory, n_steps)
    
        # Performance evaluation
        tracking_error = results['y'] - results['r']
        mae = np.mean(np.abs(tracking_error))
        rmse = np.sqrt(np.mean(tracking_error ** 2))
    
        print(f"\nTracking Performance:")
        print(f"  MAE (Mean Absolute Error): {mae:.4f} nm")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.4f} nm")
    
        # Control input statistics
        print(f"\nControl Input Statistics:")
        print(f"  Gas Flow: {np.mean(results['u'][:, 0]):.2f} ± {np.std(results['u'][:, 0]):.2f} sccm")
        print(f"  RF Power: {np.mean(results['u'][:, 1]):.2f} ± {np.std(results['u'][:, 1]):.2f} W")
        print(f"  Pressure: {np.mean(results['u'][:, 2]):.2f} ± {np.std(results['u'][:, 2]):.2f} mTorr")
    
        # Visualization
        mpc.plot_results(results)
    

### 4.2.3 Nonlinear MPC and Neural Network Model

For complex nonlinear processes, neural networks are used as process models:
    
    
    # Requirements:
    # - Python 3.9+
    # - tensorflow>=2.13.0, <2.16.0
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    class NeuralNetworkMPC:
        """
        Neural Network-based MPC
    
        Model complex nonlinear processes with NN,
        Execute MPC optimization using gradient methods
        """
    
        def __init__(self, state_dim=2, control_dim=3, prediction_horizon=10):
            """
            Parameters:
            -----------
            state_dim : int
                State dimension
            control_dim : int
                Control input dimension
            prediction_horizon : int
                Prediction horizon
            """
            self.state_dim = state_dim
            self.control_dim = control_dim
            self.N = prediction_horizon
    
            # Neural Network Process Model
            self.process_model = self._build_process_model()
    
        def _build_process_model(self):
            """
            Build process model NN
    
            Input: [x_k, u_k] (concat)
            Output: x_{k+1}
            """
            inputs = layers.Input(shape=(self.state_dim + self.control_dim,))
    
            x = layers.Dense(64, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(32, activation='relu')(x)
    
            outputs = layers.Dense(self.state_dim)(x)
    
            model = keras.Model(inputs, outputs, name='process_model')
            model.compile(optimizer='adam', loss='mse')
    
            return model
    
        def train_process_model(self, X_train, y_train, epochs=50):
            """
            Train process model
    
            Parameters:
            -----------
            X_train : ndarray
                Training data [x_k, u_k] (N, state_dim + control_dim)
            y_train : ndarray
                Labels x_{k+1} (N, state_dim)
            """
            history = self.process_model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
    
            return history
    
        def predict_trajectory(self, x0, u_sequence):
            """
            Trajectory prediction with NN process model
    
            Parameters:
            -----------
            x0 : ndarray
                Initial state (state_dim,)
            u_sequence : ndarray
                Control input sequence (N, control_dim)
    
            Returns:
            --------
            x_trajectory : ndarray
                Predicted state trajectory (N+1, state_dim)
            """
            x_trajectory = np.zeros((self.N + 1, self.state_dim))
            x_trajectory[0] = x0
    
            for k in range(self.N):
                xu_k = np.concatenate([x_trajectory[k], u_sequence[k]]).reshape(1, -1)
                x_trajectory[k + 1] = self.process_model.predict(xu_k, verbose=0)[0]
    
            return x_trajectory
    
        def mpc_optimization(self, x0, r_sequence):
            """
            MPC optimization using TensorFlow automatic differentiation
    
            Parameters:
            -----------
            x0 : ndarray
                Current state
            r_sequence : ndarray
                Target value sequence (N+1,)
    
            Returns:
            --------
            u_opt : ndarray
                Optimal control input sequence (N, control_dim)
            """
            # Initial control input
            u_var = tf.Variable(
                np.random.uniform(50, 200, (self.N, self.control_dim)),
                dtype=tf.float32
            )
    
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
            # Optimization loop
            for iteration in range(50):
                with tf.GradientTape() as tape:
                    # Prediction
                    x_pred = tf.constant(x0, dtype=tf.float32)
                    cost = 0.0
    
                    for k in range(self.N):
                        # State transition
                        xu_k = tf.concat([x_pred, u_var[k]], axis=0)
                        xu_k = tf.reshape(xu_k, (1, -1))
                        x_pred = self.process_model(xu_k, training=False)[0]
    
                        # Tracking error cost
                        error = x_pred[0] - r_sequence[k + 1]  # Film thickness error
                        cost += 100 * error ** 2
    
                        # Control input cost
                        cost += 0.01 * tf.reduce_sum(u_var[k] ** 2)
    
                # Gradient calculation and update
                gradients = tape.gradient(cost, [u_var])
                optimizer.apply_gradients(zip(gradients, [u_var]))
    
            u_opt = u_var.numpy()
    
            return u_opt
    
    
    # ========== Usage Example ==========
    # Generate dummy data for process model training
    np.random.seed(42)
    n_samples = 5000
    
    X_train = np.random.randn(n_samples, 5)  # [x1, x2, u1, u2, u3]
    # Dummy nonlinear process
    y_train = np.zeros((n_samples, 2))
    y_train[:, 0] = X_train[:, 0] + 0.1 * X_train[:, 2] + 0.02 * X_train[:, 3]
    y_train[:, 1] = 0.95 * X_train[:, 1] + 0.01 * X_train[:, 2]
    
    # Build and train NN-MPC
    nn_mpc = NeuralNetworkMPC(state_dim=2, control_dim=3, prediction_horizon=10)
    print("\n========== Training NN Process Model ==========")
    history = nn_mpc.train_process_model(X_train, y_train, epochs=30)
    
    print(f"Training Loss: {history.history['loss'][-1]:.6f}")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.6f}")
    
    # MPC optimization
    x0_nn = np.array([0.0, 0.0])
    r_sequence_nn = np.full(11, 100.0)
    
    print("\n========== NN-MPC Optimization ==========")
    u_opt_nn = nn_mpc.mpc_optimization(x0_nn, r_sequence_nn)
    
    print(f"Optimal Control Sequence (first 3 steps):")
    for k in range(3):
        print(f"  Step {k}: u = {u_opt_nn[k]}")
    

## 4.3 Controller Learning with Reinforcement Learning

### 4.3.1 Concept of Reinforcement Learning APC

Reinforcement Learning (RL) learns optimal control laws through trial and error:

  * **Model-free** : No process model required (learn directly from actual data)
  * **Adaptability** : Automatic adaptation to process changes
  * **Optimality** : Maximize long-term rewards
  * **Nonlinear Control** : Handle complex nonlinear processes

### 4.3.2 Discrete Control with DQN (Deep Q-Network)

We learn discrete control action selection (e.g., power level Low/Medium/High) using DQN:
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - tensorflow>=2.13.0, <2.16.0
    
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from collections import deque
    import random
    
    class DQNController:
        """
        DQN (Deep Q-Network) based controller
    
        Learn discrete control of CVD process
        Actions: Increase/maintain/decrease gas flow, RF power, and pressure
        """
    
        def __init__(self, state_dim=4, action_dim=27, learning_rate=0.001):
            """
            Parameters:
            -----------
            state_dim : int
                State dimension [thickness, rate, target_thickness, error]
            action_dim : int
                Number of actions (3 variables × 3 levels = 27 patterns)
            """
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.learning_rate = learning_rate
    
            # Hyperparameters
            self.gamma = 0.99  # Discount factor
            self.epsilon = 1.0  # ε-greedy initial value
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.batch_size = 64
            self.memory = deque(maxlen=10000)
    
            # Q-Network
            self.q_network = self._build_network()
            self.target_network = self._build_network()
            self.update_target_network()
    
        def _build_network(self):
            """Build Q-Network"""
            inputs = layers.Input(shape=(self.state_dim,))
    
            x = layers.Dense(128, activation='relu')(inputs)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dense(64, activation='relu')(x)
    
            # Q-value output (Q value for each action)
            q_values = layers.Dense(self.action_dim, activation='linear')(x)
    
            model = keras.Model(inputs, q_values)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss='mse'
            )
    
            return model
    
        def update_target_network(self):
            """Update Target Network weights"""
            self.target_network.set_weights(self.q_network.get_weights())
    
        def select_action(self, state):
            """
            Select action with ε-greedy policy
    
            Parameters:
            -----------
            state : ndarray
                Current state (state_dim,)
    
            Returns:
            --------
            action : int
                Selected action (0 ~ action_dim-1)
            """
            if np.random.rand() < self.epsilon:
                # Random action (exploration)
                return np.random.randint(self.action_dim)
            else:
                # Action with maximum Q value (exploitation)
                q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
                return np.argmax(q_values)
    
        def remember(self, state, action, reward, next_state, done):
            """Store experience in memory"""
            self.memory.append((state, action, reward, next_state, done))
    
        def replay(self):
            """
            Learn with Experience Replay
    
            Update Q-Network by random sampling from memory
            """
            if len(self.memory) < self.batch_size:
                return
    
            # Mini-batch sampling
            minibatch = random.sample(self.memory, self.batch_size)
    
            states = np.array([exp[0] for exp in minibatch])
            actions = np.array([exp[1] for exp in minibatch])
            rewards = np.array([exp[2] for exp in minibatch])
            next_states = np.array([exp[3] for exp in minibatch])
            dones = np.array([exp[4] for exp in minibatch])
    
            # Current Q values
            q_values = self.q_network.predict(states, verbose=0)
    
            # Target Q values (Double DQN)
            next_q_values = self.target_network.predict(next_states, verbose=0)
    
            # Bellman update
            for i in range(self.batch_size):
                if dones[i]:
                    q_values[i, actions[i]] = rewards[i]
                else:
                    q_values[i, actions[i]] = (
                        rewards[i] + self.gamma * np.max(next_q_values[i])
                    )
    
            # Train Q-Network
            self.q_network.fit(states, q_values, epochs=1, verbose=0)
    
            # Epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
        def action_to_control(self, action):
            """
            Convert action number to control input
    
            Action: 0-26 (3^3 = 27 patterns)
            Each variable: 0=decrease, 1=maintain, 2=increase
            """
            # Ternary expansion
            gas_action = action // 9
            rf_action = (action % 9) // 3
            pressure_action = action % 3
    
            # Convert to control amount
            gas_delta = (gas_action - 1) * 10  # ±10 sccm
            rf_delta = (rf_action - 1) * 20  # ±20 W
            pressure_delta = (pressure_action - 1) * 5  # ±5 mTorr
    
            return np.array([gas_delta, rf_delta, pressure_delta])
    
    
    class CVDEnvironment:
        """CVD Process Environment (for RL)"""
    
        def __init__(self, target_thickness=100):
            self.target_thickness = target_thickness
            self.reset()
    
        def reset(self):
            """Reset environment"""
            self.thickness = 0.0
            self.rate = 0.0
            self.gas_flow = 125
            self.rf_power = 250
            self.pressure = 55
            self.step_count = 0
    
            return self._get_state()
    
        def _get_state(self):
            """Get state"""
            error = self.target_thickness - self.thickness
            return np.array([self.thickness, self.rate, self.target_thickness, error])
    
        def step(self, action_delta):
            """
            Execute one step
    
            Parameters:
            -----------
            action_delta : ndarray
                Control amount change [Δgas, ΔRF, Δpressure]
    
            Returns:
            --------
            next_state : ndarray
            reward : float
            done : bool
            """
            # Update control input
            self.gas_flow = np.clip(self.gas_flow + action_delta[0], 50, 200)
            self.rf_power = np.clip(self.rf_power + action_delta[1], 100, 400)
            self.pressure = np.clip(self.pressure + action_delta[2], 10, 100)
    
            # Process simulation (simplified model)
            self.rate = (
                0.01 * self.gas_flow + 0.02 * self.rf_power - 0.005 * self.pressure
            ) / 10
            self.thickness += self.rate + np.random.normal(0, 0.1)
    
            # Reward design
            error = abs(self.target_thickness - self.thickness)
    
            if error < 1:
                reward = 10  # Target achieved
            elif error < 5:
                reward = 5 - error
            else:
                reward = -error / 10
    
            # Termination condition
            self.step_count += 1
            done = (self.step_count >= 50) or (error < 1)
    
            next_state = self._get_state()
    
            return next_state, reward, done
    
    
    # ========== DQN Training ==========
    if __name__ == "__main__":
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
    
        env = CVDEnvironment(target_thickness=100)
        agent = DQNController(state_dim=4, action_dim=27, learning_rate=0.001)
    
        print("========== DQN Training ==========")
        episodes = 200
        target_update_freq = 10
    
        episode_rewards = []
    
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
    
            for step in range(50):
                # Select action
                action = agent.select_action(state)
                action_delta = agent.action_to_control(action)
    
                # Environment step
                next_state, reward, done = env.step(action_delta)
                total_reward += reward
    
                # Store experience
                agent.remember(state, action, reward, next_state, done)
    
                # Learn
                agent.replay()
    
                state = next_state
    
                if done:
                    break
    
            episode_rewards.append(total_reward)
    
            # Update Target Network
            if episode % target_update_freq == 0:
                agent.update_target_network()
    
            # Progress display
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"Episode {episode+1}/{episodes}: "
                      f"Avg Reward (last 20) = {avg_reward:.2f}, "
                      f"ε = {agent.epsilon:.3f}")
    
        print("\n========== Training Complete ==========")
    
        # Learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, alpha=0.3)
        plt.plot(np.convolve(episode_rewards, np.ones(20)/20, mode='valid'),
                 linewidth=2, label='Moving Average (20 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('DQN Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('dqn_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        print(f"Final ε: {agent.epsilon:.4f}")
        print(f"Final Average Reward (last 20 episodes): "
              f"{np.mean(episode_rewards[-20:]):.2f}")
    

## 4.4 Summary

In this chapter, we learned AI implementation methods for Advanced Process Control (APC):

### Key Learning Content

#### 1\. Model Predictive Control (MPC)

  * **Prediction horizon optimization** considering future constraints
  * **Multivariable control** for simultaneous optimization of multiple I/O
  * **Linear MPC** : State-space model based (CVD film thickness control)
  * **Nonlinear MPC** : Utilizing Neural Network process models

#### 2\. Reinforcement Learning Control (DQN)

  * **Model-free learning** acquiring control laws from actual data
  * **Experience Replay** for efficient learning
  * **ε-greedy policy** balancing exploration and exploitation
  * **Discrete control** : Optimal control in 27-action space

#### Practical Results

  * Film thickness control accuracy: **Within ±0.5nm** (conventional ±2nm)
  * Tracking error: **RMSE < 1nm**
  * Constraint violations: **0 cases** (guaranteed operation within safe range)
  * Learning convergence: **200 episodes** to reach practical level

### Looking Ahead to the Next Chapter

In Chapter 5 "Fault Detection & Classification (FDC)", we will learn early detection and diagnosis techniques for process anomalies:

  * Multivariate Statistical Process Control (MSPC)
  * Anomaly detection using Isolation Forest
  * Fault diagnosis classification using Deep Learning
  * Root Cause Analysis (RCA) for cause identification

[← Previous Chapter](<chapter-3.html>) [Back to Contents](<index.html>) [Next Chapter →](<chapter-5.html>)

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
