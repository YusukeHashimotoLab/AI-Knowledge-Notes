---
title: "Chapter 5: Case Studies"
chapter_title: "Chapter 5: Case Studies"
subtitle: Case Studies in Food Process AI
---

[AI Terakoya Top](<../index.html>)›[Process Informatics](<../../index.html>)›Food Process AI›Chapter 5

🌐 EN | [🇯🇵 日本語](<../../../jp/PI/food-process-ai/chapter-5.html>) Last sync: 2025-11-16

[← Back to Series Index](<index.html>)

## 📖 Chapter Overview

In this chapter, we present concrete case studies in which the AI technologies learned in Chapters 1 through 4 are applied to actual food manufacturing processes. Through AI implementation examples across various food categories—dairy products, beverages, snack foods, seasonings, and more—you will learn practical methods for applying these technologies and their effects. For each case, we explain in detail the entire process from problem identification, AI technology selection, and implementation to effectiveness measurement. 

### 🎯 Learning Objectives

  * Understanding the AI implementation process in actual food manufacturing sites
  * Approaches to technology selection based on industry and product characteristics
  * Methods for evaluating ROI (Return on Investment)
  * Challenges during implementation and their solutions
  * Organizational transformation and cultivating a data culture

## 🥛 5.1 Case Study 1: Quality Control AI for Yogurt Manufacturing

### 📋 Company Profile

  * **Industry** : Dairy manufacturer (300 employees)
  * **Products** : Fermented dairy products (yogurt, drinkable yogurt)
  * **Production volume** : 50 tons per day

### 🚨 Challenges

  * Quality instability due to fermentation process variability (acidity, viscosity, flavor)
  * Difficulty responding to compositional changes in raw milk caused by seasonal variation
  * Skill transfer problems due to the retirement of experienced operators
  * 3.5% waste rate due to lot defects (approximately 6 million yen in annual losses)

### 💡 Implemented AI Solutions

  1. **Fermentation Condition Optimization AI** : Automatic adjustment of temperature and time using Bayesian optimization
  2. **Quality Prediction Model** : Predicting final product quality in advance from raw milk composition
  3. **Anomaly Detection System** : Real-time monitoring of temperature and pH changes during fermentation

### 💻 Code Example 5.1: Optimization of the Yogurt Fermentation Process
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Simulation of the yogurt fermentation process
    class YogurtFermentationSimulator:
        """Yogurt fermentation process simulator"""
    
        def __init__(self):
            # Optimal conditions (true values, unknown in experiments)
            self.optimal_temp = 42.5  # °C
            self.optimal_time = 5.0   # hours
            self.optimal_pH = 6.2
    
        def simulate_quality(self, temperature, fermentation_time, initial_pH, lactose_content=4.5):
            """
            Simulate the quality score from fermentation conditions
    
            Args:
                temperature: Fermentation temperature (°C)
                fermentation_time: Fermentation time (hours)
                initial_pH: Initial pH
                lactose_content: Lactose content (%)
    
            Returns:
                quality_score: Quality score (0-100)
            """
            # Effect of temperature (40-45°C is optimal)
            temp_factor = 1.0 - 0.2 * ((temperature - self.optimal_temp) / 5) ** 2
    
            # Effect of time (4-6 hours is optimal)
            time_factor = 1.0 - 0.15 * ((fermentation_time - self.optimal_time) / 2) ** 2
    
            # Effect of pH (6.0-6.5 is optimal)
            pH_factor = 1.0 - 0.1 * ((initial_pH - self.optimal_pH) / 0.5) ** 2
    
            # Effect of lactose content (4.0-5.0% is optimal)
            lactose_factor = 1.0 - 0.05 * ((lactose_content - 4.5) / 0.5) ** 2
    
            # Overall quality score
            base_quality = 85
            quality_score = base_quality * temp_factor * time_factor * pH_factor * lactose_factor
    
            # Random noise (process variation)
            noise = np.random.normal(0, 2)
            quality_score += noise
    
            # Clip to the 0-100 range
            quality_score = np.clip(quality_score, 0, 100)
    
            return quality_score
    
        def simulate_acidity(self, temperature, fermentation_time):
            """Calculate post-fermentation acidity (°T)"""
            # Higher temperature and time increase acidity
            acidity = 60 + (temperature - 40) * 2 + fermentation_time * 5
            acidity += np.random.normal(0, 3)
            return np.clip(acidity, 40, 100)
    
        def simulate_viscosity(self, temperature, protein_content=3.5):
            """Calculate viscosity (mPa·s)"""
            # Higher temperature lowers viscosity
            viscosity = 5000 - (temperature - 42) * 200 + protein_content * 300
            viscosity += np.random.normal(0, 200)
            return np.clip(viscosity, 2000, 8000)
    
    # Initialize the simulator
    simulator = YogurtFermentationSimulator()
    
    # Generate experimental data (simulating past production data)
    np.random.seed(42)
    n_experiments = 50
    
    experimental_data = []
    for i in range(n_experiments):
        temp = np.random.uniform(38, 46)
        time = np.random.uniform(3, 7)
        pH = np.random.uniform(5.8, 6.6)
        lactose = np.random.uniform(4.0, 5.0)
    
        quality = simulator.simulate_quality(temp, time, pH, lactose)
        acidity = simulator.simulate_acidity(temp, time)
        viscosity = simulator.simulate_viscosity(temp)
    
        experimental_data.append({
            'temperature': temp,
            'fermentation_time': time,
            'initial_pH': pH,
            'lactose_content': lactose,
            'quality_score': quality,
            'acidity': acidity,
            'viscosity': viscosity
        })
    
    df_experiments = pd.DataFrame(experimental_data)
    
    # Implementation of Bayesian optimization
    class BayesianOptimizationYogurt:
        """Bayesian optimization of yogurt fermentation conditions"""
    
        def __init__(self, bounds, simulator, n_init=10):
            self.bounds = np.array(bounds)
            self.simulator = simulator
            self.n_init = n_init
            self.X_sample = []
            self.y_sample = []
    
            # Gaussian process regression model
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                               alpha=1e-6, normalize_y=True)
    
        def acquisition_function(self, X, xi=0.01):
            """Expected Improvement acquisition function"""
            X = np.atleast_2d(X)
            mu, sigma = self.gp.predict(X, return_std=True)
    
            if len(self.y_sample) > 0:
                mu_sample_opt = np.max(self.y_sample)
            else:
                mu_sample_opt = 0
    
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)
                ei[sigma == 0.0] = 0.0
    
            return ei
    
        def _norm_pdf(self, x):
            """PDF of the standard normal distribution"""
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
        def _norm_cdf(self, x):
            """CDF of the standard normal distribution"""
            return 0.5 * (1 + np.vectorize(lambda t: np.sign(t) * np.sqrt(1 - np.exp(-2*t**2/np.pi)))(x))
    
        def propose_location(self):
            """Propose the next experiment point"""
            def min_obj(X):
                return -self.acquisition_function(X)
    
            min_val = float('inf')
            min_x = None
    
            # Optimize with random starts
            for _ in range(25):
                x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
    
                if res.fun < min_val:
                    min_val = res.fun
                    min_x = res.x
    
            return min_x
    
        def optimize(self, n_iter=20, initial_pH=6.2, lactose_content=4.5):
            """Run the optimization"""
            # Initial random sampling
            for _ in range(self.n_init):
                x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                y = self.simulator.simulate_quality(x[0], x[1], initial_pH, lactose_content)
                self.X_sample.append(x)
                self.y_sample.append(y)
    
            # Main loop of Bayesian optimization
            for iteration in range(n_iter):
                # Update the GP model
                self.gp.fit(np.array(self.X_sample), np.array(self.y_sample))
    
                # Propose the next experiment point
                x_next = self.propose_location()
    
                # Conduct the experiment (simulation)
                y_next = self.simulator.simulate_quality(x_next[0], x_next[1], initial_pH, lactose_content)
    
                # Add data
                self.X_sample.append(x_next)
                self.y_sample.append(y_next)
    
                if (iteration + 1) % 5 == 0:
                    best_idx = np.argmax(self.y_sample)
                    best_x = self.X_sample[best_idx]
                    best_y = self.y_sample[best_idx]
                    print(f"Iteration {iteration + 1}: current best = quality score {best_y:.2f} "
                          f"(temperature: {best_x[0]:.1f}°C, time: {best_x[1]:.1f}h)")
    
            # Extract the optimal conditions
            best_idx = np.argmax(self.y_sample)
            best_params = self.X_sample[best_idx]
            best_quality = self.y_sample[best_idx]
    
            return best_params, best_quality
    
    # Run Bayesian optimization
    print("=" * 60)
    print("Yogurt Fermentation Process Optimization (Bayesian Optimization)")
    print("=" * 60)
    
    bounds = [[38, 46], [3, 7]]  # [temperature range, time range]
    optimizer = BayesianOptimizationYogurt(bounds, simulator, n_init=10)
    
    print("\nStarting optimization...")
    best_params, best_quality = optimizer.optimize(n_iter=20, initial_pH=6.2, lactose_content=4.5)
    
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Optimal temperature: {best_params[0]:.2f} °C")
    print(f"Optimal fermentation time: {best_params[1]:.2f} hours")
    print(f"Predicted quality score: {best_quality:.2f}")
    print(f"\nReference: true optimal conditions")
    print(f"Optimal temperature: {simulator.optimal_temp} °C")
    print(f"Optimal fermentation time: {simulator.optimal_time} hours")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Optimization history
    iterations = range(1, len(optimizer.y_sample) + 1)
    cumulative_best = [max(optimizer.y_sample[:i+1]) for i in range(len(optimizer.y_sample))]
    
    axes[0, 0].plot(iterations, optimizer.y_sample, 'o-', color='#11998e', alpha=0.6, label='Quality of each experiment')
    axes[0, 0].plot(iterations, cumulative_best, 'r-', linewidth=2, label='Cumulative best')
    axes[0, 0].set_xlabel('Experiment number')
    axes[0, 0].set_ylabel('Quality score')
    axes[0, 0].set_title('Convergence of Bayesian Optimization', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Temperature-time map
    temp_grid = np.linspace(38, 46, 50)
    time_grid = np.linspace(3, 7, 50)
    T, Ti = np.meshgrid(temp_grid, time_grid)
    Z = np.zeros_like(T)
    
    for i in range(len(temp_grid)):
        for j in range(len(time_grid)):
            Z[j, i] = simulator.simulate_quality(T[j, i], Ti[j, i], 6.2, 4.5)
    
    contour = axes[0, 1].contourf(T, Ti, Z, levels=20, cmap='RdYlGn')
    axes[0, 1].scatter([x[0] for x in optimizer.X_sample],
                       [x[1] for x in optimizer.X_sample],
                       c='blue', s=50, edgecolor='black', linewidth=1, label='Experiment points', zorder=5)
    axes[0, 1].scatter(best_params[0], best_params[1], c='red', s=200, marker='*',
                       edgecolor='black', linewidth=2, label='Optimal point', zorder=6)
    axes[0, 1].set_xlabel('Fermentation temperature (°C)')
    axes[0, 1].set_ylabel('Fermentation time (hours)')
    axes[0, 1].set_title('Contour Plot of Quality Score', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    plt.colorbar(contour, ax=axes[0, 1], label='Quality score')
    
    # 3. Effect of temperature
    temp_range = np.linspace(38, 46, 30)
    quality_temp = [simulator.simulate_quality(t, best_params[1], 6.2, 4.5) for t in temp_range]
    
    axes[1, 0].plot(temp_range, quality_temp, color='#38ef7d', linewidth=2)
    axes[1, 0].axvline(x=best_params[0], color='red', linestyle='--', linewidth=2, label=f'Optimal temperature: {best_params[0]:.1f}°C')
    axes[1, 0].set_xlabel('Fermentation temperature (°C)')
    axes[1, 0].set_ylabel('Quality score')
    axes[1, 0].set_title('Relationship Between Temperature and Quality', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Effect of time
    time_range = np.linspace(3, 7, 30)
    quality_time = [simulator.simulate_quality(best_params[0], t, 6.2, 4.5) for t in time_range]
    
    axes[1, 1].plot(time_range, quality_time, color='#11998e', linewidth=2)
    axes[1, 1].axvline(x=best_params[1], color='red', linestyle='--', linewidth=2, label=f'Optimal time: {best_params[1]:.1f}h')
    axes[1, 1].set_xlabel('Fermentation time (hours)')
    axes[1, 1].set_ylabel('Quality score')
    axes[1, 1].set_title('Relationship Between Fermentation Time and Quality', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yogurt_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 📊 Implementation Results (After 6 Months)

Metric | Before | After | Improvement  
---|---|---|---  
Waste rate | 3.5% | 1.2% | ▼ 65.7%  
Average quality score | 78.5 | 89.2 | ▲ 13.6%  
Variability (standard deviation) | 8.3 | 3.1 | ▼ 62.7%  
Annual cost reduction | - | Approx. 4 million yen | -  
  
### 🔑 Keys to Success

  * Curbing initial investment by utilizing existing sensor data
  * Discovering optimal conditions with few experiments via Bayesian optimization
  * Emphasizing communication with on-site operators and visualizing AI decision-making
  * Risk management through phased implementation (one line → deployment to all lines)

## 🥤 5.2 Case Study 2: Predictive Maintenance System for Soft Drinks

### 📋 Company Profile

  * **Industry** : Beverage manufacturer (500 employees)
  * **Products** : Carbonated drinks, juice, sports drinks
  * **Production volume** : 2 million bottles per day

### 🚨 Challenges

  * Production stoppages due to sudden filling machine failures (15 times per year, average downtime 4 hours)
  * Opportunity losses and delivery delays due to unplanned stoppages
  * Increased maintenance costs due to excessive preventive maintenance
  * Time-consuming cause identification during equipment stoppages (average 1.5 hours)

### 💡 Implemented AI Solutions

  1. **Equipment Failure Prediction AI** : Failure risk prediction up to 24 hours ahead using Random Forest
  2. **Anomaly Detection System** : Real-time monitoring using Isolation Forest
  3. **Root Cause Analysis Tool** : Automatic identification of failure factors using decision trees

### 💻 Code Example 5.2: Filling Machine Failure Prediction Dashboard
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Simulation of filling machine sensor data (24 hours)
    np.random.seed(42)
    hours = 24
    data_points_per_hour = 60  # every minute
    total_points = hours * data_points_per_hour
    
    # Generate timestamps
    start_time = datetime(2025, 10, 27, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(total_points)]
    
    # Generate normal operation data (0-18 hours)
    normal_hours = 18 * data_points_per_hour
    
    temp_normal = np.random.normal(65, 2, normal_hours)
    vibration_normal = np.random.normal(0.3, 0.05, normal_hours)
    pressure_normal = np.random.normal(4.0, 0.1, normal_hours)
    flow_rate_normal = np.random.normal(1000, 20, normal_hours)
    motor_current_normal = np.random.normal(25, 1, normal_hours)
    
    # Signs of anomaly (18-24 hours: gradual degradation)
    degradation_hours = total_points - normal_hours
    t_degrade = np.linspace(0, 1, degradation_hours)
    
    temp_degrade = 65 + t_degrade * 15 + np.random.normal(0, 3, degradation_hours)
    vibration_degrade = 0.3 + t_degrade * 0.5 + np.random.normal(0, 0.1, degradation_hours)
    pressure_degrade = 4.0 - t_degrade * 0.8 + np.random.normal(0, 0.15, degradation_hours)
    flow_rate_degrade = 1000 - t_degrade * 150 + np.random.normal(0, 30, degradation_hours)
    motor_current_degrade = 25 + t_degrade * 10 + np.random.normal(0, 2, degradation_hours)
    
    # Create the dataframe
    sensor_data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.concatenate([temp_normal, temp_degrade]),
        'vibration': np.concatenate([vibration_normal, vibration_degrade]),
        'pressure': np.concatenate([pressure_normal, pressure_degrade]),
        'flow_rate': np.concatenate([flow_rate_normal, flow_rate_degrade]),
        'motor_current': np.concatenate([motor_current_normal, motor_current_degrade])
    })
    
    # Calculate the failure risk score (simplified version)
    def calculate_failure_risk(row):
        """Calculate the failure risk score from sensor values (0-100)"""
        # Deviation of each parameter from its normal range
        temp_risk = max(0, (row['temperature'] - 65) / 20) * 100
        vib_risk = max(0, (row['vibration'] - 0.3) / 0.7) * 100
        pressure_risk = max(0, (4.0 - row['pressure']) / 2.0) * 100
        flow_risk = max(0, (1000 - row['flow_rate']) / 200) * 100
        current_risk = max(0, (row['motor_current'] - 25) / 15) * 100
    
        # Overall risk score (take the maximum)
        total_risk = max(temp_risk, vib_risk, pressure_risk, flow_risk, current_risk)
        return min(100, total_risk)
    
    sensor_data['failure_risk'] = sensor_data.apply(calculate_failure_risk, axis=1)
    
    # Classify the risk level
    def classify_risk_level(risk_score):
        if risk_score < 20:
            return 'Low'
        elif risk_score < 50:
            return 'Medium'
        elif risk_score < 80:
            return 'High'
        else:
            return 'Critical'
    
    sensor_data['risk_level'] = sensor_data['failure_risk'].apply(classify_risk_level)
    
    # Statistical summary
    print("=" * 60)
    print("Filling Machine Predictive Maintenance Dashboard")
    print("=" * 60)
    print(f"\nMonitoring period: {timestamps[0].strftime('%Y-%m-%d %H:%M')} ~ {timestamps[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"Total data points: {total_points}")
    
    current_risk = sensor_data.iloc[-1]['failure_risk']
    current_level = sensor_data.iloc[-1]['risk_level']
    print(f"\nCurrent failure risk: {current_risk:.1f} ({current_level})")
    
    # Aggregate by risk level
    risk_counts = sensor_data['risk_level'].value_counts()
    print(f"\nRisk level distribution:")
    for level in ['Low', 'Medium', 'High', 'Critical']:
        if level in risk_counts.index:
            count = risk_counts[level]
            percentage = count / total_points * 100
            print(f"  {level}: {count} points ({percentage:.1f}%)")
    
    # Warning messages
    if current_risk >= 80:
        print(f"\n🚨 [CRITICAL WARNING] Failure risk has reached the danger zone!")
        print(f"   Recommended action: Stop production immediately and perform equipment inspection")
    elif current_risk >= 50:
        print(f"\n⚠️ [WARNING] Failure risk is rising")
        print(f"   Recommended action: Plan equipment inspection during the next break")
    elif current_risk >= 20:
        print(f"\n📝 [CAUTION] Slight signs of anomaly have been detected")
        print(f"   Recommended action: Continue monitoring")
    else:
        print(f"\n✅ [NORMAL] The equipment is operating normally")
    
    # Visualization dashboard
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Time data (for the X-axis)
    time_hours = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]
    
    # 1. Temperature trend
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_hours, sensor_data['temperature'], color='#ff6b6b', linewidth=1)
    ax1.axhline(y=65, color='green', linestyle='--', alpha=0.5, label='Normal value')
    ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    ax1.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='Danger threshold')
    ax1.set_xlabel('Elapsed time (hours)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Filling Head Temperature', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    
    # 2. Vibration trend
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_hours, sensor_data['vibration'], color='#4ecdc4', linewidth=1)
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Elapsed time (hours)')
    ax2.set_ylabel('Vibration (mm/s)')
    ax2.set_title('Vibration Level', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Pressure trend
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_hours, sensor_data['pressure'], color='#95e1d3', linewidth=1)
    ax3.axhline(y=4.0, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=3.5, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Elapsed time (hours)')
    ax3.set_ylabel('Pressure (MPa)')
    ax3.set_title('Filling Pressure', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Flow rate trend
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_hours, sensor_data['flow_rate'], color='#f38181', linewidth=1)
    ax4.axhline(y=1000, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=900, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=800, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Elapsed time (hours)')
    ax4.set_ylabel('Flow rate (bottles/min)')
    ax4.set_title('Filling Flow Rate', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. Motor current trend
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time_hours, sensor_data['motor_current'], color='#aa96da', linewidth=1)
    ax5.axhline(y=25, color='green', linestyle='--', alpha=0.5)
    ax5.axhline(y=30, color='orange', linestyle='--', alpha=0.5)
    ax5.axhline(y=35, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Elapsed time (hours)')
    ax5.set_ylabel('Current (A)')
    ax5.set_title('Motor Current', fontsize=11, fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # 6. Failure risk score
    ax6 = fig.add_subplot(gs[2, 1])
    colors = []
    for risk in sensor_data['failure_risk']:
        if risk < 20:
            colors.append('green')
        elif risk < 50:
            colors.append('yellow')
        elif risk < 80:
            colors.append('orange')
        else:
            colors.append('red')
    
    ax6.scatter(time_hours, sensor_data['failure_risk'], c=colors, s=5, alpha=0.6)
    ax6.axhline(y=20, color='yellow', linestyle='--', alpha=0.5, label='Caution')
    ax6.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax6.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Danger')
    ax6.set_xlabel('Elapsed time (hours)')
    ax6.set_ylabel('Failure risk score')
    ax6.set_title('Integrated Failure Risk Assessment', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    # 7. Risk level transition (stacked bar chart)
    ax7 = fig.add_subplot(gs[3, :])
    
    # Resample by hour
    sensor_data['hour'] = sensor_data['timestamp'].dt.hour
    risk_by_hour = sensor_data.groupby(['hour', 'risk_level']).size().unstack(fill_value=0)
    
    # Stacked bar chart
    risk_by_hour.plot(kind='bar', stacked=True, ax=ax7,
                      color={'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Critical': 'red'},
                      width=0.8)
    ax7.set_xlabel('Hour')
    ax7.set_ylabel('Number of data points')
    ax7.set_title('Risk Level Distribution by Time of Day', fontsize=11, fontweight='bold')
    ax7.legend(title='Risk level', fontsize=8)
    ax7.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Filling Machine Predictive Maintenance Dashboard - Real-time Monitoring', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig('filling_machine_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Recommended maintenance schedule
    print("\n" + "=" * 60)
    print("Recommended Maintenance Schedule")
    print("=" * 60)
    
    if current_risk >= 80:
        print("⏰ Emergency maintenance: within 2 hours")
        print("📝 Inspection items: Comprehensive system inspection, prepare parts replacement")
    elif current_risk >= 50:
        print("⏰ Planned maintenance: within 24 hours")
        print("📝 Inspection items: Around vibration/temperature sensors, motor bearings")
    elif current_risk >= 20:
        print("⏰ Preventive maintenance: within 1 week")
        print("📝 Inspection items: Routine cleaning, lubricant replenishment")
    else:
        print("⏰ Next scheduled maintenance: as per the normal schedule")
        print("📝 Inspection items: Standard inspection items")
    

### 📊 Implementation Results (After 12 Months)

Metric | Before | After | Improvement  
---|---|---|---  
Unplanned stoppages | 15/year | 3/year | ▼ 80%  
Average failure response time | 4.0 hours | 1.5 hours | ▼ 62.5%  
Maintenance cost | 12 million yen/year | 8.5 million yen/year | ▼ 29.2%  
Equipment uptime | 92.5% | 97.8% | ▲ 5.7%  
  
### 🔑 Keys to Success

  * Achieving real-time data collection through additional investment in IoT sensors
  * A dashboard for maintenance personnel that lets the situation be grasped even without specialized knowledge
  * Suppressing false positives through graduated alert settings
  * Root cause analysis combining maintenance history data with AI predictions

## 🍪 5.3 Summary of Other Food Category Cases

### Snack Food Manufacturing (Frying Process)

**Challenge** : Quality degradation due to frying oil deterioration, difficulty optimizing oil change timing

**AI Technology** : Product color tone analysis via image recognition + oil deterioration prediction via time series forecasting

**Effect** : 20% reduction in oil change frequency, reduced waste oil costs (2.5 million yen annually)

### Bread Manufacturing (Fermentation and Baking)

**Challenge** : Changes in fermentation speed due to climate variation, occurrence of uneven baking

**AI Technology** : Automatic fermentation time adjustment via machine learning + thermography image analysis

**Effect** : Reduced the uneven-baking defect rate from 2.8% to 0.6%

### Seasoning Manufacturing (Fermented Seasonings)

**Challenge** : Difficulty predicting quality in long-term fermentation processes (6 months to 1 year)

**AI Technology** : Prediction of end-of-fermentation quality via deep learning (LSTM)

**Effect** : Predicting final quality with ±5% accuracy at the 3-month fermentation point, early detection of defective lots

## 🎯 5.4 Success Factors and Lessons from AI Implementation

### Common Success Factors

  1. **Management Commitment** : Top-down promotion of AI strategy
  2. **Collaboration with the Field** : Integrating operator expertise with AI
  3. **Small Start** : Starting from a single line or process and horizontally expanding success cases
  4. **Ensuring Data Quality** : Sensor calibration, data cleansing
  5. **Continuous Improvement** : Periodic model retraining and accuracy improvement

### Lessons Learned from Failures

  * **Excessive Expectations** : AI is not a panacea. Clarifying the scope of application is important
  * **Insufficient Data** : At least 6 months to 1 year of data is required (including seasonal variation)
  * **Black-Boxing** : A lack of explainability breeds distrust on the shop floor
  * **Inadequate Maintenance Structure** : A post-implementation maintenance plan is essential

### Key Points for ROI Evaluation

#### Return on Investment Formula

$$ \text{ROI} = \frac{\text{Annual Cost Reduction} + \text{Added Profit from Productivity Gains}}{\text{Initial Investment} + \text{Annual Operating Cost}} \times 100 (\%) $$ 

  * **Initial Investment** : Sensor installation, system development, education and training
  * **Cost Reduction** : Waste reduction, maintenance cost reduction, energy reduction
  * **Added Profit** : Added value from quality improvement, increased production volume

**Benchmark** : ROI payback within 2-3 years is a common target

## 📚 Summary

In this chapter, we studied actual AI implementation cases in food manufacturing processes.

### Key Points

  * Selection and application of AI technologies according to industry and product characteristics
  * Integrated use of multiple technologies such as Bayesian optimization, failure prediction, and anomaly detection
  * The importance of quantitative effectiveness measurement and ROI evaluation
  * Collaboration with the field and cultivating a culture of continuous improvement
  * Risk management through a small start

**🎓 Series Complete**  
In this series, "Introduction to Food Process AI," we comprehensively learned about AI utilization in food manufacturing sites, from the fundamentals through to practice. Going forward, according to your own organization's challenges, please select and implement appropriate AI technologies and drive data-driven improvement of your manufacturing processes.  
  
Resources for continued learning:  
・Other series in the Process Informatics Dojo  
・The fundamentals series in the Machine Learning Dojo  
・Participation in industry conferences and workshops 

[← Chapter 4: Predictive Maintenance](<chapter-4.html>) [Series Index →](<index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
