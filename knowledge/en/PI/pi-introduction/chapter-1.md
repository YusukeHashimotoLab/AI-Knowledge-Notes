---
title: "Chapter 1: Fundamental Concepts of PI and Data Utilization in Process Industries"
chapter_title: "Chapter 1: Fundamental Concepts of PI and Data Utilization in Process Industries"
subtitle: Foundation of Digital Transformation in Process Industries
version: 1.0
created_at: "by:"
---

# Chapter 1: Fundamental Concepts of PI and Data Utilization in Process Industries

Understand the basic concepts of Process Informatics (PI) and learn about the characteristics of process industries and types of data. Through real improvement cases with data-driven approaches, experience the value of PI firsthand.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Explain the definition and purpose of Process Informatics (PI)
  * ✅ Understand the characteristics of process industries and differences from conventional materials development
  * ✅ Classify major types of process data (sensor, operation, quality data)
  * ✅ Explain specific examples of data-driven process improvement
  * ✅ Perform basic process data visualization using Python

* * *

## 1.1 What is Process Informatics (PI)?

### Definition of PI

**Process Informatics (PI)** is an academic field that uses **data-driven approaches** to understand, optimize, and control processes in chemical plants, pharmaceuticals, food, semiconductors, and other process industries.

Specifically, PI encompasses five core activities: **process data collection and analysis** (gathering and analyzing sensor data, operating conditions, and quality data), **process modeling** (constructing quality prediction models using machine learning and soft sensors), **process optimization** (improving yield, reducing energy consumption, and enhancing quality through optimal conditions), **process control** (building automatic control systems based on real-time data), and **anomaly detection** (early identification of abnormal states through data analysis).

### Difference from Materials Informatics (MI)

PI is often confused with Materials Informatics (MI). Let's clearly understand the difference between them.

Item | Materials Informatics (MI) | Process Informatics (PI)  
---|---|---  
**Target** | Materials themselves (composition, structure, properties) | Manufacturing processes (operating conditions, control, quality)  
**Purpose** | Discovery and design of new materials | Process optimization and control  
**Data types** | Physical property values, crystal structures, composition data | Time-series sensor data, operating conditions  
**Time axis** | Static (intrinsic material properties) | Dynamic (changing moment by moment)  
**Main methods** | Descriptors, materials databases, screening | Time-series analysis, soft sensors, process control  
**Typical challenges** | Search for materials with bandgap of 2.5 eV | Reduce energy consumption by 10% while maintaining product purity above 98%  
  
**Important point** : MI focuses on "what to make," while PI focuses on "how to make it." The two are complementary, collaborating in ways such as efficiently manufacturing new materials discovered by MI using PI.

* * *

## 1.2 Characteristics of Process Industries

To understand PI, you first need to know the characteristics of process industries.

### Classification of Processes

Processes in process industries are broadly classified into two types:

Aspect | Continuous Process | Batch Process  
---|---|---  
**Characteristics** | 24/7, 365 days operation with continuous raw material feed and product extraction | Cyclic operation: raw material input → reaction → product extraction  
**Examples** | Oil refining, chemical plants (ethylene), papermaking | Pharmaceutical manufacturing, food processing, fine chemicals  
**Advantages** | High productivity, stable quality | Flexible product switching, suitable for small-volume high-variety production  
**Challenges** | High shutdown/restart costs, low flexibility | Batch-to-batch variability, lower productivity  
  
### Major Fields of Process Industries

Industry Field | Main Processes | Typical PI Applications  
---|---|---  
**Chemical** | Distillation, reaction, separation | Yield optimization, energy reduction  
**Petrochemical** | Refining, cracking | Product quality prediction, equipment anomaly detection  
**Pharmaceutical** | Synthesis, crystallization, drying | Batch quality control, accelerated process development  
**Food** | Fermentation, sterilization, mixing | Quality consistency improvement, shelf-life prediction  
**Semiconductor** | CVD, etching, cleaning | Yield improvement, real-time control  
  
### Process Complexity

Processes in process industries are very complex due to the following characteristics:

  1. **Multivariate nature**
     * Dozens to hundreds of variables mutually influence each other
     * Example: In a distillation column, temperature (multiple stages), pressure, flow rate, and composition are interconnected
  2. **Nonlinearity**
     * The relationship between inputs and outputs is not linear
     * Example: Raising reaction temperature by 10°C increases yield by 5%, but raising it another 10°C causes side reactions, reducing yield
  3. **Time delay (time lag)**
     * It takes time for the effect of operational changes to appear
     * Example: Changing the reboiler temperature in a distillation column affects the overhead product only after several minutes to tens of minutes
  4. **Presence of disturbances**
     * Raw material composition variations, ambient temperature changes, etc.
     * These need to be compensated for in real-time

**Conclusion** : Due to this complexity, **data-driven approaches (PI)** are effective. Machine learning and data analysis can capture complex relationships that cannot be handled by conventional rules of thumb or simple models.

* * *

## 1.3 Types of Process Data

The data handled in PI is mainly classified into three categories.

### 1\. Sensor Data (Measurement Data)

Data that measures process states in real-time.

**Major sensor types** :

Sensor Type | Measurement Target | Typical Measurement Frequency | Usage Examples  
---|---|---|---  
**Temperature sensor** | Process temperature | 1 second to 1 minute | Reaction temperature monitoring, furnace control  
**Pressure sensor** | Process pressure | 1 second to 1 minute | Distillation column pressure, reactor pressure  
**Flow meter** | Fluid flow rate | 1 second to 1 minute | Raw material feed rate, product extraction rate  
**Level meter** | Tank liquid level | 10 seconds to 1 minute | Reactor level, storage tank  
**Concentration meter** | Component concentration | 1 minute to 1 hour | Product purity, reaction progress  
**pH meter** | pH value | 1 second to 1 minute | Chemical reaction control  
  
Sensor data is characterized by high-frequency generation (second to minute intervals), accumulation as time-series data, and often contains missing values, outliers, and noise that require preprocessing.

### 2\. Operation Condition Data (Setpoints and Control Data)

Values set by humans or DCS (Distributed Control System) to control the process.

Major operation variables include **temperature setpoints** (reactor and furnace settings), **pressure setpoints** (distillation column pressure), **flow rate setpoints** (raw material feed and cooling water), **valve openings** (0-100%), and **agitation speeds** (reactor agitator rotation). These variables are directly linked with sensor data for setpoint-vs-measured comparisons, and finding optimal operating conditions is a core objective of PI.

### 3\. Quality Data (Product Properties and Analytical Data)

Data for evaluating product quality. Often obtained through offline analysis.

**Major quality indicators** :

Quality Indicator | Measurement Method | Measurement Frequency | Importance  
---|---|---|---  
**Product purity** | Gas chromatography (GC) | 1 hour to 1 day | Main item of product specifications  
**Yield** | Material balance calculation | Per batch | Economic indicator  
**Viscosity** | Viscometer | Several hours to 1 day | Product quality  
**Color** | Spectrophotometer | Several hours to 1 day | Product appearance  
**Impurity content** | HPLC, GC-MS | 1 day to 1 week | Quality specifications, safety  
  
Quality data has low measurement frequency (hour to day intervals) and requires significant time and cost. An important PI challenge is building **"soft sensors"** that predict quality in real-time from sensor data, enabling continuous quality monitoring without expensive offline analysis.

### 4\. Event Data (Auxiliary Data)

Event data records various occurrences in the process, including **alarms** (warnings of abnormal conditions such as temperature or pressure anomalies), **operation logs** (manual records by operators), **equipment maintenance records** (maintenance history), and **batch records** (start/end times and raw material information for each batch).

* * *

## 1.4 Case Studies of Data-Driven Process Improvement

Beyond theory, let's look at specific examples of how PI is being utilized in practice.

### Case Study 1: Chemical Plant Yield Improvement (5% improvement)

**Background** :

At a chemical plant, product C was manufactured from raw materials A and B. The theoretical yield was 95%, but the actual yield remained at an average of 85%. Traditionally, veteran operators adjusted reaction temperature and pressure based on experience, but it was not clear why the yield varied.

**PI Approach** :

  1. **Data collection (1 month)**
     * Collected 2 years of operational data (temperature, pressure, flow rate, raw material composition)
     * Collected product quality data for the same period (yield, purity)
     * Number of data points: approximately 1 million
  2. **Exploratory Data Analysis (EDA)**
     * Correlation analysis between variables
     * Discovery: Not only reaction temperature, but **the ratio of raw material A feed rate to reactor agitation speed** strongly affected yield
     * This was a previously unknown relationship
  3. **Machine learning model construction**
     * Built a yield prediction model using Random Forest
     * Prediction accuracy: R² = 0.82
     * Identified important operation variables through feature importance analysis
  4. **Optimization and implementation**
     * Searched for optimal operating conditions based on the model
     * Optimal conditions: reaction temperature 175°C, raw material A flow rate 2.5 m³/h, agitation speed 300 rpm
     * Conducted trial operation in actual plant

**Results** :

  * **Yield improvement** : 85% → 90% (**+5%**)
  * **Economic effect** : Profit increase of approximately 500 million yen per year (in product value terms)
  * **Secondary effect** : Energy consumption also reduced by 3%

### Case Study 2: Energy Consumption Reduction (15% reduction)

**Background** :

The distillation column at a petrochemical plant used large amounts of steam to refine products. Energy costs were over 1 billion yen per year, a major business challenge.

**PI Approach** :

  1. **Energy consumption analysis**
     * Collected data on reboiler heat duty and product quality in the distillation column
     * Discovery: Operation at unnecessarily high reflux ratio for many time periods
     * Target product purity was 99.5%, but actual operation was around 99.8% (excessive quality)
  2. **Soft sensor construction**
     * Purpose: Real-time prediction of overhead product purity
     * Conventional: Purity could only be measured once a day by GC analysis
     * Built PLS model to predict purity from temperature profile
     * Prediction accuracy: R² = 0.88
  3. **Optimal control**
     * Dynamically adjusted reflux ratio using soft sensor predictions
     * Minimized energy usage while maintaining product purity at 99.5-99.6%

**Results** :

  * **Energy reduction** : **15% reduction**
  * **Economic effect** : Cost reduction of approximately 150 million yen per year
  * **CO₂ reduction** : Approximately 5,000 tons of CO₂ emissions reduced per year
  * **Payback period** : Approximately 1 year (including soft sensor development and implementation costs)

### ROI (Return on Investment) Analysis

Let's look at the typical ROI of PI projects.

Item | Case Study 1 (Yield improvement) | Case Study 2 (Energy reduction)  
---|---|---  
**Initial investment** | Data infrastructure: 20 million yen  
Model development: 10 million yen | Soft sensor development: 15 million yen  
Control system modification: 30 million yen  
**Annual benefit** | 500 million yen/year (profit increase) | 150 million yen/year (cost reduction)  
**ROI** | First year **1,567%** | First year **233%**  
**Payback period** | **Approximately 2 months** | **Approximately 4 months**  
  
**Conclusion** : When properly implemented, PI projects can achieve **extremely high ROI**.

* * *

## 1.5 Introduction to Process Data Visualization with Python

Now, let's actually visualize process data using Python. Through the following five code examples, you'll acquire fundamental PI skills.

### Environment Setup

First, install the necessary libraries.
    
    
    # Install required libraries
    pip install pandas matplotlib seaborn numpy plotly
    

### Code Example 1: Time-Series Sensor Data Visualization

Plot time-series temperature data from a chemical reactor.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Plot time-series temperature data from a chemical reactor.
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate sample data: 1 day of reactor temperature data (1-minute intervals)
    np.random.seed(42)
    time_points = pd.date_range('2025-01-01 00:00', periods=1440, freq='1min')
    temperature = 175 + np.random.normal(0, 2, 1440) + \
                  5 * np.sin(np.linspace(0, 4*np.pi, 1440))  # Simulate temperature fluctuation
    
    # Store in DataFrame
    df = pd.DataFrame({
        'timestamp': time_points,
        'temperature': temperature
    })
    
    # Visualization
    plt.figure(figsize=(14, 5))
    plt.plot(df['timestamp'], df['temperature'], linewidth=0.8, color='#11998e')
    plt.axhline(y=175, color='red', linestyle='--', label='Target: 175°C')
    plt.fill_between(df['timestamp'], 173, 177, alpha=0.2, color='green', label='Acceptable range')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Reactor Temperature - 24 Hour Trend', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    print(f"Average temperature: {df['temperature'].mean():.2f}°C")
    print(f"Standard deviation: {df['temperature'].std():.2f}°C")
    print(f"Maximum temperature: {df['temperature'].max():.2f}°C")
    print(f"Minimum temperature: {df['temperature'].min():.2f}°C")
    

**Output example** :
    
    
    Average temperature: 174.98°C
    Standard deviation: 3.45°C
    Maximum temperature: 183.12°C
    Minimum temperature: 166.54°C
    

**Explanation** : In this example, we plot time-series data of reactor temperature. By displaying the target temperature (175°C) and acceptable range (173-177°C), we can visually confirm the process stability.

### Code Example 2: Simultaneous Plotting of Multiple Sensors

Display distillation column temperature, pressure, and flow rate simultaneously.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    
    """
    Example: Display distillation column temperature, pressure, and flow 
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    time_points = pd.date_range('2025-01-01 00:00', periods=1440, freq='1min')
    
    df = pd.DataFrame({
        'timestamp': time_points,
        'temperature': 85 + np.random.normal(0, 1.5, 1440),
        'pressure': 1.2 + np.random.normal(0, 0.05, 1440),
        'flow_rate': 50 + np.random.normal(0, 3, 1440)
    })
    
    # Display in 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Temperature
    axes[0].plot(df['timestamp'], df['temperature'], color='#11998e', linewidth=0.8)
    axes[0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0].set_title('Distillation Column - Multi-Sensor Data', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Pressure
    axes[1].plot(df['timestamp'], df['pressure'], color='#f59e0b', linewidth=0.8)
    axes[1].set_ylabel('Pressure (MPa)', fontsize=11)
    axes[1].grid(alpha=0.3)
    
    # Flow rate
    axes[2].plot(df['timestamp'], df['flow_rate'], color='#7b2cbf', linewidth=0.8)
    axes[2].set_ylabel('Flow Rate (m³/h)', fontsize=11)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation between variables
    print("Correlation coefficients between variables:")
    print(df[['temperature', 'pressure', 'flow_rate']].corr())
    

**Explanation** : By displaying multiple sensor data on the same time axis, you can visually grasp relationships between variables and anomaly patterns. Correlation coefficients also confirm quantitative relationships between variables.

### Code Example 3: Correlation Matrix Heatmap

Grasp correlation relationships among multiple variables at a glance.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Grasp correlation relationships among multiple variables at 
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate sample data: 8 variables in distillation column
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'Feed_Temp': np.random.normal(60, 5, n),
        'Reflux_Ratio': np.random.uniform(1.5, 3.5, n),
        'Reboiler_Duty': np.random.normal(1500, 200, n),
        'Top_Temp': np.random.normal(85, 3, n),
        'Bottom_Temp': np.random.normal(155, 5, n),
        'Pressure': np.random.normal(1.2, 0.1, n),
        'Purity': np.random.uniform(95, 99.5, n),
        'Yield': np.random.uniform(85, 95, n)
    })
    
    # Adjust data based on correlations (create realistic correlations)
    df['Top_Temp'] = df['Top_Temp'] + 0.3 * df['Reflux_Ratio']
    df['Purity'] = df['Purity'] + 0.5 * df['Reflux_Ratio'] - 0.2 * df['Top_Temp']
    df['Yield'] = df['Yield'] + 0.3 * df['Reboiler_Duty'] / 100
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Distillation Column Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Display strongly correlated pairs
    print("\nStrongly correlated variable pairs (|r| > 0.5):")
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.5:
                print(f"{corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i, j]:.3f}")
    

**Explanation** : The correlation matrix heatmap allows you to visually grasp which variables are strongly related to each other. This is useful for preprocessing and feature selection in modeling.

### Code Example 4: Scatter Plot Matrix (Pair Plot)

Observe relationships between variables in detail with scatter plots.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - seaborn>=0.12.0
    
    """
    Example: Observe relationships between variables in detail with scatt
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n = 300
    
    df = pd.DataFrame({
        'Temperature': np.random.normal(175, 5, n),
        'Pressure': np.random.normal(1.5, 0.2, n),
        'Flow_Rate': np.random.normal(50, 5, n),
        'Yield': np.random.uniform(80, 95, n)
    })
    
    # Add relationships (realistic correlations)
    df['Yield'] = df['Yield'] + 0.5 * (df['Temperature'] - 175) + 2 * (df['Pressure'] - 1.5)
    
    # Create pair plot
    sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.6, 'color': '#11998e'},
                 diag_kws={'color': '#11998e'})
    plt.suptitle('Pairplot - Process Variables vs Yield', y=1.01, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    

**Explanation** : The pair plot allows you to check scatter plots of all variable pairs and the distribution of each variable at once. It's effective for detecting nonlinear relationships and outliers.

### Code Example 5: Interactive Visualization (Plotly)

Create interactive graphs with zoom and hover display capabilities in a web browser.
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pandas>=2.0.0, <2.2.0
    # - plotly>=5.14.0
    
    """
    Example: Create interactive graphs with zoom and hover display capabi
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    time_points = pd.date_range('2025-01-01', periods=1440, freq='1min')
    
    df = pd.DataFrame({
        'timestamp': time_points,
        'temperature': 175 + np.random.normal(0, 2, 1440),
        'pressure': 1.5 + np.random.normal(0, 0.1, 1440),
        'yield': 90 + np.random.normal(0, 3, 1440)
    })
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Reactor Temperature', 'Reactor Pressure', 'Yield'),
        vertical_spacing=0.12
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['temperature'],
                   mode='lines', name='Temperature',
                   line=dict(color='#11998e', width=1.5)),
        row=1, col=1
    )
    
    # Pressure
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['pressure'],
                   mode='lines', name='Pressure',
                   line=dict(color='#f59e0b', width=1.5)),
        row=2, col=1
    )
    
    # Yield
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['yield'],
                   mode='lines', name='Yield',
                   line=dict(color='#7b2cbf', width=1.5)),
        row=3, col=1
    )
    
    # Layout settings
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (MPa)", row=2, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=3, col=1)
    
    fig.update_layout(
        title_text="Interactive Process Data Visualization",
        height=900,
        showlegend=False
    )
    
    # Display graph (for Jupyter Notebook)
    fig.show()
    
    # Save as HTML file
    fig.write_html("process_data_interactive.html")
    print("Interactive graph saved to 'process_data_interactive.html'.")
    

**Explanation** : Using Plotly, you can create graphs with interactive features like zoom, pan, and hover display. This is very convenient when exploring large amounts of data.

* * *

## 1.6 Chapter Summary

### What We Learned

**1\. Definition of Process Informatics (PI)** — PI enables process understanding, optimization, and control through data-driven approaches. While MI focuses on "what to make," PI focuses on "how to make it."

**2\. Characteristics of process industries** — Processes are classified as continuous or batch, with inherent complexity arising from multivariate interactions, nonlinear relationships, and time delays.

**3\. Three main categories of process data** — Sensor data (temperature, pressure, flow rate at high frequency), operation condition data (setpoints and control parameters), and quality data (purity, yield at low frequency via offline analysis).

**4\. Real examples of data-driven improvement** — A 5% yield improvement delivered ¥500M annual profit increase; 15% energy reduction achieved ¥150M annual cost savings. PI projects consistently demonstrate high ROI (hundreds to thousands of percent) with short payback periods (2-4 months).

**5\. Data visualization with Python** — We covered time-series plots, multi-variable displays, correlation matrices, pair plots, and interactive visualization with Plotly.

### Important Points

PI is a **practical technology with immediate effectiveness** , typically achieving ROI within months. Data-driven approaches are particularly effective given the inherent complexity of industrial processes. Data visualization serves as the foundational first step in any PI project, while soft sensors (real-time quality prediction) represent one of PI's core enabling technologies.

### To the Next Chapter

In Chapter 2, we will explore **process data preprocessing and visualization** in detail, covering time-series data handling (resampling and rolling statistics), practical methods for missing value handling and outlier detection, data scaling and normalization techniques, advanced visualization approaches, and strategies for addressing challenges specific to process data.
