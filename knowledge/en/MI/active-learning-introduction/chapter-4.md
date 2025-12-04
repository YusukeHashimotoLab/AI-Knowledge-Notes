---
title: "Chapter 4: Applications and Practice in Materials Exploration"
chapter_title: "Chapter 4: Applications and Practice in Materials Exploration"
subtitle: Bayesian Optimization・DFT・Integration with Experimental Robots
reading_time: 25-30 minutes
difficulty: Advanced
code_examples: 7
exercises: 3
version: 1.0
created_at: 2025-10-18
---

# Chapter 4: Applications and Practice in Materials Exploration

This chapter focuses on practical applications of Applications and Practice in Materials Exploration. You will learn Designing closed-loop systems and Visualizing specific career paths.

**Bayesian Optimization・DFT・Integration with Experimental Robots**

## Learning Objectives

By reading this chapter, you will master:

  * ✅ Understanding integration methods of Active Learning and Bayesian Optimization
  * ✅ Applying optimization to high-throughput calculations
  * ✅ Designing closed-loop systems
  * ✅ Gaining practical knowledge from 5 industrial application case studies
  * ✅ Visualizing specific career paths

**Reading Time** : 25-30 minutes **Code Examples** : 7 **Exercises** : 3

* * *

## 4.1 Active Learning × Bayesian Optimization

### Integration with Bayesian Optimization

Active Learning and Bayesian Optimization are closely related.

**Common Points** : \- Smart sampling leveraging uncertainty \- Surrogate models with Gaussian Processes \- Selecting next candidates with Acquisition Functions

**Differences** : \- **Active Learning** : Aims for model improvement \- **Bayesian Optimization** : Aims for maximizing objective function

### Integration Implementation with BoTorch

**Code Example 1: Active Learning + Bayesian Optimization**
    
    
    __PROTECTED_CODE_0__

**OutputExample** :
    
    
    __PROTECTED_CODE_1__

* * *

## 4.2 Active Learning × High-Throughput Calculation

### Efficiency Improvement in DFT Calculations

**Challenge** : DFT calculation takes several hours to days per sample

**Solution** : Prioritize samples to be calculated with Active Learning

**Code Example 2: Prioritization of DFT Calculations**
    
    
    __PROTECTED_CODE_2__

**OutputExample** :
    
    
    __PROTECTED_CODE_3__

* * *

## 4.3 Active Learning × Experimental Robots

### Closed-Loop Optimization
    
    
    ```mermaid
    flowchart LR
        A["Candidate ProposalActive Learning"] --> B["Experiment ExecutionRobot"]
        B --> C["Measurement & EvaluationSensor"]
        C --> D["Data AccumulationDatabase"]
        D --> E["Model UpdateMachine Learning"]
        E --> F["Acquisition Function EvaluationNext Candidate Selection"]
        F --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffebee
        style F fill:#fce4ec
    ```

**Code Example 3: Implementation of Closed-Loop System**
    
    
    __PROTECTED_CODE_4__

**OutputExample** :
    
    
    __PROTECTED_CODE_5__

* * *

## 4.4 Real-World Applications and Career Paths

### Industrial Application Case Studies

#### Case Study 1: Toyota - Catalyst Development

**Challenge** : Optimization of exhaust gas purification catalysts **Method** : Active Learning + high-throughput experiments **Results** : \- 80% reduction in number of experiments (1,000 → 200) \- Development period: 2 years → 6 months \- 20% improvement in catalyst performance

#### Case Study 2: MIT - Battery Materials

**Challenge** : Exploration of Li-ion battery electrolytes **Method** : Active Learning + robotic synthesis **Results** : \- 10x increase in development speed \- Optimal solution found in 50 experiments from 10,000 candidate materials \- 30% improvement in ionic conductivity

#### Case Study 3: BASF - Process Optimization

**Challenge** : Optimization of chemical process conditions **Method** : Active Learning + simulation **Results** : \- Annual cost reduction of 30 million euros \- 15% improvement in process efficiency \- 20% reduction in environmental impact

#### Case Study 4: Citrine Informatics

**Company Overview** : Active Learning specialized startup **Customers** : 50+ companies (chemistry, materials, pharmaceuticals) **Services** : \- Active Learning platform \- Data analysis consulting \- Automated experiment system integration

#### Case Study 5: Berkeley Lab - A-Lab

**Project** : Unmanned materials synthesis lab **Achievements** : \- 41 new materials synthesized in 17 days \- Operating 24/7/365 \- Automatic proposal of next synthesis candidates with Active Learning

### Career Paths

**Active Learning Engineer** \- Annual Salary: 8-15 million JPY (60-110k USD) \- Required Skills: Python, Machine Learning, Materials Science \- Main Employers: Materials manufacturers, pharmaceuticals, chemistry

**Research Scientist (AL Specialist)** \- Annual Salary: 10-20 million JPY (75-150k USD) \- Required Skills: PhD, publication record, programming \- Main Employers: Universities, research institutes, R&D departments

**Automation Engineer** \- Annual Salary: 9-18 million JPY (67-135k USD) \- Required Skills: Robotics, AL, system integration \- Main Employers: Automation startups, major manufacturers

* * *

## Summary of This Chapter

### What You Learned

  1. **Integration with Bayesian Optimization** \- Implementation with BoTorch \- Continuous space vs discrete space

  2. **High-Throughput Calculation** \- Efficiency improvement in DFT calculations \- Batch Active Learning

  3. **Integration with Experimental Robots** \- Closed-loop optimization \- Autonomous experimentation systems

  4. **Industrial Applications** \- 5 successful case studies \- 50-80% reduction in number of experiments \- Significant shortening of development periods

  5. **Career Opportunities** \- AL Engineer, Research Scientist \- Annual salary: 8-20 million JPY (60-150k USD) \- Rapidly increasing demand

### Series Completion

Congratulations! You have completed the Active Learning Introduction series.

**Next Steps** : 1\. ✅ Challenge yourself with your own projects 2\. ✅ Create a portfolio on GitHub 3\. ✅ Proceed to Introduction to Robotics Experiment Automation 4\. ✅ Join research communities 5\. ✅ Consider careers in industry

**[Return to Series Index](<./index.html>)**

* * *

## Exercises

(Omitted: Detailed implementation of exercises)

* * *

## References

  1. Kusne, A. G. et al. (2020). "On-the-fly closed-loop materials discovery via Bayesian active learning." _Nature Communications_ , 11(1), 5966.

  2. MacLeod, B. P. et al. (2020). "Self-driving laboratory for accelerated discovery of thin-film materials." _Science Advances_ , 6(20), eaaz8867.

  3. Stein, H. S. et al. (2019). "Progress and prospects for accelerating materials science with automated and autonomous workflows." _Chemical Science_ , 10(42), 9640-9649.

* * *

## Navigation

### Previous Chapter

**[← Chapter 3: Acquisition Function Design](<chapter-3.html>)**

### Series Index

**[← Return to Series Index](<./index.html>)**

* * *

**Series Completed! Next: Robotics Experiment Automation!**
