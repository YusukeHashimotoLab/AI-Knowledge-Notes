---
title: Bayesian Optimization・Active Learning Beginner Series v1.0
chapter_title: Bayesian Optimization・Active Learning Beginner Series v1.0
subtitle: Pioneering the future of next-generation materials development through efficient materials exploration
reading_time: 100-120 minutes
difficulty: Beginner - Intermediate
code_examples: 32
exercises: 12
---

## Series Overview

This series is an educational content with a 4-chapter structure that allows you to learn progressively, from beginners learning Bayesian Optimization and Active Learning for the first time to those who want to acquire practical materials exploration skills.

Bayesian Optimization is the most efficient exploration technique for discovering optimal materials and process conditions with a limited number of experiments. As an innovative technology that can now achieve in just dozens of experiments what traditionally required thousands of experiments in materials development using random search or exhaustive exploration, it is rapidly spreading across all materials fields including Li-ion batteries, catalysts, alloys, and organic semiconductors.

### Why This Series is Necessary

**Background and Challenges** : The greatest challenge in materials development is the vastness of the search space. For example, in composition optimization of 5-element alloys, there are more than 1010 candidates, making it physically impossible to try them all. With traditional methods relying on random search or empirical rules, development periods extend from years to decades, and costs become enormous.

**What You Can Learn in This Series** : In this series, you will systematically learn from theory to practice of Bayesian Optimization and Active Learning through executable Code Examples and materials science case studies. You can acquire immediately practical skills including efficient exploration methods of search space, proposing next experiments by exploiting uncertainty, handling constraints and multi-objective optimization, and automatic integration with experimental equipment.

## Content of All 4 Chapters

[ Chapter 1: Why Optimization is Essential for Materials Discovery 📖 20-30 minutes 💻 6 examples 📝 3 exercises 📊 Beginner Learn about the vastness of search space in materials exploration, limitations of traditional methods, and success cases of Bayesian Optimization. From the combinatorial explosion problem to achieving 95% reduction in experiment count, deepen your understanding through concrete case studies.  ](<chapter-1.html>) [ Chapter 2: Theory of Bayesian Optimization 📖 25-30 minutes 💻 10 examples 📝 3 exercises 📊 Beginner Learn the theoretical foundations of Bayesian Optimization including Gaussian Process regression, Acquisition Functions (EI, UCB, PI), and the exploration-exploitation trade-off. Master quantifying uncertainty and its exploitation methods through mathematical formulas and implementation code.  ](<chapter-2.html>) [ Chapter 3: Practice: Application to Materials Discovery 📖 25-30 minutes 💻 12 examples 📝 3 exercises 📊 Intermediate Learn practical Python implementations including data retrieval from Materials Project, ML model integration, constrained and multi-objective optimization, and batch Bayesian Optimization. Complete implementation examples for Li-ion battery cathode materials are also provided.  ](<chapter-3.html>) [ Chapter 4: Active Learning Strategies 📖 20-25 minutes 💻 8 examples 📝 3 exercises 📊 Intermediate Learn Active Learning strategies including uncertainty sampling, diversity sampling, and closed-loop optimization. Success cases of real-world autonomous experiment systems such as Berkeley A-Lab and RoboRXN are also introduced.  ](<chapter-4.html>)

## Target Audience

  * Undergraduate and graduate students in materials science (those who want to learn efficient experimental planning)
  * R&D engineers in companies (those who want to reduce development time and costs)
  * Data scientists (those aiming to apply to materials science)
  * Computational chemists (those interested in fusion of experiments and simulations)

## Prerequisite Knowledge

**Required** :

  * Python basics (variables, functions, lists, dictionaries, NumPy arrays)
  * Basic statistics (normal distribution, mean, variance, standard deviation)

**Recommended** :

  * Machine learning basics (overfitting, generalization performance, cross-validation)
  * Optimization theory (gradient descent, concept of local optima)

## Key Tools

  * **scikit-optimize** : Basic Bayesian Optimization
  * **BoTorch** : PyTorch-based advanced optimization
  * **Ax** : Meta-developed A/B testing compatible optimization
  * **GPyOpt** : Bayesian Optimization for academic research

## Learning Path

**For Beginners** (those with no knowledge of Bayesian Optimization):  
Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 (all chapters recommended)  
Time required: 100-120 minutes

**For Intermediate Learners** (those with experience in optimization theory):  
Chapter 2 → Chapter 3 → Chapter 4  
Time required: 70-90 minutes

**Practical Skill Enhancement** (implementation-focused over theory):  
Chapter 3 (intensive learning) → Chapter 4  
Time required: 50-70 minutes

## Next Steps

Recommended learning after series completion:

  * **Robotics Experiment Automation Beginner** : Implementing closed-loop optimization
  * **Reinforcement Learning Beginner** : Multi-step optimization
  * **GNN Beginner** : Graph representation of molecules and materials

## Let's Get Started!

Are you ready? Start with Chapter 1 and begin your journey to innovate materials exploration with Bayesian Optimization!

[Read Chapter 1 →](<chapter-1.html>)
