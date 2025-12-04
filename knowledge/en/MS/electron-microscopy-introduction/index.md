---
title: Electron Microscopy Introduction Series
chapter_title: Electron Microscopy Introduction Series
subtitle: From SEM/TEM Principles to Practical Analytical Techniques - Your First Step to Nanoscale Observation
difficulty: Beginner to Intermediate
code_examples: 35
version: 1.0
created_at: "by:"
---

## Series Overview

This series is an introductory course covering the fundamental principles to practical analytical techniques of electron microscopy (SEM/TEM), with a hands-on approach using Python. You will acquire the knowledge and skills necessary for nanoscale structural analysis of materials.

### Learning Flow
    
    
    ```mermaid
    flowchart LR
        A[Chapter 1Electron Microscopy Basics] --> B[Chapter 2SEM Introduction]
        B --> C[Chapter 3TEM Introduction]
        C --> D[Chapter 4STEM and Analytical Techniques]
        D --> E[Chapter 5Integrated Analysis Practice]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

## Series Structure

Chapter 1

Fundamentals of Electron Microscopy

Learn the basic principles of electron optics, comparison with optical microscopy, resolution theory, electron-matter interactions, and the types and characteristics of electron microscopes. 

⏱️ 30-35 min 💻 7 code examples 📊 Beginner

[Start Learning →](<chapter-1.html>)

Chapter 2

SEM Introduction

Learn SEM instrument configuration, differences between secondary electrons (SE) and backscattered electrons (BSE), elemental analysis by EDS, ZAF correction method, and practical observation and analytical techniques. 

⏱️ 30-35 min 💻 7 code examples 📊 Beginner to Intermediate

[Start Learning →](<chapter-2.html>)

Chapter 3

Transmission Electron Microscopy (TEM) Introduction

Learn TEM imaging theory, bright-field/dark-field imaging, selected area electron diffraction (SAED), lattice imaging/high-resolution TEM, and aberration correction techniques, mastering the basics of atomic-level analysis. 

⏱️ 25-35 min 💻 7 code examples 📊 Intermediate

[Start Learning →](<chapter-3.html>)

Chapter 4

STEM and Analytical Techniques

Learn STEM principles, Z-contrast imaging, electron energy loss spectroscopy (EELS), elemental mapping, atomic-resolution analysis, and the basics and applications of tomography. 

⏱️ 25-35 min 💻 7 code examples 📊 Intermediate to Advanced

[Start Learning →](<chapter-4.html>)

Chapter 5

EDS, EELS, and EBSD Integrated Analysis Practice

Practice integrated analysis workflows using Python, data processing with HyperSpy, machine learning classification, phase identification, crystallographic orientation analysis, and troubleshooting. 

⏱️ 30-40 min 💻 7 code examples 📊 Advanced

[Start Learning →](<chapter-5.html>)

## Learning Objectives

By completing this series, you will acquire the following skills and knowledge:

  * ✅ Understand the fundamentals of electron optics and resolution theory to optimize observation conditions
  * ✅ Explain the principles and applications of SEM, TEM, and STEM
  * ✅ Understand the physical origins of secondary electrons, backscattered electrons, diffraction, and EELS signals
  * ✅ Perform EDS quantitative analysis (ZAF correction) and correctly interpret results
  * ✅ Index electron diffraction patterns (SAED)
  * ✅ Perform FFT analysis and lattice spacing measurements of high-resolution TEM images
  * ✅ Process spectral data using HyperSpy
  * ✅ Perform phase classification and elemental mapping using machine learning
  * ✅ Perform EBSD orientation analysis and KAM/GND calculations
  * ✅ Elucidate microstructure-properties-process correlations from integrated data

## Recommended Learning Patterns

### Pattern 1: Standard Learning - Balance of Theory and Practice (5-7 days)

  * Day 1: Chapter 1 (Fundamentals of Electron Microscopy)
  * Day 2: Chapter 2 (SEM Introduction)
  * Day 3: Chapter 3 (TEM Introduction)
  * Day 4: Chapter 4 (STEM Techniques)
  * Day 5: Chapter 5 (Integrated Analysis Practice) + Comprehensive Review

### Pattern 2: Intensive Learning - Electron Microscopy Mastery (3 days)

  * Day 1: Chapters 1-2 (Basic Theory and SEM)
  * Day 2: Chapters 3-4 (TEM and STEM)
  * Day 3: Chapter 5 (Practical Analysis) + Exercise Problems from All Chapters

### Pattern 3: Practice-Focused - Data Analysis Skills Acquisition (1 day)

  * Chapters 1-4: Execute only code examples (refer to theory as needed)
  * Chapter 5: Study thoroughly and practice analysis with actual electron microscopy data
  * Return to theoretical sections as needed for clarification

## Prerequisites

Field | Required Level | Description  
---|---|---  
**Materials Science Basics** | Beginner level completed | Understanding of crystal structures, chemical bonding, and materials classification  
**Physics** | University 1-2 year level | Fundamentals of electromagnetism, wave optics, and quantum mechanics  
**Mathematics** | University 1st year level | Calculus, linear algebra, and basics of Fourier transforms  
**Python** | Intermediate | Basic operations with numpy, matplotlib, pandas, scikit-image, and HyperSpy  
  
## Python Libraries Used

Key libraries used in this series:

  * **numpy** : Numerical computation and array operations
  * **matplotlib** : 2D plotting and image display
  * **scipy** : Scientific computing (FFT, optimization, signal processing)
  * **pandas** : Data processing and analysis
  * **scikit-image** : Image processing
  * **HyperSpy** : Electron microscopy spectral data analysis (EDS, EELS)
  * **pyxem** : Electron diffraction data analysis
  * **orix** : EBSD crystallographic orientation analysis
  * **kikuchipy** : EBSD pattern analysis
  * **scikit-learn** : Machine learning (classification, clustering)

## FAQ - Frequently Asked Questions

### Q1: Is it okay if I have no hands-on experience with actual electron microscopes?

Yes, it's perfectly fine. This series focuses on theory, computation, and data analysis. While we don't cover actual instrument operation, you will gain deep understanding through data interpretation and simulation.

### Q2: What's the difference between SEM and TEM?

SEM (Scanning Electron Microscopy) scans the sample surface with a beam to observe surface morphology. TEM (Transmission Electron Microscopy) observes internal structures at atomic resolution using electrons transmitted through the sample. Chapters 2 and 3 explain these in detail.

### Q3: How does this relate to Materials Informatics (MI)?

Electron microscopy data is a treasure trove of materials microstructural information. The data processing and machine learning techniques learned in this series can be directly applied to materials database construction, microstructure-property correlation modeling, and automatic phase classification in MI.

### Q4: Is mastery of HyperSpy essential?

Chapter 5 focuses intensively on it, but you can learn with basic numpy and matplotlib knowledge. HyperSpy is widely used in the electron microscopy community, making it very useful in practical work.

### Q5: Can this be applied to biological samples?

This series focuses on materials science (metals, ceramics, semiconductors), but the basic principles are common to biological samples. However, sample preparation methods (fixation, staining, embedding) are significantly different.

## Key Learning Points

  * **Develop scale awareness** : Be conscious of the scales - SEM (μm to nm), TEM (nm to Å), STEM (atomic level)
  * **Understand physical origins of signals** : Always consider where SE, BSE, diffraction, and EELS come from
  * **Importance of quantification** : Develop the habit of treating data numerically rather than just "beautiful images"
  * **Execute code and change parameters** : Run all code examples and understand behavior by varying parameters
  * **Practice with real data** : In Chapter 5, practice analysis with your own research data or public datasets if possible

## Next Steps

After completing this series, we recommend the following advanced learning:

  * Advanced Electron Microscopy Techniques - Environmental TEM, in-situ observation, 4D-STEM
  * X-ray Analysis Methods Introduction - XRD, XRF, XPS
  * Atomic-Resolution Image Analysis - Image simulation, atomic arrangement determination
  * Materials Informatics Practice - Microstructure database construction and machine learning modeling
  * Process Informatics Practice - Electron microscopy data-driven process optimization
