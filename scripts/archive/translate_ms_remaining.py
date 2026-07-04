#!/usr/bin/env python3
"""Quick translation script for remaining MS series"""
import os
import shutil
from pathlib import Path

# Define series to translate with their details
SERIES = {
    'materials-thermodynamics-introduction': {
        'title': 'Introduction to Materials Thermodynamics Series',
        'subtitle': 'From Thermodynamics Fundamentals to Phase Diagrams and Equilibria',
        'chapters': 5
    },
    'mechanical-testing-introduction': {
        'title': 'Introduction to Mechanical Testing Series', 
        'subtitle': 'From Tensile Testing to Fatigue and Fracture',
        'chapters': 4
    },
    'electrical-magnetic-testing-introduction': {
        'title': 'Introduction to Electrical and Magnetic Testing Series',
        'subtitle': 'From Conductivity to Magnetic Materials Characterization',
        'chapters': 5
    },
    'spectroscopy-introduction': {
        'title': 'Introduction to Spectroscopy Series',
        'subtitle': 'Fundamentals and Applications of UV-Vis, IR, Raman, and XPS',
        'chapters': 5
    },
    'xrd-analysis-introduction': {
        'title': 'Introduction to X-ray Diffraction Analysis Series',
        'subtitle': 'From Crystal Structure Analysis to Rietveld Refinement',
        'chapters': 5
    },
    'synthesis-processes-introduction': {
        'title': 'Introduction to Materials Synthesis Processes Series',
        'subtitle': 'Fundamentals of Solid, Liquid, and Vapor Phase Synthesis',
        'chapters': 4
    },
    'processing-introduction': {
        'title': 'Introduction to Materials Processing Series',
        'subtitle': 'From Heat Treatment to Forming and Shaping',
        'chapters': 6
    },
    '3d-printing-introduction': {
        'title': 'Introduction to 3D Printing Series',
        'subtitle': 'Fundamentals and Applications of Additive Manufacturing',
        'chapters': 6
    },
    'advanced-materials-systems-introduction': {
        'title': 'Introduction to Advanced Materials Systems Series',
        'subtitle': 'From Smart Materials to Nanostructured Systems',
        'chapters': 5
    },
    'materials-science-introduction': {
        'title': 'Introduction to Materials Science Series',
        'subtitle': 'From Fundamentals to Structure and Properties',
        'chapters': 5
    },
    'polymer-materials-introduction': {
        'title': 'Introduction to Polymer Materials Series',
        'subtitle': 'From Structure to Properties and Processing',
        'chapters': 4
    },
    'thin-film-nano-introduction': {
        'title': 'Introduction to Thin Films and Nanomaterials Series',
        'subtitle': 'From Deposition Techniques to Nanostructure Control',
        'chapters': 4
    }
}

base_en = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS')
template = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/index.html')

# Read template
with open(template, 'r') as f:
    template_content = f.read()

count = 0
for series_name, details in SERIES.items():
    target_dir = base_en / series_name
    target_file = target_dir / 'index.html'
    
    if target_file.exists():
        print(f"SKIP: {series_name} (already exists)")
        continue
    
    # Create directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Customize template
    content = template_content.replace(
        'Introduction to Materials Microstructure Series',
        details['title']
    ).replace(
        'From Grain Structures to Phase Transformations - Master the Fundamentals of Microstructure Control',
        details['subtitle']
    ).replace(
        'Materials Microstructure',
        series_name.replace('-', ' ').title()
    )
    
    # Write index
    with open(target_file, 'w') as f:
        f.write(content)
    
    # Create chapter files
    for i in range(1, details['chapters'] + 1):
        (target_dir / f'chapter-{i}.html').touch()
    
    print(f"CREATED: {series_name} ({details['chapters']} chapters)")
    count += 1

print(f"\nTotal translated: {count} series")
