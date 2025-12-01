# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Terakoya is a bilingual (Japanese/English) educational knowledge base for materials science, machine learning, and related fields. The site contains lecture series organized by "Dojo" (discipline) with HTML chapters and Markdown sources.

## Directory Structure

```
wp/
├── knowledge/
│   ├── en/                    # English content
│   │   ├── FM/               # Foundational Mathematics
│   │   ├── MI/               # Materials Informatics
│   │   ├── ML/               # Machine Learning
│   │   ├── MS/               # Materials Science
│   │   ├── PI/               # Process Informatics
│   │   └── assets/           # Shared CSS, JS, media
│   └── jp/                    # Japanese content (same structure)
├── scripts/                   # Content manipulation scripts
├── tools/                     # Markdown/HTML pipeline tools
├── endowed/                   # Endowed chair pages
└── 執筆ガイドライン.md         # Writing guidelines (Japanese)
```

Series directories use kebab-case (e.g., `transformer-introduction/`) and contain `index.html` plus `chapter-*.html` files.

## Development Commands

### Local Preview
```bash
# English site (port 4000)
python -m http.server 4000 --directory knowledge/en

# Japanese site (port 4100)
python -m http.server 4100 --directory knowledge/jp
```

### Content Pipeline
```bash
# Convert Markdown to HTML
python tools/convert_md_to_html_en.py knowledge/en/ML/transformer-introduction/

# Convert HTML back to Markdown
python tools/html_to_md.py knowledge/en/ML/transformer-introduction/

# Bidirectional sync (auto-detects newer file)
python tools/sync_md_html.py knowledge/en/ML/transformer-introduction/

# Watch mode for live development
python tools/sync_md_html.py knowledge/en/ML/transformer-introduction/ --watch
```

### Validation
```bash
# Markdown lint
npx markdownlint "**/*.md"

# HTML validation
npx html-validate knowledge/en/FM/calculus-vector-analysis/index.html

# Link checking
python scripts/check_links.py knowledge/en/
python scripts/fix_broken_links.py knowledge/en/
```

### Dependencies
```bash
pip install markdown pyyaml beautifulsoup4 html2text watchdog
```

## Key Conventions

### File Naming
- Series: `series-name-introduction/`
- Chapters: `chapter-1.html`, `chapter-2.html` (consistent pattern per series)
- Index: `index.html` for series overview

### Asset Paths
- Shared assets in `knowledge/en/assets/`
- Reference from both locales via relative paths: `../../assets/css/knowledge-base.css`

### HTML Structure
- Two-space indentation
- MathJax for equations (inline `$...$`, display `$$...$$`)
- Mermaid for diagrams (initialized in `<head>`)

### YAML Frontmatter (for Markdown sources)
```yaml
---
title: "Chapter 1: Introduction"
chapter_title: "Chapter 1: Self-Attention"
subtitle: "Understanding the core architecture"
reading_time: "25-30 minutes"
difficulty: "Intermediate"
code_examples: 8
exercises: 5
---
```

## Git Workflow

- Conventional Commits: `feat:`, `fix:`, `docs:`
- Scope by dojo when helpful: `feat(ML): add diffusion lecture`
- Update `TRANSLATION_STATUS.md` when chapter counts change

## Writing Guidelines

See `執筆ガイドライン.md` for detailed Japanese writing conventions. Key points:
- All code examples must be 100% working
- Explain technical terms on first use: "機械学習（Machine Learning）"
- Standard 4-chapter structure: Introduction → Fundamentals → Hands-on → Real-world
