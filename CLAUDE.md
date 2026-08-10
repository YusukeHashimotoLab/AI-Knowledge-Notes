# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Terakoya is a bilingual (Japanese/English) educational knowledge base for materials science, machine learning, and related fields. The site contains lecture series organized by "Dojo" (discipline) with HTML chapters and Markdown sources.

This directory is also its own git repository (the public deploy repo). See the root repo's `DEPLOYMENT.md` for the dual-commit rule summarized under "Git Workflow" below.

## Directory Structure

```
wp/
├── knowledge/                 # AI Terakoya knowledge base
│   ├── en/                    # English content
│   │   ├── FM/               # Foundational Mathematics
│   │   ├── MI/               # Materials Informatics
│   │   ├── ML/               # Machine Learning
│   │   ├── MS/               # Materials Science
│   │   ├── PI/               # Process Informatics
│   │   └── assets/           # Shared CSS, JS, media, search-index.json (generated)
│   ├── jp/                    # Japanese content (same structure)
│   └── AGENTS.md              # Knowledge-base-specific conventions
├── scripts/                   # QA + generated-asset scripts, unit tests
│   └── archive/               # Retired one-off scripts — do not use or cite
├── tools/                     # Markdown/HTML pipeline + Mermaid validator
├── endowed/                   # Endowed chair pages (en/, jp/, images/)
├── profile/                   # Researcher's personal pages (en/, jp/, assets/)
├── private/                   # Redirect stubs -> profile/ (legacy URLs only)
├── claudedocs/                # Working notes and archived reports
├── index.html, 404.html       # Site root landing + error page
├── sitemap.xml, robots.txt    # sitemap.xml is generated
├── requirements.txt           # THE dependency list for this repo
└── 執筆ガイドライン.md         # Writing guidelines (Japanese)
```

Series directories use kebab-case (e.g., `transformer-introduction/`) and contain `index.html` plus chapter files (see File Naming below for the two chapter-name patterns in use). `profile/` was renamed from `private/` on 2026-08-10; the old tree now holds nothing but redirect stubs, so edit `profile/`.

Never hand-edit generated files: `sitemap.xml`, `knowledge/{en,jp}/assets/search-index.json`, and the statistics numbers on dojo/landing pages. Regenerate them with the scripts under "Validation".

## Development Commands

Run every command below from `wp/` (this directory).

### Local Preview
```bash
# English site (port 4000)
python3 -m http.server 4000 --directory knowledge/en

# Japanese site (port 4100)
python3 -m http.server 4100 --directory knowledge/jp
```

### Dependencies

One list for everything: `requirements.txt`. Reference interpreter is Python 3.11 (what CI uses).

```bash
python3 -m pip install -r requirements.txt
```

Many scripts are stdlib-only (see the header of `requirements.txt`). `npx` (Node 20) is only needed for `html-validate`.

### Content Pipeline

These tools write HTML/Markdown in place. Use `--dry-run` where offered and review `git diff` before committing.

```bash
# Convert Markdown to HTML. One positional target: a .md file, a series dir, or a
# bare dojo name; omit it for a whole locale. Locale comes from the path when the
# path contains knowledge/<en|jp>/, otherwise it defaults to en — so pass --lang.
python3 tools/convert_md_to_html.py knowledge/en/ML/transformer-introduction/
python3 tools/convert_md_to_html.py knowledge/jp/MI/gnn-introduction/chapter-1.md
python3 tools/convert_md_to_html.py ML --lang jp        # whole dojo, Japanese
python3 tools/convert_md_to_html.py --lang jp           # entire jp locale

# Convert HTML back to Markdown (requires html2text; writes .bak unless --no-backup)
python3 tools/html_to_md.py knowledge/en/ML/transformer-introduction/
python3 tools/html_to_md.py knowledge/en/ML/transformer-introduction/ --output-dir /tmp/md/

# Bidirectional sync (direction auto-detected from mtime — can overwrite hand-written
# Markdown, so always dry-run first)
python3 tools/sync_md_html.py knowledge/en/ML/transformer-introduction/ --dry-run
python3 tools/sync_md_html.py knowledge/en/ML/transformer-introduction/
python3 tools/sync_md_html.py knowledge/en/ML/transformer-introduction/ --force-direction md2html

# Watch mode for live development (requires watchdog)
python3 tools/sync_md_html.py knowledge/en/ML/transformer-introduction/ --watch
```

### Validation

Everything CI enforces, reproducible locally. All of these are read-only.

```bash
# Link + anchor check — require "Broken links: 0" and "Missing anchors: 0"
python3 scripts/check_links.py --path knowledge/en --output /tmp/lc_en.txt
python3 scripts/check_links.py --path knowledge/jp --output /tmp/lc_jp.txt

# Generated assets and content gates (stdlib only; exit 1 on drift)
python3 scripts/update_index_stats.py --check
python3 scripts/build_search_index.py --check
python3 scripts/build_sitemap.py --check
python3 scripts/check_translation_residue.py --check   # Japanese residue in EN pages
python3 scripts/check_stale_coming_soon.py --check     # 準備中 navs pointing at real chapters

# Mermaid diagram syntax (exit 1 on error; no argument = whole knowledge base)
python3 tools/validate_mermaid.py knowledge/jp
python3 tools/validate_mermaid.py knowledge/en

# Unit tests (converter, link fixer, Mermaid validator)
python3 -m unittest discover -s scripts -p 'test_*.py'
```

To fix drift, rerun the same script in write mode: `update_index_stats.py --write`,
`build_search_index.py`, `build_sitemap.py` (no flag = regenerate).

Link repairs are `scripts/fix_broken_links.py`, which takes flags rather than a path —
`--base-dir` defaults to the current directory, so run it from `wp/`:

```bash
python3 scripts/fix_broken_links.py --dry-run          # report only
python3 scripts/fix_broken_links.py                    # apply, writing .bak backups
python3 scripts/fix_broken_links.py --restore          # undo from backups
```

HTML validation needs the workflow's rule set; bare `html-validate:recommended` reports
rules CI turns off (`void-style`, `no-inline-style`, `long-title`). Copy the
`.htmlvalidate.json` block out of `.github/workflows/html-validate.yml`, then use the
pinned version (8.9.0+ adds `unique-landmark`, which flags 80+ existing landmarks):

```bash
npx html-validate@8.8.0 -c .htmlvalidate.json knowledge/en/FM/calculus-vector-analysis/index.html
```

Markdown linting is not part of CI and has no config in this repo; `npx markdownlint "**/*.md"`
is available ad hoc but its findings are not gating.

## Key Conventions

### File Naming
- Series: `series-name-introduction/`
- Chapters: `chapter-1.html`, `chapter-2.html` — or the older `chapter1-topic-name.html` style used by some series (e.g. `ML/transformer-introduction/`). Be consistent *within* a series; navigation links are generated from the filenames.
- Index: `index.html` for series overview

### Asset Paths
- CSS is per locale: `knowledge/en/assets/css/` and `knowledge/jp/assets/css/` both hold
  `knowledge-base.css` and `dojo.css`. Pages link `../../assets/css/knowledge-base.css`, which
  resolves inside their own locale — a shared CSS change must be applied to both trees.
- Images are centralized in `knowledge/en/assets/images/`; `jp/` pages reference them
  cross-locale (`../en/assets/images/...`) rather than duplicating them.
- `knowledge/{en,jp}/assets/search-index.json` is generated by `scripts/build_search_index.py`.

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
version: "1.0"          # optional
created_at: "2026-08-01" # optional
---
```

All keys are optional — the converter falls back to per-locale defaults for anything missing.

## Git Workflow

- Conventional Commits: `feat:`, `fix:`, `docs:`
- Scope by dojo when helpful: `feat(ML): add diffusion lecture`
- **Dual-commit rule:** this directory is the public deploy repo (`AI-Knowledge-Notes`), and the
  private root monorepo (`AI_Homepage`) tracks the same files. Every change here must be committed
  in both, with matching messages: once from the repository root (paths prefixed `wp/`) and once
  from `wp/`. Check `git rev-parse --show-toplevel` first — the current directory decides which
  repo you commit to. Pushing `main` from `wp/` is what redeploys the live site, so run the
  Validation commands above before pushing. Full procedure: `DEPLOYMENT.md` in the root repo.
- Regenerate and commit derived files alongside content changes (stats, search index, sitemap);
  CI fails on drift.
- `TRANSLATION_STATUS.md` per dojo is produced by `python3 scripts/generate_translation_status.py`
  (writes `knowledge/en/<dojo>/TRANSLATION_STATUS.md`); refresh it when chapter counts change.

## Writing Guidelines

See `執筆ガイドライン.md` for detailed Japanese writing conventions. Key points:
- All code examples must be 100% working
- Explain technical terms on first use: "機械学習（Machine Learning）"
- Standard 4-chapter structure: Introduction → Fundamentals → Hands-on → Real-world
