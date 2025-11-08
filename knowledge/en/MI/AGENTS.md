# Repository Guidelines

## Project Structure & Module Organization
Each topic in the MI knowledge base lives in its own directory (for example, `mi-introduction`, `battery-mi-application`, `gnn-introduction`). Inside each module you’ll find an `index.html` landing page plus numbered chapter files (`chapter-1.html`, `chapter-2.html`, etc.) that follow a consistent naming pattern to simplify cross-linking. Any helper utilities, such as the translation scripts in `bayesian-optimization-introduction/`, stay alongside the content they manage. The root-level `index.html` aggregates links to every series, so add new modules there whenever you introduce a topic.

## Build, Test, and Development Commands
Use lightweight tooling so contributors can preview changes quickly:

```bash
python3 -m http.server 4100
```
Serves the repository root locally; open `http://localhost:4100` to review navigation and styling.

```bash
tidy -qe path/to/file.html
```
Runs HTML linting (fails on structural issues; omit `-e` while drafting).

```bash
python3 bayesian-optimization-introduction/translate_chapter1.py
```
Example of regenerating derived content—keep similar scripts with their modules and document them in-line when you add new automation.

## Coding Style & Naming Conventions
Keep markup semantic (`<section>`, `<article>`, `<figure>`), indent nested blocks with two spaces, and wrap lines at ~100 characters for diffs that are easy to review. Images or downloadable assets should be stored beside the referencing article in a subfolder such as `images/` to avoid path confusion. Use sentence case headings and hyphenated filenames (`chapter3-hands-on.html`) to match the existing convention.

## Testing Guidelines
Before opening a PR, load the site via the local server, click every new/updated navigation link, and confirm math or code snippets render as intended. For link integrity, run `tidy -qe` on each modified file; if you have `linkchecker` installed, point it at the local server to catch broken anchors. Document any manual validation (e.g., “replayed chapter flow on Safari + Chrome”) inside the PR description.

## Commit & Pull Request Guidelines
Follow the conventional style used in `git log`: `<scope>: <summary>` (e.g., `feat: complete bayesian-optimization-introduction translation`). Keep commits scoped to a single module whenever possible to ease cherry-picking. PRs should describe the motivation, list affected modules, attach before/after screenshots for visual tweaks, and link tracking issues or localization tickets. Request review from another content maintainer when touching shared assets like `index.html`.

## Localization & Content Updates
When translating or localizing chapters, note the source language, tooling, and any terminology decisions in a short comment block at the top of the affected HTML file. Preserve glossary consistency across modules by reusing established phrasing (see `mi-introduction` for canonical definitions) and update related modules in the same PR when terminology changes ripple outward.
