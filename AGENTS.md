# Repository Guidelines

## Project Structure & Module Organization
The site is split into static-content domains. `endowed/` holds the Endowed Division landing pages, with localized copies in `en/` and `jp/` plus shared assets under `endowed/images/`. `private/` mirrors that structure for internal-facing pages; keep the `en/` and `jp/` trees aligned when updating copy. `knowledge/` is the knowledge base: Markdown sources sit beside their generated HTML so reviewers can see both the content (`*.md`) and the rendered output (`*.html`). Japanese learning series live under `knowledge/jp/<domain>/series/`, while English material resides in `knowledge/en/`. Run tooling from the repository root to preserve relative asset paths.

## Build, Test, and Development Commands
Use Python 3.9+ with `markdown` and `PyYAML` installed (`pip install markdown pyyaml`). Generate Japanese series pages with `python3 knowledge/jp/convert_md_to_html.py`; update the `BASE_PATH` constant and rerun if you clone the repo to a different location. Inject breadcrumbs after regeneration via `python3 knowledge/jp/add_breadcrumbs.py --dry-run` to preview and rerun without `--dry-run` to apply. Validate diagram markup before publishing with `python3 knowledge/jp/validate_mermaid.py knowledge/jp`, which reports unclosed or malformed Mermaid blocks.

## Coding Style & Naming Conventions
Markdown chapters start with YAML front matter bounded by `---` and use sentence-case headings. Name new chapters `chapter<n>-<topic>.md` (for example, `chapter3-hands-on.md`) so they align with existing navigation. Keep generated HTML in sync: indent with two spaces, favor semantic sections (`<header>`, `<section>`, `<footer>`), and mirror the language-specific directory structure. CSS lives inline; group related rules with block comments as in the existing templates.

## Testing Guidelines
There is no automated CI today, so treat script runs as required checks. After converting Markdown, open the corresponding HTML in a browser to confirm layout and localized strings. Run `validate_mermaid.py` whenever a chapter introduces or edits Mermaid diagrams, and inspect its warnings. For substantial styling updates, capture before/after screenshots from both `en` and `jp` variants to confirm parity.

## Commit & Pull Request Guidelines
Recent history mixes short statements and scoped commits (`feat(endowed): …`). Prefer the latter: `<type>(<area>): concise summary`, using `feat`, `fix`, or `content` for copy edits. Reference tracking issues where possible. Pull requests should include: a one-paragraph summary, the scripts you executed (with flags), and visual diffs for page-facing changes. Link to related knowledge-series tickets and note any follow-up translation tasks so reviewers can schedule updates across locales.
