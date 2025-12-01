# Repository Guidelines

## Project Structure & Module Organization
English content lives in `en/`, Japanese in `jp/`. Each locale root keeps `index.md`/`index.html` landing pages plus five dojo folders (`FM`, `MI`, `ML`, `MS`, `PI`). Series directories follow kebab-case (`en/FM/calculus-vector-analysis/`) and contain `index.html` plus `chapter-*.html`. Shared CSS/JS/media sit in `en/assets/`; keep the existing relative paths (for example `../../assets/css/knowledge-base.css`). Update the per-dojo `TRANSLATION_STATUS.md` files whenever chapters move, counts change, or a locale falls behind.

## Build, Test, and Development Commands
- `python -m http.server 4000 --directory en` previews the English tree with the production-relative paths.
- `python -m http.server 4100 --directory jp` previews Japanese pages; run both servers to validate locale switchers.
- `npx markdownlint "**/*.md"` validates front matter spacing and heading hierarchy on Markdown landing pages.
- `npx html-validate en/FM/calculus-vector-analysis/index.html` (swap in any touched file) catches malformed tags and missing attributes before publishing.

## Coding Style & Naming Conventions
YAML front matter mirrors `en/index.md`: double-quoted values, snake_case keys, and counts that match the visible cards. Keep prose concise (U.S. English in `en/`, formal Japanese in `jp/`) and reuse emoji headers only where they already exist. HTML uses two-space indentation, kebab-case filenames (`chapter-3.html`), and relative navigation (`../index.html` for dojo roots, `../../../jp/...` cross-locale). Place shared Mermaid initialization in the `<head>` block as shown in current files.

## Testing Guidelines
Run the Markdown and HTML checks above for every change, then click through breadcrumbs, cards, and locale toggles while the local server runs. Compare metadata blocks (chapter counts, runtimes, code examples) between locales and adjust `TRANSLATION_STATUS.md` whenever parity shifts. When adding Mermaid diagrams or media, confirm the asset path resolves in both locales before committing.

## Commit & Pull Request Guidelines
Use the Conventional Commits style already in history (`feat:`, `fix:`, `docs:`) and scope messages by dojo when helpful (`feat(ML): add diffusion lecture`). Pull requests should summarize affected locales, list verification commands, and attach screenshots or localhost URLs for visual changes. Reference related issues or translation tickets and flag whether the companion locale update is included or deferred.

## Localization & Asset Tips
Central assets (CSS, diagrams, shared JS) live in `en/assets/`; reuse them from Japanese pages via the same relative climb depth instead of duplicating. Finalize the English Markdown or HTML first, then apply translations so metadata, sections, and scripts stay aligned. Document partial translations in `TRANSLATION_STATUS.md`, and leave inline TODO markers only when a placeholder preserves layout.
