# Repository Guidelines

## Project Structure & Module Organization
English content lives in `en/`, Japanese in `jp/`. Each locale root keeps `index.md`/`index.html` landing pages, `search.html`, `start-here.html`, and five dojo folders (`FM`, `MI`, `ML`, `MS`, `PI`). Series directories follow kebab-case (`en/FM/calculus-vector-analysis/`) and contain `index.html` plus chapter files, with the Markdown source next to the generated HTML where one exists. Two chapter-naming patterns exist — `chapter-1.html` (most series) and `chapter1-topic-name.html` (e.g. `en/ML/transformer-introduction/`); match the series you are editing, since navigation is generated from the filenames. Each locale has its own `assets/css/` (`knowledge-base.css`, `dojo.css`), so pages link `../../assets/css/knowledge-base.css` and resolve inside their own locale — keep those relative paths. Images are centralized in `en/assets/images/` and referenced cross-locale from `jp/` (`../en/assets/images/...`). `en/assets/search-index.json` and `jp/assets/search-index.json` are generated — never edit them by hand. Update the per-dojo `TRANSLATION_STATUS.md` files whenever chapters move, counts change, or a locale falls behind.

## Build, Test, and Development Commands
Preview servers run from this directory (`wp/knowledge`); every QA script runs from `wp/` — see `../CLAUDE.md` for the full list and `../requirements.txt` for the one dependency list.

- `python3 -m http.server 4000 --directory en` previews the English tree with the production-relative paths.
- `python3 -m http.server 4100 --directory jp` previews Japanese pages; run both servers to validate locale switchers.
- From `wp/`: `python3 scripts/check_links.py --path knowledge/en --output /tmp/lc_en.txt` (and `knowledge/jp`) must report `Broken links: 0` and `Missing anchors: 0`; `python3 tools/validate_mermaid.py knowledge/jp` checks diagram syntax; `python3 scripts/update_index_stats.py --check`, `build_search_index.py --check` and `build_sitemap.py --check` catch stale generated data.
- HTML validation mirrors CI only with the workflow's rule set — copy the `.htmlvalidate.json` block out of `wp/.github/workflows/html-validate.yml` and run the pinned version from `wp/`: `npx html-validate@8.8.0 -c .htmlvalidate.json knowledge/en/FM/calculus-vector-analysis/index.html`. Without `-c` you will see rules CI disables (`void-style`, `no-inline-style`, `long-title`).
- `npx markdownlint "**/*.md"` is available for ad-hoc front-matter/heading checks, but it has no config in this repo and CI does not run it, so its findings are advisory.

## Coding Style & Naming Conventions
YAML front matter mirrors `en/index.md`: double-quoted values, snake_case keys, and counts that match the visible cards. Keep prose concise (U.S. English in `en/`, formal Japanese in `jp/`) and reuse emoji headers only where they already exist. HTML uses two-space indentation, kebab-case filenames (`chapter-3.html`), and relative navigation (`../index.html` for dojo roots, `../../../jp/...` cross-locale). Place shared Mermaid initialization in the `<head>` block as shown in current files.

## Testing Guidelines
Run the checks above for every change (they are what CI runs), then click through breadcrumbs, cards, and locale toggles while the local server runs. If you touched anything under `wp/scripts/` or `wp/tools/`, also run the unit tests from `wp/`: `python3 -m unittest discover -s scripts -p 'test_*.py'`. Compare metadata blocks (chapter counts, runtimes, code examples) between locales and adjust `TRANSLATION_STATUS.md` whenever parity shifts. When adding Mermaid diagrams or media, confirm the asset path resolves in both locales before committing.

## Commit & Pull Request Guidelines
Use the Conventional Commits style already in history (`feat:`, `fix:`, `docs:`) and scope messages by dojo when helpful (`feat(ML): add diffusion lecture`). Everything under `wp/` must be committed twice — once in the private root monorepo and once in the public deploy repo that `wp/` itself is — with matching messages; the push from `wp/` is what deploys the site. See `../../DEPLOYMENT.md` for the procedure. Pull requests should summarize affected locales, list verification commands, and attach screenshots or localhost URLs for visual changes. Reference related issues or translation tickets and flag whether the companion locale update is included or deferred.

## Localization & Asset Tips
Stylesheets are per locale (`en/assets/css/`, `jp/assets/css/`): `dojo.css` is currently identical in both and `knowledge-base.css` is locale-specific, so a shared-CSS change has to be applied in both trees. Images are not duplicated — they live in `en/assets/images/` and `jp/` pages point at them cross-locale. Finalize the English Markdown or HTML first, then apply translations so metadata, sections, and scripts stay aligned. Document partial translations in `TRANSLATION_STATUS.md`, and leave inline TODO markers only when a placeholder preserves layout.
