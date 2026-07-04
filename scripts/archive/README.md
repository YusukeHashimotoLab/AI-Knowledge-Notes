# Archived one-off scripts

This directory holds scripts and session reports that served a single
migration, translation batch, or fix and are not expected to run again
(most contain hardcoded series paths, absolute local paths, or inline
translation dictionaries). They are kept for reference only.

Reusable tools remain in `wp/scripts/`:

- `check_links.py` / `analyze_broken_links.py` / `fix_broken_links.py` (+ `test_fix_broken_links.py`) — link checking (used by CI)
- `qa_check.py` — content QA
- `translate_html.py` / `translate_series.py` — Claude-API translation pipeline
- `add_locale_switcher.py` / `add_bidirectional_locale_links.py` — locale switcher injection
- `generate_translation_status.py` — EN/JP parity status
- `add_meta_description.py`, `fix_asset_paths.py` — generic batch fixers
