# Link Checker for AI Terakoya Knowledge Base

## Overview

Comprehensive link validation script for HTML files in `/knowledge/en/` directory. Checks internal links, anchors, and cross-references.

## Features

- ✅ Scans all HTML and Markdown files recursively
- ✅ Validates internal links and relative paths
- ✅ Checks anchor existence in target files
- ✅ Categorizes broken links by pattern
- ✅ Generates detailed reports with line numbers
- ✅ Provides fix suggestions
- ✅ Progress bar for large codebases

## Installation

Requires Python 3.7+ with the following packages:

```bash
python3 -m pip install -r requirements.txt   # from wp/ — the single dependency list
```

The script will auto-install dependencies if missing.

## Usage

### Basic Usage

Both defaults are derived from the script's own location, so a bare invocation works from any
cwd in any checkout: it scans `<wp>/knowledge/en` and writes `<wp>/linkcheck_en_local.txt`
(the `linkcheck*.txt` pattern is gitignored).

```bash
python3 scripts/check_links.py                                   # en, default report path
python3 scripts/check_links.py --path knowledge/en --output /tmp/lc_en.txt
python3 scripts/check_links.py --path knowledge/jp --output /tmp/lc_jp.txt
```

Note that `--output` has its own default: passing only `--path knowledge/jp` writes the Japanese
report to `<wp>/linkcheck_en_local.txt`, so pass `--output` too whenever you scan a non-`en`
tree.

A healthy run reports `Broken links: 0` and `Missing anchors: 0` — this is exactly what CI
checks (`.github/workflows/link-check.yml`).

### Auto-fix Mode

`--fix-auto` is accepted but does nothing: the handler prints "Auto-fix mode not yet
implemented." Use `scripts/fix_broken_links.py` for actual repairs (see
`README_fix_broken_links.md`):

```bash
python3 scripts/fix_broken_links.py --dry-run
```

## Output Format

The report includes:

1. **Summary Statistics**
   - Total files scanned
   - Total links checked
   - Broken links count
   - Missing anchors count

2. **Broken Links by Pattern**
   - Missing chapters (incomplete series)
   - Missing Dojo prefixes
   - Non-existent series references
   - Other issues

3. **Detailed Report**
   - File path (relative to base)
   - Line number
   - Broken URL
   - Reason for failure
   - Target path
   - Suggested fixes

4. **Recommendations**
   - Actionable steps to fix issues
   - Batch operation suggestions

## Sample Output

```
Link Checker Report - Generated: 2025-11-16 21:58:11
================================================================================

Summary:
- Total HTML files: 582
- Total MD files: 10
- Total links checked: 6745
- Broken links: 497
- Missing anchors: 9
- Warnings: 0

Broken Links by Pattern:
--------------------------------------------------------------------------------
1. Missing Chapters (208 instances)
   Example: chapter-5.html

2. Missing Index (38 instances)
   Example: ../../../index.html

3. Other (240 instances)
   Example: ../../../assets/css/variables.css
```

## Common Issues & Fixes

### 1. Missing Chapters

**Issue**: Links to chapter files that don't exist yet

**Example**:
```
File: FM/equilibrium-thermodynamics/index.html
Line 349: chapter-2.html
Status: BROKEN
Target: .../FM/equilibrium-thermodynamics/chapter-2.html
```

**Fix**: Either create the missing chapter files or remove links from navigation

### 2. Missing Index Files

**Issue**: Incorrect relative path to parent index

**Example**:
```
File: FM/index.html
Line 185: ../../index.html
Status: BROKEN
Target: .../knowledge/index.html
```

**Fix**: Correct path should be `../index.html` (one level up, not two)

### 3. Missing Dojo Prefix

**Issue**: Links missing the Dojo category prefix (FM, MI, ML, MS, PI)

**Example**:
```
File: index.md
Line 190: ./ml-introduction/index.html
Status: BROKEN
```

**Fix**: Add Dojo prefix: `./ML/ml-introduction/index.html`

### 4. Non-Existent Series

**Issue**: References to series that don't exist

**Example**:
```
File: index.md
Line 200: ./PI/pi-introduction/
Status: BROKEN
```

**Fix**: Remove reference or create the series

### 5. Missing Anchors

**Issue**: Anchor links to IDs that don't exist in target file

**Example**:
```
File: MI/experimental-data-analysis-introduction/index.html
Line 1016: #how-to-learn
Status: MISSING ANCHOR
```

**Fix**: Add the anchor ID to the target file or update the link

## Link Validation Logic

### Internal vs External Links

- **Internal**: Relative paths (`.`, `..`, `/`) - validated
- **External**: `http://`, `https://`, `mailto:` - skipped
- **Special**: `javascript:`, `data:`, `#` - skipped

### Path Resolution

1. Anchor-only links (`#section`) - validated in same file
2. Relative paths (`../series/chapter.html`) - resolved from source file
3. Absolute paths (`/AI-Knowledge-Notes/...`) - handled for GitHub Pages
4. Combined (`../chapter.html#section`) - validates both file and anchor

### Anchor Extraction

Finds anchors from:
- `<element id="anchor-name">` attributes
- `<a name="anchor-name">` tags (legacy)

## Performance

- ~582 HTML files processed in 3-4 seconds
- ~6,745 links validated
- Progress bar shows real-time status
- Memory efficient (processes files sequentially)

## Limitations

1. **External Links**: Not validated by default (would require HTTP requests)
2. **Dynamic Content**: JavaScript-generated links not detected
3. **Complex URLs**: URL parameters and fragments may need manual review

## Future Enhancements

- [ ] Auto-fix common patterns (Dojo prefix, path corrections)
- [ ] External link validation (optional)
- [ ] HTML validation
- [ ] Broken image detection
- [ ] JSON output format
- [ ] Integration with CI/CD

## Troubleshooting

### "Parser not found" error

```bash
python3 -m pip install -r requirements.txt
```

`check_links.py` parses with the stdlib `html.parser`; `beautifulsoup4` and `lxml` come from
`requirements.txt`. `html5lib` is not used by this repository.

### "Permission denied"

```bash
chmod +x scripts/check_links.py
```

### "Module not found"

```bash
# Run from wp/ (the site root inside your AI_Homepage checkout)
cd <repo>/wp
python3 scripts/check_links.py --path knowledge/en --output /tmp/lc_en.txt
```

## Files

- **Script**: `scripts/check_links.py`
- **Output**: whatever you pass to `--output` (CI writes `linkcheck_en_report.txt` /
  `linkcheck_jp_report.txt`; both patterns are gitignored)
- **Documentation**: `scripts/README_LINK_CHECKER.md`
- **Dependencies**: `wp/requirements.txt`

## Author

Dr. Yusuke Hashimoto, Tohoku University

## License

CC BY 4.0
