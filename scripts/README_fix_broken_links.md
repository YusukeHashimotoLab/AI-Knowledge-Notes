# Broken Link Fixer Script

## Overview

`fix_broken_links.py` is a comprehensive Python script that automatically fixes broken links in HTML files based on common patterns. It uses BeautifulSoup4 for accurate HTML parsing and provides robust backup/restore functionality.

## Features

- **Pattern-based fixing**: Automatically detects and fixes 5 common link patterns
- **Dry-run mode**: Preview changes before applying them
- **Automatic backups**: Creates `.bak` files before modification
- **Restore capability**: Undo all changes with `--restore`
- **Detailed reporting**: Generates comprehensive reports of all fixes
- **Progress tracking**: Shows progress bar during processing
- **Safe operation**: Preserves HTML structure and formatting

## Requirements

```bash
pip install beautifulsoup4 lxml tqdm
```

## Usage

### Basic Usage

```bash
# Preview fixes (dry-run)
python3 scripts/fix_broken_links.py --dry-run

# Apply fixes
python3 scripts/fix_broken_links.py

# Apply fixes with verbose logging
python3 scripts/fix_broken_links.py --verbose

# Specify custom base directory
python3 scripts/fix_broken_links.py --base-dir /path/to/project

# Custom output report location
python3 scripts/fix_broken_links.py --output my_report.txt
```

### Undo/Restore

```bash
# Restore all files from backups (undo changes)
python3 scripts/fix_broken_links.py --restore

# Clean all backup files
python3 scripts/fix_broken_links.py --clean-backups
```

## Fix Patterns

### Pattern 1a: Absolute `/knowledge/en/` paths → Relative paths

**Problem**: Files use absolute paths like `/knowledge/en/MI/gnn-introduction/`

**Solution**: Convert to relative paths based on file depth

**Examples**:
```html
<!-- File: knowledge/en/MI/gnn-introduction/chapter-1.html -->
Before: <a href="/knowledge/en/">Knowledge Base</a>
After:  <a href="../../">Knowledge Base</a>

Before: <a href="/knowledge/en/MI/">Materials Informatics</a>
After:  <a href="../../MI/">Materials Informatics</a>

Before: <a href="/knowledge/en/MI/gnn-introduction/">GNN Introduction</a>
After:  <a href="../../MI/gnn-introduction/">GNN Introduction</a>
```

### Pattern 1b: Absolute `/en/` paths → Relative paths

**Problem**: Files use site-root absolute paths like `/en/`

**Solution**: Convert to relative paths accounting for 2-level depth from site root

**Examples**:
```html
<!-- File: knowledge/en/MI/gnn-introduction/chapter-1.html -->
Before: <a href="/en/">Home</a>
After:  <a href="../../../en/">Home</a>

Before: <a href="/en/#research">Research</a>
After:  <a href="../../../en/#research">Research</a>
```

### Pattern 2: Path depth issues

**Problem**: Incorrect `../` depth in relative paths, especially in breadcrumbs

**Solution**: Fix path depth based on file location

**Examples**:
```html
<!-- File: knowledge/en/MI/gnn-introduction/chapter-1.html (depth=2) -->
Before: <a href="../../../index.html">Home</a>
After:  <a href="../../index.html">Home</a>

<!-- File: knowledge/en/MI/gnn-introduction/index.html (depth=2) -->
Before: <a href="../../index.html">Knowledge Base</a>
After:  <a href="../index.html">Knowledge Base</a>

<!-- File: knowledge/en/MI/index.html (depth=1) -->
Before: <a href="../../index.html">Knowledge Base</a>
After:  <a href="./index.html">Knowledge Base</a>

<!-- File: knowledge/en/FM/index.html (depth=1) -->
Before: <a href="../../FM/index.html">FM</a>
After:  <a href="../FM/index.html">FM</a>
```

### Pattern 3: Asset paths `/assets/` → Relative paths

**Problem**: Assets use absolute paths

**Solution**: Convert to relative paths based on file depth

**Examples**:
```html
<!-- File: knowledge/en/MI/gnn-introduction/chapter-1.html (depth=2) -->
Before: <link href="/assets/css/variables.css">
After:  <link href="../../assets/css/variables.css">

Before: <script src="/assets/js/main.js">
After:  <script src="../../assets/js/main.js">

<!-- File: knowledge/en/MI/index.html (depth=1) -->
Before: <link href="/assets/css/base.css">
After:  <link href="../assets/css/base.css">
```

### Pattern 4: Non-existent series links

**Problem**: Links reference series that don't exist

**Solution**: Either fix the path (for cross-dojo references) or comment out

**Examples**:
```html
<!-- Cross-dojo reference fix -->
Before: <a href="../pi-introduction/">PI Introduction</a>
After:  <a href="../../PI/pi-introduction/">PI Introduction</a>

<!-- Non-existent series (commented out) -->
Before: <a href="../llm-basics/">LLM Basics</a>
After:  <!-- TODO: Add when series exists - <a href="../llm-basics/">LLM Basics</a> -->
```

**Known missing series**:
- `llm-basics`
- `machine-learning-basics`
- `robotic-lab-automation-introduction`
- `inferential-bayesian-statistics`

### Pattern 5: Wrong filename references

**Problem**: Link references wrong filename (typo or rename)

**Solution**: Find and suggest correct filename based on similarity

**Examples**:
```html
Before: <a href="chapter4-deep-learning-interpretation.html">
After:  <a href="chapter4-deep-learning-interpretability.html">

Before: <a href="chapter2-q-learning.html">
After:  <a href="chapter2-q-learning-sarsa.html">

Before: <a href="chapter5-transfer-learning.html">
After:  <a href="chapter3-transfer-learning.html">
```

## File Structure Context

The script understands the following file structure:

```
wp/
├── knowledge/
│   └── en/
│       ├── index.html           (depth=0, type=knowledge_index)
│       ├── assets/              (asset directory)
│       ├── FM/                  (Dojo: Foundational Mathematics)
│       │   ├── index.html       (depth=1, type=dojo_index)
│       │   └── calculus-vector-analysis/
│       │       ├── index.html   (depth=2, type=series_index)
│       │       ├── chapter-1.html (depth=2, type=chapter)
│       │       └── chapter-2.html (depth=2, type=chapter)
│       ├── MI/                  (Dojo: Materials Informatics)
│       ├── ML/                  (Dojo: Machine Learning)
│       ├── MS/                  (Dojo: Materials Science)
│       ├── PI/                  (Dojo: Process Informatics)
│       └── NM/                  (Dojo: Nanomaterials)
└── scripts/
    └── fix_broken_links.py
```

## Report Format

The script generates a detailed report at `link_fix_report.txt` (or custom location):

```
Link Fix Report - Generated: 2025-11-16 22:06:49
================================================================================

Statistics:
- Files Processed: 582
- Files Modified: 408
- Total Fixes Applied: 499

Fixes by Pattern:
  absolute_knowledge_path: 5
  absolute_site_path: 39
  asset_path: 35
  breadcrumb_depth: 418
  wrong_filename: 2

Detailed Fixes:
--------------------------------------------------------------------------------

File: MI/gnn-introduction/chapter-1.html
  Line 8: [asset_path]
    /assets/css/variables.css → ../../assets/css/variables.css
  Line 57: [absolute_knowledge_path]
    /knowledge/en/ → ../../
```

## Safety Features

1. **Automatic Backups**: Every modified file gets a `.bak` backup
2. **Dry-run Mode**: Preview all changes before applying
3. **Restore Capability**: Undo all changes with `--restore`
4. **HTML Preservation**: Uses BeautifulSoup for accurate parsing
5. **Error Logging**: All errors are logged for review

## Workflow

### Recommended Workflow

```bash
# 1. Run dry-run to see what will be fixed
python3 scripts/fix_broken_links.py --dry-run --verbose > preview.txt

# 2. Review the report
cat link_fix_report.txt

# 3. Apply fixes
python3 scripts/fix_broken_links.py

# 4. Test the site
# (Manual testing here)

# 5. If issues found, restore
python3 scripts/fix_broken_links.py --restore

# 6. Once confirmed working, clean backups
python3 scripts/fix_broken_links.py --clean-backups
```

### Integration with Git

```bash
# Before fixing: commit current state
git add .
git commit -m "Before link fixes"

# Run fixer
python3 scripts/fix_broken_links.py

# Review changes
git diff

# Commit fixes
git add .
git commit -m "Fix broken links with automated script"

# Or rollback if needed
git reset --hard HEAD
```

## Advanced Usage

### Filter by Pattern Type

You can modify the script to only apply certain pattern types by commenting out unwanted patterns in the `process_html_file` method.

### Custom Similarity Threshold

Adjust the `threshold` parameter in `_files_similar` method to change filename matching sensitivity (default: 0.8):

```python
def _files_similar(self, name1: str, name2: str, threshold: float = 0.9):
    # More strict matching
```

### Add Custom Patterns

Add new fix patterns by:

1. Creating a new fix method (e.g., `fix_custom_pattern`)
2. Adding pattern detection in `process_html_file`
3. Registering the pattern type in statistics

## Performance

- **Speed**: ~2-3 seconds for 582 HTML files
- **Memory**: Minimal (processes one file at a time)
- **Backup Size**: ~1:1 ratio with original files

## Troubleshooting

### Issue: "Required packages not installed"

**Solution**: Install dependencies
```bash
pip install beautifulsoup4 lxml tqdm
```

### Issue: "Knowledge directory not found"

**Solution**: Ensure you're running from the correct directory or use `--base-dir`
```bash
python3 scripts/fix_broken_links.py --base-dir /path/to/wp/
```

### Issue: Backup files accumulating

**Solution**: Clean backups after confirming fixes work
```bash
python3 scripts/fix_broken_links.py --clean-backups
```

### Issue: Too many/few fixes detected

**Solution**: Adjust similarity threshold or review patterns in code

## Limitations

1. **HTML Only**: Only processes `.html` files
2. **No JavaScript**: Doesn't fix links in JavaScript code
3. **Pattern-Based**: May miss edge cases not covered by patterns
4. **Similarity Matching**: Filename matching may have false positives/negatives

## Future Enhancements

Potential improvements:

- [ ] Add JSON output format for programmatic consumption
- [ ] Support for Markdown files
- [ ] Parallel processing for large file sets
- [ ] Custom pattern configuration via YAML/JSON
- [ ] Integration with link checker tools
- [ ] Automatic detection of new patterns

## Contributing

To add new fix patterns:

1. Add pattern detection method
2. Register in `process_html_file`
3. Update documentation
4. Test with dry-run
5. Submit changes

## License

Part of AI Homepage project.

## Author

Generated by Claude Code, 2025-11-16
