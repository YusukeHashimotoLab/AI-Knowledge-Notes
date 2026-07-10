# AI Terakoya Locale Switcher Implementation

Production-ready implementation for adding bilingual language switchers to the AI Terakoya knowledge base.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [CSS Customization](#css-customization)
- [Sync Date Management](#sync-date-management)
- [Troubleshooting](#troubleshooting)
- [Integration Guide](#integration-guide)

## Overview

The locale switcher system provides seamless navigation between English and Japanese versions of knowledge base articles. It automatically detects corresponding files, displays sync status, and provides elegant UI/UX for language switching.

### Architecture

```
knowledge/
├── en/                          # English knowledge base
│   ├── ML/
│   │   └── transformer-introduction/
│   │       ├── index.html       # With locale switcher
│   │       └── chapter1.html    # With locale switcher
│   └── assets/
│       └── css/
│           └── knowledge-base.css  # With switcher styles
└── jp/                          # Japanese knowledge base
    └── ML/
        └── transformer-introduction/
            ├── index.html       # Corresponding JP file
            └── chapter1.html    # Corresponding JP file
```

### Visual Design

The locale switcher appears immediately after the breadcrumb navigation:

```
┌─────────────────────────────────────────────────┐
│ Breadcrumb: Home › ML › Transformer › Chapter 1 │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│ 🌐 EN | 日本語        Last sync: 2025-11-16      │
└─────────────────────────────────────────────────┘
```

## Features

### Core Features

- **Automatic Path Detection**: Intelligently finds corresponding Japanese files
- **Sync Date Tracking**: Uses git history or file mtime to track translation sync
- **Safe Operations**: Atomic writes with automatic backup creation
- **Smart Insertion**: Preserves HTML structure and formatting
- **Comprehensive Validation**: Pre and post-validation of HTML integrity
- **Progress Reporting**: Real-time progress with tqdm integration

### Safety Features

- **Backup Creation**: Automatic `.bak` files before modification
- **Atomic Writes**: Uses temporary files to prevent corruption
- **HTML Validation**: Validates structure before and after changes
- **Dry Run Mode**: Preview changes without modifying files
- **Force Mode**: Safely update existing switchers

### Design Features

- **Responsive Design**: Mobile-optimized layout
- **Accessibility**: High contrast mode support, proper ARIA labels
- **Print Friendly**: Hidden in print styles
- **Graceful Degradation**: Shows "準備中" when translation unavailable

## Installation

### Prerequisites

```bash
# Python 3.8 or higher required
python3 --version

# Install dependencies
pip install beautifulsoup4 tqdm
```

### Verification

```bash
# Verify installation
python3 scripts/add_locale_switcher.py --help
python3 scripts/update_css_locale.py --help
```

## Usage

### Quick Start

```bash
# Step 1: Update CSS (one-time operation)
cd /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp
python3 scripts/update_css_locale.py

# Step 2: Add switchers to all English files
python3 scripts/add_locale_switcher.py
```

### Advanced Usage

#### Dry Run (Preview Changes)

```bash
# Preview what would be changed
python3 scripts/add_locale_switcher.py --dry-run --verbose
```

#### Process Specific Directory

```bash
# Only process ML dojo
python3 scripts/add_locale_switcher.py knowledge/en/ML/

# Only process transformer series
python3 scripts/add_locale_switcher.py knowledge/en/ML/transformer-introduction/
```

#### Force Update Existing Switchers

```bash
# Update all switchers even if they exist
python3 scripts/add_locale_switcher.py --force
```

#### Custom Sync Date

```bash
# Set specific sync date for all files
python3 scripts/add_locale_switcher.py --sync-date 2025-11-16
```

#### No Backup Mode (Advanced)

```bash
# Skip backup creation (use with caution!)
python3 scripts/add_locale_switcher.py --no-backup
```

### Command-Line Options

#### add_locale_switcher.py

```
positional arguments:
  path                  Target path (default: knowledge/en/)

optional arguments:
  --dry-run            Preview changes without modifying files
  --no-backup          Don't create .bak files
  --force              Overwrite existing switchers
  --sync-date DATE     Set custom sync date (YYYY-MM-DD)
  --verbose            Show detailed logging
```

#### update_css_locale.py

```
optional arguments:
  --css-path PATH      Path to knowledge-base.css (default: auto-detect)
  --dry-run            Preview changes without modifying files
  --no-backup          Don't create .bak backup file
  --verbose            Show detailed logging
```

## CSS Customization

### Default Styles

The locale switcher uses CSS variables for easy customization:

```css
.locale-switcher {
    /* Customize these variables */
    --color-accent: #7b2cbf;      /* Current locale color */
    --color-border: #cbd5e0;      /* Separator color */
    --color-link: #3182ce;        /* Link color */
    --color-link-hover: #2c5aa0;  /* Link hover color */
    --color-text-light: #718096;  /* Meta text color */
}
```

### Color Schemes

#### Purple Accent (Default)

```css
:root {
    --color-accent: #7b2cbf;
    --color-link: #667eea;
}
```

#### Blue Accent

```css
:root {
    --color-accent: #3182ce;
    --color-link: #2b6cb0;
}
```

#### Green Accent

```css
:root {
    --color-accent: #38a169;
    --color-link: #2f855a;
}
```

### Layout Customization

#### Compact Layout

```css
.locale-switcher {
    padding: 0.3rem 0.8rem;
    font-size: 0.85rem;
}
```

#### Full Width Layout

```css
.locale-switcher {
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
}
```

#### Vertical Layout (Mobile-First)

```css
.locale-switcher {
    flex-direction: column;
    align-items: flex-start;
}
```

### Theme Integration

#### Dark Mode Support

```css
@media (prefers-color-scheme: dark) {
    .locale-switcher {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #e2e8f0;
    }

    .current-locale {
        color: #9f7aea;
    }

    .locale-link {
        color: #63b3ed;
    }
}
```

## Sync Date Management

### How Sync Dates Work

The system determines sync dates in this priority order:

1. **Custom Date**: If `--sync-date` is provided
2. **Git History**: Last commit date for the file
3. **File Mtime**: File modification timestamp

### Git Integration

#### Prerequisites

```bash
# Ensure git is available
git --version

# Ensure you're in a git repository
cd /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp
git status
```

#### Sync Date from Git

```bash
# Get last commit date for specific file
git log -1 --format=%ai knowledge/en/ML/transformer-introduction/chapter1.html

# Output: 2025-11-16 12:34:56 +0900
# Displayed as: Last sync: 2025-11-16
```

#### Update Sync Dates

```bash
# After updating translations, commit changes
git add knowledge/en/ML/ knowledge/jp/ML/
git commit -m "feat: Update ML transformer chapter translations"

# Re-run locale switcher to update sync dates
python3 scripts/add_locale_switcher.py --force knowledge/en/ML/transformer-introduction/
```

### Manual Sync Date Management

#### Set Specific Date

```bash
# Set today's date as sync date
python3 scripts/add_locale_switcher.py --sync-date $(date +%Y-%m-%d)

# Set specific historical date
python3 scripts/add_locale_switcher.py --sync-date 2025-10-01
```

#### Batch Update Sync Dates

```bash
# Update all ML dojo files with current date
python3 scripts/add_locale_switcher.py --force --sync-date 2025-11-16 knowledge/en/ML/
```

## Troubleshooting

### Common Issues

#### Issue: "No HTML files found"

**Cause**: Invalid target path or no HTML files in directory

**Solution**:
```bash
# Verify path exists
ls -la knowledge/en/ML/

# Check for HTML files
find knowledge/en/ML/ -name "*.html"

# Use absolute path
python3 scripts/add_locale_switcher.py /absolute/path/to/knowledge/en/
```

#### Issue: "Invalid HTML structure"

**Cause**: Malformed HTML in source file

**Solution**:
```bash
# Validate HTML with verbose logging
python3 scripts/add_locale_switcher.py --verbose --dry-run problem_file.html

# Check for common issues
# - Missing <!DOCTYPE html>
# - Unclosed tags
# - Malformed attributes
```

#### Issue: "Could not find insertion point"

**Cause**: HTML missing breadcrumb navigation or body tag

**Solution**:
```bash
# The script will fallback to inserting after <body>
# Verify the HTML has proper structure:
grep -n '<nav class="breadcrumb">' file.html
grep -n '<body>' file.html
```

#### Issue: "Switcher already exists"

**Cause**: File already has locale switcher

**Solution**:
```bash
# Skip the file (default behavior)
# OR force update
python3 scripts/add_locale_switcher.py --force
```

#### Issue: "Git not available"

**Cause**: Git not installed or not in PATH

**Solution**:
```bash
# Install git (macOS)
brew install git

# OR use custom sync date
python3 scripts/add_locale_switcher.py --sync-date 2025-11-16
```

### Recovery Procedures

#### Restore from Backup

```bash
# Find backup files
find knowledge/en/ -name "*.html.bak"

# Restore specific file
cp knowledge/en/ML/transformer-introduction/chapter1.html.bak \
   knowledge/en/ML/transformer-introduction/chapter1.html

# Restore all backups in directory
find knowledge/en/ML/ -name "*.html.bak" | while read backup; do
    original="${backup%.bak}"
    cp "$backup" "$original"
done
```

#### Clean Backups

```bash
# Remove all backup files after verification
find knowledge/en/ -name "*.html.bak" -delete

# Preview what would be deleted
find knowledge/en/ -name "*.html.bak"
```

#### Re-run from Scratch

```bash
# Restore from backups
find knowledge/en/ -name "*.html.bak" | while read backup; do
    cp "$backup" "${backup%.bak}"
done

# Re-run with fresh settings
python3 scripts/add_locale_switcher.py --force
```

### Debugging

#### Enable Verbose Logging

```bash
# See detailed processing information
python3 scripts/add_locale_switcher.py --verbose --dry-run
```

#### Test Single File

```bash
# Test on one file first
python3 scripts/add_locale_switcher.py \
    knowledge/en/ML/transformer-introduction/chapter1.html \
    --verbose --dry-run
```

#### Validate Output

```bash
# Check HTML validity after processing
python3 -c "
from bs4 import BeautifulSoup
with open('knowledge/en/ML/transformer-introduction/chapter1.html') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')
    print('Valid HTML:', soup.find('html') is not None)
    print('Has switcher:', soup.find(class_='locale-switcher') is not None)
"
```

## Integration Guide

### With Existing Workflows

#### Git Workflow Integration

```bash
# Add to pre-commit hook (.git/hooks/pre-commit)
#!/bin/bash
# Update locale switchers before commit
python3 scripts/add_locale_switcher.py --force --no-backup

# Stage updated files
git add knowledge/en/**/*.html
```

#### CI/CD Integration

```yaml
# GitHub Actions example (.github/workflows/locale-switcher.yml)
name: Update Locale Switchers

on:
  push:
    paths:
      - 'knowledge/en/**/*.html'

jobs:
  update-switchers:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install beautifulsoup4 tqdm
      - name: Update locale switchers
        run: python3 scripts/add_locale_switcher.py --force --no-backup
      - name: Commit changes
        run: |
          git config user.name "Locale Switcher Bot"
          git config user.email "bot@example.com"
          git add knowledge/en/**/*.html
          git commit -m "chore: Update locale switchers" || true
          git push
```

#### Make Integration

```makefile
# Makefile
.PHONY: locale-switcher locale-css locale-all

locale-css:
	python3 scripts/update_css_locale.py

locale-switcher:
	python3 scripts/add_locale_switcher.py --force

locale-all: locale-css locale-switcher
	@echo "Locale switcher setup complete"

locale-dry-run:
	python3 scripts/add_locale_switcher.py --dry-run --verbose
```

### With Translation Workflows

#### Translation Sync Process

```bash
#!/bin/bash
# sync_translations.sh

# 1. Update Japanese translations
python3 scripts/translate_content.py knowledge/en/ML/ knowledge/jp/ML/

# 2. Update locale switchers with current date
python3 scripts/add_locale_switcher.py \
    --force \
    --sync-date $(date +%Y-%m-%d) \
    knowledge/en/ML/

# 3. Commit changes
git add knowledge/en/ML/ knowledge/jp/ML/
git commit -m "feat: Sync ML translations - $(date +%Y-%m-%d)"
```

#### Per-Dojo Updates

```bash
# Update specific dojo after translation
update_dojo() {
    local dojo=$1
    echo "Updating ${dojo} dojo..."

    # Update locale switchers
    python3 scripts/add_locale_switcher.py \
        --force \
        --sync-date $(date +%Y-%m-%d) \
        "knowledge/en/${dojo}/"

    echo "✓ ${dojo} locale switchers updated"
}

# Usage
update_dojo "ML"
update_dojo "PI"
update_dojo "FM"
```

### Browser Testing

#### Manual Testing Checklist

- [ ] Switcher appears on all pages
- [ ] Japanese link works (when file exists)
- [ ] "準備中" shows when JP missing
- [ ] Sync date displays correctly
- [ ] Mobile layout works
- [ ] Print hides switcher
- [ ] High contrast mode works
- [ ] Links have proper focus indicators

#### Automated Testing

```python
# test_locale_switcher.py
from bs4 import BeautifulSoup
from pathlib import Path

def test_locale_switcher_present():
    """Test that all EN files have locale switchers."""
    en_files = Path('knowledge/en').rglob('*.html')
    for file in en_files:
        with open(file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            switcher = soup.find(class_='locale-switcher')
            assert switcher is not None, f"Missing switcher in {file}"

def test_locale_links_valid():
    """Test that locale links point to valid paths."""
    en_files = Path('knowledge/en').rglob('*.html')
    for file in en_files:
        with open(file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            link = soup.select_one('.locale-link[href]')
            if link:
                # Resolve relative path
                jp_path = (file.parent / link['href']).resolve()
                # Should be either valid or show "準備中"
                if not jp_path.exists():
                    assert 'disabled' in link.get('class', [])
```

## Performance Considerations

### Processing Speed

- **Small project** (<100 files): ~5-10 seconds
- **Medium project** (100-1000 files): ~30-60 seconds
- **Large project** (>1000 files): ~2-5 minutes

### Optimization Tips

```bash
# Process in parallel (GNU parallel)
find knowledge/en -name "*.html" | \
    parallel -j 4 python3 scripts/add_locale_switcher.py {}

# Process by dojo
for dojo in FM MI ML MS PI; do
    python3 scripts/add_locale_switcher.py "knowledge/en/${dojo}/" &
done
wait
```

## Changelog

### Version 1.0.0 (2025-11-16)

- Initial release
- HTML locale switcher implementation
- CSS styling with responsive design
- Git integration for sync dates
- Comprehensive error handling
- Full documentation

## License

Part of AI Terakoya Knowledge Base project.

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review script logs with `--verbose` flag
3. Test with `--dry-run` first
4. Create an issue with detailed error logs

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
