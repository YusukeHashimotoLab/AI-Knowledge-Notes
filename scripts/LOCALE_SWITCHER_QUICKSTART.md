# Locale Switcher Quick Start Guide

Fast track to adding language switchers to your AI Terakoya knowledge base.

## TL;DR

```bash
# 1. Update CSS (once)
python3 scripts/update_css_locale.py

# 2. Add switchers to all English files
python3 scripts/add_locale_switcher.py

# Done! 🎉
```

## Step-by-Step

### Step 1: Update CSS (One-Time Setup)

Add locale switcher styles to your knowledge-base.css:

```bash
cd /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp

# Test first (dry run)
python3 scripts/update_css_locale.py --dry-run

# Apply changes
python3 scripts/update_css_locale.py
```

**Expected Output**:
```
✓ Successfully updated knowledge-base.css
Size of added CSS: 1912 characters
```

**Backup Created**: `knowledge/en/assets/css/knowledge-base.css.bak`

### Step 2: Add Locale Switchers to HTML Files

Add language switchers to all English knowledge base files:

```bash
# Test on one directory first
python3 scripts/add_locale_switcher.py --dry-run knowledge/en/ML/transformer-introduction/

# If all looks good, apply to all
python3 scripts/add_locale_switcher.py
```

**Expected Output**:
```
Processing files: 100%|██████████| 500/500 [00:20<00:00, 24.35it/s]

Total files found:     500
Successfully updated:  500
Skipped (exists):      0
Failed:                0
```

**Backups Created**: All original files saved as `*.html.bak`

### Step 3: Verify in Browser

1. Open any English knowledge base page
2. Look for the locale switcher after breadcrumb navigation
3. Click "日本語" to switch to Japanese (if available)
4. Verify "Last sync" date is correct

### Step 4: Commit Changes

```bash
# Review changes
git status
git diff knowledge/en/assets/css/knowledge-base.css

# Stage changes
git add knowledge/en/assets/css/knowledge-base.css
git add knowledge/en/**/*.html

# Commit
git commit -m "feat: Add locale switcher to knowledge base

- Add locale switcher styles to knowledge-base.css
- Add language switcher to all English HTML files
- Enable seamless EN/JP navigation
- Display last sync dates

Generated with locale_switcher scripts v1.0.0"

# Push
git push
```

### Step 5: Clean Up Backups (Optional)

Once you've verified everything works:

```bash
# Remove all backup files
find knowledge/en -name "*.html.bak" -delete
find knowledge/en -name "*.css.bak" -delete
```

## Common Tasks

### Update Switchers After Translation

When you add new Japanese translations:

```bash
# Update specific dojo
python3 scripts/add_locale_switcher.py --force \
    --sync-date $(date +%Y-%m-%d) \
    knowledge/en/ML/

# Update all dojos
python3 scripts/add_locale_switcher.py --force \
    --sync-date $(date +%Y-%m-%d)
```

### Preview Changes First

Always test before applying:

```bash
# Dry run with verbose output
python3 scripts/add_locale_switcher.py --dry-run --verbose knowledge/en/ML/
```

### Process Single File

Test on one file:

```bash
python3 scripts/add_locale_switcher.py \
    knowledge/en/ML/transformer-introduction/chapter1-self-attention.html
```

### Update Only New Files

Skip files that already have switchers:

```bash
# Without --force, existing switchers are preserved
python3 scripts/add_locale_switcher.py knowledge/en/ML/
```

## Troubleshooting

### Problem: "No HTML files found"

**Solution**: Check your path
```bash
ls knowledge/en/ML/transformer-introduction/*.html
```

### Problem: "Invalid HTML structure"

**Solution**: Validate HTML
```bash
# Check specific file
python3 -c "
from bs4 import BeautifulSoup
soup = BeautifulSoup(open('path/to/file.html').read(), 'html.parser')
print('Valid:', soup.find('html') is not None)
"
```

### Problem: Switcher not appearing

**Solution**: Clear browser cache and verify CSS is loaded
```bash
# Check CSS was updated
grep -A5 "Locale Switcher Styles" knowledge/en/assets/css/knowledge-base.css
```

### Problem: Wrong sync date

**Solution**: Set custom date
```bash
python3 scripts/add_locale_switcher.py --force \
    --sync-date 2025-11-16 \
    knowledge/en/ML/
```

## What Gets Changed

### HTML Files

**Before**:
```html
<nav class="breadcrumb">...</nav>
<header>...</header>
```

**After**:
```html
<nav class="breadcrumb">...</nav>
<div class="locale-switcher">
    <span class="current-locale">🌐 EN</span>
    <span class="locale-separator">|</span>
    <a href="../../jp/..." class="locale-link">日本語</a>
    <span class="locale-meta">Last sync: 2025-11-16</span>
</div>
<header>...</header>
```

### CSS File

Adds section 15 (Locale Switcher Styles) before section 13 (Accessibility):

```css
/* ========================================
   15. Locale Switcher Styles
   ======================================== */

.locale-switcher { ... }
.current-locale { ... }
.locale-link { ... }
/* ... more styles ... */
```

## Safety Features

- ✓ Automatic backups (`.bak` files)
- ✓ Atomic writes (no partial updates)
- ✓ HTML validation (before and after)
- ✓ Dry run mode (preview changes)
- ✓ Idempotent (safe to re-run)

## Performance

| Files | Time | Speed |
|-------|------|-------|
| 10 | ~0.5s | 20 files/s |
| 100 | ~4s | 25 files/s |
| 500 | ~20s | 25 files/s |
| 1000 | ~40s | 25 files/s |

## Next Steps

1. **Review Changes**: Open a few files in browser
2. **Test Links**: Click Japanese links to verify paths
3. **Mobile Test**: Check responsive behavior
4. **Commit**: Add to version control
5. **Deploy**: Push to production

## Advanced: Integration

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
python3 scripts/add_locale_switcher.py --force --no-backup
git add knowledge/en/**/*.html
```

### Makefile

```makefile
.PHONY: locale-all locale-css locale-html

locale-all: locale-css locale-html

locale-css:
	python3 scripts/update_css_locale.py

locale-html:
	python3 scripts/add_locale_switcher.py --force
```

### GitHub Actions

```yaml
# .github/workflows/locale.yml
name: Update Locale Switchers
on:
  push:
    paths: ['knowledge/en/**/*.html']
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install beautifulsoup4 tqdm
      - run: python3 scripts/add_locale_switcher.py --force --no-backup
      - run: git add knowledge/en/**/*.html && git commit -m "chore: Update locale switchers" && git push
```

## Need Help?

- Full documentation: `README_LOCALE_SWITCHER.md`
- Examples: `LOCALE_SWITCHER_EXAMPLES.md`
- Script help: `python3 scripts/add_locale_switcher.py --help`

---

**Quick Reference Card**

| Command | Purpose |
|---------|---------|
| `--dry-run` | Preview without changes |
| `--force` | Update existing switchers |
| `--sync-date DATE` | Set custom sync date |
| `--verbose` | Detailed logging |
| `--no-backup` | Skip backup creation |

**File Locations**

| Item | Path |
|------|------|
| Scripts | `scripts/add_locale_switcher.py`, `scripts/update_css_locale.py` |
| CSS | `knowledge/en/assets/css/knowledge-base.css` |
| English Files | `knowledge/en/**/*.html` |
| Japanese Files | `knowledge/jp/**/*.html` |
| Backups | `*.html.bak`, `*.css.bak` |

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
