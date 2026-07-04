# Quick Reference: fix_broken_links.py

## TL;DR

```bash
# See what will be fixed
python3 scripts/fix_broken_links.py --dry-run

# Fix it
python3 scripts/fix_broken_links.py

# Undo if needed
python3 scripts/fix_broken_links.py --restore
```

## Command Cheat Sheet

| Command | Description |
|---------|-------------|
| `--dry-run` | Preview without making changes |
| `--verbose` | Show detailed logging |
| `--restore` | Undo all changes from backups |
| `--clean-backups` | Remove all .bak files |
| `--output FILE` | Custom report location |
| `--base-dir PATH` | Specify project directory |

## What It Fixes

| Pattern | Example | Count |
|---------|---------|-------|
| Absolute paths | `/knowledge/en/` → `../../` | 5 |
| Site paths | `/en/` → `../../../en/` | 39 |
| Assets | `/assets/css/` → `../../assets/css/` | 35 |
| Breadcrumbs | `../../../index.html` → `../../index.html` | 418 |
| Wrong names | `chapter2-q-learning.html` → `chapter2-q-learning-sarsa.html` | 12 |

**Total: 509 fixes across 412 files**

## Safety Checks

- ✅ Creates `.bak` backups before changes
- ✅ Dry-run mode shows preview
- ✅ Restore capability to undo
- ✅ Preserves HTML structure
- ✅ Detailed logging and reporting

## Typical Workflow

```bash
# 1. Preview
python3 scripts/fix_broken_links.py --dry-run

# 2. Review report
cat link_fix_report.txt

# 3. Apply
python3 scripts/fix_broken_links.py

# 4. Test manually
# (open some fixed pages in browser)

# 5. If OK, commit
git add knowledge/en/
git commit -m "fix: Correct broken links"

# 6. Clean up
python3 scripts/fix_broken_links.py --clean-backups
```

## Emergency Rollback

```bash
# Method 1: Script restore
python3 scripts/fix_broken_links.py --restore

# Method 2: Git reset (if not committed)
git checkout -- knowledge/en/

# Method 3: Git revert (if committed)
git revert HEAD
```

## Output Files

- `link_fix_report.txt` - Detailed fix report
- `*.html.bak` - Backup files (one per modified file)

## Common Issues

| Issue | Solution |
|-------|----------|
| "Knowledge directory not found" | Run from `/wp/` directory |
| Permission denied | `chmod u+w knowledge/en/**/*.html` |
| Too many backups | `--clean-backups` |

## Performance

- **582 files** processed in **~5 seconds**
- **Memory usage:** < 100MB
- **Disk space:** ~1:1 ratio for backups

## Testing

```bash
# Run unit tests
python3 scripts/test_fix_broken_links.py

# Expected: 18 tests, all passing
```

## Help

```bash
# Full help
python3 scripts/fix_broken_links.py --help

# Documentation
cat scripts/README_fix_broken_links.md
cat scripts/USAGE_EXAMPLE.md
```

## One-Liner Examples

```bash
# Fix with verbose output saved to log
python3 scripts/fix_broken_links.py --verbose 2>&1 | tee fix.log

# Dry-run with custom output
python3 scripts/fix_broken_links.py --dry-run --output reports/links_$(date +%Y%m%d).txt

# Fix from different directory
cd /path/to/anywhere && python3 /path/to/wp/scripts/fix_broken_links.py --base-dir /path/to/wp
```

## Exit Codes

- `0` - Success
- `1` - Error (check logs)

## Dependencies

```bash
pip install beautifulsoup4 lxml tqdm
```

## Version

- **Version:** 1.0
- **Date:** 2025-11-16
- **Python:** 3.7+
- **Status:** Production Ready ✅
