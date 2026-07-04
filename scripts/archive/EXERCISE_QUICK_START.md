# PI Exercise Conversion - Quick Start Guide

## TL;DR

Convert PI exercise sections from plain HTML to interactive `<details>` format.

## Quick Commands

```bash
# Find files with exercises
./scripts/find_pi_exercises.py

# Test conversion on one file
./scripts/convert_exercises_pi.py --dry-run --file knowledge/en/PI/process-data-analysis/chapter-1.html

# Preview all changes
./scripts/convert_exercises_pi.py --dry-run

# Run conversion (creates backups)
./scripts/convert_exercises_pi.py

# Or use automated workflow
./scripts/run_exercise_conversion.sh
```

## What It Does

**Before:**
```html
<h4>Exercise 1 (Basic): Title</h4>
<p>Exercise content...</p>
```

**After:**
```html
<h4>Exercise 1 (Easy): Title</h4>
<p>Exercise content...</p>

<details>
<summary>💡 Hint</summary>
<p>Helpful hint...</p>
</details>

<details>
<summary>📝 Sample Solution</summary>
<p><em>Implementation approach:</em></p>
<ul>
<li>Step 1: ...</li>
<li>Step 2: ...</li>
<li>Step 3: ...</li>
</ul>
</details>
```

## Safety Features

- ✅ Dry-run mode (no files modified)
- ✅ Automatic backups (`.bak` files)
- ✅ Single file testing
- ✅ Validation tests

## Recovery

```bash
# Revert one file
mv file.html.bak file.html

# Revert all files
find knowledge/en/PI -name "*.html.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;
```

## Files

- `convert_exercises_pi.py` - Main script
- `test_exercise_conversion.py` - Validation
- `find_pi_exercises.py` - File finder
- `run_exercise_conversion.sh` - Complete workflow
- `EXERCISE_CONVERSION_README.md` - Full docs

## Status

**Current:** 9 exercises in 3 files need conversion

**Ready:** ✅ All tests passing, production-ready

## Help

```bash
# Show help
./scripts/convert_exercises_pi.py --help

# Read full documentation
cat scripts/EXERCISE_CONVERSION_README.md
```

---

**Created:** 2025-11-17
**Location:** `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/scripts/`
