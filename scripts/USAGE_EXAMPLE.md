# Usage Example: fix_broken_links.py

## Quick Start

```bash
# 1. Preview what will be fixed (recommended first step)
python3 scripts/fix_broken_links.py --dry-run

# 2. Review the report
cat link_fix_report.txt

# 3. Apply fixes
python3 scripts/fix_broken_links.py

# 4. If something went wrong, restore from backups
python3 scripts/fix_broken_links.py --restore

# 5. Once confirmed working, clean up backups
python3 scripts/fix_broken_links.py --clean-backups
```

## Example Session

### Step 1: Dry Run

```bash
$ python3 scripts/fix_broken_links.py --dry-run
2025-11-16 22:10:15 - INFO - Starting link fixing (dry_run=True)...
2025-11-16 22:10:15 - INFO - Found 582 HTML files to process
Processing files: 100%|████████████████████| 582/582 [00:03<00:00, 194.23it/s]
2025-11-16 22:10:18 - INFO - Report written to link_fix_report.txt

================================================================================
Link Fix Report
================================================================================
Files Processed: 582
Files Modified: 412
Total Fixes Applied: 509

Fixes by Pattern:
  absolute_knowledge_path: 5
  absolute_site_path: 39
  asset_path: 35
  breadcrumb_depth: 418
  wrong_filename: 12
================================================================================

DRY RUN: No files were modified
Run without --dry-run to apply fixes
```

### Step 2: Review Report

```bash
$ head -100 link_fix_report.txt
Link Fix Report - Generated: 2025-11-16 22:10:18
================================================================================

Statistics:
- Files Processed: 582
- Files Modified: 412
- Total Fixes Applied: 509

Fixes by Pattern:
  absolute_knowledge_path: 5
  absolute_site_path: 39
  asset_path: 35
  breadcrumb_depth: 418
  wrong_filename: 12

Detailed Fixes:
--------------------------------------------------------------------------------

File: MI/gnn-introduction/chapter-1.html
  Line 8: [asset_path]
    /assets/css/variables.css → ../../assets/css/variables.css
  Line 9: [asset_path]
    /assets/css/base.css → ../../assets/css/base.css
  Line 57: [absolute_knowledge_path]
    /knowledge/en/ → ../../
  Line 58: [absolute_knowledge_path]
    /knowledge/en/MI/ → ../../MI/
...
```

### Step 3: Apply Fixes

```bash
$ python3 scripts/fix_broken_links.py
2025-11-16 22:12:30 - INFO - Starting link fixing (dry_run=False)...
2025-11-16 22:12:30 - INFO - Found 582 HTML files to process
Processing files: 100%|████████████████████| 582/582 [00:05<00:00, 115.42it/s]
2025-11-16 22:12:35 - INFO - Applied 8 fixes to knowledge/en/MI/gnn-introduction/chapter-1.html
2025-11-16 22:12:35 - INFO - Applied 5 fixes to knowledge/en/MI/gnn-introduction/chapter-2.html
...
2025-11-16 22:12:40 - INFO - Report written to link_fix_report.txt

================================================================================
Link Fix Report
================================================================================
Files Processed: 582
Files Modified: 412
Total Fixes Applied: 509

Fixes by Pattern:
  absolute_knowledge_path: 5
  absolute_site_path: 39
  asset_path: 35
  breadcrumb_depth: 418
  wrong_filename: 12
================================================================================

Backup files created with .bak extension
To undo changes, run: scripts/fix_broken_links.py --restore
```

### Step 4: Verify Changes

```bash
# Check a sample file
$ git diff knowledge/en/MI/gnn-introduction/chapter-1.html | head -30
diff --git a/knowledge/en/MI/gnn-introduction/chapter-1.html b/knowledge/en/MI/gnn-introduction/chapter-1.html
index abc1234..def5678 100644
--- a/knowledge/en/MI/gnn-introduction/chapter-1.html
+++ b/knowledge/en/MI/gnn-introduction/chapter-1.html
@@ -5,8 +5,8 @@
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <title>Chapter 1: Why GNN for Materials Science | GNN Introduction Series</title>
     <meta name="description" content="Learn why Graph Neural Networks (GNN) are essential for materials science.">
-    <link rel="stylesheet" href="/assets/css/variables.css">
-    <link rel="stylesheet" href="/assets/css/base.css">
+    <link rel="stylesheet" href="../../assets/css/variables.css">
+    <link rel="stylesheet" href="../../assets/css/base.css">
...

# Run link checker to verify
$ python3 scripts/linkchecker.py knowledge/en --format text --output linkcheck_after_fix.txt
```

### Step 5 (Optional): Restore if Needed

```bash
# If something went wrong, restore from backups
$ python3 scripts/fix_broken_links.py --restore
2025-11-16 22:15:00 - INFO - Restoring files from backups...
Restoring backups: 100%|██████████████| 412/412 [00:01<00:00, 289.33it/s]
2025-11-16 22:15:02 - INFO - All backups restored
```

### Step 6: Clean Up

```bash
# Once verified everything works, remove backups
$ python3 scripts/fix_broken_links.py --clean-backups
2025-11-16 22:20:00 - INFO - Cleaning backup files...
2025-11-16 22:20:01 - INFO - Found 412 backup files to remove
2025-11-16 22:20:01 - INFO - All backups removed
```

## Advanced Usage

### Verbose Mode

```bash
# See detailed logging of every fix
python3 scripts/fix_broken_links.py --dry-run --verbose 2>&1 | tee fix_verbose.log

# Example output:
# 2025-11-16 22:10:15 - DEBUG - MI/gnn-introduction/chapter-1.html:8 [asset_path] /assets/css/variables.css → ../../assets/css/variables.css
# 2025-11-16 22:10:15 - DEBUG - MI/gnn-introduction/chapter-1.html:57 [absolute_knowledge_path] /knowledge/en/ → ../../
```

### Custom Output Location

```bash
# Save report to custom location
python3 scripts/fix_broken_links.py --dry-run --output reports/link_fix_$(date +%Y%m%d).txt
```

### Custom Base Directory

```bash
# Run from different directory
python3 scripts/fix_broken_links.py --base-dir /path/to/wp/ --dry-run
```

## Integration with Git

### Safe Workflow

```bash
# 1. Ensure clean working directory
git status

# 2. Create feature branch
git checkout -b fix/broken-links

# 3. Run dry-run and review
python3 scripts/fix_broken_links.py --dry-run
cat link_fix_report.txt

# 4. Apply fixes
python3 scripts/fix_broken_links.py

# 5. Review changes
git diff --stat
git diff knowledge/en/MI/gnn-introduction/chapter-1.html

# 6. Stage changes
git add knowledge/en/

# 7. Commit with detailed message
git commit -m "fix: Correct broken links in HTML files

- Fix absolute /knowledge/en/ paths to relative paths
- Fix asset paths from /assets/ to relative paths
- Correct breadcrumb depth issues
- Fix wrong filename references
- Total: 509 fixes across 412 files

Generated with fix_broken_links.py script.
Backed up files with .bak extension."

# 8. Verify with link checker
python3 scripts/linkchecker.py knowledge/en

# 9. Push changes
git push origin fix/broken-links

# 10. Clean backups after merge
python3 scripts/fix_broken_links.py --clean-backups
```

### Rollback Strategy

```bash
# Method 1: Using script restore
python3 scripts/fix_broken_links.py --restore

# Method 2: Using git (if committed)
git reset --hard HEAD~1

# Method 3: Using git (if not committed)
git checkout -- knowledge/en/
```

## Common Issues

### Issue 1: "Knowledge directory not found"

**Symptom:**
```
ValueError: Knowledge directory not found: /path/to/knowledge/en
```

**Solution:**
```bash
# Ensure you're in the correct directory
cd /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp

# Or use --base-dir flag
python3 scripts/fix_broken_links.py --base-dir /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp
```

### Issue 2: Permission Denied

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'knowledge/en/MI/index.html'
```

**Solution:**
```bash
# Check file permissions
ls -l knowledge/en/MI/index.html

# Fix permissions if needed
chmod u+w knowledge/en/MI/index.html

# Or fix all HTML files
find knowledge/en -name "*.html" -exec chmod u+w {} \;
```

### Issue 3: Too Many Fixes

**Symptom:**
```
Total Fixes Applied: 2000+
```

**Solution:**
```bash
# Review verbose output to understand what's being fixed
python3 scripts/fix_broken_links.py --dry-run --verbose > review.log

# Check if similarity threshold is too low
# Edit script and adjust threshold in _files_similar method
```

## Performance Notes

- **Small projects** (< 100 files): ~1 second
- **Medium projects** (100-500 files): ~2-5 seconds
- **Large projects** (500+ files): ~5-10 seconds

The script is I/O bound, so SSD vs HDD makes significant difference.

## Next Steps

After fixing links:

1. Run link checker to verify fixes
2. Test site locally
3. Check critical pages manually
4. Run any automated tests
5. Deploy to staging
6. Verify in production

## Tips

- Always run `--dry-run` first
- Keep backups until verified
- Review the report file carefully
- Test critical pages manually
- Use git for additional safety
- Document any manual fixes needed
