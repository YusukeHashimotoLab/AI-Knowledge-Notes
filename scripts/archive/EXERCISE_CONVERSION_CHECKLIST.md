# PI Exercise Conversion Checklist

Use this checklist when running the conversion to ensure safe execution.

## Pre-Conversion Checklist

- [ ] **Backup entire project** (optional but recommended)
  ```bash
  git status  # Ensure working tree is clean
  git commit -am "Checkpoint before exercise conversion"
  ```

- [ ] **Verify script location**
  ```bash
  ls scripts/convert_exercises_pi.py  # Should exist
  ```

- [ ] **Run file finder**
  ```bash
  python scripts/find_pi_exercises.py
  ```
  Expected: 8 files, 9 exercises

- [ ] **Run validation tests**
  ```bash
  python scripts/test_exercise_conversion.py
  ```
  Expected: "ALL TESTS PASSED ✓"

## Conversion Checklist

- [ ] **Run dry-run mode**
  ```bash
  python scripts/convert_exercises_pi.py --dry-run
  ```
  Expected output:
  - Files processed: 93
  - Exercises converted: 9
  - Hints converted: 3
  - Hints added: 9
  - Solutions added: 9

- [ ] **Review dry-run results**
  - Numbers match expectations?
  - No unexpected errors?

- [ ] **Test single file (optional)**
  ```bash
  python scripts/convert_exercises_pi.py --dry-run --file knowledge/en/PI/process-data-analysis/chapter-1.html
  ```
  Expected: 3 exercises converted

- [ ] **Run actual conversion**
  ```bash
  python scripts/convert_exercises_pi.py --verbose
  ```

- [ ] **Verify backup files created**
  ```bash
  find knowledge/en/PI -name "*.html.bak" | wc -l
  ```
  Expected: 3 backup files

## Post-Conversion Checklist

- [ ] **Review changes**
  ```bash
  git diff knowledge/en/PI/process-data-analysis/
  ```

- [ ] **Check one converted file manually**
  ```bash
  open knowledge/en/PI/process-data-analysis/chapter-1.html
  ```
  Verify in browser:
  - Exercises display correctly
  - Details elements are collapsible
  - Hints show/hide properly
  - Solutions show/hide properly

- [ ] **Verify all exercises converted**
  ```bash
  python scripts/find_pi_exercises.py
  ```
  Expected: "Already converted: 3" or similar

- [ ] **Run validation on converted files**
  ```bash
  grep -r "Exercise.*Basic" knowledge/en/PI/process-data-analysis/
  ```
  Expected: No matches (should be "Easy" now)

- [ ] **Test in browser**
  - Open each converted chapter
  - Click all details elements
  - Verify proper functionality

## Cleanup Checklist

- [ ] **If satisfied with conversion:**
  ```bash
  # Remove backup files
  find knowledge/en/PI -name "*.html.bak" -delete
  ```

- [ ] **If need to revert:**
  ```bash
  # Restore from backups
  find knowledge/en/PI -name "*.html.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;
  ```

- [ ] **Commit changes**
  ```bash
  git add knowledge/en/PI/
  git commit -m "Convert PI exercise sections to details format"
  ```

## Troubleshooting

### If conversion fails

1. Check error message
2. Verify file exists and is readable
3. Check for malformed HTML
4. Restore from backup if needed
5. Report issue with error message

### If output looks wrong

1. Don't delete backups yet
2. Review git diff carefully
3. Check specific file that looks wrong
4. Restore individual file: `mv file.html.bak file.html`
5. Re-run conversion with `--verbose` on that file

### If tests fail

1. Don't proceed with conversion
2. Check test output for specific failure
3. Review test file expectations
4. Investigate why conversion doesn't match expected output

## Success Criteria

✅ All checklist items completed
✅ Validation tests pass
✅ Browser testing confirms functionality
✅ Git diff shows only expected changes
✅ No errors in conversion output
✅ All exercises properly formatted

## Rollback Plan

If anything goes wrong:

```bash
# Quick rollback
find knowledge/en/PI -name "*.html.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;

# Or restore from git
git checkout knowledge/en/PI/
```

---

**Date:** _______________
**Run by:** _______________
**Status:** [ ] Success [ ] Failed [ ] Reverted

**Notes:**
_____________________________________________________________
_____________________________________________________________
_____________________________________________________________
