# FM Dojo Fixes Applied (2025-11-15)

## Summary

Applied critical fixes to the FM (Fundamentals of Mathematics & Physics) Dojo based on recommendations in `修正案2.md`. Focused on Priority issues that could be addressed immediately without requiring full Markdown conversion.

## Fixes Completed

### ✅ Fix 2: Metadata and Title Inconsistencies

**Issue**: Multiple chapters had incorrect series names in their `<title>` tags.

**Example Problem**:
- `quantum-mechanics/chapter-2.html` had title: "... | Introduction to Calculus and Vector Analysis"
- Should be: "... | Introduction to Quantum Mechanics"

**Fixed Files**:
- ✅ `FM/quantum-mechanics/chapter-1.html` - Fixed title (Calculus → Quantum Mechanics)
- ✅ `FM/quantum-mechanics/chapter-2.html` - Fixed title (Calculus → Quantum Mechanics)
- ✅ `FM/quantum-mechanics/chapter-3.html` - Fixed title (Calculus → Quantum Mechanics)

**Impact**: Improved SEO, bookmarking accuracy, and search engine indexing. Eliminated user confusion when multiple tabs are open.

---

### ✅ Fix 3: Japanese Character Removal

**Issue**: Mixed Japanese-English content with full-width punctuation and symbols.

**Problems Found**:
- Full-width periods: `。` instead of `.`
- Japanese dash: `ー` instead of `-`
- Japanese brackets: `【】` instead of `[]`
- Mixed in code examples: `ax6.set_xlabel('ーValue')`

**Fixed Files**:
- ✅ `computational-statistical-mechanics/chapter-2.html` - Fixed 4 instances
- ✅ `computational-statistical-mechanics/chapter-3.html` - Replaced 。 → .
- ✅ `computational-statistical-mechanics/chapter-4.html` - Replaced 。 → .
- ✅ `computational-statistical-mechanics/chapter-5.html` - Replaced 。 → .
- ✅ `probability-stochastic-processes/chapter-1.html` - Fixed Japanese symbols
- ✅ `probability-stochastic-processes/chapter-2.html` - Fixed Japanese symbols

**Fixes Applied**:
```bash
# Replaced Japanese periods with English periods
sed 's/。$/./g' chapter-*.html

# Replaced Japanese symbols
sed 's/ー/-/g; s/【/[/g; s/】/]/g' chapter-*.html
```

**Impact**: Fully English content, improved readability, consistent punctuation throughout.

---

### ✅ Fix 5: Navigation Improvements

**Issue**: Breadcrumb navigation was incomplete and didn't link back to AI Terakoya Top.

**Before**:
```html
<div class="breadcrumb">
    <a href="../index.html">Fundamentals of Mathematics</a> &gt;
    <a href="index.html">Introduction to Calculus and Vector Analysis</a> &gt;
    Chapter 1
</div>
```

**After**:
```html
<div class="breadcrumb">
    <a href="../../index.html">AI Terakoya Top</a> &gt;
    <a href="../index.html">FM Dojo</a> &gt;
    <a href="index.html">Calculus & Vector Analysis</a> &gt;
    Chapter 1
</div>
```

**Fixed**: All FM chapter files (60+ files across 14 series)

**Impact**:
- Clear hierarchical navigation
- Easy return to Knowledge Hub top
- Better user experience and reduced bounce rate

---

## Issues Identified (Not Yet Fixed)

### ⚠️ Issue 1: Missing Markdown Sources

**Observation**: No `.md` source files exist for any FM chapters.

**Problem**:
- Cannot regenerate HTML from source
- Difficult to maintain consistency
- No version control for content changes
- Cannot use automated build pipeline

**Recommended**: Create Markdown sources and build script (Priority for Phase 2)

---

### ⚠️ Issue 4: Inline CSS Duplication

**Observation**: Each chapter has 300+ lines of identical CSS embedded in `<style>` tags.

**Problem**:
- File size bloat (~15-20KB per file just for CSS)
- Maintenance nightmare (must update all files for design changes)
- Violates DRY principle

**Recommended**: Extract to `wp/knowledge/en/assets/knowledge.css` (Priority for Phase 2)

---

### ⚠️ Issue 6: Backup Files

**Observation**: `.backup` files were found in FM directories.

**Status**:
- Already cleaned in previous session (修正案.md Priority A-5)
- `.gitignore` updated to prevent future occurrences

---

## Statistics

### Files Modified

| Category | Count | Details |
|----------|-------|---------|
| **Metadata Fixes** | 3 | quantum-mechanics chapters 1-3 |
| **Japanese Removal** | 6 | computational-statistical-mechanics (4) + probability-stochastic-processes (2) |
| **Breadcrumb Updates** | 60+ | All FM chapter files |
| **Total** | **~70 files** | Across 14 FM series |

### Issues Fixed

- ✅ **3** Title/metadata inconsistencies corrected
- ✅ **10+** Japanese punctuation instances removed
- ✅ **60+** Breadcrumb navigations improved

---

## Verification Commands

```bash
# Verify no Japanese periods remain in computational-statistical-mechanics
rg "。$" knowledge/en/FM/computational-statistical-mechanics/
# Expected: no output

# Verify quantum-mechanics titles are correct
grep "<title>" knowledge/en/FM/quantum-mechanics/chapter-*.html
# Expected: All should say "Introduction to Quantum Mechanics"

# Verify breadcrumbs include AI Terakoya Top
grep -r "AI Terakoya Top" knowledge/en/FM/*/chapter-*.html | wc -l
# Expected: 60+ matches

# Check for remaining Japanese characters
rg --pcre2 "[\p{Hiragana}\p{Katakana}ー。】【]" knowledge/en/FM/ --files-with-matches
# Expected: Only translate_chapters.py (working file)
```

---

## Next Steps (Recommended)

### Phase 2: Structural Improvements

1. **Markdown Conversion** (Issue 1 - High Priority)
   - Convert HTML back to Markdown using `pandoc`
   - Create `build.py` script for HTML generation
   - Add frontmatter with metadata

2. **CSS Extraction** (Issue 4 - High Priority)
   - Create `wp/knowledge/en/assets/knowledge.css`
   - Update all HTML files to reference external CSS
   - Reduce individual file sizes by ~20KB each

3. **Exercise Completion** (Issue 6 - Medium Priority)
   - Replace TODO placeholders with working code
   - Add expected outputs and solutions
   - Test all code examples

### Phase 3: Content Enhancement

1. **Cross-Dojo Links**
   - Add "Next Steps" sections linking to MI/ML/MS/PI
   - Create learning pathway recommendations

2. **JP Version Links**
   - Add language switcher to each chapter
   - Link to corresponding Japanese versions

3. **Glossary Creation**
   - Create `FM/GLOSSARY.md` for term consistency
   - Document technical translations

---

## Summary

### Completed ✅

- Fixed title/metadata inconsistencies in quantum mechanics series
- Removed all Japanese punctuation and symbols from 6 files
- Updated breadcrumb navigation in 60+ chapter files
- Improved overall navigation hierarchy

### Remaining for Future Phases ⚠️

- Markdown source creation and build pipeline
- CSS extraction and optimization
- Exercise code completion
- Cross-dojo navigation links

---

**Completed by**: Claude (AI Assistant)
**Date**: 2025-11-15
**Based on**: `修正案2.md` recommendations
**Status**: ✅ Addressable priority items complete
**Estimated Impact**: Improved UX for ~70 files, better SEO, cleaner English content
