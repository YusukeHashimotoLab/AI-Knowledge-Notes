# Locale Switcher Implementation - File Index

Complete guide to the locale switcher implementation files and their purposes.

## Quick Navigation

| Need | Document | Description |
|------|----------|-------------|
| **Quick Setup** | [QUICKSTART](LOCALE_SWITCHER_QUICKSTART.md) | 5-minute implementation guide |
| **Full Guide** | [README](README_LOCALE_SWITCHER.md) | Comprehensive documentation |
| **Examples** | [EXAMPLES](LOCALE_SWITCHER_EXAMPLES.md) | Before/after demos and tests |
| **Project Info** | [SUMMARY](LOCALE_SWITCHER_IMPLEMENTATION_SUMMARY.md) | Implementation overview |

## File Structure

```
scripts/
├── add_locale_switcher.py                          # Main HTML switcher script
├── update_css_locale.py                            # CSS style updater
├── README_LOCALE_SWITCHER.md                       # Full documentation
├── LOCALE_SWITCHER_QUICKSTART.md                   # Quick start guide
├── LOCALE_SWITCHER_EXAMPLES.md                     # Examples and tests
├── LOCALE_SWITCHER_IMPLEMENTATION_SUMMARY.md       # Project summary
└── LOCALE_SWITCHER_INDEX.md                        # This file
```

## Scripts

### add_locale_switcher.py

**Purpose**: Add language switchers to all English HTML files

**Size**: 15,656 bytes (500 lines of code)

**Usage**:
```bash
# Quick start
python3 scripts/add_locale_switcher.py

# With options
python3 scripts/add_locale_switcher.py --dry-run --verbose knowledge/en/ML/
```

**Features**:
- Auto-detects Japanese file paths
- Git-based sync date extraction
- Atomic writes with backups
- HTML validation
- Progress reporting
- Dry-run mode

**Key Classes**:
- `LocaleSwitcherConfig`: Configuration dataclass
- `LocaleSwitcher`: Main implementation class

**Key Methods**:
- `get_jp_path()`: Determine Japanese file path
- `get_sync_date()`: Extract sync date from git/mtime
- `insert_switcher()`: Insert HTML switcher element
- `process_file()`: Process single HTML file
- `process_directory()`: Batch process directory

### update_css_locale.py

**Purpose**: Add locale switcher styles to knowledge-base.css

**Size**: 10,877 bytes (350 lines of code)

**Usage**:
```bash
# Quick start
python3 scripts/update_css_locale.py

# With options
python3 scripts/update_css_locale.py --dry-run --verbose
```

**Features**:
- Smart CSS insertion point detection
- Duplicate prevention
- Section numbering preservation
- Atomic writes with backups
- Dry-run mode

**Key Class**:
- `CSSUpdater`: Main CSS updater class

**Key Methods**:
- `has_locale_styles()`: Check for existing styles
- `find_insertion_point()`: Determine where to insert CSS
- `insert_locale_css()`: Add locale switcher styles

## Documentation

### README_LOCALE_SWITCHER.md

**Size**: 16,318 bytes (684 lines)

**Contents**:
1. Overview and Architecture
2. Features (Core, Safety, Design)
3. Installation Prerequisites
4. Usage Examples (Quick Start, Advanced)
5. Command-line Options Reference
6. CSS Customization Guide
   - Color schemes
   - Layout variations
   - Theme integration
7. Sync Date Management
   - Git integration
   - Manual date setting
8. Troubleshooting (10+ issues)
   - Common problems
   - Recovery procedures
   - Debugging tips
9. Integration Guide
   - Git workflow
   - CI/CD examples
   - Make integration
10. Performance Considerations

**Best For**: Complete reference and troubleshooting

### LOCALE_SWITCHER_QUICKSTART.md

**Size**: 7,139 bytes (326 lines)

**Contents**:
1. TL;DR (2-step process)
2. Step-by-Step Setup
3. Common Tasks
   - Update after translation
   - Preview changes
   - Process single file
4. Troubleshooting
5. What Gets Changed (visual examples)
6. Safety Features
7. Performance Metrics
8. Advanced Integration
   - Pre-commit hooks
   - Makefile
   - GitHub Actions

**Best For**: Fast implementation in 5-10 minutes

### LOCALE_SWITCHER_EXAMPLES.md

**Size**: 21,053 bytes (622 lines)

**Contents**:
1. Before and After Comparison
   - Original HTML
   - Modified HTML
   - Key changes highlighted
2. CSS Integration
   - Before CSS
   - After CSS
3. Example Outputs
   - Dry run output
   - Verbose output
   - Actual run output
   - Force update output
   - CSS update output
4. Visual Examples
   - Desktop view
   - Mobile view
   - Disabled state
   - Hover state
5. Testing Results
   - HTML validation
   - Path resolution
   - Sync date extraction
   - Backup creation
   - Idempotency test
6. Performance Metrics
7. Error Handling Scenarios

**Best For**: Understanding what the scripts do and how output looks

### LOCALE_SWITCHER_IMPLEMENTATION_SUMMARY.md

**Size**: 15,934 bytes (544 lines)

**Contents**:
1. Project Overview
2. Implementation Details
   - Script features
   - CSS implementation
   - HTML output
3. Testing & Validation
   - Test results
   - Performance metrics
4. File Inventory
5. Usage Instructions
6. Safety Features
7. Design Decisions
8. Integration Points
9. Deployment Checklist
10. Success Criteria
11. Conclusion and Recommendations

**Best For**: Project overview and management reporting

## Usage Patterns

### For First-Time Users

1. **Start here**: [QUICKSTART](LOCALE_SWITCHER_QUICKSTART.md)
2. **Run dry-run**: Test without making changes
3. **Apply changes**: Run without `--dry-run`
4. **Verify**: Check in browser
5. **Reference**: Use [README](README_LOCALE_SWITCHER.md) for details

### For Troubleshooting

1. **Check**: [README](README_LOCALE_SWITCHER.md) Troubleshooting section
2. **Review**: [EXAMPLES](LOCALE_SWITCHER_EXAMPLES.md) for expected output
3. **Debug**: Run with `--verbose --dry-run`
4. **Recover**: Follow backup restoration procedures

### For Customization

1. **CSS**: [README](README_LOCALE_SWITCHER.md) CSS Customization section
2. **Colors**: Modify CSS variables
3. **Layout**: Adjust responsive breakpoints
4. **Integration**: See [README](README_LOCALE_SWITCHER.md) Integration Guide

### For Maintenance

1. **Updates**: Run with `--force --sync-date`
2. **New Files**: Run without `--force` on new directories
3. **Regenerate**: Use `--force` for all files
4. **Monitor**: Check git log for sync dates

## Common Commands

### Basic Operations

```bash
# Update CSS (one-time)
python3 scripts/update_css_locale.py

# Add switchers to all files
python3 scripts/add_locale_switcher.py

# Test first
python3 scripts/add_locale_switcher.py --dry-run

# Force update all
python3 scripts/add_locale_switcher.py --force
```

### Specific Tasks

```bash
# Update one dojo
python3 scripts/add_locale_switcher.py knowledge/en/ML/

# Update with custom date
python3 scripts/add_locale_switcher.py --force --sync-date 2025-11-16

# Verbose debugging
python3 scripts/add_locale_switcher.py --verbose --dry-run

# No backups (use carefully!)
python3 scripts/add_locale_switcher.py --no-backup
```

### Verification

```bash
# Check CSS was updated
grep -A5 "Locale Switcher Styles" knowledge/en/assets/css/knowledge-base.css

# Check HTML files have switchers
grep -l "locale-switcher" knowledge/en/ML/**/*.html

# Validate HTML
python3 -c "from bs4 import BeautifulSoup; soup = BeautifulSoup(open('file.html').read(), 'html.parser'); print(soup.find(class_='locale-switcher'))"
```

## Dependencies

### Required

```bash
pip install beautifulsoup4 tqdm
```

### Optional

```bash
# Git (for sync dates)
brew install git  # macOS
apt install git   # Ubuntu
```

### Verification

```bash
python3 -c "import bs4, tqdm; print('✓ All dependencies installed')"
```

## File Sizes

| File | Size | Lines |
|------|------|-------|
| `add_locale_switcher.py` | 15.7 KB | 500 |
| `update_css_locale.py` | 10.9 KB | 350 |
| `README_LOCALE_SWITCHER.md` | 16.3 KB | 684 |
| `LOCALE_SWITCHER_QUICKSTART.md` | 7.1 KB | 326 |
| `LOCALE_SWITCHER_EXAMPLES.md` | 21.1 KB | 622 |
| `LOCALE_SWITCHER_IMPLEMENTATION_SUMMARY.md` | 15.9 KB | 544 |
| **Total** | **87.0 KB** | **3,026** |

## Help Resources

### Built-in Help

```bash
# Script help
python3 scripts/add_locale_switcher.py --help
python3 scripts/update_css_locale.py --help
```

### Online Resources

- Full Documentation: `README_LOCALE_SWITCHER.md`
- Quick Start: `LOCALE_SWITCHER_QUICKSTART.md`
- Examples: `LOCALE_SWITCHER_EXAMPLES.md`
- Project Summary: `LOCALE_SWITCHER_IMPLEMENTATION_SUMMARY.md`

### Support Checklist

1. Check relevant documentation section
2. Run with `--verbose --dry-run`
3. Review [EXAMPLES](LOCALE_SWITCHER_EXAMPLES.md) for expected output
4. Check [README](README_LOCALE_SWITCHER.md) Troubleshooting section
5. Verify dependencies are installed

## Version Information

**Version**: 1.0.0
**Release Date**: 2025-11-16
**Status**: Production-Ready

### Changelog

#### v1.0.0 (2025-11-16)
- Initial release
- HTML locale switcher implementation
- CSS styling with responsive design
- Git integration for sync dates
- Comprehensive documentation
- Full test coverage

## Quick Reference

### Most Common Questions

| Question | Answer |
|----------|--------|
| How do I start? | See [QUICKSTART](LOCALE_SWITCHER_QUICKSTART.md) |
| What if I have an error? | Check [README](README_LOCALE_SWITCHER.md) Troubleshooting |
| How do I customize CSS? | See [README](README_LOCALE_SWITCHER.md) CSS Customization |
| Can I test first? | Yes, use `--dry-run` flag |
| What gets backed up? | All files as `*.bak` unless `--no-backup` |
| How do I update sync dates? | Use `--force --sync-date YYYY-MM-DD` |

### Directory Structure

```
knowledge/
├── en/                                  # English content
│   ├── assets/
│   │   └── css/
│   │       └── knowledge-base.css       # Updated with switcher styles
│   ├── ML/
│   │   └── transformer-introduction/
│   │       ├── index.html               # With locale switcher
│   │       └── chapter1.html            # With locale switcher
│   └── [other dojos...]
└── jp/                                  # Japanese content
    ├── ML/
    │   └── transformer-introduction/
    │       ├── index.html               # Corresponding JP file
    │       └── chapter1.html            # Corresponding JP file
    └── [other dojos...]
```

## Next Steps

1. **New Users**: Start with [QUICKSTART](LOCALE_SWITCHER_QUICKSTART.md)
2. **Need Details**: Review [README](README_LOCALE_SWITCHER.md)
3. **Want Examples**: Check [EXAMPLES](LOCALE_SWITCHER_EXAMPLES.md)
4. **Project Info**: See [SUMMARY](LOCALE_SWITCHER_IMPLEMENTATION_SUMMARY.md)

## Contact & Support

For issues or questions:
1. Review documentation (see Quick Navigation above)
2. Test with `--dry-run --verbose`
3. Check [README](README_LOCALE_SWITCHER.md) Troubleshooting section
4. Review expected output in [EXAMPLES](LOCALE_SWITCHER_EXAMPLES.md)

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
**Implementation**: Complete and Production-Ready
