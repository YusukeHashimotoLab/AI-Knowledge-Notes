# GitHub Actions Workflows

This directory contains CI/CD workflows for the AI Terakoya Knowledge Base project.

## Workflows

### 1. Link Checker (`link-check.yml`)

Validates internal links and anchors across the knowledge base.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests modifying HTML/MD files
- Manual workflow dispatch

**Features:**
- Parallel validation of English and Japanese content
- Comprehensive link checking using `scripts/check_links.py`
- Anchor validation (ensures fragment identifiers exist)
- Detailed reports uploaded as artifacts
- Fails build on broken links or missing anchors

**Artifact Output:**
- `linkcheck-en-report` - English link check report
- `linkcheck-jp-report` - Japanese link check report

**Usage:**
```bash
# Manually trigger workflow
gh workflow run link-check.yml

# View recent runs
gh run list --workflow=link-check.yml

# Download artifacts
gh run download <run-id>
```

### 2. HTML Validation (`html-validate.yml`)

Validates HTML structure and syntax using `html-validate`.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests modifying HTML files
- Manual workflow dispatch

**Features:**
- Sample-based validation (10 files per language)
- HTML5 standards compliance
- Configurable validation rules
- Separate validation for English and Japanese
- Reports uploaded as artifacts

**Validation Rules:**
- Doctype required
- Proper attribute quoting
- No UTF-8 BOM
- Element-required attributes
- HTML5 element usage

**Artifact Output:**
- `html-validation-en-report` - English validation report
- `html-validation-jp-report` - Japanese validation report

## Performance Optimizations

Both workflows use:
- **Dependency caching**: Python pip and Node.js npm packages
- **Parallel execution**: EN and JP validation run concurrently
- **Path filtering**: Only trigger on relevant file changes
- **Continue-on-error**: Generate reports even when checks fail

## Configuration

### Link Checker Configuration

The link checker is configured via command-line arguments:
```yaml
python scripts/check_links.py \
  --path knowledge/en \
  --output linkcheck_en_report.txt
```

### HTML Validation Configuration

HTML validation uses `.htmlvalidate.json` (created dynamically):
```json
{
  "extends": ["html-validate:recommended"],
  "rules": {
    "void-style": "off",
    "no-trailing-whitespace": "off",
    "attr-quotes": ["error", { "style": "double" }],
    "no-inline-style": "off",
    "require-sri": "off",
    "doctype-html": "error"
  }
}
```

## Troubleshooting

### Link Check Failures

1. Download the artifact report
2. Review broken links section
3. Fix links using suggestions in report
4. Run locally to verify:
   ```bash
   python scripts/check_links.py --path knowledge/en
   ```

### HTML Validation Failures

1. Check workflow logs for specific errors
2. Download validation report artifact
3. Run locally for detailed output:
   ```bash
   npx html-validate knowledge/en/path/to/file.html
   ```
4. Fix validation errors and retest

### Common Issues

**Cache issues:**
```bash
# Clear GitHub Actions cache
gh cache delete <cache-key>
```

**Dependency installation failures:**
- Check Python/Node.js versions in workflows
- Verify package availability in public registries

## Local Testing

Before pushing, test workflows locally:

### Link Checking
```bash
# English
python scripts/check_links.py --path knowledge/en --output report_en.txt

# Japanese
python scripts/check_links.py --path knowledge/jp --output report_jp.txt
```

### HTML Validation
```bash
# Install html-validate
npm install -g html-validate

# Validate specific files
html-validate knowledge/en/ML/*/index.html

# Validate with custom config
html-validate --config .htmlvalidate.json knowledge/en/**/*.html
```

## Maintenance

### Updating Dependencies

**Python packages:**
- Update versions in workflow files
- Test locally before committing

**Node.js packages:**
- Update `node-version` in workflow
- Update html-validate version if needed

### Adding New Validation Rules

1. Modify workflow configuration sections
2. Test locally with representative files
3. Update this README with new rules
4. Commit and verify in CI

## Performance Metrics

Typical execution times:
- **Link Checker**: 2-3 minutes per language
- **HTML Validation**: 1-2 minutes per language
- **Total Pipeline**: ~5-10 minutes (parallel execution)

## Integration with Development Workflow

### Pre-commit Checks

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python scripts/check_links.py --path knowledge/en --output /tmp/linkcheck.txt
if ! grep -q "Broken links: 0" /tmp/linkcheck.txt; then
  echo "Broken links detected. Please fix before committing."
  exit 1
fi
```

### Pull Request Checklist

Before submitting a PR:
- [ ] Run link checker locally
- [ ] Validate HTML files
- [ ] Review workflow status in PR
- [ ] Fix any reported issues

## Future Enhancements

Potential improvements:
- [ ] Full HTML validation (all files, not just samples)
- [ ] Markdown linting integration
- [ ] External link validation (with rate limiting)
- [ ] Accessibility testing (axe-core)
- [ ] Performance budgets (lighthouse)
- [ ] Auto-fix capabilities for common issues
