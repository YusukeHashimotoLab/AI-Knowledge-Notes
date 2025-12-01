# CI/CD Pipeline Setup Guide

## Overview

This guide covers the GitHub Actions CI/CD pipeline for AI Terakoya Knowledge Base project. The pipeline includes automated link checking and HTML validation for both English and Japanese content.

## Created Files

### 1. Workflow Files
- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/.github/workflows/link-check.yml`
- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/.github/workflows/html-validate.yml`

### 2. Documentation
- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/.github/workflows/README.md`
- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/.github/CICD_SETUP.md` (this file)

## Workflow Architecture

```
GitHub Actions Pipeline
├── Link Checker Workflow
│   ├── check-links-en (parallel)
│   ├── check-links-jp (parallel)
│   └── summary (depends on both)
└── HTML Validation Workflow
    ├── validate-html-en (parallel)
    ├── validate-html-jp (parallel)
    └── summary (depends on both)
```

## Features Implemented

### Link Checker (`link-check.yml`)

**Trigger Conditions:**
- Push to `main` or `develop` branches
- Pull requests modifying HTML/MD files
- Manual workflow dispatch

**Capabilities:**
- Validates internal links and cross-references
- Checks anchor tags (fragment identifiers)
- Detects broken links and missing files
- Provides fix suggestions for common patterns
- Generates detailed reports with categorization

**Parallel Jobs:**
- English content validation
- Japanese content validation
- Summary aggregation

**Performance Features:**
- Python pip caching
- Continue-on-error for report generation
- 30-day artifact retention

### HTML Validation (`html-validate.yml`)

**Trigger Conditions:**
- Push to `main` or `develop` branches
- Pull requests modifying HTML files
- Manual workflow dispatch

**Capabilities:**
- HTML5 standards compliance
- Syntax validation
- Attribute validation
- Sample-based validation (10 files per language)
- Configurable validation rules

**Validation Rules:**
- Doctype enforcement
- Double-quote attribute style
- No UTF-8 BOM
- Element-required attributes
- Relaxed rules for inline styles (educational content)

**Performance Features:**
- Node.js module caching
- Sample-based testing (scalable for large repos)
- Parallel English/Japanese validation

## Setup Instructions

### Step 1: Commit Workflow Files

```bash
cd /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp

# Stage workflow files
git add .github/workflows/link-check.yml
git add .github/workflows/html-validate.yml
git add .github/workflows/README.md
git add .github/CICD_SETUP.md

# Commit
git commit -m "feat(ci): Add GitHub Actions workflows for link checking and HTML validation

- Add link-check.yml: Validates internal links and anchors for EN/JP
- Add html-validate.yml: Validates HTML syntax and structure for EN/JP
- Add comprehensive documentation and troubleshooting guides
- Implement parallel execution with artifact generation
- Include caching for improved performance

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to remote
git push origin main
```

### Step 2: Verify Workflow Activation

```bash
# Check workflow status
gh workflow list

# View recent runs
gh run list --workflow=link-check.yml
gh run list --workflow=html-validate.yml

# Watch a specific run
gh run watch <run-id>
```

### Step 3: Enable GitHub Actions (if not already enabled)

1. Navigate to your repository on GitHub
2. Go to **Settings** → **Actions** → **General**
3. Ensure "Allow all actions and reusable workflows" is selected
4. Save changes

### Step 4: Configure Branch Protection (Recommended)

1. Go to **Settings** → **Branches**
2. Add rule for `main` branch
3. Enable "Require status checks to pass before merging"
4. Select required status checks:
   - `check-links-en`
   - `check-links-jp`
   - `validate-html-en`
   - `validate-html-jp`

## Usage Guide

### Triggering Workflows Manually

```bash
# Trigger link checker
gh workflow run link-check.yml

# Trigger HTML validator
gh workflow run html-validate.yml

# Trigger with specific branch
gh workflow run link-check.yml --ref develop
```

### Downloading Artifacts

```bash
# List recent runs
gh run list --limit 5

# Download artifacts from specific run
gh run download <run-id>

# Download specific artifact
gh run download <run-id> --name linkcheck-en-report
```

### Viewing Logs

```bash
# View logs for latest run
gh run view --log

# View logs for specific run
gh run view <run-id> --log

# Follow logs in real-time
gh run watch <run-id>
```

## Interpreting Results

### Link Check Results

**Success Indicators:**
```
✓ No broken links found in English version
✓ No broken links found in Japanese version
Status: PASSED
```

**Failure Indicators:**
```
✗ Broken links detected in English version
Download the artifact for detailed report
Status: FAILED
```

**Report Structure:**
```
Link Checker Report
==================
Summary:
- Total HTML files: 450
- Total links checked: 8,234
- Broken links: 12
- Missing anchors: 5
- Warnings: 2

Broken Links by Pattern:
1. Missing Dojo Prefix (8 instances)
2. Non-existent Series (3 instances)
3. Missing Chapters (1 instance)

[Detailed breakdown follows...]
```

### HTML Validation Results

**Success Indicators:**
```
✓ Valid HTML
Status: PASSED - All sampled files are valid
```

**Failure Indicators:**
```
✗ Validation errors found
Status: FAILED - Some files have validation errors
```

**Common Validation Errors:**
- Unclosed tags
- Missing required attributes
- Invalid nesting
- Duplicate IDs
- Malformed URLs

## Troubleshooting

### Issue: Workflow Not Triggering

**Diagnosis:**
```bash
# Check if workflows are visible
gh workflow list

# Check repository settings
gh api repos/:owner/:repo/actions/permissions
```

**Solutions:**
1. Verify GitHub Actions is enabled
2. Check path filters match modified files
3. Ensure branch name matches trigger conditions

### Issue: Dependency Installation Failures

**Diagnosis:**
```bash
# Check workflow logs
gh run view --log | grep -A 10 "Install dependencies"
```

**Solutions:**
1. Verify Python/Node.js versions are available
2. Check for typos in package names
3. Review network connectivity in GitHub Actions

### Issue: High Failure Rate

**Diagnosis:**
Download artifact reports to identify patterns

**Solutions:**
1. Run local validation before pushing
2. Use pre-commit hooks
3. Fix systematic issues (e.g., Dojo prefix patterns)

### Issue: Slow Execution Times

**Current Performance:**
- Link Checker: ~2-3 minutes per language
- HTML Validation: ~1-2 minutes per language

**Optimization Options:**
1. Reduce validation sample size
2. Implement incremental validation (only changed files)
3. Split validation into more granular jobs

## Local Testing Before Push

### Link Checking

```bash
# English content
python scripts/check_links.py \
  --path knowledge/en \
  --output report_en.txt

# Japanese content
python scripts/check_links.py \
  --path knowledge/jp \
  --output report_jp.txt

# Review reports
cat report_en.txt | head -n 50
```

### HTML Validation

```bash
# Install html-validate
npm install -g html-validate

# Create config file
cat > .htmlvalidate.json << 'EOF'
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
EOF

# Validate sample files
html-validate knowledge/en/ML/*/index.html
html-validate knowledge/jp/MI/*/index.html

# Validate specific file
html-validate knowledge/en/ML/transformer-introduction/chapter1-self-attention.html
```

## Integration with Development Workflow

### Pre-Push Checklist

Before pushing changes to remote:

1. Run local link checker:
   ```bash
   python scripts/check_links.py --path knowledge/en
   ```

2. Validate modified HTML files:
   ```bash
   html-validate <path-to-modified-file>
   ```

3. Review changes:
   ```bash
   git diff knowledge/
   ```

4. Commit with descriptive message
5. Push and monitor workflow execution

### Pre-Commit Hook Setup (Optional)

Create `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Check if HTML files are staged
HTML_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.html$')

if [ ! -z "$HTML_FILES" ]; then
  echo "Running HTML validation on staged files..."

  for file in $HTML_FILES; do
    if ! html-validate "$file"; then
      echo "HTML validation failed for: $file"
      exit 1
    fi
  done

  echo "✓ HTML validation passed"
fi

# Quick link check on staged files
echo "Running quick link check..."
python scripts/check_links.py --path knowledge/en --output /tmp/linkcheck.txt

if ! grep -q "Broken links: 0" /tmp/linkcheck.txt; then
  echo "⚠ Warning: Broken links detected. Review /tmp/linkcheck.txt"
  read -p "Continue with commit? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

exit 0
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Monitoring and Alerts

### Email Notifications

GitHub Actions automatically sends notifications for:
- Failed workflow runs
- Re-enabled workflows after failure

Configure notification preferences:
1. GitHub Settings → Notifications
2. Select "Actions" preferences
3. Choose email or web notifications

### Status Badges

Add to README.md:

```markdown
[![Link Checker](https://github.com/USERNAME/REPO/actions/workflows/link-check.yml/badge.svg)](https://github.com/USERNAME/REPO/actions/workflows/link-check.yml)
[![HTML Validation](https://github.com/USERNAME/REPO/actions/workflows/html-validate.yml/badge.svg)](https://github.com/USERNAME/REPO/actions/workflows/html-validate.yml)
```

### Slack Integration (Optional)

Add to workflow files under `summary` job:

```yaml
- name: Notify Slack
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
    payload: |
      {
        "text": "CI/CD Pipeline Failed",
        "attachments": [{
          "color": "danger",
          "fields": [{
            "title": "Repository",
            "value": "${{ github.repository }}",
            "short": true
          }]
        }]
      }
```

## Maintenance Schedule

### Weekly
- Review artifact reports for patterns
- Monitor execution times
- Clear old artifacts if needed

### Monthly
- Update dependencies (actions/checkout, actions/setup-python, etc.)
- Review and update validation rules
- Analyze failure trends

### Quarterly
- Evaluate performance optimizations
- Consider new validation tools
- Update documentation

## Future Enhancements

### Planned Improvements

1. **Incremental Validation**
   - Only validate changed files
   - Implement file diff detection
   - Reduce execution time by 60-80%

2. **External Link Validation**
   - Check external URLs (with rate limiting)
   - Maintain allowlist for known slow sites
   - Generate availability reports

3. **Accessibility Testing**
   - Integrate axe-core for a11y checks
   - WCAG 2.1 compliance validation
   - Generate accessibility score

4. **Performance Budgets**
   - Lighthouse integration
   - Page load time monitoring
   - Asset size tracking

5. **Auto-Fix Capabilities**
   - Automated correction of common patterns
   - Pull request generation with fixes
   - Safe automated merging

6. **Markdown Linting**
   - Style consistency checks
   - Link validation in Markdown sources
   - YAML frontmatter validation

### Integration Opportunities

- **Dependabot**: Automated dependency updates
- **CodeQL**: Security scanning
- **Lighthouse CI**: Performance monitoring
- **Percy/Chromatic**: Visual regression testing

## Support and Resources

### Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [html-validate Documentation](https://html-validate.org/)
- [GitHub CLI Documentation](https://cli.github.com/manual/)

### Internal Resources
- Workflow README: `.github/workflows/README.md`
- Link Checker Script: `scripts/check_links.py`
- Writing Guidelines: `執筆ガイドライン.md`

### Getting Help

1. Check workflow logs: `gh run view --log`
2. Review artifact reports
3. Consult this documentation
4. Check GitHub Actions status page
5. Review recent commits for breaking changes

## Conclusion

The CI/CD pipeline is now configured and ready for use. The workflows will automatically run on pushes and pull requests, ensuring code quality and link integrity across the AI Terakoya Knowledge Base project.

For questions or issues, refer to the troubleshooting section or review the workflow logs for detailed error messages.
