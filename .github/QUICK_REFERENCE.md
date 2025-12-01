# CI/CD Quick Reference Card

## Common Commands

### Workflow Management
```bash
# List all workflows
gh workflow list

# Trigger link checker
gh workflow run link-check.yml

# Trigger HTML validator
gh workflow run html-validate.yml

# View recent runs
gh run list --limit 10

# Watch current run
gh run watch

# View logs
gh run view --log
```

### Artifact Management
```bash
# Download all artifacts from latest run
gh run download

# Download specific artifact
gh run download <run-id> --name linkcheck-en-report

# List artifacts for a run
gh api repos/:owner/:repo/actions/runs/<run-id>/artifacts
```

### Local Testing
```bash
# Link checking (English)
python scripts/check_links.py --path knowledge/en --output report_en.txt

# Link checking (Japanese)
python scripts/check_links.py --path knowledge/jp --output report_jp.txt

# HTML validation (install first)
npm install -g html-validate

# Validate specific file
html-validate knowledge/en/ML/transformer-introduction/index.html

# Validate directory
html-validate knowledge/en/ML/*/index.html
```

## Workflow Status Codes

| Status | Meaning |
|--------|---------|
| ✅ Success | All checks passed, no issues found |
| ❌ Failure | Broken links or validation errors detected |
| ⚠️ Warning | Non-critical issues (still passes) |
| 🔄 In Progress | Workflow currently running |
| ⏸️ Cancelled | Manually stopped or timeout |

## File Locations

```
.github/
├── workflows/
│   ├── link-check.yml          # Link validation workflow
│   ├── html-validate.yml       # HTML validation workflow
│   └── README.md               # Workflow documentation
├── CICD_SETUP.md               # Complete setup guide
└── QUICK_REFERENCE.md          # This file
```

## Trigger Conditions

### Link Checker
- **Auto**: Push/PR with HTML/MD changes
- **Manual**: `gh workflow run link-check.yml`
- **Branches**: main, develop

### HTML Validator
- **Auto**: Push/PR with HTML changes
- **Manual**: `gh workflow run html-validate.yml`
- **Branches**: main, develop

## Troubleshooting Quick Fixes

### Issue: Workflow doesn't trigger
```bash
# Check workflow is enabled
gh workflow view link-check.yml

# Manually trigger
gh workflow run link-check.yml
```

### Issue: Can't download artifacts
```bash
# Ensure run completed
gh run view <run-id>

# Download with explicit path
gh run download <run-id> --dir ./artifacts/
```

### Issue: Local link checker fails
```bash
# Install dependencies
pip install beautifulsoup4 lxml tqdm

# Run with absolute path
python scripts/check_links.py \
  --path /Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en \
  --output report.txt
```

### Issue: HTML validation fails locally
```bash
# Update html-validate
npm install -g html-validate@latest

# Create config file
cat > .htmlvalidate.json << 'EOF'
{
  "extends": ["html-validate:recommended"],
  "rules": {
    "doctype-html": "error",
    "no-inline-style": "off"
  }
}
EOF

# Validate with config
html-validate --config .htmlvalidate.json knowledge/en/ML/*/index.html
```

## Report Interpretation

### Link Check Report
```
Summary:
- Broken links: 0        ✅ PASS
- Missing anchors: 0     ✅ PASS
- Warnings: 2            ⚠️ REVIEW
```

### HTML Validation Report
```
Validating: knowledge/en/ML/transformer-introduction/index.html
✓ Valid                  ✅ PASS

error: attribute "id" duplicated
                         ❌ FAIL - Fix required
```

## Performance Optimization

### Caching
- Python dependencies: Automatic via pip cache
- Node modules: Automatic via npm cache
- Typical speedup: 30% on subsequent runs

### Parallel Execution
- EN and JP content validated concurrently
- Typical time: 5-10 minutes total
- Sequential would take: 15-20 minutes

## Integration Checklist

- [ ] Workflows committed and pushed
- [ ] First run completed successfully
- [ ] Artifacts downloadable
- [ ] Branch protection configured (optional)
- [ ] Team notified of new CI/CD pipeline
- [ ] Local testing validated
- [ ] Documentation reviewed

## Support Resources

| Resource | Location |
|----------|----------|
| Full Setup Guide | `.github/CICD_SETUP.md` |
| Workflow Docs | `.github/workflows/README.md` |
| Link Checker Script | `scripts/check_links.py` |
| GitHub Actions Docs | https://docs.github.com/actions |
| html-validate Docs | https://html-validate.org/ |

## Maintenance Schedule

- **Daily**: Monitor workflow execution
- **Weekly**: Review artifact reports
- **Monthly**: Update dependencies
- **Quarterly**: Evaluate optimizations

## Emergency Procedures

### Disable Workflow
```bash
# Disable specific workflow
gh workflow disable link-check.yml

# Re-enable later
gh workflow enable link-check.yml
```

### Skip CI for Commit
```bash
# Add to commit message
git commit -m "docs: Update README [skip ci]"
```

### Cancel Running Workflow
```bash
# Get run ID
gh run list --limit 1

# Cancel
gh run cancel <run-id>
```

## Best Practices

1. **Test Locally First**: Run validation before pushing
2. **Review Reports**: Download and analyze artifact reports
3. **Fix Root Causes**: Address systematic issues, not just symptoms
4. **Monitor Trends**: Track failure patterns over time
5. **Update Documentation**: Keep CI/CD docs current

## Contact & Support

For issues or questions:
1. Check workflow logs: `gh run view --log`
2. Review artifact reports
3. Consult `.github/CICD_SETUP.md`
4. Check GitHub Actions status page

---

**Last Updated**: 2025-12-01
**Version**: 1.0.0
**Maintainer**: AI Terakoya DevOps Team
