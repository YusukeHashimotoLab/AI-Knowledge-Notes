# CI/CD Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD Pipeline                 │
│                   AI Terakoya Knowledge Base                     │
└─────────────────────────────────────────────────────────────────┘

Triggers:
  ┌──────────┐  ┌──────────┐  ┌──────────────┐
  │  Push    │  │   Pull   │  │   Manual     │
  │  Event   │  │  Request │  │   Dispatch   │
  └─────┬────┘  └────┬─────┘  └──────┬───────┘
        │            │                │
        └────────────┴────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Path Filter Check    │
         │  (HTML/MD changes?)   │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Workflow Selection  │
         └───────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│  Link Checker  │       │ HTML Validator │
│   Workflow     │       │   Workflow     │
└───────┬────────┘       └───────┬────────┘
        │                         │
```

## Link Checker Workflow Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Link Checker Workflow (link-check.yml)      │
└─────────────────────────────────────────────────────────┘

Parallel Execution:

┌─────────────────────────┐    ┌─────────────────────────┐
│  Job: check-links-en    │    │  Job: check-links-jp    │
│  (ubuntu-latest)        │    │  (ubuntu-latest)        │
├─────────────────────────┤    ├─────────────────────────┤
│                         │    │                         │
│ 1. Checkout repo        │    │ 1. Checkout repo        │
│                         │    │                         │
│ 2. Setup Python 3.11    │    │ 2. Setup Python 3.11    │
│    └─ Cache pip deps    │    │    └─ Cache pip deps    │
│                         │    │                         │
│ 3. Install dependencies │    │ 3. Install dependencies │
│    - beautifulsoup4     │    │    - beautifulsoup4     │
│    - lxml               │    │    - lxml               │
│    - tqdm               │    │    - tqdm               │
│                         │    │                         │
│ 4. Run link checker     │    │ 4. Run link checker     │
│    Input: knowledge/en  │    │    Input: knowledge/jp  │
│    Output: report_en    │    │    Output: report_jp    │
│    └─ Continue on error │    │    └─ Continue on error │
│                         │    │                         │
│ 5. Upload artifact      │    │ 5. Upload artifact      │
│    Name: linkcheck-en   │    │    Name: linkcheck-jp   │
│    Retention: 30 days   │    │    Retention: 30 days   │
│                         │    │                         │
│ 6. Display summary      │    │ 6. Display summary      │
│                         │    │                         │
│ 7. Check for errors     │    │ 7. Check for errors     │
│    └─ Fail if broken    │    │    └─ Fail if broken    │
│                         │    │                         │
└────────────┬────────────┘    └────────────┬────────────┘
             │                               │
             └───────────┬───────────────────┘
                         │
                ┌────────▼─────────┐
                │  Job: summary    │
                │  Aggregate status│
                └──────────────────┘
```

## HTML Validation Workflow Architecture

```
┌─────────────────────────────────────────────────────────┐
│          HTML Validator Workflow (html-validate.yml)     │
└─────────────────────────────────────────────────────────┘

Parallel Execution:

┌─────────────────────────┐    ┌─────────────────────────┐
│  Job: validate-html-en  │    │  Job: validate-html-jp  │
│  (ubuntu-latest)        │    │  (ubuntu-latest)        │
├─────────────────────────┤    ├─────────────────────────┤
│                         │    │                         │
│ 1. Checkout repo        │    │ 1. Checkout repo        │
│                         │    │                         │
│ 2. Setup Node.js 20     │    │ 2. Setup Node.js 20     │
│    └─ Cache npm modules │    │    └─ Cache npm modules │
│                         │    │                         │
│ 3. Install html-validate│    │ 3. Install html-validate│
│                         │    │                         │
│ 4. Create config file   │    │ 4. Create config file   │
│    └─ .htmlvalidate.json│    │    └─ .htmlvalidate.json│
│                         │    │                         │
│ 5. Find HTML files      │    │ 5. Find HTML files      │
│    Path: knowledge/en   │    │    Path: knowledge/jp   │
│    └─ Select 10 samples │    │    └─ Select 10 samples │
│                         │    │                         │
│ 6. Validate samples     │    │ 6. Validate samples     │
│    └─ Continue on error │    │    └─ Continue on error │
│                         │    │                         │
│ 7. Create report        │    │ 7. Create report        │
│                         │    │                         │
│ 8. Upload artifact      │    │ 8. Upload artifact      │
│    Name: html-val-en    │    │    Name: html-val-jp    │
│    Retention: 30 days   │    │    Retention: 30 days   │
│                         │    │                         │
│ 9. Check result         │    │ 9. Check result         │
│    └─ Fail if errors    │    │    └─ Fail if errors    │
│                         │    │                         │
└────────────┬────────────┘    └────────────┬────────────┘
             │                               │
             └───────────┬───────────────────┘
                         │
                ┌────────▼─────────┐
                │  Job: summary    │
                │  Aggregate status│
                └──────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  Developer   │
│  Commits     │
└──────┬───────┘
       │
       │ git push
       │
       ▼
┌──────────────────────┐
│  GitHub Repository   │
│  (main/develop)      │
└──────┬───────────────┘
       │
       │ Trigger webhook
       │
       ▼
┌──────────────────────┐
│  GitHub Actions      │
│  Runner (ubuntu)     │
└──────┬───────────────┘
       │
       │ Clone repo
       │
       ▼
┌──────────────────────┐
│  Workspace Setup     │
│  - Python 3.11       │
│  - Node.js 20        │
│  - Dependencies      │
└──────┬───────────────┘
       │
       │ Execute workflows
       │
       ▼
┌──────────────────────┐
│  Validation Jobs     │
│  (Parallel)          │
│  ├─ Link check EN    │
│  ├─ Link check JP    │
│  ├─ HTML validate EN │
│  └─ HTML validate JP │
└──────┬───────────────┘
       │
       │ Generate reports
       │
       ▼
┌──────────────────────┐
│  Artifact Storage    │
│  (30-day retention)  │
│  ├─ linkcheck-en     │
│  ├─ linkcheck-jp     │
│  ├─ html-val-en      │
│  └─ html-val-jp      │
└──────┬───────────────┘
       │
       │ Status update
       │
       ▼
┌──────────────────────┐
│  GitHub Status       │
│  ✅ Pass / ❌ Fail   │
└──────┬───────────────┘
       │
       │ Notification
       │
       ▼
┌──────────────────────┐
│  Developer           │
│  Review & Fix        │
└──────────────────────┘
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline Components                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌─────────────────┐
│   Source Code   │◄────────┤   Git Events    │
│   Repository    │         │   (push/PR)     │
└────────┬────────┘         └─────────────────┘
         │
         │ Trigger
         │
┌────────▼─────────────────────────────────────────┐
│            GitHub Actions Platform                │
│                                                   │
│  ┌─────────────┐  ┌──────────────┐              │
│  │   Workflow  │  │   Workflow   │              │
│  │   Executor  │  │   Scheduler  │              │
│  └──────┬──────┘  └──────┬───────┘              │
│         │                 │                       │
│         └────────┬────────┘                       │
│                  │                                │
│         ┌────────▼──────────┐                    │
│         │  Job Orchestrator  │                    │
│         └────────┬──────────┘                    │
│                  │                                │
│         ┌────────┴────────┐                      │
│         │                 │                       │
│  ┌──────▼─────┐    ┌─────▼──────┐               │
│  │  Link      │    │   HTML     │               │
│  │  Checker   │    │   Validator│               │
│  └──────┬─────┘    └─────┬──────┘               │
│         │                 │                       │
│         └────────┬────────┘                       │
│                  │                                │
│         ┌────────▼──────────┐                    │
│         │   Artifact Store   │                    │
│         └────────┬──────────┘                    │
└──────────────────┼───────────────────────────────┘
                   │
                   │ Status/Reports
                   │
┌──────────────────▼───────────────────────────────┐
│            Developer Interface                    │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐        │
│  │ GitHub   │  │   Email  │  │  Badge  │        │
│  │ UI       │  │  Notify  │  │  Status │        │
│  └──────────┘  └──────────┘  └─────────┘        │
└──────────────────────────────────────────────────┘
```

## Caching Strategy

```
┌─────────────────────────────────────────┐
│          Caching Hierarchy              │
└─────────────────────────────────────────┘

Level 1: Dependency Caching
  ┌────────────────────────────┐
  │   Python pip Cache         │
  │   Key: os-pip-hash         │
  │   Location: ~/.cache/pip   │
  └────────────────────────────┘

  ┌────────────────────────────┐
  │   Node.js npm Cache        │
  │   Key: os-node-hash        │
  │   Location: node_modules   │
  └────────────────────────────┘

Level 2: Workflow Cache
  ┌────────────────────────────┐
  │   Parsed HTML Cache        │
  │   (BeautifulSoup objects)  │
  │   Location: Memory         │
  └────────────────────────────┘

Cache Hit Rates:
  ├─ Python deps: ~80-90% (stable dependencies)
  ├─ Node.js deps: ~70-80% (frequent updates)
  └─ Overall speedup: ~30% on subsequent runs
```

## Error Handling Flow

```
┌─────────────────────────────────────────┐
│         Error Handling Strategy         │
└─────────────────────────────────────────┘

Validation Execution:
  │
  ├─► Link Check
  │   ├─► File not found
  │   │   └─► Log error → Continue → Report
  │   ├─► Anchor missing
  │   │   └─► Log warning → Continue → Report
  │   └─► Script error
  │       └─► Log critical → Fail immediately
  │
  └─► HTML Validation
      ├─► Syntax error
      │   └─► Log error → Continue → Report
      ├─► Missing attribute
      │   └─► Log warning → Continue → Report
      └─► Parse failure
          └─► Log critical → Fail immediately

Report Generation:
  │
  ├─► Always attempt generation
  ├─► Upload as artifact
  └─► Set job status based on error count

Workflow Status:
  │
  ├─► Critical errors → ❌ Fail
  ├─► Validation errors → ❌ Fail
  ├─► Warnings only → ✅ Pass with warnings
  └─► No issues → ✅ Pass
```

## Security Model

```
┌─────────────────────────────────────────┐
│          Security Boundaries            │
└─────────────────────────────────────────┘

GitHub Actions Environment:
  ┌──────────────────────────────┐
  │  Isolated VM/Container       │
  │  - No external network req   │
  │  - Read-only source code     │
  │  - Minimal dependencies      │
  │  - No secrets required       │
  └──────────────────────────────┘

Permissions:
  ├─► Repository: Read-only
  ├─► Artifacts: Write (scoped)
  ├─► Actions: Execute
  └─► External: None

Dependency Security:
  ├─► Python packages (from PyPI)
  │   ├─ beautifulsoup4 (vetted)
  │   ├─ lxml (vetted)
  │   └─ tqdm (vetted)
  │
  └─► Node.js packages (from npm)
      └─ html-validate (vetted)
```

## Performance Optimization Strategy

```
┌─────────────────────────────────────────┐
│      Performance Optimizations          │
└─────────────────────────────────────────┘

1. Parallel Execution
   ├─ EN and JP content: Concurrent
   ├─ Link check and HTML validation: Independent
   └─ Estimated speedup: 50% vs sequential

2. Path Filtering
   ├─ Only trigger on relevant file changes
   ├─ Reduces unnecessary runs by ~60%
   └─ Saves CI/CD minutes

3. Caching
   ├─ Python dependencies cached
   ├─ Node.js modules cached
   └─ Typical speedup: 30% on cache hit

4. Sample-Based Validation (HTML)
   ├─ Validate 10 representative files
   ├─ Scales with repository size
   └─ Can be expanded to full validation

5. Early Termination
   ├─ Fail fast on critical errors
   ├─ Skip remaining checks if unrecoverable
   └─ Saves execution time

Typical Execution Times:
  ├─ Cold start (no cache): 8-12 minutes
  ├─ Warm start (cache hit): 5-8 minutes
  └─ Fast path (no changes): 2-3 minutes
```

## Artifact Lifecycle

```
┌─────────────────────────────────────────┐
│         Artifact Management             │
└─────────────────────────────────────────┘

Generation:
  ├─► Workflow execution
  ├─► Report generation
  └─► Upload to GitHub

Storage:
  ├─► Location: GitHub Actions Storage
  ├─► Retention: 30 days
  ├─► Size limit: 500 MB per artifact
  └─► Total limit: 2 GB per repository

Access:
  ├─► GitHub UI: Actions → Run → Artifacts
  ├─► GitHub CLI: gh run download
  └─► GitHub API: REST API access

Cleanup:
  ├─► Automatic: After 30 days
  ├─► Manual: gh api (delete endpoint)
  └─► Policy: Keep last 10 successful runs
```

## Integration Points

```
┌─────────────────────────────────────────┐
│      External Integration Points        │
└─────────────────────────────────────────┘

Current:
  ├─► GitHub Events (push, PR)
  ├─► GitHub Status API
  └─► GitHub Artifacts

Potential Future:
  ├─► Slack notifications
  ├─► Email reports
  ├─► Dashboard visualization
  ├─► Metrics collection (Prometheus)
  └─► Alerting (PagerDuty)
```

## Monitoring Dashboard (Conceptual)

```
┌─────────────────────────────────────────────────────┐
│              CI/CD Monitoring Dashboard              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Workflow Status:    ✅ Passing                      │
│  Last Run:           2025-12-01 17:30 UTC           │
│  Duration:           6m 42s                          │
│  Success Rate:       95.2% (last 30 days)           │
│                                                      │
├─────────────────────────────────────────────────────┤
│  Link Checker:                                       │
│    English:  ✅ 0 broken links                       │
│    Japanese: ✅ 0 broken links                       │
│                                                      │
│  HTML Validator:                                     │
│    English:  ✅ 10/10 files valid                    │
│    Japanese: ✅ 10/10 files valid                    │
│                                                      │
├─────────────────────────────────────────────────────┤
│  Performance Metrics:                                │
│    ┌──────────────────────────────────────┐         │
│    │ Execution Time Trend                 │         │
│    │  8m ┤                                │         │
│    │  6m ┤    ●     ●     ●              │         │
│    │  4m ┤ ●     ●     ●     ●  ●  ●     │         │
│    │  2m ┤                                │         │
│    │  0m └──────────────────────────────►│         │
│    └──────────────────────────────────────┘         │
│                                                      │
├─────────────────────────────────────────────────────┤
│  Recent Failures:                                    │
│    ├─ 2025-11-28: Broken link in ML/transformer/... │
│    └─ 2025-11-25: HTML validation in FM/quantum/... │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

**Last Updated**: 2025-12-01
**Version**: 1.0.0
**Architecture**: GitHub Actions + Ubuntu Runners + Python/Node.js
