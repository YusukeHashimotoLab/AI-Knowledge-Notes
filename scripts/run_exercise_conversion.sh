#!/bin/bash
#
# Exercise Conversion Workflow
#
# Runs the complete conversion process with safety checks.
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        PI Exercise Format Conversion Workflow              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Run validation tests
echo "Step 1: Running validation tests..."
echo "────────────────────────────────────────────────────────────"
python scripts/test_exercise_conversion.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Validation tests FAILED. Aborting."
    exit 1
fi
echo ""

# Step 2: Dry run to preview changes
echo "Step 2: Running dry-run to preview changes..."
echo "────────────────────────────────────────────────────────────"
python scripts/convert_exercises_pi.py --dry-run
echo ""

# Step 3: Confirm before proceeding
read -p "Do you want to proceed with the conversion? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Conversion cancelled."
    exit 0
fi
echo ""

# Step 4: Run actual conversion
echo "Step 3: Running conversion..."
echo "────────────────────────────────────────────────────────────"
python scripts/convert_exercises_pi.py --verbose
echo ""

# Step 5: Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Conversion Complete                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Backup files created with .bak extension"
echo ""
echo "To verify changes:"
echo "  git diff knowledge/en/PI/"
echo ""
echo "To revert all changes:"
echo "  find knowledge/en/PI -name '*.html.bak' -exec sh -c 'mv \"\$1\" \"\${1%.bak}\"' _ {} \\;"
echo ""
echo "To remove backup files after verification:"
echo "  find knowledge/en/PI -name '*.html.bak' -delete"
echo ""
