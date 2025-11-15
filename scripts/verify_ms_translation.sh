#!/bin/bash
# Verification script for MS HTML translations
# Checks for Japanese characters in translated files

echo "========================================"
echo "MS Translation Verification"
echo "========================================"
echo ""

EN_BASE="/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS"

# Count Japanese characters in a file
count_japanese() {
    local file="$1"
    if [ -f "$file" ]; then
        grep -o "[あ-ん]\|[ア-ン]\|[一-龯]" "$file" 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# Categories to check
declare -a categories=(
    "3d-printing-introduction"
    "advanced-materials-systems-introduction"
    "electrical-magnetic-testing-introduction"
    "materials-microstructure-introduction"
    "materials-science-introduction"
    "materials-thermodynamics-introduction"
    "polymer-materials-introduction"
    "processing-introduction"
    "spectroscopy-introduction"
    "xrd-analysis-introduction"
)

total_files=0
files_with_jp=0
total_jp_chars=0

for category in "${categories[@]}"; do
    echo "Checking $category..."
    for file in "$EN_BASE/$category"/*.html; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            size=$(wc -c < "$file" | tr -d ' ')

            # Only check files that are not empty
            if [ $size -gt 1000 ]; then
                total_files=$((total_files + 1))
                jp_count=$(count_japanese "$file")

                if [ $jp_count -gt 0 ]; then
                    echo "  ⚠️  $filename: $jp_count Japanese characters (size: $size bytes)"
                    files_with_jp=$((files_with_jp + 1))
                    total_jp_chars=$((total_jp_chars + jp_count))
                else
                    echo "  ✅ $filename: 0 Japanese characters (size: $size bytes)"
                fi
            fi
        fi
    done
    echo ""
done

echo "========================================"
echo "Summary"
echo "========================================"
echo "Total files checked: $total_files"
echo "Files with Japanese characters: $files_with_jp"
echo "Total Japanese characters found: $total_jp_chars"

if [ $files_with_jp -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: All files are fully translated!"
else
    echo ""
    echo "⚠️  WARNING: $files_with_jp files still contain Japanese text"
fi
