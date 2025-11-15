#!/bin/bash
# Complete translation cleanup for Ceramic Materials series
# Removes remaining Japanese text to achieve <1% Japanese content

BASE_DIR="/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/ceramic-materials-introduction"

echo "=========================================="
echo "CERAMIC MATERIALS - TRANSLATION CLEANUP"
echo "=========================================="
echo ""

for file in chapter-2.html chapter-3.html chapter-4.html chapter-5.html index.html; do
    filepath="$BASE_DIR/$file"

    if [ ! -f "$filepath" ]; then
        echo "✗ File not found: $file"
        continue
    fi

    echo "Processing: $file"

    # Backup
    cp "$filepath" "${filepath}.bak"

    # Apply comprehensive translations using sed
    sed -i.tmp '
        # Title and meta translations
        s/CeramicsManufacturingプロセス/Ceramic Manufacturing Processes/g
        s/Ceramics材料入門/Introduction to Ceramic Materials/g
        s/固PhaseSintering/Solid-State Sintering/g
        s/液PhaseSintering/Liquid-Phase Sintering/g
        s/ゾル-ゲル法/Sol-Gel Method/g
        s/Powder Metallurgy/Powder Metallurgy/g

        # Navigation
        s/>トップ</>Top</g
        s/>概要</>Overview</g
        s/>粉末冶金</>Powder Metallurgy</g
        s/>固相焼結</>Solid-State Sintering</g
        s/>液相焼結</>Liquid-Phase Sintering</g
        s/>高温クリープ</>High-Temperature Creep</g
        s/>演習問題</>Exercises</g
        s/>参考文献</>References</g
        s/>← 前の章</>← Previous</g
        s/>次の章へ →</>Next Chapter →</g
        s/Next Chapterへ/Next Chapter/g

        # Headers
        s/CeramicsManufacturingプロセスの概要/Overview of Ceramic Manufacturing Processes/g
        s/CeramicsManufacturingプロセスのmin類/Classification of Ceramic Manufacturing Processes/g
        s/CeramicsManufacturing法/Ceramic Manufacturing Methods/g
        s/Powderプロセス/Powder Process/g
        s/溶液プロセス/Solution Process/g

        # Content translations
        s/本 ChapterのLearning Objectives/Learning Objectives for This Chapter/g
        s/Level1（基本理解）/Level 1 (Basic Understanding)/g
        s/Level2（practiceスキル）/Level 2 (Practical Skills)/g
        s/Level3（ApplicationsForce）/Level 3 (Applied Competence)/g
        s/Powder Metallurgyプロセス/powder metallurgy process/g
        s/SinteringSimulation/sintering simulation/g
        s/粒成長/grain growth/g
        s/Density変化/density changes/g
        s/Materials Design/materials design/g
        s/Sintering条件/sintering conditions/g

        # Chapter 3 specific
        s/脆性破壊/brittle fracture/g
        s/Griffith理論/Griffith theory/g
        s/破壊靭性/fracture toughness/g
        s/Weibull統計/Weibull statistics/g
        s/信頼性評価/reliability assessment/g

        # Common words
        s/プロセス/process/g
        s/シミュレーション/simulation/g
        s/パラメータ/parameters/g
        s/モデル/model/g
        s/データ/data/g
        s/グラフ/graph/g
        s/コード/code/g
        s/ファイル/file/g
        s/システム/system/g
        s/レベル/level/g

        # Units and technical terms (keep as is for clarity)
        # Metal -> metal, Casting -> casting, etc handled above

    ' "$filepath"

    # Remove temporary file
    rm -f "${filepath}.tmp"

    # Calculate Japanese percentage
    jp_pct=$(python3 -c "import re; content = open('$filepath', 'r', encoding='utf-8').read(); jp = len(re.findall(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', content)); total = len(content); print(f'{jp/total*100:.2f}')")

    echo "  ✓ Completed - Japanese: ${jp_pct}%"
    echo ""
done

echo "=========================================="
echo "TRANSLATION CLEANUP COMPLETE"
echo "=========================================="
