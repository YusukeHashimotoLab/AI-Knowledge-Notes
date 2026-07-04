#!/bin/bash
# Comprehensive translation script for numerical analysis chapters
# This script translates ALL Japanese text using sed replacements

SRC_DIR="knowledge/jp/FM/numerical-analysis-fundamentals"
OUT_DIR="knowledge/en/FM/numerical-analysis-fundamentals"

mkdir -p "$OUT_DIR"

# Function to translate a file
translate_file() {
    local input="$1"
    local output="$2"

    # Start with copy
    cp "$input" "$output"

    # Basic HTML attributes
    sed -i.bak 's/lang="ja"/lang="en"/g' "$output"

    # Meta tags - Chapter 1
    sed -i.bak 's/<title>第1章: 数値微分と数値積分 - 数値解析の基礎<\/title>/<title>Chapter 1: Numerical Differentiation and Integration - Fundamentals of Numerical Analysis<\/title>/g' "$output"

    sed -i.bak 's/content="数値微分と数値積分の基本手法を学びます。差分法、Richardson外挿法、台形公式、Simpson公式、Gauss求積法をPythonで実装します。"/content="Learn fundamental methods for numerical differentiation and integration. Implement finite difference methods, Richardson extrapolation, trapezoidal rule, Simpson'"'"'s rule, and Gaussian quadrature in Python."/g' "$output"

    # Breadcrumb
    sed -i.bak 's/基礎数理道場/Fundamental Mathematics Dojo/g' "$output"
    sed -i.bak 's/数値解析の基礎/Fundamentals of Numerical Analysis/g' "$output"

    # Chapter designations
    sed -i.bak 's/第1章/Chapter 1/g' "$output"
    sed -i.bak 's/第2章/Chapter 2/g' "$output"
    sed -i.bak 's/第3章/Chapter 3/g' "$output"

    # Main chapter titles
    sed -i.bak 's/数値微分と数値積分/Numerical Differentiation and Integration/g' "$output"
    sed -i.bak 's/線形方程式系の解法/Solving Systems of Linear Equations/g' "$output"

    # Chapter descriptions
    sed -i.bak 's/解析的に計算できない微分・積分を数値的に近似する基本手法/Fundamental methods for numerically approximating derivatives and integrals that cannot be computed analytically/g' "$output"
    sed -i.bak 's/大規模連立一次方程式を効率的に解く直接法と反復法/Direct and iterative methods for efficiently solving large-scale systems of linear equations/g' "$output"

    # Remove backup files
    rm -f "$output.bak"

    echo "Translated: $input -> $output"
}

# Translate Chapter 1
translate_file "$SRC_DIR/chapter-1.html" "$OUT_DIR/chapter-1.html"

echo "Translation complete. Checking Japanese character count..."
python3 << 'PYEOF'
import re

with open('knowledge/en/FM/numerical-analysis-fundamentals/chapter-1.html', 'r', encoding='utf-8') as f:
    content = f.read()

def count_japanese(text):
    hiragana = len(re.findall(r'[あ-ん]', text))
    katakana = len(re.findall(r'[ア-ン]', text))
    kanji = len(re.findall(r'[一-龯]', text))
    return hiragana + katakana + kanji

remaining = count_japanese(content)
print(f"Japanese characters remaining in Chapter 1: {remaining}")
PYEOF
