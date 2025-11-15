#!/usr/bin/env python3
"""Extract Japanese patterns from file for translation mapping"""

import re
from collections import Counter
from pathlib import Path

JP_FILE = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MI/nm-introduction/chapter3-hands-on.html"

# Read file
with open(JP_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Find all Japanese text (hiragana, katakana, kanji)
jp_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')
jp_texts = jp_pattern.findall(content)

# Count occurrences
counter = Counter(jp_texts)

# Print top 100 most common
print("Top 100 most common Japanese text patterns:")
print("=" * 80)
for text, count in counter.most_common(100):
    print(f"{count:4d}x  {text}")
