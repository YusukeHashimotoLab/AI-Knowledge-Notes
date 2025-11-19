#!/bin/bash
# Chapter 5演習問題統合スクリプト（Option A'実行）

echo "=== Chapter 5 Option A' 実行：演習問題Q2-Q10統合 ==="

# バックアップ
cp chapter-5.html chapter-5-before-option-a-prime.html
echo "✅ バックアップ作成: chapter-5-before-option-a-prime.html"

# 統合実行
# chapter-5.htmlのline 477（Q1の後）にchapter-5-exercises.html（lines 3-498）を挿入

# Pythonで統合
python3 << 'PYEOF'
# ファイル読み込み
with open('chapter-5.html', 'r', encoding='utf-8') as f:
    main_lines = f.readlines()

with open('chapter-5-exercises.html', 'r', encoding='utf-8') as f:
    exercise_lines = f.readlines()

# 挿入位置を特定（"参考文献"の直前）
insert_idx = None
for i, line in enumerate(main_lines):
    if '<h2>参考文献</h2>' in line:
        insert_idx = i
        break

if insert_idx is None:
    print("エラー: 挿入位置が見つかりません")
    exit(1)

# 統合
new_lines = main_lines[:insert_idx]
new_lines.extend(exercise_lines[2:-1])  # ヘッダーとフッターを除外
new_lines.extend(main_lines[insert_idx:])

# 書き込み
with open('chapter-5.html', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"✅ 演習問題Q2-Q10を統合しました")
print(f"   挿入位置: line {insert_idx}")
print(f"   追加行数: {len(exercise_lines[2:-1])}行")
print(f"   新総行数: {len(new_lines)}行")
PYEOF

# 検証
echo ""
echo "=== 統合後の統計 ==="
wc -l chapter-5.html
echo ""
echo "=== Chapter 5内容確認（演習問題セクション）==="
grep -n "summary.*Q[0-9]" chapter-5.html | head -15

echo ""
echo "✅ Option A'完了"
echo "   chapter-5.html: 演習問題Q1-Q10完全統合"
echo "   次ステップ: 最終統計確認とcompleted report作成"

