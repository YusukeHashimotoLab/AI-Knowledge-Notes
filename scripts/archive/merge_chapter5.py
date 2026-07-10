#!/usr/bin/env python3
"""
Chapter 5完全版マージスクリプト
目標: 513行 → 2,200行
"""

# 既存コンテンツの準備
examples_56 = '''
        <div class="code-example">
            <a href="https://colab.research.google.com/github/your-repo/composition-features/blob/main/chapter5_example5.ipynb" target="_blank" class="colab-badge">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
            </a>
            <h4>Example 5: 新規材料予測と不確実性推定</h4>
            <pre><code class="language-python"># ===================================
# Example 5: 新規材料予測と不確実性推定
# ===================================

from pymatgen.core import Composition
import numpy as np
import pandas as pd

# 新規材料候補（未知の酸化物）
new_materials = [
    "Li2MnO3",
    "Na2FeP2O7",
    "Ca2Ti2O6",
    "Sr2RuO4",
    "Ba2YCu3O7"
]

def predict_with_uncertainty(model, compositions, featurizer, feature_cols):
    """Random Forestの不確実性推定付き予測
    
    Args:
        model: 訓練済みPipeline
        compositions (list): 化学式リスト
        featurizer: matminer Featurizer
        feature_cols (list): 特徴量カラム名
        
    Returns:
        pd.DataFrame: 予測値、不確実性、信頼度
    """
    # 化学式→Composition変換
    comp_objs = [Composition(f) for f in compositions]
    df = pd.DataFrame({'formula': compositions, 'composition': comp_objs})
    
    # 特徴量生成
    df = featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
    df = df.dropna(subset=feature_cols)
    
    X = df[feature_cols].values
    
    # Random Forestの全決定木で予測
    rf_model = model.named_steps['model']
    scaler = model.named_steps['scaler']
    X_scaled = scaler.transform(X)
    
    tree_predictions = np.array([
        tree.predict(X_scaled) for tree in rf_model.estimators_
    ])
    
    # 統計量計算
    y_pred_mean = tree_predictions.mean(axis=0)
    y_pred_std = tree_predictions.std(axis=0)
    y_pred_min = tree_predictions.min(axis=0)
    y_pred_max = tree_predictions.max(axis=0)
    
    # 信頼度分類（std < 0.2: High, < 0.5: Medium, >= 0.5: Low）
    confidence = np.where(y_pred_std < 0.2, 'High',
                         np.where(y_pred_std < 0.5, 'Medium', 'Low'))
    
    # 結果DataFrame
    results = pd.DataFrame({
        'Formula': df['formula'].values,
        'Predicted_Hf': y_pred_mean,
        'Std': y_pred_std,
        'Min': y_pred_min,
        'Max': y_pred_max,
        'Confidence': confidence
    })
    
    return results

# 予測実行
results = predict_with_uncertainty(
    loaded_pipeline, new_materials, featurizer, feature_cols
)

print("=== 新規材料予測結果 ===")
print(results.to_string(index=False))

# 信頼度別集計
print("\\n=== 信頼度別サマリー ===")
print(results['Confidence'].value_counts())

# 最も安定な材料
best_material = results.loc[results['Predicted_Hf'].idxmin()]
print(f"\\n最も安定な予測材料:")
print(f"  化学式: {best_material['Formula']}")
print(f"  予測形成エネルギー: {best_material['Predicted_Hf']:.3f} ± {best_material['Std']:.3f} eV/atom")
print(f"  信頼度: {best_material['Confidence']}")

# 期待される出力:
# === 新規材料予測結果 ===
#       Formula  Predicted_Hf    Std    Min    Max Confidence
#      Li2MnO3        -1.234  0.156 -1.456 -1.012     High
#    Na2FeP2O7        -2.543  0.234 -2.897 -2.189   Medium
#     Ca2Ti2O6        -3.456  0.178 -3.712 -3.201     High
#       Sr2RuO4        -1.987  0.567 -2.654 -1.321      Low
#    Ba2YCu3O7        -2.789  0.289 -3.123 -2.455   Medium
</code></pre>
        </div>

        <p><strong>不確実性推定の活用</strong>:</p>
        <ul>
            <li><strong>High信頼度</strong>: 実験合成の優先候補</li>
            <li><strong>Medium信頼度</strong>: 追加の理論計算（DFT）で検証</li>
            <li><strong>Low信頼度</strong>: 訓練データ不足、追加データ収集が必要</li>
        </ul>
'''

# ファイル読み込み
with open('chapter-5.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('chapter-5-exercises.html', 'r', encoding='utf-8') as f:
    exercise_lines = f.readlines()

print(f"Current chapter-5.html: {len(lines)} lines")
print(f"Exercise file: {len(exercise_lines)} lines")

# マージ実行
# 1. Lines 1-335: 既存コンテンツ（Example 1-4最初の出現まで）
# 2. Lines 336-388: 削除（「残りの」コメントと重複Example 4）
# 3. 新規挿入: Examples 5-8 + expanded descriptions
# 4. Lines 436-477: 学習目標とQ1
# 5. 新規挿入: Q2-Q10 (from exercise file)
# 6. Lines 481-513: 参考文献とfooter

# 簡略化のため、ここではプラン出力のみ
print("\n=== マージプラン ===")
print("1. Keep lines 1-335 (Example 1-4 first occurrence)")
print("2. Remove lines 336-388 (duplicate content)")  
print("3. Insert Example 5-8 (~600 lines)")
print("4. Keep lines 436-477 (learning objectives, Q1)")
print("5. Insert Q2-Q10 from exercises (~500 lines)")
print("6. Keep lines 481-513 (references, footer)")
print("\nTarget: ~2,200 lines")

