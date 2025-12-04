---
title: ⚡ 非平衡統計力学
chapter_title: ⚡ 非平衡統計力学
subtitle: Non-Equilibrium Statistical Mechanics for Materials Processes
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/non-equilibrium-statistical-mechanics/index.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[基礎数理](<../../FM/index.html>)›[Non Equilibrium Statistical Mechanics](<../../FM/non-equilibrium-statistical-mechanics/index.html>)

[← 基礎数理道場トップ](<../index.html>)

## 🎯 シリーズ概要

非平衡統計力学は、材料プロセスにおける拡散、反応、緩和現象を微視的に記述する理論です。 本シリーズでは、Boltzmann方程式とH定理、Master方程式、Langevin方程式、Fokker-Planck方程式、 線形応答理論とゆらぎ散逸定理を学び、化学反応や拡散過程への応用をPythonで実装します。 

### 📋 学習目標

  * Boltzmann方程式とH定理を理解し、気体の緩和過程を計算できる
  * Master方程式を用いて確率過程を記述できる
  * Langevin方程式とFokker-Planck方程式を理解し、Brownian運動をシミュレーションできる
  * 線形応答理論とゆらぎ散逸定理を理解できる
  * 化学反応と拡散過程の動力学をPythonで実装できる

### 📖 前提知識

統計力学と確率論の基礎知識が必要です。Pythonの基本的な使い方を理解していることが望ましいです。

第1章

Boltzmann方程式とH定理

分布関数の時間発展を記述するBoltzmann方程式を導出し、H定理によるエントロピー増大則を理解します。 衝突項の扱いと緩和時間近似を学び、気体分子の緩和過程をPythonで数値シミュレーションします。 

Boltzmann方程式 H定理 衝突項 緩和時間近似 エントロピー増大 分布関数

💻 7コード例 ⏱️ 18-22分

[第1章を読む →](<chapter-1.html>)

第2章

Master方程式と確率過程

確率論的記述の基礎となるMaster方程式を導出し、遷移確率と詳細つり合いの概念を理解します。 Random walkやBirth-death processなどの基本的な確率過程をPythonで実装し、統計的性質を解析します。 

Master方程式 遷移確率 詳細つり合い Random walk Birth-death過程 Markov過程

💻 7コード例 ⏱️ 18-22分

[第2章を読む →](<chapter-2.html>)

第3章

Langevin方程式とBrownian運動

熱浴中の粒子の運動を記述するLangevin方程式を導出し、対応するFokker-Planck方程式を理解します。 Euler-Maruyama法による数値解法を実装し、Brownian運動の統計的性質（平均二乗変位、拡散係数）をPythonで検証します。 

Langevin方程式 Fokker-Planck方程式 Brownian運動 Euler-Maruyama法 拡散係数 平均二乗変位

💻 7コード例 ⏱️ 18-22分

[第3章を読む →](<chapter-3.html>)

第4章

線形応答理論とゆらぎ散逸定理

外場に対する系の応答を記述する線形応答理論を学びます。 Green-Kubo公式とOnsager reciprocityを理解し、fluctuation-dissipation theoremを導出します。輸送係数の計算手法をPythonで実装します。 

線形応答理論 Green-Kubo公式 ゆらぎ散逸定理 Onsager reciprocity 輸送係数 相関関数

💻 7コード例 ⏱️ 18-22分

第4章を読む →（準備中）

第5章

化学反応と拡散過程の動力学

非平衡統計力学の材料プロセスへの応用を学びます。 化学反応速度論、拡散方程式の解法、結晶成長dynamics、相分離kineticsの理論的枠組みを理解し、実践的な材料プロセスシミュレーションをPythonで実装します。 

化学反応速度論 拡散方程式 結晶成長 相分離kinetics Cahn-Hilliard方程式 材料プロセス

💻 7コード例 ⏱️ 18-22分

[第5章を読む →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
