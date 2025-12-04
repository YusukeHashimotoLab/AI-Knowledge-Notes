---
title: 💻 計算統計力学
chapter_title: 💻 計算統計力学
subtitle: Computational Statistical Mechanics for Materials Simulation
---

🌐 JP | [🇬🇧 EN](<../../../en/FM/computational-statistical-mechanics/index.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[基礎数理](<../../FM/index.html>)›[Computational Statistical Mechanics](<../../FM/computational-statistical-mechanics/index.html>)

[← 基礎数理道場トップ](<../index.html>)

## 🎯 シリーズ概要

計算統計力学は、Monte Carlo法や分子動力学法を用いて材料の熱力学的性質や動的挙動をシミュレーションする手法です。 本シリーズでは、Metropolis法、重点サンプリング、Replica交換法、分子動力学法、自由エネルギー計算まで、 理論とPython実装をペアで学び、材料物性予測に応用します。 

### 📋 学習目標

  * Metropolis Monte Carlo法の原理を理解し、実装できる
  * 重点サンプリングとReplica交換法を用いて相転移をシミュレーションできる
  * 分子動力学法の基礎を理解し、Lennard-Jonesポテンシャルで原子系をシミュレーションできる
  * 拡張アンサンブル法を理解し、実装できる
  * 自由エネルギー計算法を用いて材料物性を予測できる

### 📖 前提知識

古典統計力学の基礎知識と数値計算の基本が必要です。Pythonの基本的な使い方を理解していることが望ましいです。

第1章

Monte Carlo法の基礎

統計力学計算の基本となるMonte Carlo法を学びます。Metropolis法のアルゴリズムとImportance samplingの原理を理解し、Ising modelのシミュレーションをPythonで実装します。 Acceptance ratioの最適化とエルゴード性の検証も扱います。 

Monte Carlo法 Metropolis法 Importance sampling Markov chain Ising model エルゴード性

💻 7コード例 ⏱️ 20-24分

[第1章を読む →](<chapter-1.html>)

第2章

高度なサンプリング法

エネルギーバリアを越える効率的なサンプリング手法を学びます。Wang-Landau法、Multicanonical sampling、Umbrella samplingの原理を理解し、状態密度計算や相転移検出をPythonで実装します。 各手法の適用範囲と限界も議論します。 

Wang-Landau法 Multicanonical法 Umbrella sampling 状態密度 重点サンプリング 相転移検出

💻 7コード例 ⏱️ 20-24分

[第2章を読む →](<chapter-2.html>)

第3章

分子動力学法の基礎

原子・分子の古典力学シミュレーションを学びます。Verlet積分法とその変種（Leap-frog法、速度Verlet法）を理解し、Lennard-Jones系のシミュレーションをPythonで実装します。 動径分布関数などの構造解析手法と、温度・圧力制御アルゴリズムも習得します。 

分子動力学法 Verlet積分法 Lennard-Jones 動径分布関数 温度制御 周期境界条件

💻 7コード例 ⏱️ 20-24分

[第3章を読む →](<chapter-3.html>)

第4章

Replica交換法と拡張アンサンブル法

多重極小問題を解決する高度なサンプリング手法を学びます。Parallel tempering（Replica交換法）、Replica exchange MD、Simulated annealingの原理を理解し、エネルギーlandscape探索をPythonで実装します。 材料構造最適化への応用も扱います。 

Replica交換法 Parallel tempering Simulated annealing 拡張アンサンブル エネルギーlandscape 構造最適化

💻 7コード例 ⏱️ 20-24分

第4章を読む →（準備中）

第5章

自由エネルギー計算と材料物性予測

材料の熱力学的安定性を評価する自由エネルギー計算手法を学びます。Thermodynamic integration、Bennett acceptance ratio（BAR法）、Free energy perturbationを理解し、材料相安定性や界面エネルギーをPythonで計算します。 実材料への応用例も紹介します。 

自由エネルギー計算 TI法 Bennett法 相安定性 界面エネルギー 材料物性予測

💻 7コード例 ⏱️ 20-24分

[第5章を読む →](<chapter-5.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
