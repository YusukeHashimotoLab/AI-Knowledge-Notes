---
title: Electron Microscopy Introduction Series
chapter_title: Electron Microscopy Introduction Series
subtitle: From SEM/TEM Principles to Practical Analytical Techniques - Your First Step to Nanoscale Observation
difficulty: Beginner to Intermediate
code_examples: 35
version: 1.0
created_at: 2025-10-28
---

## Series Overview

This series is an introductory course covering the fundamental principles to practical analytical techniques of electron microscopy (SEM/TEM), with a hands-on approach using Python. You will acquire the knowledge and skills necessary for nanoscale structural analysis of materials.

### Learning Flow
    
    
    ```mermaid
    flowchart LR
        A[Chapter 1Electron Microscopy Basics] --> B[Chapter 2SEM Introduction]
        B --> C[Chapter 3TEM Introduction]
        C --> D[Chapter 4STEM and Analytical Techniques]
        D --> E[Chapter 5Integrated Analysis Practice]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

## Series Structure

Chapter 1

Electron Microscopyfundamentals of

Electronics光学の基本原理、光学顕微鏡とのRatio較、min解能のTheory、Electron Beamと物質のPhase互作用、Electron Microscopyの種類と特徴will be covered。 

⏱️ 30-35min 💻 7code examples 📊 入門

[Start Learning →](<chapter-1.html>)

Chapter 2

SEM入門

SEMの装置構成、Secondary Electrons（SE）とBackscattered Electrons（BSE）の違い、EDSによるElementmin析、ZAF補正法、practice的なObservation・min析技術will be covered。 

⏱️ 30-35min 💻 7code examples 📊 入門〜Intermediate

[Start Learning →](<chapter-2.html>)

Chapter 3

Transmission Electron Microscopy（TEM）入門

TEM結ImageTheory、明視野・Dark Field Image、制限視野Diffraction（SAED）、LatticeImage・高min解能TEM、収差補正技術を学び、atomLevelのanalysis基礎masterします。 

⏱️ 25-35min 💻 7code examples 📊 Intermediate

[Start Learning →](<chapter-3.html>)

Chapter 4

STEMとmin析技術

STEM原理、Z-contrastImage、ElectronicsEnergy損失min光（EELS）、Elementマッピング、atommin解能min析、トモGraphィーfundamentals ofとApplicationswill be covered。 

⏱️ 25-35min 💻 7code examples 📊 Intermediate〜Advanced

[Start Learning →](<chapter-4.html>)

Chapter 5

EDS・EELS・EBSD統合min析practice

Pythonで学ぶ統合min析ワークフロー、HyperSpyによるData処理、機械学習min類、Phase同定、Crystal方位analysis、トラブルシューティングをpracticeします。 

⏱️ 30-40min 💻 7code examples 📊 Advanced

[Start Learning →](<chapter-5.html>)

## Learning Objectives

この Seriesを完了することで、以下のスキルと知識masterできます：

  * ✅ Electronics光学fundamentals ofとmin解能Theoryunderstandし、Observation条件をOptimizationできる
  * ✅ SEM、TEM、STEMの原理と使いminけをDescriptionできる
  * ✅ Secondary Electrons、Backscattered Electrons、Diffraction、EELS信号の物理的起源understandできる
  * ✅ EDS定量min析（ZAF補正）を実行し、結果を正しく解釈できる
  * ✅ ElectronicsDiffraction Pattern（SAED）の指数付けable to
  * ✅ 高min解能TEMImageのFFTanalysisとLattice間隔Measurementable to
  * ✅ HyperSpyを使ってスペクトルDataを処理できる
  * ✅ 機械学習を用いたPhasemin類とElementマッピングable to
  * ✅ EBSD方位analysisとKAM/GNDcalculationable to
  * ✅ 統合Dataから材料のMicrostructure-Properties-プロセスPhase関を解明できる

## Recommended Learning Patterns

### パターン1: 標準学習 - Theoryとpracticeのバランス（5-7days）

  * 1Day: Chapter 1（Electron Microscopyfundamentals of）
  * 2Day: Chapter 2（SEM入門）
  * 3Day: Chapter 3（TEM入門）
  * 4Day: Chapter 4（STEM技術）
  * 5Day: Chapter 5（統合min析practice）+ 総合復習

### パターン2: 集中学習 - 電顕マスター（3days）

  * 1Day: 第1-2 Chapter（基礎TheoryとSEM）
  * 2Day: 第3-4 Chapter（TEMとSTEM）
  * 3Day: Chapter 5（practiceanalysis）+ 各 ChapterExercises

### パターン3: practice重視 - Dataanalysisスキル習得（1日）

  * 第1-4 Chapter: code examplesのみを実行（Theoryは参照程度）
  * Chapter 5: じっくり取り組み、実際の電顕Dataでanalysis練習
  * 必要に応じてTheory部minに戻って確認

## Prerequisites

min野 | Required Level | Description  
---|---|---  
**Materials Science基礎** | 入門Level完了 | Crystal Structure、ChemistryBond、材料のmin類の理解  
**Physics** | 大学1-2年Level | 電磁気学、波動光学、量子Force学fundamentals of  
**Mathematics** | 大学1年Level | 微積min、線形代数、フーリエ変換fundamentals of  
**Python** | Intermediate | numpy、matplotlib、pandas、scikit-image、HyperSpyの基本操作  
  
## Python Libraries Used

この Seriesで使用する主要なライブラリ：

  * **numpy** : 数値calculationと配列操作
  * **matplotlib** : 2DGraph・画ImageTable示
  * **scipy** : 科学calculation（FFT、Optimization、信号処理）
  * **pandas** : Data処理とanalysis
  * **scikit-image** : 画Image処理
  * **HyperSpy** : 電顕スペクトルDataanalysis（EDS、EELS）
  * **pyxem** : ElectronicsDiffractionDataanalysis
  * **orix** : EBSDCrystal方位analysis
  * **kikuchipy** : EBSDパターンanalysis
  * **scikit-learn** : 機械学習（min類、クラスタリング）

## FAQ - Frequently Asked Questions

### Q1: Electron Microscopyの実機に触れた経験がなくても大丈夫ですか？

はい、大丈夫です。This seriesTheoryとcalculation・Dataanalysisに焦点を当てています。実際の装置操作は扱いませんが、Data解釈とSimulationを通じて深い理解が得られます。

### Q2: SEMとTEMの違いは何ですか？

SEM（走査型）はSampleTable面をビームで走査し、Table面形態をObservationします。TEM（透過型）はSampleを透過したElectron Beamで内部StructureをatomLevelでObservationします。Chapter 2とChapter 3で詳しくDescriptionします。

### Q3: Materials Informatics（MI）との関係は？

Electron MicroscopyDataは材料のミクロStructure情報の宝庫です。本 Seriesで学ぶData処理と機械学習技術は、MI における材料Dataベース構築、Microstructure-PropertiesPhase関Modeling、自動Phasemin類に直接Applicationsできます。

### Q4: HyperSpyの習得は必須ですか？

Chapter 5で重点的に扱いますが、基本的なnumpyとmatplotlibの知識があれば学習できます。HyperSpyは電顕コミュニティで広く使われているため、実務で非常に有用です。

### Q5: 生物Sampleにも適用できますか？

This seriesMaterials Science（Metal、Ceramics、Semiconductor）に焦点を当てていますが、基本原理は生物Sampleにも共通です。ただし、Sample Preparation法（固定、染色、包埋）は大きく異なります。

## Key Learning Points

  * **スケール感覚を養う** : SEM（μm〜nm）、TEM（nm〜Å）、STEM（atomLevel）のスケールを意識
  * **信号の物理的起源understand** : SE、BSE、Diffraction、EELSがどこから来るのかを常に考える
  * **定量化の重要性** : 「きれいなImage」ではなく数値Dataとして扱う習慣をつける
  * **コード実行とParameter変更** : すべてのcode examplesを実行し、Parameterを変えて挙動understand
  * **実Dataでの練習** : Chapter 5では、可能なら自minのResearchDataや公開Dataでanalysis練習

## Next Steps

この Seriesを完了した後、以下の発展学習をお勧めします：

  * 先端Electron Microscopy技術 - EnvironmentTEM、in-situObservation、4D-STEM
  * X-raymin析法入門 - XRD、XRF、XPS
  * atommin解能Imageanalysis - ImageSimulation、atom配列決定
  * Materials Informaticspractice - MicrostructureDataベース構築と機械学習Modeling
  * Process Informaticspractice - 電顕Data駆動型プロセスOptimization
