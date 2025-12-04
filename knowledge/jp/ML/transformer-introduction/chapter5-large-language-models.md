---
title: 第5章：大規模言語モデル (Large Language Models)
chapter_title: 第5章：大規模言語モデル (Large Language Models)
subtitle: LLMのスケーリング則から実践的なプロンプトエンジニアリングまで
reading_time: 30-35分
difficulty: 中級〜上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 大規模言語モデル（LLM）のスケーリング則と性能特性を理解できる
  * ✅ GPT、LLaMA、Claude、Geminiなど主要なLLMアーキテクチャを比較できる
  * ✅ Zero-shot、Few-shot、Chain-of-Thoughtなどプロンプトエンジニアリング技術を実装できる
  * ✅ In-Context Learningのメカニズムと効果的な活用方法を理解できる
  * ✅ RLHFによる人間のフィードバックを活用したモデル改善手法を理解できる
  * ✅ 実用的なLLMアプリケーションを構築し、APIを統合できる

* * *

## 5.1 LLMのスケーリング則

### 大規模言語モデルとは

**大規模言語モデル（Large Language Model, LLM）** は、膨大なテキストデータで事前学習された巨大なTransformerベースのモデルです。数十億〜数千億のパラメータを持ち、様々な自然言語処理タスクを高精度で実行できます。
    
    
    ```mermaid
    graph TB
        A[言語モデルの進化] --> B[小規模モデル~100M params2018-2019]
        A --> C[中規模モデル1B-10B params2019-2020]
        A --> D[大規模モデル100B+ params2020-present]
    
        B --> B1[BERT Base110M]
        B --> B2[GPT-2117M-1.5B]
    
        C --> C1[GPT-3175B]
        C --> C2[T511B]
    
        D --> D1[GPT-4~1.7T推定]
        D --> D2[Claude 3非公開]
        D --> D3[Gemini Ultra非公開]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#fff9c4
        style D fill:#c8e6c9
    ```

### スケーリング則（Scaling Laws）

2020年にOpenAIが発表した**スケーリング則** は、モデルの性能がパラメータ数、データ量、計算量に対してどのようにスケールするかを定量化したものです。

#### 基本的なスケーリング則

モデルの損失（Loss）$L$は以下の3つの要因で決まります：

$$ L(N, D, C) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C} $$

ここで：

  * $N$: モデルのパラメータ数（非埋め込み層）
  * $D$: 訓練データのトークン数
  * $C$: 計算量（FLOPs）
  * $N_c, D_c, C_c$: スケーリング定数
  * $\alpha_N \approx 0.076, \alpha_D \approx 0.095, \alpha_C \approx 0.050$

スケーリング要因 | 影響 | 実用的な意味  
---|---|---  
**モデルサイズ（N）** | 10倍で損失が~0.95倍 | 大きいモデルほど性能向上  
**データ量（D）** | 10倍で損失が~0.93倍 | データの重要性が最も高い  
**計算量（C）** | 10倍で損失が~0.97倍 | 効率的な計算が鍵  
  
> **Chinchilla論文の発見（2022）** : DeepMindの研究により、多くのLLMは「過剰パラメータ化」されており、同じ計算予算でより小さいモデルをより多くのデータで訓練する方が効率的であることが判明しました。最適な比率は**データトークン数 ≈ 20 × パラメータ数** です。

### Emergent Abilities（創発能力）

LLMは一定規模を超えると、明示的に訓練されていない能力が**突然出現** します。これを**創発能力** と呼びます。
    
    
    ```mermaid
    graph LR
        A[モデルサイズの増加] --> B[~1B parameters]
        B --> C[基本的な文生成]
    
        A --> D[~10B parameters]
        D --> E[Few-shot Learning簡単な推論]
    
        A --> F[~100B parameters]
        F --> G[Chain-of-Thought複雑な推論指示追従]
    
        style A fill:#e3f2fd
        style G fill:#c8e6c9
    ```

創発能力 | 出現規模 | 説明  
---|---|---  
**In-Context Learning** | ~10B+ | 例示からタスクを学習  
**Chain-of-Thought** | ~100B+ | 段階的推論が可能  
**指示追従（Instruction Following）** | ~10B+（RLHF後） | 自然言語指示の理解  
**多言語能力** | ~10B+ | 未学習言語への転移  
      
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def scaling_law_loss(N, D, C, N_c=8.8e13, D_c=5.4e13, C_c=1.3e13,
                         alpha_N=0.076, alpha_D=0.095, alpha_C=0.050):
        """
        スケーリング則に基づいて損失を計算
    
        Args:
            N: パラメータ数
            D: データトークン数
            C: 計算量（FLOPs）
            その他: スケーリング定数
    
        Returns:
            予測される損失値
        """
        loss_N = (N_c / N) ** alpha_N
        loss_D = (D_c / D) ** alpha_D
        loss_C = (C_c / C) ** alpha_C
        return loss_N + loss_D + loss_C
    
    # モデルサイズの影響を可視化
    param_counts = np.logspace(6, 12, 50)  # 1M to 1T parameters
    data_tokens = 1e12  # 1T tokens固定
    compute = 1e21  # 固定
    
    losses = [scaling_law_loss(N, data_tokens, compute) for N in param_counts]
    
    plt.figure(figsize=(12, 5))
    
    # パラメータ数 vs 損失
    plt.subplot(1, 2, 1)
    plt.loglog(param_counts, losses, linewidth=2, color='#7b2cbf')
    plt.xlabel('パラメータ数', fontsize=12)
    plt.ylabel('損失（Loss）', fontsize=12)
    plt.title('スケーリング則: モデルサイズと性能', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 主要モデルをプロット
    models = {
        'GPT-2': (1.5e9, scaling_law_loss(1.5e9, data_tokens, compute)),
        'GPT-3': (175e9, scaling_law_loss(175e9, data_tokens, compute)),
        'GPT-4 (推定)': (1.7e12, scaling_law_loss(1.7e12, data_tokens, compute)),
    }
    
    for name, (params, loss) in models.items():
        plt.scatter(params, loss, s=100, zorder=5)
        plt.annotate(name, (params, loss), xytext=(10, 10),
                    textcoords='offset points', fontsize=9)
    
    # データ量の影響
    plt.subplot(1, 2, 2)
    data_amounts = np.logspace(9, 13, 50)
    model_size = 100e9  # 100B parameters固定
    
    losses_data = [scaling_law_loss(model_size, D, compute) for D in data_amounts]
    plt.loglog(data_amounts, losses_data, linewidth=2, color='#3182ce')
    plt.xlabel('訓練データトークン数', fontsize=12)
    plt.ylabel('損失（Loss）', fontsize=12)
    plt.title('スケーリング則: データ量と性能', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Chinchilla最適点をマーク
    optimal_data = 20 * model_size  # 20x rule
    optimal_loss = scaling_law_loss(model_size, optimal_data, compute)
    plt.scatter(optimal_data, optimal_loss, s=150, c='red', marker='*',
               zorder=5, label='Chinchilla最適点')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 実用例：必要な計算リソースの推定
    def estimate_training_cost(params, tokens, efficiency=0.5):
        """
        訓練コストを推定（FLOPs）
    
        Args:
            params: パラメータ数
            tokens: 訓練トークン数
            efficiency: ハードウェア効率（0-1）
    
        Returns:
            必要なFLOPs、GPU時間の推定
        """
        # 1トークンあたり約6 × params FLOPs
        total_flops = 6 * params * tokens
    
        # A100 GPU: ~312 TFLOPS（FP16）
        a100_flops = 312e12 * efficiency
        gpu_hours = total_flops / a100_flops / 3600
    
        return total_flops, gpu_hours
    
    # GPT-3クラスのモデルのコスト推定
    params_gpt3 = 175e9
    tokens_gpt3 = 300e9
    
    total_flops, gpu_hours = estimate_training_cost(params_gpt3, tokens_gpt3)
    
    print(f"\n=== GPT-3規模モデルの訓練コスト推定 ===")
    print(f"パラメータ数: {params_gpt3/1e9:.1f}B")
    print(f"訓練トークン数: {tokens_gpt3/1e9:.1f}B")
    print(f"総計算量: {total_flops:.2e} FLOPs")
    print(f"A100 GPU時間: {gpu_hours:,.0f} 時間 ({gpu_hours/24:,.0f} 日)")
    print(f"GPU数（30日で完了）: {int(np.ceil(gpu_hours / (24 * 30)))} 台")
    

* * *

## 5.2 代表的なLLM

### GPTシリーズ（OpenAI）

#### GPT-3（2020）

**GPT-3** （Generative Pre-trained Transformer 3）は、175Bパラメータを持つ自己回帰型言語モデルで、Few-shot Learningの有効性を実証しました。

特徴 | 詳細  
---|---  
**パラメータ数** | 175B（最大版）  
**アーキテクチャ** | Decoder-only Transformer、96層、12,288次元  
**訓練データ** | Common Crawl、WebText、Books、Wikipediaなど ~300B tokens  
**コンテキスト長** | 2,048 tokens  
**革新性** | Few-shot Learningの実証、プロンプトベースの汎用性  
  
#### GPT-4（2023）

**GPT-4** は、マルチモーダル（テキスト+画像）対応の最先端モデルです。詳細は非公開ですが、推定1.7兆パラメータとされています。

能力 | GPT-3.5 | GPT-4  
---|---|---  
**米国司法試験** | 下位10% | 上位10%  
**数学オリンピック** | 不合格 | 上位500位相当  
**コーディング能力** | 基本的な実装 | 複雑なアルゴリズム設計  
**マルチモーダル** | テキストのみ | テキスト+画像理解  
  
### LLaMAシリーズ（Meta）

**LLaMA** （Large Language Model Meta AI）は、Metaがオープンソースで公開した効率的なLLMファミリーです。

#### LLaMA 2の特徴

  * **モデルサイズ** : 7B、13B、70Bの3バリエーション
  * **訓練データ** : 2兆トークン（公開データのみ）
  * **コンテキスト長** : 4,096 tokens
  * **ライセンス** : 商用利用可能（条件付き）
  * **最適化** : Chinchillaスケーリング則に基づく効率的設計

    
    
    ```mermaid
    graph TB
        A[LLaMA 2アーキテクチャ] --> B[Pre-normalizationRMSNorm]
        A --> C[SwiGLU activationPaLMから採用]
        A --> D[Rotary PositionalEmbedding RoPE]
        A --> E[Grouped QueryAttention GQA]
    
        B --> F[訓練安定性向上]
        C --> G[性能向上]
        D --> H[長文対応]
        E --> I[推論高速化]
    
        style A fill:#e3f2fd
        style F fill:#c8e6c9
        style G fill:#c8e6c9
        style H fill:#c8e6c9
        style I fill:#c8e6c9
    ```

### Claude（Anthropic）

**Claude** は、Anthropicが開発した「Constitutional AI」アプローチによる安全性重視のLLMです。

モデル | 特徴 | コンテキスト長  
---|---|---  
**Claude 3 Opus** | 最高性能モデル、複雑な推論 | 200K tokens  
**Claude 3 Sonnet** | バランス型、高速応答 | 200K tokens  
**Claude 3 Haiku** | 軽量・高速、コスト効率 | 200K tokens  
  
> **Constitutional AI** : 人間のフィードバック（RLHF）に加えて、AIが自己批判・自己改善を行う「憲法」に基づいた訓練手法。有害な出力を減らし、より安全で有益な応答を生成します。

### Gemini（Google）

**Gemini** は、Googleが開発したマルチモーダルネイティブなLLMで、テキスト、画像、音声、動画を統合的に処理します。

  * **Gemini Ultra** : 最高性能、複雑なタスク対応
  * **Gemini Pro** : 汎用的な用途に最適化
  * **Gemini Nano** : デバイス上で動作する軽量版

    
    
    # 主要LLMの比較実装例（API経由）
    import os
    from typing import List, Dict
    import time
    
    class LLMComparison:
        """
        複数のLLM APIを統一インターフェースで比較
        """
    
        def __init__(self):
            """各APIキーを環境変数から取得"""
            self.openai_key = os.getenv('OPENAI_API_KEY')
            self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            self.google_key = os.getenv('GOOGLE_API_KEY')
    
        def query_gpt4(self, prompt: str, max_tokens: int = 500) -> Dict:
            """GPT-4にクエリを送信"""
            try:
                import openai
                openai.api_key = self.openai_key
    
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                latency = time.time() - start_time
    
                return {
                    'model': 'GPT-4',
                    'response': response.choices[0].message.content,
                    'tokens': response.usage.total_tokens,
                    'latency': latency
                }
            except Exception as e:
                return {'model': 'GPT-4', 'error': str(e)}
    
        def query_claude(self, prompt: str, max_tokens: int = 500) -> Dict:
            """Claude 3にクエリを送信"""
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_key)
    
                start_time = time.time()
                message = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                latency = time.time() - start_time
    
                return {
                    'model': 'Claude 3 Opus',
                    'response': message.content[0].text,
                    'tokens': message.usage.input_tokens + message.usage.output_tokens,
                    'latency': latency
                }
            except Exception as e:
                return {'model': 'Claude 3 Opus', 'error': str(e)}
    
        def query_gemini(self, prompt: str, max_tokens: int = 500) -> Dict:
            """Gemini Proにクエリを送信"""
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.google_key)
                model = genai.GenerativeModel('gemini-pro')
    
                start_time = time.time()
                response = model.generate_content(prompt)
                latency = time.time() - start_time
    
                return {
                    'model': 'Gemini Pro',
                    'response': response.text,
                    'tokens': 'N/A',  # Gemini APIは詳細なトークン数を返さない
                    'latency': latency
                }
            except Exception as e:
                return {'model': 'Gemini Pro', 'error': str(e)}
    
        def compare_all(self, prompt: str, max_tokens: int = 500) -> List[Dict]:
            """
            全モデルに同じプロンプトを送信して比較
    
            Args:
                prompt: 入力プロンプト
                max_tokens: 最大生成トークン数
    
            Returns:
                各モデルの応答結果リスト
            """
            results = []
    
            print(f"プロンプト: {prompt}\n")
            print("=" * 80)
    
            # GPT-4
            print("GPT-4にクエリ中...")
            gpt4_result = self.query_gpt4(prompt, max_tokens)
            results.append(gpt4_result)
            self._print_result(gpt4_result)
    
            # Claude 3
            print("\nClaude 3にクエリ中...")
            claude_result = self.query_claude(prompt, max_tokens)
            results.append(claude_result)
            self._print_result(claude_result)
    
            # Gemini
            print("\nGemini Proにクエリ中...")
            gemini_result = self.query_gemini(prompt, max_tokens)
            results.append(gemini_result)
            self._print_result(gemini_result)
    
            return results
    
        def _print_result(self, result: Dict):
            """結果を整形して表示"""
            if 'error' in result:
                print(f"❌ {result['model']}: エラー - {result['error']}")
            else:
                print(f"✅ {result['model']}:")
                print(f"   応答: {result['response'][:200]}...")
                print(f"   トークン数: {result['tokens']}")
                print(f"   レイテンシ: {result['latency']:.2f}秒")
    
    # 使用例
    if __name__ == "__main__":
        # 注意: 実行には各APIキーが必要です
        comparator = LLMComparison()
    
        # 簡単な比較テスト
        test_prompt = """
        以下の問題を段階的に解いてください：
    
        問題: ある会社の売上が毎年20%ずつ増加しています。
        現在の売上が1億円の場合、5年後の売上はいくらになりますか？
        計算過程を示してください。
        """
    
        results = comparator.compare_all(test_prompt, max_tokens=300)
    
        # パフォーマンス比較
        print("\n" + "=" * 80)
        print("パフォーマンス比較:")
        for result in results:
            if 'error' not in result:
                print(f"{result['model']:20} - レイテンシ: {result['latency']:.2f}秒")
    

* * *

## 5.3 Prompt Engineering

### プロンプトエンジニアリングとは

**プロンプトエンジニアリング** は、LLMから望ましい出力を引き出すための入力設計技術です。適切なプロンプトにより、モデルの再訓練なしに性能を大幅に向上させられます。
    
    
    ```mermaid
    graph LR
        A[プロンプト技術] --> B[Zero-shot]
        A --> C[Few-shot]
        A --> D[Chain-of-Thought]
        A --> E[Self-Consistency]
    
        B --> B1[指示のみで実行]
        C --> C1[例示から学習]
        D --> D1[段階的推論]
        E --> E1[複数経路の統合]
    
        style A fill:#e3f2fd
        style D fill:#c8e6c9
    ```

### Zero-shot Learning

**Zero-shot Learning** は、例示なしに指示のみでタスクを実行させる手法です。大規模モデルの創発能力により可能になりました。
    
    
    class ZeroShotPromptEngine:
        """
        Zero-shotプロンプトの設計と実行
        """
    
        @staticmethod
        def sentiment_analysis(text: str) -> str:
            """感情分析のZero-shotプロンプト"""
            prompt = f"""
    以下のテキストの感情を「ポジティブ」「ネガティブ」「中立」のいずれかに分類してください。
    分類結果のみを返してください。
    
    テキスト: {text}
    
    分類:"""
            return prompt
    
        @staticmethod
        def text_summarization(text: str, max_words: int = 50) -> str:
            """要約のZero-shotプロンプト"""
            prompt = f"""
    以下のテキストを{max_words}語以内で要約してください。
    重要なポイントを簡潔にまとめてください。
    
    テキスト:
    {text}
    
    要約:"""
            return prompt
    
        @staticmethod
        def question_answering(context: str, question: str) -> str:
            """質問応答のZero-shotプロンプト"""
            prompt = f"""
    以下の文脈に基づいて質問に答えてください。
    文脈に情報がない場合は「情報が不足しています」と答えてください。
    
    文脈:
    {context}
    
    質問: {question}
    
    回答:"""
            return prompt
    
        @staticmethod
        def language_translation(text: str, target_lang: str = "英語") -> str:
            """翻訳のZero-shotプロンプト"""
            prompt = f"""
    以下のテキストを{target_lang}に翻訳してください。
    自然で正確な翻訳を心がけてください。
    
    テキスト: {text}
    
    翻訳:"""
            return prompt
    
    # 使用例
    engine = ZeroShotPromptEngine()
    
    # 感情分析
    text1 = "この製品は期待以上の性能でした。購入して本当に良かったです。"
    prompt1 = engine.sentiment_analysis(text1)
    print("=== Zero-shot 感情分析 ===")
    print(prompt1)
    print()
    
    # 要約
    text2 = """
    人工知能（AI）の発展により、様々な分野で革新が起きています。
    特に自然言語処理の分野では、大規模言語モデル（LLM）の登場により、
    機械翻訳、文章生成、質問応答などのタスクで人間に近い性能を達成しています。
    今後、AIは医療、教育、ビジネスなど、さらに多くの領域で活用されていくでしょう。
    """
    prompt2 = engine.text_summarization(text2, max_words=30)
    print("=== Zero-shot 要約 ===")
    print(prompt2)
    

### Few-shot Learning

**Few-shot Learning** は、少数の例示（通常1〜10個）を提示することで、モデルにタスクのパターンを学習させる手法です。
    
    
    class FewShotPromptEngine:
        """
        Few-shotプロンプトの設計と実行
        """
    
        @staticmethod
        def sentiment_analysis(text: str, num_examples: int = 3) -> str:
            """感情分析のFew-shotプロンプト"""
            # 例示データ
            examples = [
                ("この映画は素晴らしかった！感動しました。", "ポジティブ"),
                ("料理が冷めていて、サービスも悪かった。", "ネガティブ"),
                ("普通の商品です。特に良くも悪くもありません。", "中立"),
            ]
    
            # プロンプト構築
            prompt = "以下の例を参考に、テキストの感情を分類してください。\n\n"
    
            for i, (example_text, label) in enumerate(examples[:num_examples], 1):
                prompt += f"例{i}:\nテキスト: {example_text}\n分類: {label}\n\n"
    
            prompt += f"テキスト: {text}\n分類:"
            return prompt
    
        @staticmethod
        def entity_extraction(text: str) -> str:
            """固有表現抽出のFew-shotプロンプト"""
            prompt = """
    以下の例を参考に、テキストから人名、組織名、場所を抽出してください。
    
    例1:
    テキスト: 田中太郎さんは東京大学で機械学習を研究しています。
    抽出: 人名=田中太郎, 組織名=東京大学, 場所=なし
    
    例2:
    テキスト: AppleのCEOティム・クックがシリコンバレーで講演しました。
    抽出: 人名=ティム・クック, 組織名=Apple, 場所=シリコンバレー
    
    例3:
    テキスト: GoogleとMicrosoftが新しいAI技術を発表しました。
    抽出: 人名=なし, 組織名=Google, Microsoft, 場所=なし
    
    テキスト: {text}
    抽出:"""
            return prompt.format(text=text)
    
        @staticmethod
        def code_generation(task_description: str) -> str:
            """コード生成のFew-shotプロンプト"""
            prompt = """
    以下の例を参考に、タスクを実行するPython関数を生成してください。
    
    例1:
    タスク: リストの要素を2倍にする
    コード:
    def double_elements(lst):
        return [x * 2 for x in lst]
    
    例2:
    タスク: 文字列を逆順にする
    コード:
    def reverse_string(s):
        return s[::-1]
    
    例3:
    タスク: リストの平均値を計算する
    コード:
    def calculate_average(lst):
        return sum(lst) / len(lst) if lst else 0
    
    タスク: {task}
    コード:"""
            return prompt.format(task=task_description)
    
        @staticmethod
        def analogical_reasoning(question: str) -> str:
            """類推推論のFew-shotプロンプト"""
            prompt = """
    以下のパターンを理解し、質問に答えてください。
    
    例1: 東京:日本 = パリ:?
    答え: フランス
    理由: 東京が日本の首都であるように、パリはフランスの首都です。
    
    例2: 医者:病院 = 教師:?
    答え: 学校
    理由: 医者が病院で働くように、教師は学校で働きます。
    
    例3: 犬:哺乳類 = 鷹:?
    答え: 鳥類
    理由: 犬が哺乳類に属するように、鷹は鳥類に属します。
    
    質問: {question}
    答え:"""
            return prompt.format(question=question)
    
    # 使用例
    few_shot_engine = FewShotPromptEngine()
    
    # Few-shot感情分析
    test_text = "期待していたほどではなかったが、まあまあです。"
    prompt = few_shot_engine.sentiment_analysis(test_text)
    print("=== Few-shot 感情分析 ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # Few-shot固有表現抽出
    test_entity = "NHKの記者がニューヨークで山田花子氏にインタビューした。"
    prompt = few_shot_engine.entity_extraction(test_entity)
    print("=== Few-shot 固有表現抽出 ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # Few-shot類推推論
    test_analogy = "本:著者 = 映画:?"
    prompt = few_shot_engine.analogical_reasoning(test_analogy)
    print("=== Few-shot 類推推論 ===")
    print(prompt)
    

### Chain-of-Thought (CoT) Prompting

**Chain-of-Thought（思考の連鎖）** は、モデルに段階的な推論過程を生成させることで、複雑な問題の精度を向上させる手法です。

$$ \text{Accuracy}_{\text{CoT}} \approx \text{Accuracy}_{\text{standard}} + \Delta_{\text{reasoning}} $$

ここで、$\Delta_{\text{reasoning}}$は推論による精度向上分で、複雑な問題ほど大きくなります。
    
    
    class ChainOfThoughtEngine:
        """
        Chain-of-Thought (CoT) プロンプトエンジニアリング
        """
    
        @staticmethod
        def math_problem_basic(problem: str) -> str:
            """数学問題の基本的なCoT"""
            prompt = f"""
    以下の問題を段階的に解いてください。
    各ステップで何を計算しているか説明してください。
    
    問題: {problem}
    
    解答:
    ステップ1:"""
            return prompt
    
        @staticmethod
        def math_problem_with_examples(problem: str) -> str:
            """Few-shot CoTの例"""
            prompt = """
    以下の例を参考に、問題を段階的に解いてください。
    
    例題1:
    問題: りんごが15個、みかんが23個あります。果物は全部で何個ですか？
    解答:
    ステップ1: りんごの数を確認 → 15個
    ステップ2: みかんの数を確認 → 23個
    ステップ3: 合計を計算 → 15 + 23 = 38個
    答え: 38個
    
    例題2:
    問題: 1冊500円の本を3冊と、1本120円のペンを5本買いました。合計金額は？
    解答:
    ステップ1: 本の合計金額を計算 → 500円 × 3冊 = 1,500円
    ステップ2: ペンの合計金額を計算 → 120円 × 5本 = 600円
    ステップ3: 総合計を計算 → 1,500円 + 600円 = 2,100円
    答え: 2,100円
    
    問題: {problem}
    解答:
    ステップ1:"""
            return prompt.format(problem=problem)
    
        @staticmethod
        def logical_reasoning(scenario: str, question: str) -> str:
            """論理的推論のCoT"""
            prompt = f"""
    以下の状況を段階的に分析し、論理的に結論を導いてください。
    
    状況: {scenario}
    
    質問: {question}
    
    分析:
    観察1:"""
            return prompt
    
        @staticmethod
        def self_consistency_cot(problem: str, num_paths: int = 3) -> str:
            """
            Self-Consistency CoT: 複数の推論経路を生成し、
            最も一貫性のある答えを選択
            """
            prompt = f"""
    以下の問題を{num_paths}つの異なるアプローチで解いてください。
    各アプローチで段階的に推論し、最終的に最も確実な答えを選んでください。
    
    問題: {problem}
    
    アプローチ1:
    """
            return prompt
    
    # 実践例：複雑な数学問題
    cot_engine = ChainOfThoughtEngine()
    
    problem1 = """
    ある店舗の売上が1年目は100万円でした。
    2年目は前年比20%増、3年目は前年比15%減、4年目は前年比25%増でした。
    4年目の売上は何万円ですか？
    """
    
    print("=== Chain-of-Thought: 数学問題 ===")
    prompt1 = cot_engine.math_problem_with_examples(problem1)
    print(prompt1)
    print("\n" + "="*80 + "\n")
    
    # 論理的推論の例
    scenario = """
    会議室A、B、Cがあります。
    - 田中さんは会議室Aにいません
    - 佐藤さんは会議室Bにいます
    - 会議室Cには誰もいません
    - 田中さんと佐藤さん以外に山田さんがいます
    """
    
    question = "山田さんはどの会議室にいますか？"
    
    print("=== Chain-of-Thought: 論理的推論 ===")
    prompt2 = cot_engine.logical_reasoning(scenario, question)
    print(prompt2)
    print("\n" + "="*80 + "\n")
    
    # Self-Consistency CoT
    problem2 = """
    袋の中に赤い玉が5個、青い玉が3個入っています。
    2個の玉を同時に取り出すとき、両方とも赤い玉である確率は？
    """
    
    print("=== Self-Consistency CoT ===")
    prompt3 = cot_engine.self_consistency_cot(problem2, num_paths=3)
    print(prompt3)
    

> **CoTの効果** : GoogleのPaLMモデルでの実験では、標準プロンプトで34%だった算数問題の正答率が、CoTプロンプトで79%まで向上しました。特に多段階推論が必要な問題で大きな改善が見られます。

### プロンプトテンプレート設計
    
    
    from typing import Dict, List, Optional
    from dataclasses import dataclass
    
    @dataclass
    class PromptTemplate:
        """プロンプトテンプレートの構造化管理"""
        name: str
        instruction: str
        examples: Optional[List[Dict[str, str]]] = None
        output_format: Optional[str] = None
    
        def render(self, **kwargs) -> str:
            """テンプレートを実際のプロンプトに変換"""
            prompt = f"{self.instruction}\n\n"
    
            # Few-shot例を追加
            if self.examples:
                prompt += "例:\n"
                for i, example in enumerate(self.examples, 1):
                    prompt += f"\n例{i}:\n"
                    for key, value in example.items():
                        prompt += f"{key}: {value}\n"
    
            # 出力フォーマットを追加
            if self.output_format:
                prompt += f"\n出力形式:\n{self.output_format}\n"
    
            # 変数を挿入
            prompt += "\n入力:\n"
            for key, value in kwargs.items():
                prompt += f"{key}: {value}\n"
    
            prompt += "\n出力:"
            return prompt
    
    class PromptLibrary:
        """再利用可能なプロンプトテンプレートライブラリ"""
    
        @staticmethod
        def get_classification_template() -> PromptTemplate:
            """分類タスク用テンプレート"""
            return PromptTemplate(
                name="classification",
                instruction="以下のテキストを指定されたカテゴリに分類してください。",
                examples=[
                    {
                        "テキスト": "新しいスマートフォンが発売されました。",
                        "カテゴリ": "テクノロジー"
                    },
                    {
                        "テキスト": "株価が急騰しています。",
                        "カテゴリ": "ビジネス"
                    }
                ],
                output_format="カテゴリ名のみを返してください。"
            )
    
        @staticmethod
        def get_extraction_template() -> PromptTemplate:
            """情報抽出用テンプレート"""
            return PromptTemplate(
                name="extraction",
                instruction="テキストから指定された情報を抽出してください。",
                output_format="JSON形式で返してください: {\"項目1\": \"値1\", \"項目2\": \"値2\"}"
            )
    
        @staticmethod
        def get_generation_template() -> PromptTemplate:
            """生成タスク用テンプレート"""
            return PromptTemplate(
                name="generation",
                instruction="以下の条件に基づいて、創造的なコンテンツを生成してください。",
                output_format="自然で読みやすい文章で出力してください。"
            )
    
        @staticmethod
        def get_reasoning_template() -> PromptTemplate:
            """推論タスク用テンプレート"""
            return PromptTemplate(
                name="reasoning",
                instruction="""
    以下の問題を段階的に分析し、論理的に解決してください。
    各ステップで推論過程を明確に示してください。
                """,
                output_format="""
    ステップ1: [最初の分析]
    ステップ2: [次の推論]
    ...
    結論: [最終的な答え]
                """
            )
    
    # 使用例
    library = PromptLibrary()
    
    # 分類タスク
    classification_template = library.get_classification_template()
    prompt = classification_template.render(
        テキスト="人工知能の研究が加速しています。",
        カテゴリ="テクノロジー, ビジネス, 政治, スポーツ, エンターテイメント"
    )
    print("=== 分類プロンプト ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # 情報抽出タスク
    extraction_template = library.get_extraction_template()
    prompt = extraction_template.render(
        テキスト="田中太郎（35歳）はABC株式会社のCTOで、東京在住です。",
        抽出項目="名前, 年齢, 会社名, 役職, 居住地"
    )
    print("=== 抽出プロンプト ===")
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    # 推論タスク
    reasoning_template = library.get_reasoning_template()
    prompt = reasoning_template.render(
        問題="3つの箱A, B, Cがあり、1つだけに宝物が入っています。箱Aには「宝物はここにある」、箱Bには「宝物はAにない」、箱Cには「宝物はBにある」と書かれています。1つだけが真実を述べている場合、宝物はどの箱にありますか？"
    )
    print("=== 推論プロンプト ===")
    print(prompt)
    

* * *

## 5.4 In-Context Learning

### In-Context Learningのメカニズム

**In-Context Learning (ICL)** は、LLMがプロンプト内の例示から直接学習し、パラメータ更新なしに新しいタスクを実行する能力です。
    
    
    ```mermaid
    graph TB
        A[In-Context Learning] --> B[入力プロンプト]
        B --> C[タスク指示]
        B --> D[Few-shot例示]
        B --> E[クエリ]
    
        C --> F[TransformerのSelf-Attention]
        D --> F
        E --> F
    
        F --> G[コンテキスト内でパターン学習]
        G --> H[出力生成]
    
        style A fill:#e3f2fd
        style F fill:#fff3e0
        style G fill:#c8e6c9
    ```

#### ICLがなぜ機能するか

最近の研究（2023）により、ICLは以下のメカニズムで動作することが判明しました：

  1. **潜在的な概念の活性化** : 事前学習で獲得した知識が例示により活性化される
  2. **タスクベクトルの形成** : 例示から抽出されたパターンが内部表現として保持される
  3. **アナロジーベースの推論** : 新しい入力を例示とのアナロジーで処理する

$$ P(y|x, \text{examples}) \approx \sum_{i=1}^{k} \alpha_i \cdot P(y|x, \text{example}_i) $$

ここで、$\alpha_i$は各例示の関連度重みです。
    
    
    import numpy as np
    from typing import List, Tuple, Dict
    
    class InContextLearningSimulator:
        """
        In-Context Learningの動作をシミュレート
        （簡略化されたメカニズムの再現）
        """
    
        def __init__(self, embedding_dim: int = 128):
            """
            Args:
                embedding_dim: 埋め込みベクトルの次元数
            """
            self.embedding_dim = embedding_dim
            self.task_vector = None
    
        def create_example_embedding(self, input_text: str, output_text: str) -> np.ndarray:
            """
            入力-出力ペアから例示埋め込みを作成
            （実際にはTransformerで処理されるが、ここでは簡略化）
            """
            # 簡略化: 文字列をハッシュして埋め込みに変換
            input_hash = hash(input_text) % 10000
            output_hash = hash(output_text) % 10000
    
            np.random.seed(input_hash)
            input_emb = np.random.randn(self.embedding_dim)
    
            np.random.seed(output_hash)
            output_emb = np.random.randn(self.embedding_dim)
    
            # タスクベクトル: 出力 - 入力の差分
            task_emb = output_emb - input_emb
            return task_emb / (np.linalg.norm(task_emb) + 1e-8)
    
        def learn_from_examples(self, examples: List[Tuple[str, str]]) -> None:
            """
            Few-shot例示からタスクベクトルを学習
    
            Args:
                examples: [(入力1, 出力1), (入力2, 出力2), ...]
            """
            task_vectors = []
    
            for input_text, output_text in examples:
                task_vec = self.create_example_embedding(input_text, output_text)
                task_vectors.append(task_vec)
    
            # 複数例の平均としてタスク表現を獲得
            self.task_vector = np.mean(task_vectors, axis=0)
            self.task_vector /= (np.linalg.norm(self.task_vector) + 1e-8)
    
        def predict(self, query: str, candidates: List[str]) -> Dict[str, float]:
            """
            学習したタスクベクトルに基づいて予測
    
            Args:
                query: 入力クエリ
                candidates: 候補となる出力リスト
    
            Returns:
                各候補のスコア辞書
            """
            if self.task_vector is None:
                raise ValueError("先にlearn_from_examples()を呼び出してください")
    
            # クエリの埋め込み
            query_hash = hash(query) % 10000
            np.random.seed(query_hash)
            query_emb = np.random.randn(self.embedding_dim)
            query_emb /= (np.linalg.norm(query_emb) + 1e-8)
    
            # 各候補のスコア計算
            scores = {}
            for candidate in candidates:
                candidate_hash = hash(candidate) % 10000
                np.random.seed(candidate_hash)
                candidate_emb = np.random.randn(self.embedding_dim)
                candidate_emb /= (np.linalg.norm(candidate_emb) + 1e-8)
    
                # タスクベクトルとの整合性
                predicted_output = query_emb + self.task_vector
                similarity = np.dot(predicted_output, candidate_emb)
                scores[candidate] = float(similarity)
    
            # スコアを正規化
            total = sum(np.exp(s) for s in scores.values())
            scores = {k: np.exp(v)/total for k, v in scores.items()}
    
            return scores
    
    # 使用例：感情分析タスク
    print("=== In-Context Learning シミュレーション ===\n")
    
    simulator = InContextLearningSimulator(embedding_dim=128)
    
    # Few-shot例示
    examples = [
        ("この映画は素晴らしかった！", "ポジティブ"),
        ("最悪の体験でした。二度と行きません。", "ネガティブ"),
        ("普通です。特に印象に残りませんでした。", "中立"),
        ("期待以上の品質で大満足です！", "ポジティブ"),
        ("ひどいサービスにがっかりしました。", "ネガティブ"),
    ]
    
    # タスクベクトルを学習
    simulator.learn_from_examples(examples)
    print("✅ Few-shot例示から学習完了\n")
    
    # 新しい入力で予測
    test_queries = [
        "素晴らしい製品です。おすすめします。",
        "期待外れでした。",
        "まあまあの出来栄えです。"
    ]
    
    candidates = ["ポジティブ", "ネガティブ", "中立"]
    
    for query in test_queries:
        print(f"クエリ: {query}")
        scores = simulator.predict(query, candidates)
    
        # スコア順にソート
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
        for label, score in sorted_scores:
            print(f"  {label}: {score:.4f} {'★' * int(score * 10)}")
        print(f"  → 予測: {sorted_scores[0][0]}\n")
    

### ICLの効果的な活用

#### 例示の選択戦略

戦略 | 方法 | 利点  
---|---|---  
**ランダム選択** | 訓練データからランダムに選ぶ | シンプル、バイアスが少ない  
**類似度ベース** | クエリに類似した例を選ぶ | タスク性能が高い  
**多様性重視** | 多様な例を含める | 汎化性能が向上  
**難易度調整** | 簡単→難しい順に並べる | 学習効率が良い  
  
* * *

## 5.5 RLHF（Reinforcement Learning from Human Feedback）

### RLHFとは

**RLHF** は、人間のフィードバックを活用してLLMを改善する手法です。ChatGPTやClaude、Geminiなど、ほぼすべての商用LLMで採用されています。
    
    
    ```mermaid
    graph TB
        A[RLHF 3段階プロセス] --> B[Step 1事前学習済みLLM]
    
        B --> C[Step 2Reward Model訓練]
        C --> C1[人間による応答の評価]
        C --> C2[好ましい/好ましくない応答ペアを作成]
        C --> C3[Reward Model学習]
    
        C3 --> D[Step 3PPOによる強化学習]
        D --> D1[LLMが応答生成]
        D --> D2[Reward Modelがスコアリング]
        D --> D3[高スコア応答を強化]
    
        D3 --> E[最適化されたLLM]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#fff9c4
        style E fill:#c8e6c9
    ```

### RLHFの3ステップ

#### Step 1: 事前学習（Pre-training）

大規模コーパスでTransformerを訓練し、言語の基本的なパターンを学習します。

$$ \mathcal{L}_{\text{pretrain}} = -\mathbb{E}_{x \sim \mathcal{D}} \left[\log P_{\theta}(x)\right] $$

#### Step 2: Reward Modelの訓練

人間の評価者が複数の応答を比較し、好ましさをランク付けします。このデータから**Reward Model** を訓練します。

$$ \mathcal{L}_{\text{reward}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l))\right] $$

ここで：

  * $r_{\phi}$: Reward Model（パラメータ$\phi$）
  * $y_w$: 好ましい応答（winner）
  * $y_l$: 好ましくない応答（loser）
  * $\sigma$: シグモイド関数

#### Step 3: PPOによる強化学習

**PPO（Proximal Policy Optimization）** アルゴリズムでLLMを最適化します。KLダイバージェンス制約により、元の事前学習モデルから大きく逸脱しないようにします。

$$ \mathcal{L}_{\text{RLHF}} = \mathbb{E}_{x, y} \left[r_{\phi}(x, y) - \beta \cdot D_{KL}(\pi_{\theta} || \pi_{\text{ref}})\right] $$

ここで：

  * $\pi_{\theta}$: 最適化中のポリシー（LLM）
  * $\pi_{\text{ref}}$: 参照ポリシー（元のモデル）
  * $\beta$: KL制約の強さ

    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    
    class RewardModel(nn.Module):
        """
        RLHF用のReward Model
        """
    
        def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
            """
            Args:
                input_dim: 入力埋め込み次元（例: BERT埋め込み）
                hidden_dim: 隠れ層の次元
            """
            super().__init__()
    
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)  # スカラー報酬を出力
            )
    
        def forward(self, embeddings):
            """
            Args:
                embeddings: [batch_size, seq_len, input_dim]
    
            Returns:
                rewards: [batch_size] スカラー報酬
            """
            # 平均プーリング
            pooled = embeddings.mean(dim=1)  # [batch_size, input_dim]
            rewards = self.network(pooled).squeeze(-1)  # [batch_size]
            return rewards
    
    class RLHFTrainer:
        """
        RLHF訓練プロセスのシミュレーション
        """
    
        def __init__(self, reward_model: RewardModel, beta: float = 0.1):
            """
            Args:
                reward_model: 訓練済みReward Model
                beta: KLダイバージェンス制約の強さ
            """
            self.reward_model = reward_model
            self.beta = beta
    
        def compute_reward(self, response_embeddings: torch.Tensor) -> torch.Tensor:
            """
            応答に対する報酬を計算
    
            Args:
                response_embeddings: [batch_size, seq_len, embed_dim]
    
            Returns:
                rewards: [batch_size]
            """
            with torch.no_grad():
                rewards = self.reward_model(response_embeddings)
            return rewards
    
        def compute_kl_penalty(self,
                              current_logprobs: torch.Tensor,
                              reference_logprobs: torch.Tensor) -> torch.Tensor:
            """
            KLダイバージェンスペナルティを計算
    
            Args:
                current_logprobs: 現在のモデルのlog確率
                reference_logprobs: 参照モデルのlog確率
    
            Returns:
                kl_penalty: KLダイバージェンス
            """
            kl = current_logprobs - reference_logprobs
            return kl.mean()
    
        def ppo_loss(self,
                     old_logprobs: torch.Tensor,
                     new_logprobs: torch.Tensor,
                     advantages: torch.Tensor,
                     epsilon: float = 0.2) -> torch.Tensor:
            """
            PPO（Proximal Policy Optimization）損失を計算
    
            Args:
                old_logprobs: 古いポリシーのlog確率
                new_logprobs: 新しいポリシーのlog確率
                advantages: アドバンテージ（報酬 - ベースライン）
                epsilon: クリッピング範囲
    
            Returns:
                ppo_loss: PPO損失
            """
            # 確率比
            ratio = torch.exp(new_logprobs - old_logprobs)
    
            # クリップされた目的関数
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
            # PPO損失（最小化）
            loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
    
            return loss
    
    # 使用例
    print("=== RLHF Reward Model デモ ===\n")
    
    # Reward Modelの初期化
    reward_model = RewardModel(input_dim=768, hidden_dim=256)
    
    # ダミーデータ: 2つの応答の埋め込み
    torch.manual_seed(42)
    response1_emb = torch.randn(1, 20, 768)  # 好ましい応答
    response2_emb = torch.randn(1, 20, 768)  # 好ましくない応答
    
    # 報酬を計算
    reward1 = reward_model(response1_emb)
    reward2 = reward_model(response2_emb)
    
    print(f"応答1の報酬: {reward1.item():.4f}")
    print(f"応答2の報酬: {reward2.item():.4f}")
    
    # ペアワイズ損失の計算（訓練時）
    pairwise_loss = -F.logsigmoid(reward1 - reward2).mean()
    print(f"\nペアワイズランキング損失: {pairwise_loss.item():.4f}")
    
    # RLHFトレーナーの初期化
    trainer = RLHFTrainer(reward_model, beta=0.1)
    
    # PPO損失の計算例
    old_logprobs = torch.randn(16)  # [batch_size]
    new_logprobs = old_logprobs + torch.randn(16) * 0.1
    advantages = torch.randn(16)
    
    ppo_loss = trainer.ppo_loss(old_logprobs, new_logprobs, advantages)
    print(f"\nPPO損失: {ppo_loss.item():.4f}")
    
    # KLペナルティの計算
    kl_penalty = trainer.compute_kl_penalty(new_logprobs, old_logprobs)
    print(f"KLペナルティ: {kl_penalty.item():.4f}")
    
    # 総合目的関数
    total_loss = ppo_loss + trainer.beta * kl_penalty
    print(f"\n総合損失（PPO + KL制約）: {total_loss.item():.4f}")
    

### RLHFの課題と改善

課題 | 説明 | 解決策  
---|---|---  
**報酬ハッキング** | モデルが報酬を最大化する非意図的な方法を見つける | 多様な評価者、KL制約の調整  
**評価者のバイアス** | 人間の評価に一貫性がない | 複数評価者の合意、ガイドライン整備  
**計算コスト** | PPO訓練は計算量が大きい | DPO（Direct Preference Optimization）などの代替手法  
**過度の安全性** | 過剰に慎重な応答になる | 報酬モデルの微調整  
  
* * *

## 5.6 実践プロジェクト

### プロジェクト1: Few-shot テキスト分類システム
    
    
    import openai
    import os
    from typing import List, Dict, Tuple
    from collections import Counter
    
    class FewShotClassifier:
        """
        Few-shot Learning による汎用テキスト分類器
        """
    
        def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
            """
            Args:
                api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
                model: 使用するモデル名
            """
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model
            openai.api_key = self.api_key
    
        def create_few_shot_prompt(self,
                                   examples: List[Tuple[str, str]],
                                   query: str,
                                   labels: List[str]) -> str:
            """
            Few-shotプロンプトを構築
    
            Args:
                examples: [(テキスト1, ラベル1), (テキスト2, ラベル2), ...]
                query: 分類対象のテキスト
                labels: 可能なラベルのリスト
    
            Returns:
                構築されたプロンプト
            """
            prompt = "以下は文章をカテゴリに分類するタスクです。\n\n"
            prompt += f"利用可能なカテゴリ: {', '.join(labels)}\n\n"
    
            # Few-shot例を追加
            for i, (text, label) in enumerate(examples, 1):
                prompt += f"例{i}:\n"
                prompt += f"文章: {text}\n"
                prompt += f"カテゴリ: {label}\n\n"
    
            # クエリを追加
            prompt += f"次の文章を分類してください：\n"
            prompt += f"文章: {query}\n"
            prompt += f"カテゴリ:"
    
            return prompt
    
        def classify(self,
                    query: str,
                    examples: List[Tuple[str, str]],
                    labels: List[str],
                    temperature: float = 0.3) -> Dict[str, any]:
            """
            テキストを分類
    
            Args:
                query: 分類対象のテキスト
                examples: Few-shot例
                labels: 可能なラベル
                temperature: 生成の多様性（0-1）
    
            Returns:
                分類結果辞書
            """
            prompt = self.create_few_shot_prompt(examples, query, labels)
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=50
                )
    
                predicted_label = response.choices[0].message.content.strip()
    
                return {
                    'query': query,
                    'predicted_label': predicted_label,
                    'prompt': prompt,
                    'success': True
                }
    
            except Exception as e:
                return {
                    'query': query,
                    'error': str(e),
                    'success': False
                }
    
        def batch_classify(self,
                          queries: List[str],
                          examples: List[Tuple[str, str]],
                          labels: List[str]) -> List[Dict]:
            """
            複数のテキストを一括分類
    
            Args:
                queries: 分類対象のテキストリスト
                examples: Few-shot例
                labels: 可能なラベル
    
            Returns:
                分類結果のリスト
            """
            results = []
            for query in queries:
                result = self.classify(query, examples, labels)
                results.append(result)
    
            return results
    
    # 使用例: ニュース記事の分類
    print("=== Few-shot テキスト分類デモ ===\n")
    
    # 注意: 実行にはOpenAI APIキーが必要です
    # classifier = FewShotClassifier()
    
    # Few-shot訓練例
    examples = [
        ("新型スマートフォンが発売され、予約が殺到しています。", "テクノロジー"),
        ("株式市場が急騰し、過去最高値を更新しました。", "ビジネス"),
        ("サッカーワールドカップで日本代表が勝利しました。", "スポーツ"),
        ("新しい映画が公開され、興行収入記録を樹立しました。", "エンターテイメント"),
        ("政府が新しい経済政策を発表しました。", "政治"),
    ]
    
    # ラベル
    labels = ["テクノロジー", "ビジネス", "スポーツ", "エンターテイメント", "政治"]
    
    # テストクエリ
    test_queries = [
        "人工知能の研究開発に巨額の投資が行われています。",
        "プロ野球の優勝チームが決定しました。",
        "新作ゲームが世界的にヒットしています。",
    ]
    
    # 分類実行（デモ用の擬似コード）
    print("Few-shot例:")
    for text, label in examples:
        print(f"  [{label}] {text}")
    
    print("\n分類対象:")
    for query in test_queries:
        print(f"  - {query}")
    
    # 実際のAPI呼び出しはコメントアウト
    # results = classifier.batch_classify(test_queries, examples, labels)
    #
    # print("\n結果:")
    # for result in results:
    #     if result['success']:
    #         print(f"✅ [{result['predicted_label']}] {result['query']}")
    #     else:
    #         print(f"❌ エラー: {result['error']}")
    

### プロジェクト2: Chain-of-Thought 推論エンジン
    
    
    class ChainOfThoughtReasoner:
        """
        Chain-of-Thought推論を実装した問題解決エンジン
        """
    
        def __init__(self, api_key: str = None, model: str = "gpt-4"):
            """
            Args:
                api_key: OpenAI APIキー
                model: 使用するモデル（CoTにはGPT-4推奨）
            """
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model
            openai.api_key = self.api_key
    
        def solve_math_problem(self, problem: str) -> Dict[str, any]:
            """
            数学問題をCoTで解く
    
            Args:
                problem: 問題文
    
            Returns:
                解答と推論過程
            """
            prompt = f"""
    以下の数学問題を段階的に解いてください。
    各ステップで何を計算しているか明確に説明してください。
    
    問題: {problem}
    
    解答手順:
    ステップ1:"""
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
    
                reasoning = response.choices[0].message.content
    
                # 最終答えを抽出（簡易版）
                lines = reasoning.split('\n')
                answer_line = [l for l in lines if '答え' in l or '解答' in l]
                final_answer = answer_line[-1] if answer_line else "抽出失敗"
    
                return {
                    'problem': problem,
                    'reasoning': reasoning,
                    'final_answer': final_answer,
                    'success': True
                }
    
            except Exception as e:
                return {
                    'problem': problem,
                    'error': str(e),
                    'success': False
                }
    
        def solve_logic_puzzle(self, puzzle: str, question: str) -> Dict[str, any]:
            """
            論理パズルをCoTで解く
    
            Args:
                puzzle: パズルの状況説明
                question: 解くべき質問
    
            Returns:
                解答と推論過程
            """
            prompt = f"""
    以下の論理パズルを段階的に分析して解いてください。
    
    状況:
    {puzzle}
    
    質問: {question}
    
    分析手順:
    観察1:"""
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=600
                )
    
                reasoning = response.choices[0].message.content
    
                return {
                    'puzzle': puzzle,
                    'question': question,
                    'reasoning': reasoning,
                    'success': True
                }
    
            except Exception as e:
                return {
                    'puzzle': puzzle,
                    'error': str(e),
                    'success': False
                }
    
    # 使用例
    print("=== Chain-of-Thought 推論エンジン デモ ===\n")
    
    # reasoner = ChainOfThoughtReasoner()
    
    # 数学問題
    math_problem = """
    ある商品の定価は10,000円です。
    セールで20%オフになり、さらにクーポンで500円引きされました。
    最終的な支払額を計算してください。
    """
    
    print("数学問題:")
    print(math_problem)
    print("\n期待される推論:")
    print("ステップ1: 20%オフの金額を計算 → 10,000 × 0.2 = 2,000円")
    print("ステップ2: セール後の価格 → 10,000 - 2,000 = 8,000円")
    print("ステップ3: クーポン適用 → 8,000 - 500 = 7,500円")
    print("答え: 7,500円")
    
    # result = reasoner.solve_math_problem(math_problem)
    # if result['success']:
    #     print("\nCoT推論結果:")
    #     print(result['reasoning'])
    

### プロジェクト3: 統合型LLMチャットボットシステム
    
    
    from datetime import datetime
    from typing import Optional
    
    class LLMChatbot:
        """
        複数のプロンプト技術を統合したチャットボット
        """
    
        def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model
            self.conversation_history = []
            openai.api_key = self.api_key
    
        def set_system_prompt(self, persona: str, capabilities: List[str]):
            """
            システムプロンプトを設定（ボットのペルソナ）
    
            Args:
                persona: ボットの性格・役割
                capabilities: ボットの能力リスト
            """
            system_prompt = f"""
    あなたは{persona}です。
    
    あなたの能力:
    {chr(10).join('- ' + cap for cap in capabilities)}
    
    ユーザーとの対話では、親切で正確な情報を提供してください。
    不確実な情報については、その旨を明示してください。
    """
            self.conversation_history = [
                {"role": "system", "content": system_prompt}
            ]
    
        def chat(self,
                user_message: str,
                use_cot: bool = False,
                temperature: float = 0.7) -> Dict[str, any]:
            """
            ユーザーメッセージに応答
    
            Args:
                user_message: ユーザーの入力
                use_cot: Chain-of-Thoughtを使用するか
                temperature: 応答の多様性
    
            Returns:
                応答結果
            """
            # CoTプロンプトの追加
            if use_cot:
                user_message = f"""
    {user_message}
    
    上記の質問に答える際は、段階的に考えて説明してください。
    """
    
            # 会話履歴に追加
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
    
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=temperature,
                    max_tokens=500
                )
    
                assistant_message = response.choices[0].message.content
    
                # アシスタントの応答を履歴に追加
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
    
                return {
                    'user': user_message,
                    'assistant': assistant_message,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
    
            except Exception as e:
                return {
                    'user': user_message,
                    'error': str(e),
                    'success': False
                }
    
        def get_conversation_summary(self) -> str:
            """会話履歴の要約を取得"""
            if len(self.conversation_history) <= 1:
                return "会話はまだ開始されていません。"
    
            summary = "会話履歴:\n"
            for i, msg in enumerate(self.conversation_history[1:], 1):  # systemプロンプトをスキップ
                role = "ユーザー" if msg["role"] == "user" else "アシスタント"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                summary += f"{i}. [{role}] {content}\n"
    
            return summary
    
        def clear_history(self, keep_system: bool = True):
            """
            会話履歴をクリア
    
            Args:
                keep_system: システムプロンプトは保持するか
            """
            if keep_system and self.conversation_history:
                self.conversation_history = [self.conversation_history[0]]
            else:
                self.conversation_history = []
    
    # 使用例
    print("=== 統合型LLMチャットボット デモ ===\n")
    
    # チャットボットの初期化
    # bot = LLMChatbot(model="gpt-3.5-turbo")
    
    # ペルソナ設定
    persona = "親切で知識豊富なAIアシスタント"
    capabilities = [
        "一般的な質問への回答",
        "プログラミングのサポート",
        "数学・論理問題の解決",
        "文章の要約・翻訳",
    ]
    
    # bot.set_system_prompt(persona, capabilities)
    
    print(f"ボットペルソナ: {persona}\n")
    print("会話シミュレーション:\n")
    
    # 会話例（デモ）
    demo_conversations = [
        ("こんにちは！Pythonでリストを逆順にする方法を教えてください。", False),
        ("リストの要素数が100万個の場合、最も効率的な方法は何ですか？", True),  # CoT使用
    ]
    
    for user_msg, use_cot in demo_conversations:
        print(f"👤 ユーザー: {user_msg}")
    
        if use_cot:
            print("   (Chain-of-Thought推論を使用)")
    
        # 実際のAPI呼び出しはコメントアウト
        # result = bot.chat(user_msg, use_cot=use_cot)
        # if result['success']:
        #     print(f"🤖 アシスタント: {result['assistant']}")
        # else:
        #     print(f"❌ エラー: {result['error']}")
    
        print()
    
    # print(bot.get_conversation_summary())
    

* * *

## 演習問題

**演習5.1: スケーリング則の理解**

**問題** : 以下の条件で2つのモデルを訓練する場合、どちらがより高性能になると予想されますか？Chinchillaスケーリング則に基づいて説明してください。

  * モデルA: 200Bパラメータ、1兆トークンで訓練
  * モデルB: 70Bパラメータ、4兆トークンで訓練

**ヒント** : Chinchilla最適比率は「データトークン数 ≈ 20 × パラメータ数」です。

解答例

**分析** :

  * モデルA: 最適データ量 = 200B × 20 = 4兆トークン → 実際は1兆トークン（不足）
  * モデルB: 最適データ量 = 70B × 20 = 1.4兆トークン → 実際は4兆トークン（過剰だが許容範囲）

**結論** : モデルBの方が高性能になる可能性が高い。モデルAは「過剰パラメータ化」されており、データ不足により性能が頭打ちになります。Chinchilla論文が示すように、同じ計算予算ならより小さいモデルをより多くのデータで訓練する方が効率的です。

**演習5.2: Few-shotプロンプトの設計**

**問題** : 以下のタスクに対する効果的なFew-shotプロンプトを設計してください。

**タスク** : 商品レビューから「評価スコア（1-5）」と「主な理由」を抽出する

**要件** :

  * 3つの例示を含める
  * 出力形式を明確に指定する
  * エッジケース（曖昧な評価）も考慮する

解答例
    
    
    以下の商品レビューから「評価スコア（1-5）」と「主な理由」を抽出してください。
    
    例1:
    レビュー: この掃除機は吸引力が強く、軽量で使いやすいです。価格も手頃で大満足です。
    出力: {"score": 5, "reason": "吸引力、軽量性、コストパフォーマンス"}
    
    例2:
    レビュー: デザインは良いのですが、バッテリーの持ちが悪く、すぐに充電が必要です。
    出力: {"score": 2, "reason": "バッテリー持続時間の短さ"}
    
    例3:
    レビュー: 普通の商品です。特に良くも悪くもありません。
    出力: {"score": 3, "reason": "特筆すべき特徴なし"}
    
    レビュー: {入力レビュー}
    出力:
    

**設計のポイント** :

  * ポジティブ（例1）、ネガティブ（例2）、中立（例3）の例をバランス良く含める
  * JSON形式で構造化出力を指定し、パースしやすくする
  * 「理由」は簡潔な要約とし、レビュー全文をコピーしないよう誘導

**演習5.3: Chain-of-Thought推論の実装**

**問題** : 以下の論理パズルを解くCoTプロンプトを作成してください。

**パズル** :

> 3人の容疑者A、B、Cがいます。 
> 
>   * Aは「Bが犯人だ」と言っています
>   * Bは「私は無実だ」と言っています
>   * Cは「Aが犯人だ」と言っています
> 
犯人は1人で、その人だけが嘘をついています。犯人は誰ですか？ 

解答例
    
    
    以下の論理パズルを段階的に分析して解いてください。
    
    パズル:
    3人の容疑者A、B、Cがいます。
    - Aは「Bが犯人だ」と言っています
    - Bは「私は無実だ」と言っています
    - Cは「Aが犯人だ」と言っています
    犯人は1人で、その人だけが嘘をついています。犯人は誰ですか？
    
    段階的分析:
    
    仮定1: Aが犯人の場合
     - Aの発言「Bが犯人」は嘘 → ✓ 犯人は嘘をつく
     - Bの発言「私は無実」は真実 → ✓ BとCは真実を言う
     - Cの発言「Aが犯人」は真実 → ✓ 矛盾なし
     結論: Aが犯人の可能性あり
    
    仮定2: Bが犯人の場合
     - Aの発言「Bが犯人」は真実 → ✗ 矛盾（犯人以外も嘘？）
     - Bの発言「私は無実」は嘘 → ✓ 犯人は嘘をつく
     - Cの発言「Aが犯人」は嘘 → ✗ 矛盾（2人が嘘？）
     結論: 条件に矛盾
    
    仮定3: Cが犯人の場合
     - Aの発言「Bが犯人」は嘘 → ✗ 矛盾（犯人以外も嘘？）
     - Bの発言「私は無実」は真実 → ✓
     - Cの発言「Aが犯人」は嘘 → ✓ 犯人は嘘をつく
     結論: 条件に矛盾
    
    最終結論: Aが犯人です。
    理由: 仮定1のみがすべての条件を満たします。
    

**CoT設計のポイント** :

  * すべての可能性を体系的に検証する
  * 各仮定で矛盾の有無を明確にチェックする
  * 記号（✓, ✗）で視覚的に分かりやすくする

**演習5.4: RLHFの理解**

**問題** : RLHFで使用されるKLダイバージェンス制約 $\beta \cdot D_{KL}(\pi_{\theta} || \pi_{\text{ref}})$ の役割を説明してください。また、$\beta$が大きすぎる場合と小さすぎる場合の問題点を述べてください。

解答例

**KLダイバージェンス制約の役割** :

  1. **モードの崩壊防止** : 最適化中のモデル $\pi_{\theta}$ が参照モデル $\pi_{\text{ref}}$（事前学習済み）から大きく逸脱しないよう制約します。
  2. **言語能力の保持** : 報酬を最大化する過程で、文法や一貫性などの基本的な言語能力が失われるのを防ぎます。
  3. **報酬ハッキング回避** : モデルが報酬モデルの脆弱性を悪用する極端な戦略を学習するのを防ぎます。

**$\beta$が大きすぎる場合** :

  * 問題: モデルが参照モデルに過度に近くなり、RLHFの効果が薄れる
  * 結果: 人間のフィードバックがほとんど反映されず、改善が見られない

**$\beta$が小さすぎる場合** :

  * 問題: モデルが参照モデルから大きく逸脱し、不自然な出力を生成
  * 結果: 文法崩壊、意味不明な応答、報酬ハッキング

**実用的な$\beta$の選択** :

  * 一般的な範囲: 0.01〜0.1
  * 調整方法: 検証セットでの人間評価に基づいて最適値を探索

**演習5.5: LLMアプリケーション設計**

**問題** : カスタマーサポート用のLLMチャットボットを設計してください。以下の要件を満たすシステムアーキテクチャとプロンプト戦略を提案してください。

**要件** :

  * よくある質問（FAQ）には即座に回答
  * 複雑な問題には段階的に対応
  * 不確実な場合は人間のオペレーターにエスカレーション
  * 会話履歴を考慮した文脈理解

解答例

**システムアーキテクチャ** :
    
    
    class CustomerSupportChatbot:
        """
        カスタマーサポート用LLMチャットボット
        """
    
        def __init__(self):
            self.faq_database = self.load_faq()
            self.conversation_history = []
            self.escalation_threshold = 0.3  # 信頼度しきい値
    
        def load_faq(self):
            """FAQデータベースをロード"""
            return {
                "配送にかかる日数": "通常3-5営業日でお届けします。",
                "返品ポリシー": "購入から30日以内であれば返品可能です。",
                "支払い方法": "クレジットカード、銀行振込、代引きに対応しています。",
            }
    
        def check_faq(self, query: str) -> Optional[str]:
            """FAQに一致する質問をチェック"""
            # 簡略化: 実際は埋め込みベースの類似度検索を使用
            for question, answer in self.faq_database.items():
                if question in query:
                    return answer
            return None
    
        def classify_complexity(self, query: str) -> str:
            """問い合わせの複雑度を分類"""
            complexity_prompt = f"""
    以下の問い合わせを「簡単」「中程度」「複雑」に分類してください。
    
    問い合わせ: {query}
    
    分類:"""
            # LLM呼び出し（擬似コード）
            # complexity = call_llm(complexity_prompt)
            return "中程度"  # デモ用
    
        def handle_query(self, query: str) -> Dict:
            """問い合わせを処理"""
            # Step 1: FAQチェック
            faq_answer = self.check_faq(query)
            if faq_answer:
                return {
                    'type': 'faq',
                    'answer': faq_answer,
                    'confidence': 1.0
                }
    
            # Step 2: 複雑度評価
            complexity = self.classify_complexity(query)
    
            # Step 3: 複雑度に応じた処理
            if complexity == "簡単":
                return self.simple_response(query)
            elif complexity == "中程度":
                return self.cot_response(query)
            else:
                return self.escalate_to_human(query)
    
        def simple_response(self, query: str):
            """シンプルなZero-shot応答"""
            prompt = f"""
    あなたはカスタマーサポートのアシスタントです。
    以下の質問に簡潔に答えてください。
    
    質問: {query}
    
    回答:"""
            # response = call_llm(prompt)
            return {'type': 'simple', 'answer': "応答内容"}
    
        def cot_response(self, query: str):
            """Chain-of-Thoughtで段階的に対応"""
            prompt = f"""
    あなたはカスタマーサポートのアシスタントです。
    以下の問題を段階的に分析し、解決策を提案してください。
    
    問題: {query}
    
    分析:
    ステップ1:"""
            # response = call_llm(prompt)
            return {'type': 'cot', 'answer': "段階的な応答"}
    
        def escalate_to_human(self, query: str):
            """人間のオペレーターにエスカレーション"""
            return {
                'type': 'escalation',
                'message': "この問題は複雑なため、専門のオペレーターにお繋ぎします。",
                'query': query
            }
    

**プロンプト戦略** :

  1. **システムプロンプト** : ボットの役割、トーン、制約を明確に定義
  2. **Few-shot FAQ** : 類似質問の例を提示して精度向上
  3. **CoT for Complex Issues** : 複雑な問題は段階的に分析
  4. **Confidence Scoring** : 応答の信頼度を評価し、低い場合はエスカレーション

**評価指標** :

  * FAQ一致率: 70%以上
  * エスカレーション率: 15%以下
  * 顧客満足度: 4.0/5.0以上

* * *

## まとめ

この章では、大規模言語モデル（LLM）の本質と実践的な活用方法を学びました：

  * ✅ **スケーリング則** : モデルサイズ、データ量、計算量の関係を理解し、Chinchilla最適比率を把握しました
  * ✅ **主要LLM** : GPT、LLaMA、Claude、GeminiなどのアーキテクチャとDifferentiatorを比較しました
  * ✅ **プロンプトエンジニアリング** : Zero-shot、Few-shot、Chain-of-Thoughtなどの技術を実装しました
  * ✅ **In-Context Learning** : パラメータ更新なしに新タスクを学習するメカニズムを理解しました
  * ✅ **RLHF** : 人間のフィードバックを活用したモデル改善プロセスを学びました
  * ✅ **実践プロジェクト** : Few-shot分類、CoT推論、統合チャットボットを構築しました

> **次のステップ** : LLMの基礎を理解したら、次は特定ドメインへのFine-tuning、RAG（Retrieval-Augmented Generation）、マルチモーダルLLMなど、より高度なトピックに進みましょう。また、責任あるAI開発のための倫理的配慮とバイアス軽減技術も重要です。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
