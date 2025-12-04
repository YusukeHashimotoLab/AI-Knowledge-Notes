---
title: "第1章: AIエージェント基礎"
chapter_title: "第1章: AIエージェント基礎"
---

## AIエージェントとは

### 定義と特徴

**AIエージェント** は、環境を認識し、自律的に意思決定を行い、目標達成のために行動するAIシステムです。従来の静的なAIモデルとは異なり、エージェントは動的に状況を判断し、複数のステップを経て複雑なタスクを完遂します。

**AIエージェントの主要な特徴** ：

  * **自律性（Autonomy）** : 人間の指示なしに独立して行動
  * **反応性（Reactivity）** : 環境の変化を認識し適切に対応
  * **目標指向（Goal-oriented）** : 明確な目標に向けて行動を計画
  * **学習能力（Learning）** : 経験から学習し改善
  * **ツール使用（Tool Use）** : 外部ツールやAPIを活用

### 従来のAIとの違い

観点 | 従来のAI | AIエージェント  
---|---|---  
入出力 | 1回の入力→1回の出力 | 複数ステップの対話と行動  
意思決定 | 即座に回答生成 | 推論→行動→観察のループ  
外部連携 | 限定的または不可 | ツール・API・検索を活用  
タスク複雑性 | 単純な質問応答 | 多段階の複雑なタスク  
適応性 | 固定的な振る舞い | 状況に応じて戦略変更  
  
## エージェントアーキテクチャ

### 基本ループ: 認識・推論・行動

AIエージェントは以下のサイクルを繰り返すことで目標を達成します：
    
    
    ```mermaid
    graph LR
        A[認識Perception] --> B[推論Reasoning]
        B --> C[行動Action]
        C --> D[環境Environment]
        D --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

  1. **認識（Perception）** : 環境の状態、ユーザーの入力、前回の行動結果を観察
  2. **推論（Reasoning）** : 観察した情報を基に次の行動を計画
  3. **行動（Action）** : ツールの実行、回答の生成、タスクの実行
  4. **環境（Environment）** : 行動の結果が環境に反映され、次の認識へ

### エージェントの主要コンポーネント
    
    
    # エージェントの基本構造
    class Agent:
        def __init__(self, llm, tools, memory):
            self.llm = llm              # 大規模言語モデル（推論エンジン）
            self.tools = tools          # 利用可能なツール群
            self.memory = memory        # 会話履歴と状態
            self.max_iterations = 10    # 最大実行回数
    
        def run(self, task):
            """エージェント実行ループ"""
            self.memory.add_message("user", task)
    
            for i in range(self.max_iterations):
                # 1. 推論: 次のアクションを決定
                thought = self.think()
    
                # 2. 行動: ツールを実行または回答
                if thought.action:
                    observation = self.act(thought.action)
                    self.memory.add_observation(observation)
                else:
                    return thought.answer
    
            return "タスクを完了できませんでした"
    
        def think(self):
            """LLMで次のアクションを推論"""
            prompt = self.build_prompt()
            response = self.llm.generate(prompt)
            return self.parse_response(response)
    
        def act(self, action):
            """ツールを実行して結果を取得"""
            tool = self.tools[action.tool_name]
            result = tool.execute(action.parameters)
            return result
    

## ReActパターン

### Reasoning（推論）とActing（行動）の統合

**ReAct** （Reasoning and Acting）は、Yao et al. (2022)が提案したエージェントパターンで、推論の過程を言語化しながら行動を決定します。

**ReActのステップ** ：

  1. **Thought（思考）** : 現状を分析し、次のアクションを考える
  2. **Action（行動）** : ツールを選択し、パラメータを決定
  3. **Observation（観察）** : ツールの実行結果を確認
  4. 繰り返し → 最終的に Answer（回答）

### ReActプロンプトの例
    
    
    REACT_PROMPT = """あなたは質問に答えるアシスタントです。以下の形式で思考と行動を繰り返してください。
    
    利用可能なツール:
    - search: Web検索を実行 (入力: 検索クエリ)
    - calculator: 数式を計算 (入力: 数式)
    - finish: 最終回答を返す (入力: 回答文)
    
    形式:
    Question: ユーザーの質問
    Thought: 何を考えているか
    Action: ツール名[入力]
    Observation: ツールの実行結果
    ... (必要に応じて繰り返し)
    Thought: 最終的な結論
    Action: finish[最終回答]
    
    Question: {question}
    Thought:"""
    
    # 実行例
    question = "2024年のノーベル物理学賞受賞者は誰ですか？"
    response = """
    Thought: ノーベル賞の最新情報はWeb検索で調べる必要があります
    Action: search[2024年 ノーベル物理学賞 受賞者]
    Observation: 2024年のノーベル物理学賞は、John HopfieldとGeoffrey Hintonが受賞しました。機械学習の基礎研究が評価されました。
    Thought: 検索結果から受賞者が判明しました
    Action: finish[2024年のノーベル物理学賞は、John HopfieldとGeoffrey Hintonが受賞しました。彼らは機械学習の基礎研究、特にニューラルネットワークの理論的基盤への貢献が評価されました。]
    """
    

### ReActエージェントの実装
    
    
    import re
    from openai import OpenAI
    
    class ReActAgent:
        def __init__(self, api_key):
            self.client = OpenAI(api_key=api_key)
            self.tools = {
                "search": self.mock_search,
                "calculator": self.calculator
            }
    
        def mock_search(self, query):
            """検索のモック（実際はSerpAPIなどを使用）"""
            # 実際の実装ではWeb検索APIを呼び出す
            return f"検索結果: {query}についての情報"
    
        def calculator(self, expression):
            """計算機ツール"""
            try:
                result = eval(expression)
                return f"計算結果: {result}"
            except Exception as e:
                return f"エラー: {str(e)}"
    
        def run(self, question, max_steps=5):
            """ReActループの実行"""
            prompt = REACT_PROMPT.format(question=question)
    
            for step in range(max_steps):
                # LLMで次のアクションを生成
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
    
                text = response.choices[0].message.content
                prompt += text
    
                # Actionを抽出
                action_match = re.search(r'Action: (\w+)\[(.*?)\]', text)
                if not action_match:
                    continue
    
                tool_name = action_match.group(1)
                tool_input = action_match.group(2)
    
                # 終了判定
                if tool_name == "finish":
                    return tool_input
    
                # ツール実行
                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_input)
                    prompt += f"\nObservation: {observation}\nThought:"
                else:
                    prompt += f"\nObservation: エラー - ツール {tool_name} は存在しません\nThought:"
    
            return "最大ステップ数に到達しました"
    
    # 使用例
    agent = ReActAgent(api_key="your-api-key")
    answer = agent.run("123 + 456 の計算結果を教えてください")
    print(answer)  # 出力: 579
    

## Chain-of-Thought（思考の連鎖）

### 段階的な推論プロセス

**Chain-of-Thought (CoT)** は、複雑な問題を段階的に分解して推論する手法です。Wei et al. (2022)が提案し、特に数学的推論や論理的思考を要するタスクで効果を発揮します。

**CoTの利点** ：

  * ✅ **精度向上** : 複雑な問題の正解率が向上
  * ✅ **解釈可能性** : 推論過程が可視化される
  * ✅ **エラー検出** : 思考過程を確認できる
  * ✅ **デバッグ容易性** : どこで間違えたか特定可能

### Few-shot CoTプロンプト
    
    
    COT_PROMPT = """以下の例のように、段階的に考えて問題を解いてください。
    
    Q: カフェでコーヒーが1杯300円、ケーキが1個450円です。コーヒー2杯とケーキ3個を買うといくらですか？
    A: まず、コーヒーの合計を計算します: 300円 × 2杯 = 600円
    次に、ケーキの合計を計算します: 450円 × 3個 = 1,350円
    最後に合計します: 600円 + 1,350円 = 1,950円
    答え: 1,950円
    
    Q: {question}
    A: 段階的に考えましょう。"""
    
    # 実装例
    from openai import OpenAI
    
    def chain_of_thought(question, api_key):
        """CoTを使った推論"""
        client = OpenAI(api_key=api_key)
    
        prompt = COT_PROMPT.format(question=question)
    
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    
        return response.choices[0].message.content
    
    # 使用例
    question = "りんごが12個あります。友達に3個あげて、新たに8個買いました。今、りんごは何個ありますか？"
    answer = chain_of_thought(question, "your-api-key")
    print(answer)
    # 出力:
    # まず、友達にあげた後のりんごの数を計算します: 12個 - 3個 = 9個
    # 次に、新たに買ったりんごを足します: 9個 + 8個 = 17個
    # 答え: 17個
    

### Zero-shot CoT（「Let's think step by step」）

Kojima et al. (2022)は、例を示さなくても「Let's think step by step」という魔法のフレーズで段階的推論を引き出せることを発見しました。
    
    
    def zero_shot_cot(question, api_key):
        """Zero-shot CoT: 例なしでステップバイステップ推論"""
        client = OpenAI(api_key=api_key)
    
        # ステップ1: 推論プロセスを生成
        prompt1 = f"{question}\n\nLet's think step by step."
    
        response1 = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt1}],
            temperature=0
        )
    
        reasoning = response1.choices[0].message.content
    
        # ステップ2: 推論から最終回答を抽出
        prompt2 = f"{question}\n\n{reasoning}\n\nTherefore, the answer is:"
    
        response2 = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt2}],
            temperature=0
        )
    
        answer = response2.choices[0].message.content
    
        return {
            "reasoning": reasoning,
            "answer": answer
        }
    
    # 使用例
    result = zero_shot_cot(
        "ある数の3倍に5を足すと23になります。ある数はいくつですか？",
        "your-api-key"
    )
    print(f"推論: {result['reasoning']}")
    print(f"答え: {result['answer']}")
    

## 基本的なエージェントの実装

### シンプルなエージェントループ
    
    
    import json
    from openai import OpenAI
    from typing import List, Dict, Any
    
    class SimpleAgent:
        """シンプルなAIエージェント実装"""
    
        def __init__(self, api_key: str, tools: Dict[str, callable]):
            self.client = OpenAI(api_key=api_key)
            self.tools = tools
            self.conversation_history = []
            self.system_prompt = """あなたは有能なAIアシスタントです。
    ユーザーの質問に答えるために、必要に応じてツールを使用してください。
    
    利用可能なツール:
    {tool_descriptions}
    
    思考プロセス:
    1. 質問を理解する
    2. 必要なツールを選択する
    3. ツールを実行する
    4. 結果を統合して回答する"""
    
        def get_tool_descriptions(self) -> str:
            """ツールの説明を生成"""
            descriptions = []
            for name, func in self.tools.items():
                desc = func.__doc__ or "説明なし"
                descriptions.append(f"- {name}: {desc}")
            return "\n".join(descriptions)
    
        def run(self, user_input: str, max_iterations: int = 5) -> str:
            """エージェントを実行"""
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
    
            for iteration in range(max_iterations):
                # LLMに問い合わせ
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt.format(
                                tool_descriptions=self.get_tool_descriptions()
                            )
                        }
                    ] + self.conversation_history,
                    temperature=0
                )
    
                assistant_message = response.choices[0].message.content
    
                # ツール呼び出しをパース
                tool_call = self.parse_tool_call(assistant_message)
    
                if tool_call:
                    # ツールを実行
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
    
                    if tool_name in self.tools:
                        result = self.tools[tool_name](**tool_args)
    
                        # 結果を会話履歴に追加
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": f"ツール {tool_name} を実行しました"
                        })
                        self.conversation_history.append({
                            "role": "user",
                            "content": f"結果: {result}"
                        })
                    else:
                        # 未知のツール
                        self.conversation_history.append({
                            "role": "user",
                            "content": f"エラー: ツール {tool_name} は存在しません"
                        })
                else:
                    # ツール呼び出しなし = 最終回答
                    return assistant_message
    
            return "最大反復回数に達しました"
    
        def parse_tool_call(self, message: str) -> Dict[str, Any]:
            """メッセージからツール呼び出しを抽出（簡易版）"""
            # 実際にはより堅牢なパーサーが必要
            import re
            match = re.search(r'TOOL: (\w+)\((.*?)\)', message)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                # 簡易的な引数パース
                args = {}
                if args_str:
                    for arg in args_str.split(','):
                        key, value = arg.split('=')
                        args[key.strip()] = value.strip().strip('"\'')
                return {"name": tool_name, "args": args}
            return None
    
    # ツールの定義
    def get_weather(location: str) -> str:
        """指定された場所の天気を取得"""
        # 実際にはAPIを呼び出す
        weather_data = {
            "東京": "晴れ、気温22度",
            "大阪": "曇り、気温20度",
            "札幌": "雨、気温15度"
        }
        return weather_data.get(location, "データなし")
    
    def calculate(expression: str) -> float:
        """数式を計算"""
        try:
            result = eval(expression)
            return result
        except Exception as e:
            return f"計算エラー: {str(e)}"
    
    # エージェント実行
    agent = SimpleAgent(
        api_key="your-api-key",
        tools={
            "get_weather": get_weather,
            "calculate": calculate
        }
    )
    
    response = agent.run("東京の天気を教えてください")
    print(response)
    

## エージェント向けプロンプトエンジニアリング

### 効果的なシステムプロンプトの設計

エージェントの振る舞いは、システムプロンプトによって大きく左右されます。以下は効果的なプロンプト設計のベストプラクティスです。

#### 1\. 明確な役割定義
    
    
    SYSTEM_PROMPT = """あなたはカスタマーサポートエージェントです。
    
    役割:
    - ユーザーの問題を正確に理解する
    - 適切なツールを使って情報を収集する
    - 親切で専門的な回答を提供する
    
    重要な制約:
    - 不確実な情報は推測しない
    - 個人情報は慎重に扱う
    - エラー時は適切にエスカレーションする
    """
    

#### 2\. ツール使用ガイドライン
    
    
    TOOL_USAGE_GUIDE = """
    利用可能なツール:
    
    1. search_database(query: str) -> List[Dict]
       - データベースから関連情報を検索
       - 使用例: 商品情報、注文履歴の検索
    
    2. send_email(to: str, subject: str, body: str) -> bool
       - メールを送信
       - 使用例: 確認メール、通知の送信
    
    3. escalate_to_human(reason: str) -> None
       - 人間のオペレーターにエスカレーション
       - 使用例: 複雑な問題、クレーム対応
    
    ツール選択の原則:
    - まず必要な情報を search_database で収集
    - 自動対応可能なら回答を生成
    - 複雑または重要な case は escalate_to_human
    """
    

#### 3\. Few-shot Examples
    
    
    FEW_SHOT_EXAMPLES = """
    例1:
    ユーザー: 注文番号12345の配送状況を教えてください
    思考: 注文情報をデータベースから検索する必要がある
    行動: search_database(query="注文番号:12345")
    観察: {order_id: 12345, status: "配送中", tracking: "ABC123"}
    回答: ご注文12345は現在配送中です。追跡番号はABC123です。
    
    例2:
    ユーザー: 返金したいのですが
    思考: 返金は重要な金銭的処理なので、人間に引き継ぐべき
    行動: escalate_to_human(reason="返金リクエスト")
    回答: 返金のご相談ですね。担当者におつなぎいたします。
    """
    

## まとめ

### この章で学んだこと

  * ✅ **AIエージェントの定義** : 自律性、反応性、目標指向、ツール使用
  * ✅ **エージェントアーキテクチャ** : 認識→推論→行動のループ
  * ✅ **ReActパターン** : 推論と行動を統合したエージェント設計
  * ✅ **Chain-of-Thought** : 段階的な推論プロセスで精度向上
  * ✅ **基本実装** : シンプルなエージェントループの構築
  * ✅ **プロンプト設計** : エージェント向けの効果的なプロンプト

### 重要な概念

> **エージェント = LLM + ツール + 推論ループ**
> 
> AIエージェントは、大規模言語モデルの推論能力と外部ツールを組み合わせ、複数ステップの思考と行動を繰り返すことで、複雑なタスクを自律的に解決します。

### 次のステップ

第2章では、Function CallingとTool Useについて詳しく学びます：

  * OpenAI/AnthropicのFunction Calling API
  * ツールスキーマの定義
  * 外部API統合とエラーハンドリング
  * 実用的なツール実装パターン

[← シリーズ概要](<./index.html>) [第2章: ツール使用 →](<./chapter2-tool-use.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
