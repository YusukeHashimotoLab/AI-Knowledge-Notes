---
title: 第6章：大規模言語モデルとの統合
chapter_title: 第6章：大規模言語モデルとの統合
subtitle: LLMエージェントによる高度な協調制御とプロセス診断
---

[← シリーズ目次に戻る](<./index.html>)

## 本章の概要

本章では、大規模言語モデル（LLM）を強化学習エージェントと統合し、高度なプロセス制御システムを構築する方法を学びます。LLMの自然言語理解能力と推論能力を活用することで、従来のRLエージェントでは困難だった柔軟な意思決定、異常診断、人間との協調作業が可能になります。

**💡 本章で学ぶこと**

  * Claude API / OpenAI APIを用いたLLMエージェントの構築
  * LangChainによるエージェントフレームワークの実装
  * LLMとRLエージェントのハイブリッドシステム設計
  * Tool Use（Function Calling）による外部システム連携
  * プロセス異常診断とLLM説明生成
  * 実プラントデプロイ時のベストプラクティス

## 6.1 LLMエージェントの基礎

大規模言語モデル（LLM）は、自然言語を理解し生成する能力を持つAIシステムです。プロセス制御への応用では、センサーデータの解釈、異常の診断、制御戦略の提案など、従来のルールベースシステムでは困難だったタスクを実行できます。

### 6.1.1 LLMエージェントの特徴

特徴 | RLエージェント | LLMエージェント | ハイブリッド  
---|---|---|---  
学習方法 | 試行錯誤（報酬最大化） | 事前学習済み（推論のみ） | RLで最適化、LLMで推論  
推論速度 | ⚡⚡⚡ 高速（ms） | ⚡ 中速（1-5秒） | ⚡⚡ 役割分担で最適化  
説明可能性 | ⭐ 低い | ⭐⭐⭐ 高い（自然言語） | ⭐⭐⭐ LLMが説明  
柔軟性 | ⭐⭐ 学習範囲内 | ⭐⭐⭐ 汎用推論可能 | ⭐⭐⭐ 両方の長所  
コスト | 初期学習コスト高 | API呼び出しコスト | バランス型  
  
### Example 1: Claude APIによるプロセス状態分析

Claude API（Anthropic）を使用して、CSTR（連続攪拌槽反応器）の状態を分析し、運転状況を自然言語で説明するエージェントを実装します。
    
    
    import anthropic
    import os
    import json
    from typing import Dict, List
    
    # ===================================
    # Example 1: Claude APIでCSTR状態分析
    # ===================================
    
    class ClaudeProcessAnalyzer:
        """Claude APIを用いたプロセス分析エージェント"""
    
        def __init__(self, api_key: str = None):
            """
            Args:
                api_key: Anthropic API Key（環境変数ANTHROPIC_API_KEYから自動取得）
            """
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("API key required. Set ANTHROPIC_API_KEY environment variable.")
    
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = "claude-3-5-sonnet-20241022"
    
        def analyze_cstr_state(self, state: Dict, target: Dict) -> Dict:
            """CSTR状態を分析し、運転状況を説明
    
            Args:
                state: 現在の状態 {'temperature': float, 'concentration': float, ...}
                target: 目標値 {'temperature': float, 'concentration': float, ...}
    
            Returns:
                {'status': str, 'analysis': str, 'recommendations': List[str]}
            """
            # プロンプトを構築
            prompt = f"""あなたは化学プロセスエンジニアです。以下のCSTR（連続攪拌槽反応器）の状態を分析してください。
    
    **現在の状態:**
    - 温度: {state['temperature']:.1f} K
    - 濃度: {state['concentration']:.3f} mol/L
    - 流量: {state['flow_rate']:.2f} L/min
    - 加熱電力: {state['heating_power']:.2f} kW
    
    **目標値:**
    - 目標温度: {target['temperature']:.1f} K
    - 目標濃度: {target['concentration']:.3f} mol/L
    
    以下の形式でJSON形式で回答してください:
    {{
        "status": "正常 or 注意 or 異常",
        "analysis": "現在の運転状況の説明（100文字以内）",
        "recommendations": ["推奨アクション1", "推奨アクション2", ...]
    }}"""
    
            # Claude APIを呼び出し
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
    
            # レスポンスをパース
            response_text = message.content[0].text
    
            # JSONをパース（```json```で囲まれている場合に対応）
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text
    
            result = json.loads(json_str)
            return result
    
    
    # 使用例
    if __name__ == "__main__":
        # CSTR状態データ
        current_state = {
            'temperature': 365.0,  # K（目標350Kから+15K高い）
            'concentration': 0.32,  # mol/L
            'flow_rate': 1.0,
            'heating_power': 5.2
        }
    
        target_state = {
            'temperature': 350.0,
            'concentration': 0.30
        }
    
        # 分析を実行
        analyzer = ClaudeProcessAnalyzer()
        result = analyzer.analyze_cstr_state(current_state, target_state)
    
        print("=== CSTR状態分析結果 ===")
        print(f"ステータス: {result['status']}")
        print(f"\n分析: {result['analysis']}")
        print(f"\n推奨アクション:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # 期待される出力例:
    # === CSTR状態分析結果 ===
    # ステータス: 注意
    #
    # 分析: 温度が目標値より15K高く、オーバーシュート状態です。濃度も目標より高めですが許容範囲内です。
    #
    # 推奨アクション:
    #   1. 加熱電力を3.5kW程度まで減少させる
    #   2. 冷却水流量を10%増加させる
    #   3. 温度が340K以下に低下しないよう監視を継続
    

**💡 Pro Tip: API Key管理**

本番環境では、API Keyを環境変数やSecret Managerに保存し、コードに直接埋め込まないようにしましょう。また、API呼び出しのレート制限（Anthropic: 50 requests/min）に注意してください。

## 6.2 Tool Use（Function Calling）の実装

Claude APIのTool Use機能を使用すると、LLMが外部関数を呼び出してリアルタイムデータを取得したり、制御アクションを実行したりできます。これにより、LLMエージェントが実プロセスと直接連携できるようになります。
    
    
    ```mermaid
    flowchart LR
        A[ユーザー/センサー] -->|状態データ| B[LLMエージェント]
        B -->|Tool Call| C[get_sensor_data]
        C -->|センサー値| B
        B -->|Tool Call| D[execute_control_action]
        D -->|制御信号| E[CSTR]
        E -->|フィードバック| A
    
        style B fill:#e3f2fd
        style C fill:#e8f5e9
        style D fill:#fff3e0
    ```

### Example 2: Tool Useによるプロセス制御

LLMが実際のプロセス制御関数を呼び出せるようにします。
    
    
    # ===================================
    # Example 2: Tool Useでプロセス制御
    # ===================================
    
    class ClaudeProcessController:
        """Tool Useを使ったプロセス制御エージェント"""
    
        def __init__(self, api_key: str = None):
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022"
    
            # 制御対象のCSTR（シミュレーション）
            self.cstr_state = {
                'temperature': 355.0,
                'concentration': 0.35,
                'heating_power': 4.5,
                'flow_rate': 1.0
            }
    
        def get_sensor_data(self) -> Dict:
            """センサーデータを取得（Tool関数）"""
            return self.cstr_state
    
        def set_heating_power(self, power: float) -> Dict:
            """加熱電力を設定（Tool関数）
    
            Args:
                power: 加熱電力 [kW]（0-10の範囲）
            """
            power = max(0.0, min(10.0, power))  # 安全範囲に制限
            self.cstr_state['heating_power'] = power
            return {
                'success': True,
                'new_power': power,
                'message': f'加熱電力を{power:.1f}kWに設定しました'
            }
    
        def set_flow_rate(self, flow: float) -> Dict:
            """流量を設定（Tool関数）
    
            Args:
                flow: 流量 [L/min]（0.5-2.0の範囲）
            """
            flow = max(0.5, min(2.0, flow))
            self.cstr_state['flow_rate'] = flow
            return {
                'success': True,
                'new_flow': flow,
                'message': f'流量を{flow:.2f}L/minに設定しました'
            }
    
        def define_tools(self) -> List[Dict]:
            """LLMが使用可能なツールを定義"""
            return [
                {
                    "name": "get_sensor_data",
                    "description": "CSTRのセンサーデータ（温度、濃度、加熱電力、流量）を取得します。",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "set_heating_power",
                    "description": "CSTRの加熱電力を設定します。0-10kWの範囲で指定してください。",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "power": {
                                "type": "number",
                                "description": "設定する加熱電力 [kW]"
                            }
                        },
                        "required": ["power"]
                    }
                },
                {
                    "name": "set_flow_rate",
                    "description": "CSTRの流量を設定します。0.5-2.0 L/minの範囲で指定してください。",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "flow": {
                                "type": "number",
                                "description": "設定する流量 [L/min]"
                            }
                        },
                        "required": ["flow"]
                    }
                }
            ]
    
        def execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
            """ツールを実行"""
            if tool_name == "get_sensor_data":
                return self.get_sensor_data()
            elif tool_name == "set_heating_power":
                return self.set_heating_power(tool_input['power'])
            elif tool_name == "set_flow_rate":
                return self.set_flow_rate(tool_input['flow'])
            else:
                return {'error': f'Unknown tool: {tool_name}'}
    
        def run_control_task(self, user_request: str, max_iterations: int = 5) -> str:
            """制御タスクを実行
    
            Args:
                user_request: ユーザーの要求（例: "温度を350Kに下げてください"）
                max_iterations: 最大Tool呼び出し回数
    
            Returns:
                最終的なLLMの応答
            """
            messages = [
                {"role": "user", "content": user_request}
            ]
    
            for iteration in range(max_iterations):
                # Claude APIを呼び出し
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    tools=self.define_tools(),
                    messages=messages
                )
    
                # 応答をログ
                print(f"\n=== Iteration {iteration + 1} ===")
                print(f"Stop reason: {response.stop_reason}")
    
                # Tool呼び出しがある場合
                if response.stop_reason == "tool_use":
                    # Assistantの応答をメッセージ履歴に追加
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
    
                    # 各Tool callを実行
                    tool_results = []
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input
    
                            print(f"Tool call: {tool_name}({tool_input})")
    
                            # Toolを実行
                            result = self.execute_tool(tool_name, tool_input)
                            print(f"Tool result: {result}")
    
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": json.dumps(result)
                            })
    
                    # Tool結果をメッセージ履歴に追加
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
    
                # 終了条件
                elif response.stop_reason == "end_turn":
                    # 最終応答を抽出
                    final_response = ""
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            final_response += content_block.text
    
                    return final_response
    
            return "最大反復回数に到達しました"
    
    
    # 使用例
    if __name__ == "__main__":
        controller = ClaudeProcessController()
    
        # タスク: 温度を下げる
        result = controller.run_control_task(
            "現在のCSTR温度が355Kで目標は350Kです。温度を下げてください。"
        )
    
        print("\n=== 最終結果 ===")
        print(result)
        print(f"\n現在の状態: {controller.cstr_state}")
    
    # 期待される出力例:
    # === Iteration 1 ===
    # Stop reason: tool_use
    # Tool call: get_sensor_data({})
    # Tool result: {'temperature': 355.0, 'concentration': 0.35, ...}
    #
    # === Iteration 2 ===
    # Stop reason: tool_use
    # Tool call: set_heating_power({'power': 3.5})
    # Tool result: {'success': True, 'new_power': 3.5, ...}
    #
    # === Iteration 3 ===
    # Stop reason: end_turn
    #
    # === 最終結果 ===
    # 加熱電力を4.5kWから3.5kWに減少させました。この調整により、
    # 反応器温度は徐々に350Kの目標値に近づいていくはずです。
    # 10分後に再度温度を確認し、必要に応じて微調整を行うことを推奨します。
    

**⚠️ 安全上の注意**

Tool Useで実プロセスを制御する場合、以下の安全対策が必須です：

  * すべてのTool関数に入力値の範囲チェックを実装
  * 危険な操作（緊急停止など）は人間の承認を必須とする
  * LLMの推奨アクションをシミュレーションで検証してから実行
  * API障害時のフォールバック機構（PID制御など）を用意

## 6.3 LangChainによるエージェントフレームワーク

LangChainは、LLMアプリケーション開発のためのフレームワークで、エージェント構築、メモリ管理、ツール統合を簡潔に実装できます。

### 6.3.1 LangChainエージェントの構成要素
    
    
    ```mermaid
    flowchart TD
        A[ユーザー入力] --> B[Agent Executor]
        B --> C[LLMClaude/GPT-4]
        C --> D{Tool必要?}
        D -->|Yes| E[Toolget_sensor_dataset_control]
        E --> F[Tool結果]
        F --> C
        D -->|No| G[最終応答]
    
        H[Memory会話履歴] -.-> B
        B -.->|更新| H
    
        style C fill:#e3f2fd
        style E fill:#e8f5e9
        style H fill:#fff3e0
    ```

### Example 3: LangChainでプロセス診断エージェント

LangChainを使用して、会話履歴を保持しながらプロセス診断を行うエージェントを構築します。
    
    
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_anthropic import ChatAnthropic
    from langchain.tools import Tool
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    import numpy as np
    
    # ===================================
    # Example 3: LangChainプロセス診断エージェント
    # ===================================
    
    class ProcessDiagnosticAgent:
        """LangChainを用いたプロセス診断エージェント"""
    
        def __init__(self, api_key: str = None):
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
            # LLMを初期化
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                anthropic_api_key=self.api_key,
                temperature=0.0
            )
    
            # プロセス状態（シミュレーション）
            self.process_history = []
            self.current_state = {
                'temperature': 348.0,
                'pressure': 2.1,  # bar
                'flow_rate': 1.0,
                'concentration': 0.28
            }
    
            # メモリ（会話履歴を保存）
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
    
            # エージェントを構築
            self.agent_executor = self._build_agent()
    
        def _get_current_state(self, query: str = "") -> str:
            """現在のプロセス状態を取得"""
            state_str = f"""現在のCSTR状態:
    - 温度: {self.current_state['temperature']:.1f} K
    - 圧力: {self.current_state['pressure']:.2f} bar
    - 流量: {self.current_state['flow_rate']:.2f} L/min
    - 濃度: {self.current_state['concentration']:.3f} mol/L"""
            return state_str
    
        def _get_historical_trend(self, query: str = "") -> str:
            """過去のトレンドデータを取得"""
            if len(self.process_history) == 0:
                return "履歴データがありません。"
    
            temps = [h['temperature'] for h in self.process_history[-10:]]
            avg_temp = np.mean(temps)
            std_temp = np.std(temps)
    
            trend = "上昇" if temps[-1] > temps[0] else "低下"
    
            return f"""過去10ステップのトレンド:
    - 平均温度: {avg_temp:.1f} K
    - 温度標準偏差: {std_temp:.2f} K
    - トレンド: {trend}傾向
    - 変動範囲: {min(temps):.1f} - {max(temps):.1f} K"""
    
        def _diagnose_anomaly(self, query: str = "") -> str:
            """異常を診断"""
            state = self.current_state
            issues = []
    
            # 温度異常チェック
            if state['temperature'] < 340 or state['temperature'] > 360:
                issues.append(f"温度異常: {state['temperature']:.1f} K（正常範囲: 340-360K）")
    
            # 圧力異常チェック
            if state['pressure'] < 1.8 or state['pressure'] > 2.5:
                issues.append(f"圧力異常: {state['pressure']:.2f} bar（正常範囲: 1.8-2.5bar）")
    
            # 濃度異常チェック
            if state['concentration'] < 0.25 or state['concentration'] > 0.35:
                issues.append(f"濃度異常: {state['concentration']:.3f} mol/L（正常範囲: 0.25-0.35）")
    
            if len(issues) == 0:
                return "現在、異常は検出されていません。すべてのパラメータが正常範囲内です。"
            else:
                return "検出された異常:\n" + "\n".join([f"- {issue}" for issue in issues])
    
        def _build_agent(self) -> AgentExecutor:
            """エージェントを構築"""
    
            # ツールを定義
            tools = [
                Tool(
                    name="get_current_state",
                    func=self._get_current_state,
                    description="現在のプロセス状態（温度、圧力、流量、濃度）を取得します。"
                ),
                Tool(
                    name="get_historical_trend",
                    func=self._get_historical_trend,
                    description="過去のプロセスデータのトレンドを分析します。平均値、標準偏差、トレンド方向を返します。"
                ),
                Tool(
                    name="diagnose_anomaly",
                    func=self._diagnose_anomaly,
                    description="プロセスパラメータの異常を診断します。正常範囲外の値を検出します。"
                )
            ]
    
            # ReActプロンプトテンプレート
            template = """あなたは化学プロセス診断の専門家です。利用可能なツールを使用して、ユーザーの質問に答えてください。
    
    利用可能なツール:
    {tools}
    
    ツール名: {tool_names}
    
    以下の形式で応答してください:
    
    Question: 入力された質問
    Thought: 何をすべきか考える
    Action: 実行するツール名
    Action Input: ツールへの入力
    Observation: ツールの結果
    ... (この Thought/Action/Action Input/Observation を必要なだけ繰り返す)
    Thought: 最終的な答えがわかった
    Final Answer: ユーザーへの最終回答
    
    質問: {input}
    
    {agent_scratchpad}"""
    
            prompt = PromptTemplate.from_template(template)
    
            # エージェントを作成
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
    
            # Agent Executorを作成
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
    
            return agent_executor
    
        def diagnose(self, query: str) -> str:
            """診断クエリを実行
    
            Args:
                query: ユーザーからの質問
    
            Returns:
                診断結果
            """
            # 履歴に現在の状態を追加
            self.process_history.append(self.current_state.copy())
    
            # エージェントを実行
            result = self.agent_executor.invoke({
                "input": query
            })
    
            return result['output']
    
    
    # 使用例
    if __name__ == "__main__":
        agent = ProcessDiagnosticAgent()
    
        # クエリ1: 現在の状態確認
        print("=== Query 1 ===")
        result1 = agent.diagnose("現在のプロセス状態を教えてください。")
        print(f"\n応答: {result1}\n")
    
        # 状態を変更（異常を発生させる）
        agent.current_state['temperature'] = 365.0  # 異常値
        agent.current_state['pressure'] = 2.7  # 異常値
    
        # クエリ2: 異常診断
        print("\n=== Query 2 ===")
        result2 = agent.diagnose("何か異常がありますか？もしあれば原因と対策を教えてください。")
        print(f"\n応答: {result2}\n")
    
    # 期待される出力例:
    # === Query 1 ===
    #
    # > Entering new AgentExecutor chain...
    # Thought: プロセス状態を確認する必要がある
    # Action: get_current_state
    # Action Input:
    # Observation: 現在のCSTR状態:
    # - 温度: 348.0 K
    # - 圧力: 2.10 bar
    # ...
    # Thought: 最終的な答えがわかった
    # Final Answer: 現在のプロセスは正常に運転されています。温度348K、圧力2.1bar、
    # 流量1.0L/min、濃度0.28mol/Lで、すべてのパラメータが正常範囲内にあります。
    #
    # === Query 2 ===
    #
    # > Entering new AgentExecutor chain...
    # Thought: 異常診断ツールを使用すべき
    # Action: diagnose_anomaly
    # Action Input:
    # Observation: 検出された異常:
    # - 温度異常: 365.0 K（正常範囲: 340-360K）
    # - 圧力異常: 2.70 bar（正常範囲: 1.8-2.5bar）
    # Thought: 最終的な答えがわかった
    # Final Answer: 現在、2つの異常が検出されています。
    #
    # 1. 温度異常（365K）: 正常範囲（340-360K）を超過
    #    原因: 加熱電力過剰、冷却不足、または発熱反応の加速
    #    対策: 加熱電力を20%減少、冷却水流量を15%増加
    #
    # 2. 圧力異常（2.7bar）: 正常範囲（1.8-2.5bar）を超過
    #    原因: 流出バルブの詰まり、または気相生成物の増加
    #    対策: 流出バルブの開度を10%増加、温度低下により気相生成を抑制
    

**💡 LangChainのメリット**

  * **メモリ管理** : 会話履歴を自動的に保存・参照
  * **ツール統合** : Pythonさえあればどんな関数でもLLMツールに変換可能
  * **エラーハンドリング** : LLMの出力パース失敗を自動リトライ
  * **モジュール性** : LLM、Tools、Memoryを独立して変更可能

## 6.4 LLM-RL Hybridシステムの構築

LLMの推論能力とRLエージェントの最適化能力を組み合わせることで、より高度なプロセス制御システムを構築できます。LLMは高レベルの戦略決定を担当し、RLエージェントは低レベルの制御を実行します。

### 6.4.1 Hierarchical Architecture
    
    
    ```mermaid
    flowchart TD
        A[人間オペレーター] -->|指示/問い合わせ| B[LLM CoordinatorClaude/GPT-4]
        B -->|戦略決定目標設定| C[RL Agent 1反応器制御]
        B -->|戦略決定目標設定| D[RL Agent 2分離器制御]
        B -->|戦略決定目標設定| E[RL Agent 3熱交換器制御]
    
        C -->|状態レポート| B
        D -->|状態レポート| B
        E -->|状態レポート| B
    
        C -.->|協調| D
        D -.->|協調| E
    
        C --> F[CSTR]
        D --> G[蒸留塔]
        E --> H[熱交換器]
    
        style B fill:#e3f2fd
        style C fill:#e8f5e9
        style D fill:#e8f5e9
        style E fill:#e8f5e9
    ```

### Example 4: LLM CoordinatorとRL Workers

LLMが複数のRLエージェントを統括し、全体最適化を実現するシステムを実装します。
    
    
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from typing import Dict, List
    
    # ===================================
    # Example 4: LLM-RL Hybridシステム
    # ===================================
    
    class RLWorkerAgent:
        """RL制御ワーカーエージェント"""
    
        def __init__(self, name: str, env: gym.Env, model_path: str = None):
            self.name = name
            self.env = env
    
            # 事前学習済みモデルをロード（またはダミー）
            if model_path:
                self.model = PPO.load(model_path)
            else:
                # デモ用: ランダムエージェント
                self.model = None
    
            self.current_state = None
            self.current_reward = 0.0
            self.cumulative_reward = 0.0
    
        def reset(self):
            """環境をリセット"""
            self.current_state = self.env.reset()
            self.cumulative_reward = 0.0
            return self.current_state
    
        def step(self, target_setpoint: Dict = None) -> Dict:
            """1ステップ実行
    
            Args:
                target_setpoint: LLMから指示された目標値
    
            Returns:
                {'state': ..., 'reward': ..., 'done': ...}
            """
            # RLモデルでアクションを選択
            if self.model:
                action, _ = self.model.predict(self.current_state, deterministic=True)
            else:
                # デモ用: ランダムアクション
                action = self.env.action_space.sample()
    
            # 環境で実行
            next_state, reward, done, info = self.env.step(action)
    
            self.current_state = next_state
            self.current_reward = reward
            self.cumulative_reward += reward
    
            return {
                'state': next_state,
                'reward': reward,
                'cumulative_reward': self.cumulative_reward,
                'done': done,
                'info': info
            }
    
        def get_status_report(self) -> str:
            """状態レポートを生成（LLMへの報告用）"""
            if self.current_state is None:
                return f"{self.name}: 未初期化"
    
            state_str = ", ".join([f"{k}={v:.2f}" for k, v in
                                   enumerate(self.current_state)])
    
            return f"""{self.name}の状態:
    - State: [{state_str}]
    - 直近報酬: {self.current_reward:.3f}
    - 累積報酬: {self.cumulative_reward:.3f}"""
    
    
    class LLMCoordinator:
        """LLMによる全体調整エージェント"""
    
        def __init__(self, workers: List[RLWorkerAgent], api_key: str = None):
            self.workers = workers
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022"
            self.conversation_history = []
    
        def collect_worker_reports(self) -> str:
            """全ワーカーから状態レポートを収集"""
            reports = []
            for worker in self.workers:
                reports.append(worker.get_status_report())
    
            return "\n\n".join(reports)
    
        def coordinate(self, user_goal: str = None) -> Dict:
            """ワーカーを調整し、戦略を決定
    
            Args:
                user_goal: オペレーターからの目標（例: "生産量を10%増加"）
    
            Returns:
                各ワーカーへの指示
            """
            # ワーカーレポートを収集
            worker_reports = self.collect_worker_reports()
    
            # プロンプトを構築
            if user_goal:
                prompt = f"""あなたはプロセス全体を管理するスーパーバイザーです。
    
    **オペレーターの目標:**
    {user_goal}
    
    **現在の各ユニットの状態:**
    {worker_reports}
    
    各ワーカーエージェント（RL制御）に対して、以下のJSON形式で指示を出してください:
    {{
        "strategy": "全体戦略の説明",
        "worker_instructions": {{
            "worker_0": {{"action": "具体的な指示"}},
            "worker_1": {{"action": "具体的な指示"}},
            "worker_2": {{"action": "具体的な指示"}}
        }}
    }}"""
            else:
                prompt = f"""現在の各ユニットの状態を分析し、改善提案をしてください:
    
    {worker_reports}
    
    JSON形式で応答してください。"""
    
            # Claude APIを呼び出し
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
    
            response_text = message.content[0].text
    
            # JSONをパース
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text
    
            try:
                instructions = json.loads(json_str)
                return instructions
            except json.JSONDecodeError:
                return {
                    'strategy': response_text,
                    'worker_instructions': {}
                }
    
        def run_coordinated_control(self, user_goal: str, n_steps: int = 10):
            """調整制御を実行
    
            Args:
                user_goal: 全体目標
                n_steps: 実行ステップ数
            """
            print(f"=== 調整制御開始 ===")
            print(f"目標: {user_goal}\n")
    
            # ワーカーを初期化
            for worker in self.workers:
                worker.reset()
    
            for step in range(n_steps):
                print(f"\n--- Step {step + 1}/{n_steps} ---")
    
                # 5ステップごとにLLMで調整
                if step % 5 == 0:
                    instructions = self.coordinate(user_goal)
                    print(f"\nLLM戦略: {instructions.get('strategy', 'N/A')}")
                    print(f"指示: {instructions.get('worker_instructions', {})}")
    
                # 各ワーカーを1ステップ実行
                for i, worker in enumerate(self.workers):
                    result = worker.step()
                    print(f"{worker.name}: 報酬={result['reward']:.3f}, "
                          f"累積={result['cumulative_reward']:.3f}")
    
            print(f"\n=== 調整制御完了 ===")
            final_report = self.collect_worker_reports()
            print(f"\n{final_report}")
    
    
    # 使用例
    if __name__ == "__main__":
        # デモ用の簡易環境
        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,))
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
                self.state = np.array([0.0, 0.0, 0.0, 0.0])
    
            def reset(self):
                self.state = np.random.randn(4)
                return self.state
    
            def step(self, action):
                self.state += 0.1 * action[0]  # 簡易ダイナミクス
                reward = -np.sum(self.state**2)  # 原点に近いほど高報酬
                done = False
                return self.state, reward, done, {}
    
        # ワーカーエージェントを作成
        workers = [
            RLWorkerAgent("Worker_Reactor", DummyEnv()),
            RLWorkerAgent("Worker_Separator", DummyEnv()),
            RLWorkerAgent("Worker_HeatExchanger", DummyEnv())
        ]
    
        # LLM Coordinatorを作成
        coordinator = LLMCoordinator(workers)
    
        # 調整制御を実行
        coordinator.run_coordinated_control(
            user_goal="全ユニットの安定性を高めつつ、エネルギー消費を10%削減してください。",
            n_steps=10
        )
    
    # 期待される出力例:
    # === 調整制御開始 ===
    # 目標: 全ユニットの安定性を高めつつ、エネルギー消費を10%削減してください。
    #
    # --- Step 1/10 ---
    # LLM戦略: 各ユニットの状態を安定化させるため、まず反応器の温度変動を抑え、
    # 分離器の還流比を最適化し、熱交換器の効率を向上させます。
    # 指示: {'worker_0': {'action': '温度を±2K以内に維持'},
    #        'worker_1': {'action': '還流比を0.7に調整'},
    #        'worker_2': {'action': '伝熱係数を5%向上'}}
    #
    # Worker_Reactor: 報酬=-2.134, 累積=-2.134
    # Worker_Separator: 報酬=-1.876, 累積=-1.876
    # Worker_HeatExchanger: 報酬=-3.021, 累積=-3.021
    # ...
    

**🎯 LLM-RL Hybridの利点**

  * **戦略と戦術の分離** : LLMが高レベル戦略、RLが低レベル制御を担当
  * **説明可能性** : LLMが決定理由を自然言語で説明
  * **適応性** : 新しい目標や制約にLLMが柔軟に対応
  * **人間協調** : オペレーターとの自然言語対話が可能

## 6.5 本番環境デプロイのベストプラクティス

LLMエージェントを実プロセスにデプロイする際には、RLエージェント以上に慎重な設計が必要です。

### 6.5.1 安全性とフォールバック

リスク | 対策 | 実装方法  
---|---|---  
API障害 | Fallback制御 | PID/MPCへの自動切替  
レート制限 | キャッシング | 同一クエリは再利用（TTL: 5分）  
不適切な推奨 | 範囲チェック | 全ての制御値に min/max制約  
レイテンシ | 非同期実行 | LLMを別スレッドで実行  
コスト超過 | 予算管理 | 月間$1000上限、超過時はアラート  
  
### Example 5: 本番対応統合システム
    
    
    # ===================================
    # Example 5: 本番対応LLM-RL統合システム
    # ===================================
    
    import time
    import logging
    from functools import lru_cache
    import hashlib
    
    # ロギング設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
    class ProductionLLMController:
        """本番環境対応のLLM制御システム"""
    
        def __init__(self, api_key: str = None):
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022"
    
            # 統計情報
            self.api_call_count = 0
            self.cache_hit_count = 0
            self.fallback_count = 0
            self.total_cost = 0.0  # ドル
    
            # 制約
            self.monthly_budget = 1000.0  # ドル
            self.max_retries = 3
            self.timeout = 10.0  # 秒
    
            # フォールバック制御器（PID）
            self.fallback_controller = SimplePIDController()
    
        @lru_cache(maxsize=100)
        def _cached_llm_call(self, prompt_hash: str, prompt: str) -> str:
            """キャッシュ付きLLM呼び出し"""
            logger.info(f"Cache miss for hash {prompt_hash[:8]}. Calling API...")
    
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout
                )
    
                self.api_call_count += 1
    
                # コスト計算（概算: $3/1M input tokens, $15/1M output tokens）
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
                self.total_cost += cost
    
                logger.info(f"API call #{self.api_call_count}, Cost: ${cost:.4f}, "
                           f"Total: ${self.total_cost:.2f}")
    
                # 予算チェック
                if self.total_cost > self.monthly_budget:
                    logger.error(f"Monthly budget ${self.monthly_budget} exceeded!")
                    raise ValueError("Budget exceeded")
    
                return message.content[0].text
    
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise
    
        def llm_control_decision(self, state: Dict, target: Dict) -> Dict:
            """LLMで制御判断（キャッシュ・フォールバック付き）
    
            Args:
                state: 現在の状態
                target: 目標値
    
            Returns:
                {'action': ..., 'reasoning': ..., 'fallback_used': bool}
            """
            # プロンプトを構築
            prompt = f"""CSTR制御判断:
    現在: 温度{state['temperature']:.1f}K, 濃度{state['concentration']:.3f}
    目標: 温度{target['temperature']:.1f}K, 濃度{target['concentration']:.3f}
    
    加熱電力（0-10kW）を推奨してください。JSON形式で回答:
    {{"heating_power": float, "reasoning": "理由"}}"""
    
            # プロンプトのハッシュ化（キャッシュキー）
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    
            try:
                # キャッシュから取得またはAPI呼び出し
                response = self._cached_llm_call(prompt_hash, prompt)
    
                # JSONパース
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = response
    
                result = json.loads(json_str)
    
                # 安全範囲チェック
                power = max(0.0, min(10.0, result['heating_power']))
    
                if power != result['heating_power']:
                    logger.warning(f"Clamped heating power from {result['heating_power']} to {power}")
    
                return {
                    'action': {'heating_power': power},
                    'reasoning': result.get('reasoning', 'N/A'),
                    'fallback_used': False
                }
    
            except Exception as e:
                # フォールバック: PID制御
                logger.warning(f"LLM failed, using fallback PID controller: {e}")
                self.fallback_count += 1
    
                pid_action = self.fallback_controller.compute(
                    state['temperature'],
                    target['temperature']
                )
    
                return {
                    'action': {'heating_power': pid_action},
                    'reasoning': 'Fallback PID control due to LLM failure',
                    'fallback_used': True
                }
    
        def get_statistics(self) -> Dict:
            """統計情報を取得"""
            return {
                'api_calls': self.api_call_count,
                'cache_hits': self.cache_hit_count,
                'fallback_uses': self.fallback_count,
                'total_cost_usd': self.total_cost,
                'budget_remaining_usd': self.monthly_budget - self.total_cost
            }
    
    
    class SimplePIDController:
        """フォールバック用PID制御器"""
    
        def __init__(self, Kp: float = 0.5, Ki: float = 0.1, Kd: float = 0.05):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.integral = 0.0
            self.prev_error = 0.0
    
        def compute(self, current: float, setpoint: float) -> float:
            """PID制御計算"""
            error = setpoint - current
            self.integral += error
            derivative = error - self.prev_error
            self.prev_error = error
    
            output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    
            # 加熱電力の範囲制限
            return max(0.0, min(10.0, output))
    
    
    # 使用例
    if __name__ == "__main__":
        controller = ProductionLLMController()
    
        # シミュレーション: 20ステップ実行
        state = {'temperature': 348.0, 'concentration': 0.30}
        target = {'temperature': 350.0, 'concentration': 0.30}
    
        for step in range(20):
            print(f"\n=== Step {step + 1} ===")
    
            # 制御判断
            decision = controller.llm_control_decision(state, target)
    
            print(f"Action: {decision['action']}")
            print(f"Reasoning: {decision['reasoning']}")
            print(f"Fallback: {decision['fallback_used']}")
    
            # 状態更新（簡易モデル）
            heating_power = decision['action']['heating_power']
            state['temperature'] += (heating_power - 5.0) * 0.5  # 簡易ダイナミクス
    
            # 10ステップごとに統計表示
            if (step + 1) % 10 == 0:
                stats = controller.get_statistics()
                print(f"\n=== Statistics ===")
                print(f"API Calls: {stats['api_calls']}")
                print(f"Fallback Uses: {stats['fallback_uses']}")
                print(f"Total Cost: ${stats['total_cost_usd']:.2f}")
                print(f"Budget Remaining: ${stats['budget_remaining_usd']:.2f}")
    
        # 最終統計
        final_stats = controller.get_statistics()
        print(f"\n=== Final Statistics ===")
        for key, value in final_stats.items():
            print(f"{key}: {value}")
    
    # 期待される出力例:
    # === Step 1 ===
    # Cache miss for hash a3b5f21c. Calling API...
    # API call #1, Cost: $0.0032, Total: $0.00
    # Action: {'heating_power': 5.5}
    # Reasoning: 温度が目標より2K低いため、加熱電力を中程度に増加
    # Fallback: False
    #
    # === Step 2 ===
    # Action: {'heating_power': 5.5}  # キャッシュヒット
    # Reasoning: 温度が目標より2K低いため、加熱電力を中程度に増加
    # Fallback: False
    # ...
    #
    # === Statistics ===
    # API Calls: 5
    # Fallback Uses: 0
    # Total Cost: $0.02
    # Budget Remaining: $999.98
    

**⚠️ 本番デプロイの重要ポイント**

  1. **必ずフォールバック機構を実装** : API障害時でもプロセスは安全に運転継続
  2. **予算管理** : 月間コスト上限を設定し、超過時はアラート
  3. **レイテンシ対策** : LLM呼び出しを非同期化、キャッシュ活用
  4. **監査ログ** : 全てのLLM判断と理由を記録（規制対応）
  5. **段階的デプロイ** : まずアドバイザリーモード（提案のみ）→監視付き自動制御→完全自動

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ LLMエージェントとRLエージェントの違いと相補性を説明できる
  * ✅ Tool Use（Function Calling）の仕組みを理解している
  * ✅ LangChainのエージェントフレームワークの構成要素を知っている
  * ✅ ハイブリッドアーキテクチャの利点を列挙できる

### 実践スキル

  * ✅ Claude API / OpenAI APIを用いたLLMエージェントを実装できる
  * ✅ Tool Useで外部関数（センサー取得、制御実行）を統合できる
  * ✅ LangChainでメモリ付きエージェントを構築できる
  * ✅ LLM CoordinatorとRL Workersのハイブリッドシステムを実装できる
  * ✅ 本番環境対応の安全機構（フォールバック、予算管理）を実装できる

### 応用力

  * ✅ プロセス異常診断にLLMを適用できる
  * ✅ 人間オペレーターとの協調作業システムを設計できる
  * ✅ LLMのコストとレイテンシを考慮したシステムアーキテクチャを選択できる
  * ✅ 実プラント展開時の安全性とロバスト性を確保できる

## 演習問題

### Easy（基礎確認）

**Q1** : LLMエージェントとRLエージェントの主な違いは何ですか？

解答を見る

**正解** :

  * **学習方法** : RLは試行錯誤で学習、LLMは事前学習済み（推論のみ）
  * **推論速度** : RLは高速（ms単位）、LLMは中速（1-5秒）
  * **説明可能性** : RLはブラックボックス、LLMは自然言語で説明可能
  * **柔軟性** : RLは学習範囲内、LLMは汎用推論可能

**解説** : それぞれの長所を活かし、LLMで高レベル戦略、RLで低レベル制御を担当するハイブリッドアーキテクチャが有効です。

**Q2** : Tool Use（Function Calling）とは何ですか？

解答を見る

**正解** : LLMが外部関数（API、データベース、制御システムなど）を呼び出して、リアルタイム情報を取得したり、アクションを実行したりする機能。

**解説** : 例えば、LLMが「現在の温度は？」という質問に答えるため、get_sensor_data()関数を呼び出してセンサー値を取得します。これにより、LLMは静的な知識だけでなく、動的な環境情報にも基づいて推論できます。

### Medium（応用）

**Q3** : Example 5の本番対応システムで実装されている3つの安全機構を挙げ、それぞれの目的を説明してください。

解答を見る

**正解** :

  1. **フォールバック制御（PID）** : API障害時でもプロセスを安全に運転継続
  2. **予算管理** : 月間コスト上限を設定し、超過時はアラート・停止
  3. **範囲チェック** : LLMの推奨値を物理的安全範囲（0-10kW）に制限

**解説** : これらの多層防御により、LLMの不確実性を許容しつつ、プロセスの安全性を保証します。

**Q4** : LLM-RL Hybridシステムで、LLM Coordinatorが5ステップごとにのみ実行される理由を2つ挙げてください。

解答を見る

**正解** :

  1. **コスト削減** : LLM API呼び出しはコストが高い（$0.003/call程度）ため、頻度を減らす
  2. **レイテンシ回避** : LLM推論は1-5秒かかるため、リアルタイム制御（100ms周期）には不向き

**解説** : LLMは戦略的な判断（長期目標設定）に特化し、RLエージェントが高頻度の制御ループを担当することで、両者の長所を活かします。

### Hard（発展）

**Q5** : 3ユニット（反応器、分離器、熱交換器）のプロセスで、LLM Coordinatorが「エネルギー消費を10%削減しつつ生産量を維持」という目標を達成するための戦略を、以下の制約を考慮して設計してください。

  * 反応器: 温度↓ → 反応速度↓ → 生産量↓
  * 分離器: 還流比↓ → エネルギー↓、純度↓
  * 熱交換器: 伝熱係数↑ → エネルギー効率↑

解答を見る

**推奨戦略** :

  1. **反応器** : 温度を2K下げる（340K → 338K） 
     * 理由: わずかな反応速度低下（~5%）でエネルギー節約
     * 対策: 触媒濃度を5%増加させて反応速度を補償
  2. **分離器** : 還流比を0.8から0.75に削減 
     * 理由: リボイラー負荷を6%削減
     * 対策: 純度低下分（98% → 97%）は下流プロセスで許容範囲内
  3. **熱交換器** : 伝熱面積を10%増加（追加投資） 
     * 理由: 廃熱回収効率を15%向上
     * 長期的なエネルギー削減効果

**総合効果** :

  * エネルギー削減: 5% + 6% + (-1%) = 10%達成
  * 生産量: 触媒増加で補償、維持可能

**LLMの役割** : この複雑なトレードオフ分析を自然言語で推論し、各RLエージェントに具体的な設定値を指示。

**Q6（コーディング）** : Example 2のClaudeProcessControllerを拡張し、以下の機能を追加してください：

  * 新しいTool: check_safety_constraints() - 安全制約違反を検出
  * 安全制約違反時は、LLMが自動的に緊急停止手順を提案

ヒント

Tool定義に以下を追加：
    
    
    {
        "name": "check_safety_constraints",
        "description": "安全制約（温度、圧力、濃度）の違反をチェックします。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

execute_tool()に実装を追加し、違反があれば緊急停止手順を含むメッセージを返してください。

## 参考文献

  1. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." _NeurIPS 2022_.
  2. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." _ICLR 2023_.
  3. Schick, T., et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." _arXiv:2302.04761_.
  4. Anthropic. (2024). "Claude API Documentation - Tool Use." https://docs.anthropic.com/claude/docs/tool-use
  5. Chase, H. (2023). "LangChain: Building Applications with LLMs." https://python.langchain.com
  6. Ahn, M., et al. (2022). "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances." _CoRL 2022_.
  7. Liang, J., et al. (2023). "Code as Policies: Language Model Programs for Embodied Control." _ICRA 2023_.
  8. Song, C. H., et al. (2023). "LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models." _ICCV 2023_.
  9. Huang, W., et al. (2022). "Inner Monologue: Embodied Reasoning through Planning with Language Models." _CoRL 2022_.
  10. Kirk, R., et al. (2023). "A Survey of Zero-Shot Generalisation in Deep Reinforcement Learning." _JAIR_ , 76, 201-264.

## 次のステップ

おめでとうございます！本シリーズ全6章を完了しました。

**🎓 習得したスキル**

  * ✅ エージェントアーキテクチャ（Reactive, Deliberative, Hybrid）
  * ✅ プロセス環境のモデリング（OpenAI Gym準拠）
  * ✅ 報酬設計と最適化目的
  * ✅ マルチエージェント協調制御（QMIX, Communication）
  * ✅ 実プラントデプロイと安全性（CBF, CQL）
  * ✅ LLMとRLの統合（Tool Use, LangChain, Hybrid Architecture）

### 推奨される次のアクション

**短期（1-2週間）:**

  * ✅ 自社プロセスのシミュレーション環境を構築
  * ✅ Chapter 1-6のコード例を実際のデータで実行
  * ✅ LLMエージェントでプロセス診断レポートを自動生成

**中期（1-3ヶ月）:**

  * ✅ RLエージェントをシミュレーション環境で訓練
  * ✅ LLM Coordinatorと統合したハイブリッドシステムを構築
  * ✅ 段階的デプロイ計画を策定（アドバイザリーモード → 自動制御）

**長期（6ヶ月以上）:**

  * ✅ パイロットプラントでの実証試験
  * ✅ 実プラント展開と運用データ収集
  * ✅ 学会発表・論文執筆

[← 第5章：実プラントデプロイ](<./chapter-5.html>) [シリーズ目次に戻る](<./index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
