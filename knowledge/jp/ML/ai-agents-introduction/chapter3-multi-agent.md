---
title: "第3章: マルチエージェントシステム"
chapter_title: "第3章: マルチエージェントシステム"
---

## マルチエージェントシステムとは

### なぜ複数のエージェントが必要か

単一のエージェントでは解決が困難な複雑タスクを、専門化された複数のエージェントが協調して処理することで、より効率的かつ高品質な結果を得られます。

**マルチエージェントの利点** ：

  * ✅ **専門化** : 各エージェントが特定の役割に特化
  * ✅ **並列処理** : 独立したタスクを同時実行
  * ✅ **スケーラビリティ** : エージェント数を動的に調整
  * ✅ **フォールトトレランス** : 一部の失敗を他がカバー
  * ✅ **モジュール性** : エージェントの追加・削除が容易

### マルチエージェントアーキテクチャの種類

アーキテクチャ | 特徴 | 適用場面  
---|---|---  
**並列型** | エージェントが独立して並行実行 | データ収集、複数視点の分析  
**順次型** | エージェントが順番に処理を引き継ぐ | パイプライン処理、段階的改善  
**階層型** | マネージャーが配下のワーカーを制御 | 複雑な計画と実行の分離  
**対話型** | エージェント間で議論・交渉 | 意思決定、合意形成  
  
## マルチエージェント設計

### エージェントの役割分担
    
    
    ```mermaid
    graph TD
        M[Manager Agentタスク分解と調整] --> R[Researcher Agent情報収集]
        M --> W[Writer Agent文書作成]
        M --> C[Critic Agentレビューと評価]
    
        R --> M
        W --> M
        C --> M
    
        style M fill:#e3f2fd
        style R fill:#fff3e0
        style W fill:#f3e5f5
        style C fill:#e8f5e9
    ```

### 役割ベース設計の例
    
    
    from typing import List, Dict, Any
    from dataclasses import dataclass
    
    @dataclass
    class AgentRole:
        """エージェントの役割定義"""
        name: str
        description: str
        capabilities: List[str]
        system_prompt: str
    
    # エージェント役割の定義
    RESEARCHER_ROLE = AgentRole(
        name="Researcher",
        description="情報収集と分析の専門家",
        capabilities=["web_search", "database_query", "data_analysis"],
        system_prompt="""あなたは優秀なリサーチャーです。
    
    役割:
    - Web検索やデータベースから関連情報を収集
    - 収集した情報の信頼性を評価
    - 重要なポイントを要約してチームに報告
    
    重要事項:
    - 情報源を明記する
    - 複数の情報源をクロスチェック
    - 不確実な情報は明示する"""
    )
    
    WRITER_ROLE = AgentRole(
        name="Writer",
        description="文書作成の専門家",
        capabilities=["content_generation", "formatting", "editing"],
        system_prompt="""あなたは優秀なライターです。
    
    役割:
    - リサーチャーの情報を基に高品質な文書を作成
    - 読みやすく、論理的な構成
    - 対象読者に適した文体とトーン
    
    重要事項:
    - 明確で簡潔な表現
    - 適切な見出しと段落構成
    - 引用と参照を適切に使用"""
    )
    
    CRITIC_ROLE = AgentRole(
        name="Critic",
        description="品質レビューの専門家",
        capabilities=["quality_check", "fact_verification", "feedback"],
        system_prompt="""あなたは批評的思考力のあるレビュアーです。
    
    役割:
    - 作成された文書を批判的にレビュー
    - 事実の正確性を検証
    - 改善点を具体的に提示
    
    重要事項:
    - 建設的なフィードバック
    - 具体的な改善提案
    - 重大な問題は明確に指摘"""
    )
    

## 通信プロトコル

### メッセージパッシング

エージェント間の通信は、構造化されたメッセージを介して行います。
    
    
    from dataclasses import dataclass
    from typing import Optional, Dict, Any
    from datetime import datetime
    from enum import Enum
    
    class MessageType(Enum):
        """メッセージタイプ"""
        TASK = "task"              # タスク指示
        RESULT = "result"          # タスク結果
        QUERY = "query"            # 情報要求
        RESPONSE = "response"      # 情報応答
        ERROR = "error"            # エラー通知
        STATUS = "status"          # 状態更新
    
    @dataclass
    class Message:
        """エージェント間メッセージ"""
        type: MessageType
        sender: str
        receiver: str
        content: Dict[str, Any]
        timestamp: datetime
        message_id: str
        reply_to: Optional[str] = None
    
    class MessageBus:
        """メッセージバス（エージェント間通信）"""
    
        def __init__(self):
            self.messages: List[Message] = []
            self.subscribers: Dict[str, List[callable]] = {}
    
        def subscribe(self, agent_name: str, callback: callable):
            """エージェントをメッセージ受信に登録"""
            if agent_name not in self.subscribers:
                self.subscribers[agent_name] = []
            self.subscribers[agent_name].append(callback)
    
        def publish(self, message: Message):
            """メッセージを配信"""
            self.messages.append(message)
    
            # 受信者にメッセージを配信
            if message.receiver in self.subscribers:
                for callback in self.subscribers[message.receiver]:
                    callback(message)
    
        def broadcast(self, message: Message):
            """全エージェントにブロードキャスト"""
            self.messages.append(message)
    
            for agent_name, callbacks in self.subscribers.items():
                if agent_name != message.sender:
                    for callback in callbacks:
                        callback(message)
    
    # 使用例
    import uuid
    
    bus = MessageBus()
    
    def researcher_receive(message: Message):
        print(f"Researcher received: {message.type.value} from {message.sender}")
    
    def writer_receive(message: Message):
        print(f"Writer received: {message.type.value} from {message.sender}")
    
    # エージェントを登録
    bus.subscribe("researcher", researcher_receive)
    bus.subscribe("writer", writer_receive)
    
    # メッセージ送信
    task_message = Message(
        type=MessageType.TASK,
        sender="manager",
        receiver="researcher",
        content={"task": "2024年のAI動向を調査"},
        timestamp=datetime.now(),
        message_id=str(uuid.uuid4())
    )
    
    bus.publish(task_message)
    

### 共有メモリアプローチ
    
    
    from typing import Dict, Any, Optional
    import threading
    
    class SharedMemory:
        """エージェント間の共有メモリ"""
    
        def __init__(self):
            self.data: Dict[str, Any] = {}
            self.lock = threading.Lock()
            self.subscribers: Dict[str, List[callable]] = {}
    
        def write(self, key: str, value: Any, agent_name: str):
            """データを書き込み"""
            with self.lock:
                self.data[key] = {
                    "value": value,
                    "author": agent_name,
                    "timestamp": datetime.now()
                }
    
                # 変更を通知
                self._notify_subscribers(key, value, agent_name)
    
        def read(self, key: str) -> Optional[Any]:
            """データを読み込み"""
            with self.lock:
                if key in self.data:
                    return self.data[key]["value"]
                return None
    
        def subscribe_to_key(self, key: str, callback: callable):
            """特定のキーの変更を監視"""
            if key not in self.subscribers:
                self.subscribers[key] = []
            self.subscribers[key].append(callback)
    
        def _notify_subscribers(self, key: str, value: Any, agent_name: str):
            """サブスクライバーに通知"""
            if key in self.subscribers:
                for callback in self.subscribers[key]:
                    callback(key, value, agent_name)
    
    # 使用例
    memory = SharedMemory()
    
    def on_research_complete(key, value, agent_name):
        print(f"Research completed by {agent_name}: {value}")
    
    memory.subscribe_to_key("research_result", on_research_complete)
    
    # Researcherが結果を書き込み
    memory.write("research_result", "2024年のAI主要トレンド...", "researcher")
    

## 協調パターン

### 1\. 並列実行パターン
    
    
    import asyncio
    from typing import List, Dict, Any
    
    class ParallelAgentSystem:
        """並列実行エージェントシステム"""
    
        def __init__(self, agents: List[Any]):
            self.agents = agents
    
        async def execute_parallel(self, task: str) -> List[Dict[str, Any]]:
            """全エージェントを並列実行"""
            tasks = [
                agent.process(task)
                for agent in self.agents
            ]
    
            results = await asyncio.gather(*tasks, return_exceptions=True)
    
            # 結果を集約
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Agent {i} failed: {str(result)}")
                else:
                    successful_results.append({
                        "agent": self.agents[i].name,
                        "result": result
                    })
    
            return successful_results
    
    # 使用例（擬似コード）
    class ResearchAgent:
        def __init__(self, name: str, specialty: str):
            self.name = name
            self.specialty = specialty
    
        async def process(self, query: str) -> Dict[str, Any]:
            # 非同期でリサーチを実行
            await asyncio.sleep(1)  # API呼び出しをシミュレート
            return {
                "specialty": self.specialty,
                "findings": f"{self.specialty}に関する{query}の調査結果"
            }
    
    # 複数の専門エージェントを並列実行
    agents = [
        ResearchAgent("Tech Researcher", "技術動向"),
        ResearchAgent("Market Researcher", "市場分析"),
        ResearchAgent("Academic Researcher", "学術研究")
    ]
    
    system = ParallelAgentSystem(agents)
    results = asyncio.run(system.execute_parallel("2024年のAI"))
    print(results)
    

### 2\. 順次実行（パイプライン）パターン
    
    
    from typing import Any, List, Callable
    
    class SequentialAgentSystem:
        """順次実行エージェントシステム（パイプライン）"""
    
        def __init__(self):
            self.pipeline: List[Callable] = []
    
        def add_stage(self, agent: Callable):
            """パイプラインにエージェントを追加"""
            self.pipeline.append(agent)
    
        def execute(self, initial_input: Any) -> Any:
            """パイプラインを実行"""
            current_data = initial_input
    
            for i, agent in enumerate(self.pipeline):
                print(f"Stage {i+1}: {agent.__name__}")
                current_data = agent(current_data)
                print(f"  Output: {current_data}\n")
    
            return current_data
    
    # パイプライン各段階のエージェント
    def data_collector(query: str) -> Dict[str, Any]:
        """ステージ1: データ収集"""
        return {
            "query": query,
            "raw_data": f"{query}に関する生データ...",
            "sources": ["source1", "source2"]
        }
    
    def data_analyzer(data: Dict[str, Any]) -> Dict[str, Any]:
        """ステージ2: データ分析"""
        data["analysis"] = "分析結果: 主要なトレンドは..."
        data["insights"] = ["インサイト1", "インサイト2"]
        return data
    
    def report_generator(data: Dict[str, Any]) -> str:
        """ステージ3: レポート生成"""
        report = f"""
    調査レポート: {data['query']}
    
    分析結果:
    {data['analysis']}
    
    主要インサイト:
    - {data['insights'][0]}
    - {data['insights'][1]}
    
    情報源: {', '.join(data['sources'])}
        """
        return report.strip()
    
    # パイプライン構築と実行
    pipeline = SequentialAgentSystem()
    pipeline.add_stage(data_collector)
    pipeline.add_stage(data_analyzer)
    pipeline.add_stage(report_generator)
    
    final_report = pipeline.execute("AIエージェントの最新動向")
    print("=== 最終レポート ===")
    print(final_report)
    

### 3\. 階層型（Manager-Worker）パターン
    
    
    from typing import List, Dict, Any
    from openai import OpenAI
    
    class ManagerAgent:
        """マネージャーエージェント（タスク分解と調整）"""
    
        def __init__(self, api_key: str, workers: List[Any]):
            self.client = OpenAI(api_key=api_key)
            self.workers = workers
            self.task_history = []
    
        def execute(self, user_request: str) -> str:
            """ユーザーリクエストを処理"""
            # ステップ1: タスクを分解
            subtasks = self.decompose_task(user_request)
    
            # ステップ2: ワーカーに割り当て
            results = self.delegate_to_workers(subtasks)
    
            # ステップ3: 結果を統合
            final_result = self.synthesize_results(user_request, results)
    
            return final_result
    
        def decompose_task(self, request: str) -> List[Dict[str, Any]]:
            """タスクをサブタスクに分解"""
            worker_capabilities = "\n".join([
                f"- {w.name}: {w.capabilities}"
                for w in self.workers
            ])
    
            prompt = f"""以下のリクエストを、利用可能なワーカーに割り当てるサブタスクに分解してください。
    
    リクエスト: {request}
    
    利用可能なワーカー:
    {worker_capabilities}
    
    各サブタスクを以下の形式で出力してください:
    1. [ワーカー名] タスク内容
    2. [ワーカー名] タスク内容
    ..."""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
    
            # サブタスクをパース（簡易版）
            subtasks = []
            for line in response.choices[0].message.content.split('\n'):
                if line.strip() and line[0].isdigit():
                    parts = line.split(']', 1)
                    if len(parts) == 2:
                        worker_name = parts[0].split('[')[1].strip()
                        task_desc = parts[1].strip()
                        subtasks.append({
                            "worker": worker_name,
                            "task": task_desc
                        })
    
            return subtasks
    
        def delegate_to_workers(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """ワーカーにタスクを委譲"""
            results = []
    
            for subtask in subtasks:
                worker_name = subtask["worker"]
                task = subtask["task"]
    
                # 該当するワーカーを見つける
                worker = next((w for w in self.workers if w.name == worker_name), None)
    
                if worker:
                    result = worker.execute(task)
                    results.append({
                        "worker": worker_name,
                        "task": task,
                        "result": result
                    })
                else:
                    results.append({
                        "worker": worker_name,
                        "task": task,
                        "result": f"エラー: ワーカー {worker_name} が見つかりません"
                    })
    
            return results
    
        def synthesize_results(self, original_request: str, results: List[Dict[str, Any]]) -> str:
            """結果を統合して最終回答を生成"""
            results_text = "\n\n".join([
                f"{r['worker']}の結果:\n{r['result']}"
                for r in results
            ])
    
            prompt = f"""以下のリクエストに対して、各ワーカーの結果を統合して最終回答を生成してください。
    
    元のリクエスト: {original_request}
    
    ワーカーの結果:
    {results_text}
    
    統合された回答を生成してください:"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
    class WorkerAgent:
        """ワーカーエージェント"""
    
        def __init__(self, name: str, capabilities: str, system_prompt: str, api_key: str):
            self.name = name
            self.capabilities = capabilities
            self.system_prompt = system_prompt
            self.client = OpenAI(api_key=api_key)
    
        def execute(self, task: str) -> str:
            """タスクを実行"""
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task}
                ],
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
    # 使用例
    researcher = WorkerAgent(
        name="Researcher",
        capabilities="Web検索、データ収集",
        system_prompt="あなたはリサーチの専門家です。",
        api_key="your-api-key"
    )
    
    writer = WorkerAgent(
        name="Writer",
        capabilities="文書作成、編集",
        system_prompt="あなたは文書作成の専門家です。",
        api_key="your-api-key"
    )
    
    manager = ManagerAgent(
        api_key="your-api-key",
        workers=[researcher, writer]
    )
    
    result = manager.execute("2024年のAI動向について1000字のレポートを作成してください")
    print(result)
    

## オーケストレーション戦略

### 動的タスク割り当て
    
    
    from typing import List, Dict, Any
    import time
    
    class TaskOrchestrator:
        """タスクオーケストレーター"""
    
        def __init__(self, agents: List[Any]):
            self.agents = agents
            self.task_queue = []
            self.agent_status = {agent.name: "idle" for agent in agents}
    
        def add_task(self, task: Dict[str, Any]):
            """タスクをキューに追加"""
            self.task_queue.append(task)
    
        def get_available_agent(self, required_capability: str = None):
            """利用可能なエージェントを取得"""
            for agent in self.agents:
                if self.agent_status[agent.name] == "idle":
                    if required_capability is None or required_capability in agent.capabilities:
                        return agent
            return None
    
        def execute_tasks(self):
            """タスクキューを処理"""
            while self.task_queue:
                task = self.task_queue.pop(0)
    
                # 適切なエージェントを見つける
                agent = self.get_available_agent(task.get("required_capability"))
    
                if agent:
                    print(f"タスク '{task['name']}' を {agent.name} に割り当て")
                    self.agent_status[agent.name] = "busy"
    
                    # タスク実行（非同期を想定）
                    result = agent.execute(task)
    
                    self.agent_status[agent.name] = "idle"
                    print(f"{agent.name} がタスク '{task['name']}' を完了")
                else:
                    # エージェントが利用不可の場合、キューに戻す
                    self.task_queue.append(task)
                    time.sleep(1)
    

## 状態管理とコンフリクト解決

### 分散状態の同期
    
    
    from typing import Dict, Any, Optional
    from datetime import datetime
    import json
    
    class StateManager:
        """エージェント間の状態管理"""
    
        def __init__(self):
            self.state: Dict[str, Any] = {}
            self.version: Dict[str, int] = {}
            self.history: List[Dict[str, Any]] = []
    
        def update_state(self, key: str, value: Any, agent_name: str) -> bool:
            """状態を更新（バージョン管理）"""
            current_version = self.version.get(key, 0)
    
            # 更新を記録
            update_record = {
                "key": key,
                "value": value,
                "agent": agent_name,
                "version": current_version + 1,
                "timestamp": datetime.now().isoformat()
            }
    
            self.state[key] = value
            self.version[key] = current_version + 1
            self.history.append(update_record)
    
            return True
    
        def get_state(self, key: str, version: Optional[int] = None) -> Optional[Any]:
            """状態を取得（特定バージョン対応）"""
            if version is None:
                return self.state.get(key)
    
            # 履歴から特定バージョンを検索
            for record in reversed(self.history):
                if record["key"] == key and record["version"] == version:
                    return record["value"]
    
            return None
    
        def resolve_conflict(self, key: str, conflicting_values: List[Dict[str, Any]]) -> Any:
            """コンフリクトを解決"""
            # 最新タイムスタンプの値を採用（Last-Write-Wins）
            latest = max(conflicting_values, key=lambda x: x["timestamp"])
            return latest["value"]
    
    # 使用例
    state_manager = StateManager()
    
    # 複数エージェントが同じキーを更新
    state_manager.update_state("document_title", "AIエージェント入門", "agent1")
    state_manager.update_state("document_title", "AIエージェント完全ガイド", "agent2")
    
    # 最新の値を取得
    current_title = state_manager.get_state("document_title")
    print(f"現在のタイトル: {current_title}")
    
    # 履歴を確認
    print("\n更新履歴:")
    for record in state_manager.history:
        print(f"  v{record['version']}: {record['value']} (by {record['agent']})")
    

## まとめ

### この章で学んだこと

  * ✅ **マルチエージェント設計** : 役割分担と専門化
  * ✅ **通信プロトコル** : メッセージパッシングと共有メモリ
  * ✅ **協調パターン** : 並列、順次、階層型の実装
  * ✅ **オーケストレーション** : タスク割り当てと調整
  * ✅ **状態管理** : 分散状態の同期とコンフリクト解決

### 設計の原則

> **効果的なマルチエージェントシステム** は、明確な役割分担、効率的な通信、適切なオーケストレーション、そして堅牢な状態管理によって実現される

[← 第2章: ツール使用](<./chapter2-tool-use.html>) [第4章: 実践応用 →](<./chapter4-applications.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
