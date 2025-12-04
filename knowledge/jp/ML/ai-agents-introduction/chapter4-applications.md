---
title: "第4章: 実践的な応用"
chapter_title: "第4章: 実践的な応用"
---

## 実用的なエージェントアプリケーション

この章では、前の章で学んだ技術を統合して、実際のビジネスシーンで活用できるエージェントシステムを構築します。

### 主要なユースケース

ユースケース | 主要機能 | ビジネス価値  
---|---|---  
カスタマーサービス | FAQ対応、問い合わせ分類、エスカレーション | 24/7対応、コスト削減、満足度向上  
コード生成 | 要件分析、コード生成、テスト、デバッグ | 開発速度向上、品質改善  
リサーチアシスタント | 情報収集、分析、レポート生成 | 調査時間削減、洞察の質向上  
タスク自動化 | ワークフロー実行、データ処理 | 業務効率化、エラー削減  
  
## カスタマーサービスエージェント

### システム設計
    
    
    ```mermaid
    graph TD
        U[ユーザー] --> C[分類エージェント]
        C --> |FAQ| F[FAQ応答エージェント]
        C --> |技術的問題| T[技術サポートエージェント]
        C --> |注文関連| O[注文処理エージェント]
        C --> |複雑な問題| H[人間へエスカレーション]
    
        F --> R[応答生成]
        T --> R
        O --> R
        R --> U
    
        style C fill:#e3f2fd
        style F fill:#fff3e0
        style T fill:#f3e5f5
        style O fill:#e8f5e9
        style H fill:#ffebee
    ```

### 実装例
    
    
    from openai import OpenAI
    from typing import Dict, Any, Optional
    import json
    
    class CustomerServiceAgent:
        """カスタマーサービスエージェント"""
    
        def __init__(self, api_key: str):
            self.client = OpenAI(api_key=api_key)
            self.conversation_history = []
    
            # ツール定義
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_faq",
                        "description": "FAQデータベースから関連情報を検索",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "検索クエリ"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "description": "注文状況を確認",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "注文番号"
                                }
                            },
                            "required": ["order_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "escalate_to_human",
                        "description": "人間のオペレーターにエスカレーション",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "エスカレーションの理由"
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                    "description": "優先度"
                                }
                            },
                            "required": ["reason"]
                        }
                    }
                }
            ]
    
            self.system_prompt = """あなたは親切で有能なカスタマーサービスエージェントです。
    
    対応方針:
    1. ユーザーの問題を正確に理解する
    2. 適切なツールを使って情報を収集する
    3. 明確で親切な回答を提供する
    4. 複雑な問題や重要な案件は人間にエスカレーションする
    
    トーン:
    - 親しみやすく、丁寧
    - 共感を示す
    - 専門的かつわかりやすい
    
    制約:
    - 推測で答えない
    - 個人情報は慎重に扱う
    - 確実でない場合はエスカレーション"""
    
        def handle_inquiry(self, user_message: str) -> str:
            """問い合わせを処理"""
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
    
            max_iterations = 5
            for _ in range(max_iterations):
                # LLMに問い合わせ
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": self.system_prompt}
                    ] + self.conversation_history,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.7
                )
    
                message = response.choices[0].message
    
                if not message.tool_calls:
                    # 最終回答
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": message.content
                    })
                    return message.content
    
                # ツール呼び出しを処理
                self.conversation_history.append(message)
    
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
    
                    # ツールを実行
                    if function_name == "search_faq":
                        result = self.search_faq(**function_args)
                    elif function_name == "get_order_status":
                        result = self.get_order_status(**function_args)
                    elif function_name == "escalate_to_human":
                        result = self.escalate_to_human(**function_args)
                    else:
                        result = {"error": "Unknown tool"}
    
                    # 結果を追加
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
    
            return "申し訳ございません。処理に時間がかかっております。担当者におつなぎいたします。"
    
        def search_faq(self, query: str) -> Dict[str, Any]:
            """FAQを検索（モック）"""
            faq_database = {
                "返品": {
                    "answer": "商品到着後30日以内であれば返品可能です。未開封・未使用の商品に限ります。",
                    "related_links": ["返品ポリシー", "返品フォーム"]
                },
                "配送": {
                    "answer": "通常、ご注文から3-5営業日でお届けします。お急ぎの場合は特急便もご利用いただけます。",
                    "related_links": ["配送オプション", "配送料金"]
                },
                "支払い": {
                    "answer": "クレジットカード、銀行振込、コンビニ払いに対応しております。",
                    "related_links": ["支払い方法", "決済セキュリティ"]
                }
            }
    
            # 簡易的なキーワードマッチング
            for key, value in faq_database.items():
                if key in query:
                    return {
                        "found": True,
                        "answer": value["answer"],
                        "related_links": value["related_links"]
                    }
    
            return {
                "found": False,
                "message": "該当するFAQが見つかりませんでした"
            }
    
        def get_order_status(self, order_id: str) -> Dict[str, Any]:
            """注文状況を取得（モック）"""
            # 実際にはデータベースから取得
            mock_orders = {
                "ORD-12345": {
                    "status": "配送中",
                    "tracking_number": "TRK-98765",
                    "estimated_delivery": "2025年10月27日"
                },
                "ORD-67890": {
                    "status": "準備中",
                    "estimated_shipping": "2025年10月26日"
                }
            }
    
            if order_id in mock_orders:
                return {
                    "found": True,
                    "order_id": order_id,
                    **mock_orders[order_id]
                }
            else:
                return {
                    "found": False,
                    "message": f"注文番号 {order_id} が見つかりませんでした"
                }
    
        def escalate_to_human(self, reason: str, priority: str = "medium") -> Dict[str, Any]:
            """人間へエスカレーション（モック）"""
            # 実際にはチケットシステムやCRMに送信
            ticket_id = f"TICKET-{int(time.time())}"
    
            return {
                "escalated": True,
                "ticket_id": ticket_id,
                "priority": priority,
                "estimated_response": "30分以内"
            }
    
    # 使用例
    import time
    
    agent = CustomerServiceAgent(api_key="your-api-key")
    
    # 例1: FAQ質問
    response1 = agent.handle_inquiry("返品ポリシーについて教えてください")
    print(f"エージェント: {response1}\n")
    
    # 例2: 注文状況確認
    response2 = agent.handle_inquiry("注文番号ORD-12345の配送状況を教えてください")
    print(f"エージェント: {response2}\n")
    
    # 例3: 複雑な問題（エスカレーション）
    response3 = agent.handle_inquiry("注文した商品が破損していました。返金してください")
    print(f"エージェント: {response3}")
    

## コード生成エージェント

### マルチステージコード生成
    
    
    from openai import OpenAI
    from typing import Dict, Any, List
    
    class CodeGenerationAgent:
        """コード生成エージェントシステム"""
    
        def __init__(self, api_key: str):
            self.client = OpenAI(api_key=api_key)
    
        def generate_code(self, requirement: str) -> Dict[str, Any]:
            """要件からコードを生成"""
            # ステップ1: 要件分析
            analysis = self.analyze_requirements(requirement)
    
            # ステップ2: コード生成
            code = self.generate_implementation(analysis)
    
            # ステップ3: テスト生成
            tests = self.generate_tests(code, analysis)
    
            # ステップ4: レビュー
            review = self.review_code(code, tests)
    
            return {
                "analysis": analysis,
                "code": code,
                "tests": tests,
                "review": review
            }
    
        def analyze_requirements(self, requirement: str) -> Dict[str, Any]:
            """要件を分析"""
            prompt = f"""以下の要件を分析し、JSON形式で出力してください。
    
    要件: {requirement}
    
    出力形式:
    {{
        "functionality": "主要機能の説明",
        "inputs": ["入力1", "入力2"],
        "outputs": ["出力1", "出力2"],
        "edge_cases": ["エッジケース1", "エッジケース2"],
        "suggested_approach": "実装アプローチ"
    }}"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
    
            return json.loads(response.choices[0].message.content)
    
        def generate_implementation(self, analysis: Dict[str, Any]) -> str:
            """実装コードを生成"""
            prompt = f"""以下の分析結果に基づいて、Pythonコードを生成してください。
    
    分析結果:
    {json.dumps(analysis, indent=2, ensure_ascii=False)}
    
    要件:
    - 型ヒントを使用
    - Docstringを含める
    - エラーハンドリングを実装
    - PEP 8に準拠
    
    コードのみを出力してください:"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
    
            return response.choices[0].message.content
    
        def generate_tests(self, code: str, analysis: Dict[str, Any]) -> str:
            """テストコードを生成"""
            prompt = f"""以下のコードに対するpytestテストケースを生成してください。
    
    コード:
    {code}
    
    分析結果:
    {json.dumps(analysis, indent=2, ensure_ascii=False)}
    
    要件:
    - 正常系のテスト
    - エッジケースのテスト
    - エラーケースのテスト
    - テストカバレッジ80%以上を目指す
    
    pytestコードのみを出力してください:"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
    
            return response.choices[0].message.content
    
        def review_code(self, code: str, tests: str) -> Dict[str, Any]:
            """コードをレビュー"""
            prompt = f"""以下のコードとテストをレビューし、JSON形式で評価してください。
    
    コード:
    {code}
    
    テスト:
    {tests}
    
    評価項目:
    - 可読性 (1-10)
    - 保守性 (1-10)
    - パフォーマンス (1-10)
    - セキュリティ (1-10)
    - テストカバレッジ (1-10)
    
    出力形式:
    {{
        "scores": {{"readability": 8, "maintainability": 7, ...}},
        "strengths": ["強み1", "強み2"],
        "improvements": ["改善点1", "改善点2"],
        "overall_assessment": "総合評価"
    }}"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
    
            return json.loads(response.choices[0].message.content)
    
    # 使用例
    agent = CodeGenerationAgent(api_key="your-api-key")
    
    requirement = """
    メールアドレスのバリデーション関数を作成してください。
    以下の要件を満たす必要があります:
    - RFC 5322に準拠
    - 一般的な形式をサポート
    - ドメインの存在チェックはオプション
    """
    
    result = agent.generate_code(requirement)
    
    print("=== 要件分析 ===")
    print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
    
    print("\n=== 生成されたコード ===")
    print(result["code"])
    
    print("\n=== テストコード ===")
    print(result["tests"])
    
    print("\n=== コードレビュー ===")
    print(json.dumps(result["review"], indent=2, ensure_ascii=False))
    

## リサーチエージェント

### 自動調査レポート生成
    
    
    from openai import OpenAI
    import requests
    from typing import List, Dict, Any
    
    class ResearchAgent:
        """リサーチエージェント"""
    
        def __init__(self, api_key: str, serp_api_key: str = None):
            self.client = OpenAI(api_key=api_key)
            self.serp_api_key = serp_api_key
    
        def research(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
            """トピックについて調査"""
            # ステップ1: リサーチクエリを生成
            queries = self.generate_search_queries(topic, depth)
    
            # ステップ2: 情報収集
            search_results = self.gather_information(queries)
    
            # ステップ3: 情報を分析
            analysis = self.analyze_information(topic, search_results)
    
            # ステップ4: レポート生成
            report = self.generate_report(topic, analysis)
    
            return {
                "topic": topic,
                "queries": queries,
                "sources": len(search_results),
                "analysis": analysis,
                "report": report
            }
    
        def generate_search_queries(self, topic: str, depth: str) -> List[str]:
            """検索クエリを生成"""
            num_queries = {"shallow": 3, "medium": 5, "deep": 8}[depth]
    
            prompt = f"""トピック「{topic}」について調査するための検索クエリを{num_queries}個生成してください。
    
    要件:
    - 異なる視点からのクエリ
    - 最新情報と背景情報の両方
    - 具体的で検索効率の良いクエリ
    
    JSON配列形式で出力してください: ["クエリ1", "クエリ2", ...]"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
    
            return json.loads(response.choices[0].message.content)
    
        def gather_information(self, queries: List[str]) -> List[Dict[str, Any]]:
            """情報を収集"""
            results = []
    
            for query in queries:
                # 実際にはSerpAPIなどを使用
                # ここではモックデータ
                results.append({
                    "query": query,
                    "title": f"{query}に関する情報",
                    "snippet": f"{query}についての詳細な情報...",
                    "source": "example.com",
                    "relevance_score": 0.85
                })
    
            return results
    
        def analyze_information(self, topic: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
            """情報を分析"""
            # 結果を統合
            combined_info = "\n\n".join([
                f"【{r['query']}】\n{r['snippet']}"
                for r in results
            ])
    
            prompt = f"""以下の情報を分析し、トピック「{topic}」について要約してください。
    
    収集した情報:
    {combined_info}
    
    以下の形式でJSON出力してください:
    {{
        "key_findings": ["発見1", "発見2", "発見3"],
        "trends": ["トレンド1", "トレンド2"],
        "challenges": ["課題1", "課題2"],
        "opportunities": ["機会1", "機会2"],
        "summary": "全体の要約"
    }}"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
    
            return json.loads(response.choices[0].message.content)
    
        def generate_report(self, topic: str, analysis: Dict[str, Any]) -> str:
            """レポートを生成"""
            prompt = f"""以下の分析結果を基に、トピック「{topic}」についての調査レポートを生成してください。
    
    分析結果:
    {json.dumps(analysis, indent=2, ensure_ascii=False)}
    
    レポート要件:
    - エグゼクティブサマリー
    - 主要な発見
    - トレンド分析
    - 課題と機会
    - 結論と推奨事項
    - 1500-2000文字程度
    
    プロフェッショナルなビジネスレポート形式で出力してください:"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
    # 使用例
    agent = ResearchAgent(api_key="your-api-key")
    
    result = agent.research("2024年のAIエージェント技術動向", depth="medium")
    
    print("=== 調査レポート ===")
    print(result["report"])
    
    print(f"\n情報源数: {result['sources']}")
    print(f"使用クエリ: {', '.join(result['queries'])}")
    

## タスク自動化エージェント

### ワークフロー自動化
    
    
    from typing import List, Dict, Any, Callable
    from dataclasses import dataclass
    from enum import Enum
    
    class TaskStatus(Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
    
    @dataclass
    class Task:
        """タスク定義"""
        id: str
        name: str
        action: Callable
        dependencies: List[str]
        status: TaskStatus = TaskStatus.PENDING
        result: Any = None
        error: str = None
    
    class WorkflowAutomationAgent:
        """ワークフロー自動化エージェント"""
    
        def __init__(self):
            self.tasks: Dict[str, Task] = {}
            self.execution_log: List[Dict[str, Any]] = []
    
        def add_task(self, task: Task):
            """タスクを追加"""
            self.tasks[task.id] = task
    
        def execute_workflow(self) -> Dict[str, Any]:
            """ワークフローを実行"""
            # トポロジカルソートで実行順序を決定
            execution_order = self._topological_sort()
    
            for task_id in execution_order:
                task = self.tasks[task_id]
    
                # 依存タスクの完了を確認
                if not self._dependencies_satisfied(task):
                    task.status = TaskStatus.FAILED
                    task.error = "依存タスクが完了していません"
                    continue
    
                # タスク実行
                try:
                    task.status = TaskStatus.RUNNING
                    self._log_event(f"タスク開始: {task.name}")
    
                    # 依存タスクの結果を取得
                    dep_results = {
                        dep_id: self.tasks[dep_id].result
                        for dep_id in task.dependencies
                    }
    
                    # タスクを実行
                    task.result = task.action(dep_results)
    
                    task.status = TaskStatus.COMPLETED
                    self._log_event(f"タスク完了: {task.name}")
    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self._log_event(f"タスク失敗: {task.name} - {str(e)}")
    
            return self._generate_summary()
    
        def _dependencies_satisfied(self, task: Task) -> bool:
            """依存タスクが全て完了しているか確認"""
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    return False
                if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    return False
            return True
    
        def _topological_sort(self) -> List[str]:
            """タスクの実行順序を決定（トポロジカルソート）"""
            # 簡易実装（依存関係のあるタスクを後に）
            sorted_tasks = []
            visited = set()
    
            def visit(task_id):
                if task_id in visited:
                    return
                visited.add(task_id)
    
                task = self.tasks[task_id]
                for dep_id in task.dependencies:
                    if dep_id in self.tasks:
                        visit(dep_id)
    
                sorted_tasks.append(task_id)
    
            for task_id in self.tasks:
                visit(task_id)
    
            return sorted_tasks
    
        def _log_event(self, message: str):
            """イベントをログに記録"""
            import datetime
            self.execution_log.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "message": message
            })
    
        def _generate_summary(self) -> Dict[str, Any]:
            """実行結果のサマリーを生成"""
            total = len(self.tasks)
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
    
            return {
                "total_tasks": total,
                "completed": completed,
                "failed": failed,
                "success_rate": completed / total if total > 0 else 0,
                "execution_log": self.execution_log
            }
    
    # 使用例: データ処理ワークフロー
    
    def fetch_data(deps):
        """データ取得"""
        print("データを取得中...")
        return {"data": [1, 2, 3, 4, 5]}
    
    def clean_data(deps):
        """データクリーニング"""
        data = deps["fetch_data"]["data"]
        print(f"データクリーニング中: {data}")
        cleaned = [x * 2 for x in data]
        return {"cleaned_data": cleaned}
    
    def analyze_data(deps):
        """データ分析"""
        cleaned = deps["clean_data"]["cleaned_data"]
        print(f"データ分析中: {cleaned}")
        avg = sum(cleaned) / len(cleaned)
        return {"average": avg, "count": len(cleaned)}
    
    def generate_report(deps):
        """レポート生成"""
        analysis = deps["analyze_data"]
        print(f"レポート生成中...")
        report = f"平均値: {analysis['average']}, データ数: {analysis['count']}"
        return {"report": report}
    
    # ワークフロー構築
    workflow = WorkflowAutomationAgent()
    
    workflow.add_task(Task(
        id="fetch_data",
        name="データ取得",
        action=fetch_data,
        dependencies=[]
    ))
    
    workflow.add_task(Task(
        id="clean_data",
        name="データクリーニング",
        action=clean_data,
        dependencies=["fetch_data"]
    ))
    
    workflow.add_task(Task(
        id="analyze_data",
        name="データ分析",
        action=analyze_data,
        dependencies=["clean_data"]
    ))
    
    workflow.add_task(Task(
        id="generate_report",
        name="レポート生成",
        action=generate_report,
        dependencies=["analyze_data"]
    ))
    
    # 実行
    summary = workflow.execute_workflow()
    
    print("\n=== 実行サマリー ===")
    print(f"総タスク数: {summary['total_tasks']}")
    print(f"完了: {summary['completed']}")
    print(f"失敗: {summary['failed']}")
    print(f"成功率: {summary['success_rate']*100:.1f}%")
    
    # レポート出力
    report_task = workflow.tasks["generate_report"]
    if report_task.status == TaskStatus.COMPLETED:
        print(f"\n最終レポート: {report_task.result['report']}")
    

## 評価とモニタリング

### エージェント性能の測定
    
    
    from typing import Dict, Any, List
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class AgentMetrics:
        """エージェントメトリクス"""
        task_id: str
        success: bool
        latency: float  # 秒
        token_usage: int
        cost: float  # USD
        user_satisfaction: float  # 1-5
        error_type: str = None
    
    class AgentEvaluator:
        """エージェント評価システム"""
    
        def __init__(self):
            self.metrics: List[AgentMetrics] = []
    
        def record_execution(self, metrics: AgentMetrics):
            """実行メトリクスを記録"""
            self.metrics.append(metrics)
    
        def generate_report(self) -> Dict[str, Any]:
            """評価レポートを生成"""
            if not self.metrics:
                return {"error": "メトリクスデータなし"}
    
            total = len(self.metrics)
            successful = sum(1 for m in self.metrics if m.success)
    
            return {
                "overview": {
                    "total_executions": total,
                    "success_rate": successful / total,
                    "avg_latency": sum(m.latency for m in self.metrics) / total,
                    "total_cost": sum(m.cost for m in self.metrics),
                    "avg_satisfaction": sum(m.user_satisfaction for m in self.metrics) / total
                },
                "performance": self._analyze_performance(),
                "errors": self._analyze_errors(),
                "recommendations": self._generate_recommendations()
            }
    
        def _analyze_performance(self) -> Dict[str, Any]:
            """パフォーマンス分析"""
            latencies = [m.latency for m in self.metrics]
            return {
                "p50_latency": sorted(latencies)[len(latencies)//2],
                "p95_latency": sorted(latencies)[int(len(latencies)*0.95)],
                "max_latency": max(latencies)
            }
    
        def _analyze_errors(self) -> Dict[str, Any]:
            """エラー分析"""
            errors = [m for m in self.metrics if not m.success]
            error_types = {}
    
            for error in errors:
                error_type = error.error_type or "unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1
    
            return {
                "total_errors": len(errors),
                "error_distribution": error_types
            }
    
        def _generate_recommendations(self) -> List[str]:
            """改善提案を生成"""
            recommendations = []
    
            # 成功率チェック
            success_rate = sum(1 for m in self.metrics if m.success) / len(self.metrics)
            if success_rate < 0.95:
                recommendations.append("成功率が95%未満です。エラーハンドリングを強化してください")
    
            # レイテンシーチェック
            avg_latency = sum(m.latency for m in self.metrics) / len(self.metrics)
            if avg_latency > 5.0:
                recommendations.append("平均レイテンシーが5秒を超えています。最適化を検討してください")
    
            # コストチェック
            total_cost = sum(m.cost for m in self.metrics)
            if total_cost > 100:
                recommendations.append(f"総コストが${total_cost:.2f}です。コスト最適化を検討してください")
    
            return recommendations
    
    # 使用例
    evaluator = AgentEvaluator()
    
    # メトリクス記録
    evaluator.record_execution(AgentMetrics(
        task_id="task1",
        success=True,
        latency=2.3,
        token_usage=500,
        cost=0.01,
        user_satisfaction=4.5
    ))
    
    evaluator.record_execution(AgentMetrics(
        task_id="task2",
        success=False,
        latency=10.5,
        token_usage=1000,
        cost=0.02,
        user_satisfaction=2.0,
        error_type="timeout"
    ))
    
    # レポート生成
    report = evaluator.generate_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    

## プロダクション考慮事項

### スケーラビリティとコスト最適化

考慮事項 | 課題 | 対策  
---|---|---  
**スケーラビリティ** | 同時リクエスト増加 | 非同期処理、キューイング、水平スケーリング  
**コスト** | APIコストの増大 | キャッシング、モデル選択、プロンプト最適化  
**信頼性** | API障害、タイムアウト | リトライ、フォールバック、サーキットブレーカー  
**セキュリティ** | データ漏洩、不正利用 | 認証、暗号化、レート制限、監査ログ  
**モニタリング** | 問題の早期発見 | メトリクス収集、アラート、ダッシュボード  
  
### ベストプラクティス

  * ✅ **キャッシング** : 同じクエリの結果をキャッシュしてAPI呼び出しを削減
  * ✅ **非同期処理** : asyncioで並列処理を実装
  * ✅ **レート制限** : APIの利用制限を守る
  * ✅ **エラーハンドリング** : リトライとフォールバック戦略
  * ✅ **ログとモニタリング** : 詳細なログ記録と性能監視
  * ✅ **コスト追跡** : トークン使用量とコストの可視化
  * ✅ **セキュリティ** : 認証、暗号化、入力検証
  * ✅ **テスト** : ユニットテスト、統合テスト、E2Eテスト

## まとめ

### この章で学んだこと

  * ✅ **カスタマーサービス** : FAQ対応、問い合わせ分類、エスカレーション
  * ✅ **コード生成** : 要件分析、実装生成、テスト、レビュー
  * ✅ **リサーチエージェント** : 情報収集、分析、レポート生成
  * ✅ **タスク自動化** : ワークフロー実行、依存関係管理
  * ✅ **評価とモニタリング** : メトリクス収集、性能分析
  * ✅ **プロダクション** : スケーラビリティ、コスト、信頼性

### シリーズのまとめ

> AIエージェントは、LLMの推論能力とツール使用を組み合わせ、複雑なタスクを自律的に解決します。効果的なエージェントシステムは、明確な役割定義、堅牢なエラーハンドリング、適切な評価とモニタリングによって実現されます。

### 次のステップ

このシリーズを完了したあなたは、実用的なAIエージェントシステムを設計・実装できます。さらに学びを深めるために：

  * 📚 **AutoGPT/BabyAGI** を探索して自律型エージェントを理解
  * 📚 **LangChain/LlamaIndex** で高度なエージェントを構築
  * 📚 **強化学習** でエージェントを最適化
  * 📚 **実プロジェクト** で学んだ技術を適用

[← 第3章: マルチエージェント](<./chapter3-multi-agent.html>) [シリーズ概要へ戻る](<./index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
