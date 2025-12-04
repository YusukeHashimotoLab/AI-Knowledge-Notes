---
title: "第2章: ツール使用とFunction Calling"
chapter_title: "第2章: ツール使用とFunction Calling"
---

## Function Callingとは

### 概要と必要性

**Function Calling** （関数呼び出し）は、LLMが外部の関数やAPIを呼び出すための標準化されたインターフェースです。OpenAI（2023年6月）、Anthropic（2023年11月）が相次いで導入し、AIエージェントの中核技術となっています。

**なぜFunction Callingが必要か** ：

  * ✅ **最新情報へのアクセス** : LLMの訓練データは古いが、APIで最新データを取得
  * ✅ **計算能力** : 正確な計算、データ処理をツールに委譲
  * ✅ **外部システム連携** : データベース、CRM、ERPとの統合
  * ✅ **構造化された出力** : JSONスキーマで型安全な実行
  * ✅ **確実性** : 重要な操作をコードで確実に実行

### Function Callingの仕組み
    
    
    ```mermaid
    sequenceDiagram
        participant User
        participant LLM
        participant Tool
    
        User->>LLM: 質問（例: 東京の天気は？）
        LLM->>LLM: 推論（天気APIを使うべき）
        LLM-->>User: Function Call要求{name: "get_weather", args: {location: "東京"}}
        User->>Tool: ツール実行
        Tool-->>User: 結果（晴れ、22度）
        User->>LLM: 結果を渡す
        LLM-->>User: 最終回答（東京は晴れで気温22度です）
    ```

## OpenAI Function Calling

### 基本的な使い方
    
    
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key="your-api-key")
    
    # ステップ1: ツール（関数）を定義
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "指定された場所の現在の天気を取得します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "都市名（例: 東京、大阪）"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度の単位"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # ステップ2: LLMに問い合わせ
    messages = [{"role": "user", "content": "東京の天気を教えてください"}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 自動でツールを選択
    )
    
    # ステップ3: Function Callの確認
    message = response.choices[0].message
    
    if message.tool_calls:
        # LLMがツールを呼び出したい
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
    
        print(f"関数名: {function_name}")
        print(f"引数: {function_args}")
        # 出力:
        # 関数名: get_current_weather
        # 引数: {'location': '東京', 'unit': 'celsius'}
    
        # ステップ4: 実際にツールを実行
        def get_current_weather(location, unit="celsius"):
            """天気APIを呼び出す（ここではモック）"""
            weather_data = {
                "location": location,
                "temperature": 22,
                "unit": unit,
                "condition": "晴れ"
            }
            return json.dumps(weather_data, ensure_ascii=False)
    
        function_response = get_current_weather(**function_args)
    
        # ステップ5: 結果をLLMに返して最終回答を生成
        messages.append(message)  # LLMのツール呼び出しメッセージを追加
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_response
        })
    
        # 最終回答を取得
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
    
        print(final_response.choices[0].message.content)
        # 出力: 東京の天気は晴れで、気温は22度です。
    

### 複数ツールの定義と選択
    
    
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key="your-api-key")
    
    # 複数のツールを定義
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Web検索を実行して最新情報を取得します",
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
                "name": "calculate",
                "description": "数学的な計算を実行します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "計算式（例: 2 + 2, sqrt(16)）"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "株価情報を取得します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "株式シンボル（例: AAPL, GOOGL）"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]
    
    # LLMが適切なツールを選択
    def run_agent_with_tools(user_query):
        """ツールを使ったエージェント実行"""
        messages = [{"role": "user", "content": user_query}]
    
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
    
        message = response.choices[0].message
    
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
    
                print(f"選択されたツール: {function_name}")
                print(f"引数: {function_args}")
    
        return message
    
    # 異なるクエリで試す
    queries = [
        "123 + 456 を計算してください",
        "Appleの株価を教えてください",
        "2024年のノーベル賞受賞者を教えてください"
    ]
    
    for query in queries:
        print(f"\nクエリ: {query}")
        run_agent_with_tools(query)
    
    # 出力例:
    # クエリ: 123 + 456 を計算してください
    # 選択されたツール: calculate
    # 引数: {'expression': '123 + 456'}
    #
    # クエリ: Appleの株価を教えてください
    # 選択されたツール: get_stock_price
    # 引数: {'symbol': 'AAPL'}
    #
    # クエリ: 2024年のノーベル賞受賞者を教えてください
    # 選択されたツール: search_web
    # 引数: {'query': '2024年 ノーベル賞 受賞者'}
    

## Anthropic Tool Use

### Claudeでのツール使用
    
    
    import anthropic
    import json
    
    client = anthropic.Anthropic(api_key="your-api-key")
    
    # ツール定義
    tools = [
        {
            "name": "get_weather",
            "description": "指定された場所の天気情報を取得します",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "都市名"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    # エージェント実行
    def run_claude_agent(user_message):
        """Claudeエージェントの実行"""
        messages = [{"role": "user", "content": user_message}]
    
        while True:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                tools=tools,
                messages=messages
            )
    
            # レスポンスの処理
            if response.stop_reason == "tool_use":
                # ツール呼び出しを処理
                tool_use_block = next(
                    block for block in response.content
                    if block.type == "tool_use"
                )
    
                tool_name = tool_use_block.name
                tool_input = tool_use_block.input
    
                print(f"ツール使用: {tool_name}")
                print(f"入力: {tool_input}")
    
                # ツールを実行
                if tool_name == "get_weather":
                    result = get_weather(**tool_input)
                else:
                    result = "Unknown tool"
    
                # 結果をClaudeに返す
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": result
                    }]
                })
    
            elif response.stop_reason == "end_turn":
                # 最終回答
                final_answer = next(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
                return final_answer
    
    def get_weather(location):
        """天気取得関数"""
        return json.dumps({
            "location": location,
            "temperature": 22,
            "condition": "晴れ"
        }, ensure_ascii=False)
    
    # 実行
    answer = run_claude_agent("東京の天気を教えてください")
    print(f"回答: {answer}")
    

## ツール定義とスキーマ設計

### 効果的なツールスキーマの設計原則

#### 1\. 明確な説明文
    
    
    # 悪い例
    {
        "name": "search",
        "description": "検索する"
    }
    
    # 良い例
    {
        "name": "search_products",
        "description": """商品データベースを検索して関連商品を取得します。
        商品名、カテゴリ、価格範囲で検索できます。
        最大10件の商品情報を返します。"""
    }
    

#### 2\. 適切なパラメータ設計
    
    
    search_products_tool = {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "商品データベースを検索します",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索キーワード（商品名やカテゴリ）"
                    },
                    "min_price": {
                        "type": "number",
                        "description": "最低価格（円）",
                        "minimum": 0
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高価格（円）",
                        "minimum": 0
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books", "food"],
                        "description": "商品カテゴリ"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_asc", "price_desc", "popularity"],
                        "description": "ソート順",
                        "default": "popularity"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "取得件数",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
    

#### 3\. エラーケースの明示
    
    
    def search_products(query, min_price=None, max_price=None,
                             category=None, sort_by="popularity", limit=10):
        """
        商品を検索します。
    
        Returns:
            dict: 成功時は商品リスト、エラー時はエラー情報
            {
                "success": bool,
                "data": [...],  # 成功時
                "error": str,   # エラー時
                "error_code": str  # エラー時
            }
        """
        try:
            # バリデーション
            if not query or len(query) < 2:
                return {
                    "success": False,
                    "error": "検索クエリは2文字以上必要です",
                    "error_code": "INVALID_QUERY"
                }
    
            if min_price and max_price and min_price > max_price:
                return {
                    "success": False,
                    "error": "最低価格が最高価格を上回っています",
                    "error_code": "INVALID_PRICE_RANGE"
                }
    
            # 検索実行（データベースアクセスなど）
            results = perform_search(query, min_price, max_price, category, sort_by, limit)
    
            return {
                "success": True,
                "data": results,
                "total": len(results)
            }
    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "INTERNAL_ERROR"
            }
    

## エラーハンドリングとリトライ

### 堅牢なツール実行
    
    
    import time
    import logging
    from typing import Any, Dict, Callable
    
    class ToolExecutor:
        """安全なツール実行のためのラッパー"""
    
        def __init__(self, max_retries=3, timeout=30):
            self.max_retries = max_retries
            self.timeout = timeout
            self.logger = logging.getLogger(__name__)
    
        def execute(self, tool_func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
            """
            ツールを安全に実行
    
            Args:
                tool_func: 実行する関数
                args: 関数の引数
    
            Returns:
                実行結果またはエラー情報
            """
            for attempt in range(self.max_retries):
                try:
                    # タイムアウト付き実行
                    result = self._execute_with_timeout(tool_func, args)
    
                    return {
                        "success": True,
                        "result": result,
                        "attempt": attempt + 1
                    }
    
                except TimeoutError:
                    self.logger.warning(f"ツール実行がタイムアウト (試行 {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return {
                            "success": False,
                            "error": "タイムアウトしました",
                            "error_type": "timeout"
                        }
    
                except ValueError as e:
                    # バリデーションエラーはリトライしない
                    self.logger.error(f"バリデーションエラー: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "error_type": "validation"
                    }
    
                except Exception as e:
                    # その他のエラー
                    self.logger.error(f"ツール実行エラー: {str(e)} (試行 {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return {
                            "success": False,
                            "error": str(e),
                            "error_type": "execution"
                        }
    
            return {
                "success": False,
                "error": "最大リトライ回数に到達しました",
                "error_type": "max_retries"
            }
    
        def _execute_with_timeout(self, func: Callable, args: Dict[str, Any]) -> Any:
            """タイムアウト付きで関数を実行"""
            import signal
    
            def timeout_handler(signum, frame):
                raise TimeoutError()
    
            # タイムアウトを設定
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
    
            try:
                result = func(**args)
                signal.alarm(0)  # タイムアウトをキャンセル
                return result
            except:
                signal.alarm(0)
                raise
    
    # 使用例
    executor = ToolExecutor(max_retries=3, timeout=10)
    
    def risky_api_call(param):
        """不安定なAPI呼び出し"""
        import random
        if random.random() < 0.3:
            raise ConnectionError("API接続エラー")
        return {"data": f"結果: {param}"}
    
    result = executor.execute(risky_api_call, {"param": "test"})
    print(result)
    

## 外部API統合

### 実用的なAPI統合例

#### 1\. 天気API統合（OpenWeatherMap）
    
    
    import requests
    from typing import Dict, Optional
    
    class WeatherTool:
        """天気情報取得ツール"""
    
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
        def get_weather(self, location: str, unit: str = "metric") -> Dict:
            """
            指定された場所の天気情報を取得
    
            Args:
                location: 都市名
                unit: 温度単位 (metric: 摂氏, imperial: 華氏)
    
            Returns:
                天気情報の辞書
            """
            try:
                params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": unit,
                    "lang": "ja"
                }
    
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
    
                data = response.json()
    
                return {
                    "success": True,
                    "location": data["name"],
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "wind_speed": data["wind"]["speed"]
                }
    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    return {
                        "success": False,
                        "error": f"都市 '{location}' が見つかりません"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"APIエラー: {str(e)}"
                    }
    
            except requests.exceptions.Timeout:
                return {
                    "success": False,
                    "error": "APIリクエストがタイムアウトしました"
                }
    
            except Exception as e:
                return {
                    "success": False,
                    "error": f"予期しないエラー: {str(e)}"
                }
    
    # エージェントへの統合
    weather_tool = WeatherTool(api_key="your-openweathermap-api-key")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "指定された場所の現在の天気情報を取得します",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "都市名（例: Tokyo, Osaka）"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # ツール実行時
    # result = weather_tool.get_weather("Tokyo")
    

## ツールチェーンと連携

### 複数ツールの連携実行
    
    
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key="your-api-key")
    
    class AgentWithToolChain:
        """ツールチェーンを持つエージェント"""
    
        def __init__(self):
            self.tools = {
                "search_company": self.search_company,
                "get_stock_price": self.get_stock_price,
                "calculate_change": self.calculate_change
            }
    
            self.tool_definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_company",
                        "description": "企業名から株式シンボルを検索",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "company_name": {"type": "string"}
                            },
                            "required": ["company_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_stock_price",
                        "description": "株式シンボルから現在価格を取得",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"}
                            },
                            "required": ["symbol"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate_change",
                        "description": "価格変動率を計算",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "current_price": {"type": "number"},
                                "previous_price": {"type": "number"}
                            },
                            "required": ["current_price", "previous_price"]
                        }
                    }
                }
            ]
    
        def search_company(self, company_name: str) -> str:
            """企業名から株式シンボルを検索（モック）"""
            mapping = {
                "Apple": "AAPL",
                "Microsoft": "MSFT",
                "Google": "GOOGL"
            }
            symbol = mapping.get(company_name, "UNKNOWN")
            return json.dumps({"symbol": symbol})
    
        def get_stock_price(self, symbol: str) -> str:
            """株価取得（モック）"""
            prices = {
                "AAPL": {"current": 150.25, "previous": 148.50},
                "MSFT": {"current": 380.75, "previous": 375.00},
                "GOOGL": {"current": 140.50, "previous": 142.00}
            }
            data = prices.get(symbol, {"current": 0, "previous": 0})
            return json.dumps(data)
    
        def calculate_change(self, current_price: float, previous_price: float) -> str:
            """価格変動率を計算"""
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            return json.dumps({
                "change": round(change, 2),
                "change_percent": round(change_percent, 2)
            })
    
        def run(self, user_query: str, max_iterations: int = 10) -> str:
            """エージェント実行（ツールチェーン対応）"""
            messages = [{"role": "user", "content": user_query}]
    
            for i in range(max_iterations):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto"
                )
    
                message = response.choices[0].message
    
                if not message.tool_calls:
                    # 最終回答
                    return message.content
    
                # ツール呼び出しを処理
                messages.append(message)
    
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
    
                    # ツール実行
                    if function_name in self.tools:
                        result = self.tools[function_name](**function_args)
                    else:
                        result = json.dumps({"error": "Unknown tool"})
    
                    # 結果を追加
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result
                    })
    
            return "最大反復回数に達しました"
    
    # 使用例
    agent = AgentWithToolChain()
    answer = agent.run("Appleの株価は昨日と比べてどう変化しましたか？")
    print(answer)
    # LLMは自動的に: search_company → get_stock_price → calculate_change と連携実行
    

## セキュリティとレート制限

### 安全なエージェント設計

#### 1\. ツール実行の承認フロー
    
    
    class SecureAgent:
        """承認フローを持つセキュアなエージェント"""
    
        def __init__(self, require_approval_for=None):
            self.require_approval_for = require_approval_for or [
                "delete_*",
                "send_email",
                "make_payment"
            ]
    
        def requires_approval(self, tool_name: str) -> bool:
            """ツールが承認を必要とするか判定"""
            import fnmatch
            for pattern in self.require_approval_for:
                if fnmatch.fnmatch(tool_name, pattern):
                    return True
            return False
    
        def request_approval(self, tool_name: str, args: dict) -> bool:
            """ユーザーに承認を求める"""
            print(f"\n⚠️  承認が必要です")
            print(f"ツール: {tool_name}")
            print(f"引数: {args}")
            response = input("実行しますか？ (yes/no): ")
            return response.lower() == "yes"
    
        def execute_tool(self, tool_name: str, args: dict):
            """承認フロー付きツール実行"""
            if self.requires_approval(tool_name):
                if not self.request_approval(tool_name, args):
                    return {"success": False, "error": "ユーザーが拒否しました"}
    
            # ツール実行
            return self.tools[tool_name](**args)
    

#### 2\. レート制限の実装
    
    
    import time
    from collections import defaultdict
    
    class RateLimiter:
        """ツール実行のレート制限"""
    
        def __init__(self, max_calls_per_minute=10):
            self.max_calls = max_calls_per_minute
            self.calls = defaultdict(list)
    
        def allow_call(self, tool_name: str) -> bool:
            """呼び出しを許可するか判定"""
            now = time.time()
            one_minute_ago = now - 60
    
            # 1分以内の呼び出しをフィルタ
            self.calls[tool_name] = [
                t for t in self.calls[tool_name]
                if t > one_minute_ago
            ]
    
            if len(self.calls[tool_name]) >= self.max_calls:
                return False
    
            self.calls[tool_name].append(now)
            return True
    
    # 使用例
    limiter = RateLimiter(max_calls_per_minute=5)
    
    if limiter.allow_call("expensive_api"):
        result = call_expensive_api()
    else:
        print("レート制限に達しました。しばらくお待ちください。")
    

## まとめ

### この章で学んだこと

  * ✅ **Function Calling** : OpenAI/AnthropicのFunction Calling API
  * ✅ **ツールスキーマ** : JSONスキーマによる型安全な定義
  * ✅ **エラーハンドリング** : リトライ、タイムアウト、エラー処理
  * ✅ **外部API統合** : 天気API、株価APIなどの実装
  * ✅ **ツールチェーン** : 複数ツールの連携実行
  * ✅ **セキュリティ** : 承認フロー、レート制限

> **重要な設計原則** ：ツールは明確な責務を持ち、エラーを適切にハンドリングし、セキュリティを考慮して設計する

[← 第1章: エージェント基礎](<./chapter1-agent-basics.html>) [第3章: マルチエージェント →](<./chapter3-multi-agent.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
