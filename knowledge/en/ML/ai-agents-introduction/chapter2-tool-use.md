---
title: "Chapter 2: Tool Use and Function Calling"
chapter_title: "Chapter 2: Tool Use and Function Calling"
---

This chapter covers Tool Use and Function Calling. You will learn essential concepts and techniques.

## What is Function Calling

### Overview and Necessity

**Function Calling** is a standardized interface for LLMs to call external functions and APIs. Introduced by OpenAI (June 2023) and Anthropic (November 2023), it has become a core technology for AI agents.

**Why Function Calling is Necessary** :

  * ✅ **Access to Latest Information** : While LLM training data is outdated, APIs retrieve current data
  * ✅ **Computational Capability** : Delegate accurate calculations and data processing to tools
  * ✅ **External System Integration** : Integration with databases, CRM, ERP systems
  * ✅ **Structured Output** : Type-safe execution with JSON schema
  * ✅ **Reliability** : Execute critical operations reliably with code

### How Function Calling Works
    
    
    ```mermaid
    sequenceDiagram
        participant User
        participant LLM
        participant Tool
    
        User->>LLM: Question (e.g., What's the weather in Tokyo?)
        LLM->>LLM: Reasoning (Should use weather API)
        LLM-->>User: Function Call request{name: "get_weather", args: {location: "Tokyo"}}
        User->>Tool: Execute tool
        Tool-->>User: Result (Sunny, 22°C)
        User->>LLM: Pass result
        LLM-->>User: Final answer (It's sunny in Tokyo with temperature 22°C)
    ```

## OpenAI Function Calling

### Basic Usage
    
    
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key="your-api-key")
    
    # Step 1: Define tools (functions)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a specified location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name (e.g., Tokyo, Osaka)"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Step 2: Query the LLM
    messages = [{"role": "user", "content": "Please tell me the weather in Tokyo"}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Automatically select tool
    )
    
    # Step 3: Check Function Call
    message = response.choices[0].message
    
    if message.tool_calls:
        # LLM wants to call a tool
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
    
        print(f"Function name: {function_name}")
        print(f"Arguments: {function_args}")
        # Output:
        # Function name: get_current_weather
        # Arguments: {'location': 'Tokyo', 'unit': 'celsius'}
    
        # Step 4: Actually execute the tool
        def get_current_weather(location, unit="celsius"):
            """Call weather API (mock here)"""
            weather_data = {
                "location": location,
                "temperature": 22,
                "unit": unit,
                "condition": "sunny"
            }
            return json.dumps(weather_data, ensure_ascii=False)
    
        function_response = get_current_weather(**function_args)
    
        # Step 5: Return result to LLM to generate final answer
        messages.append(message)  # Add LLM's tool call message
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_response
        })
    
        # Get final answer
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
    
        print(final_response.choices[0].message.content)
        # Output: The weather in Tokyo is sunny with temperature 22 degrees.
    

### Defining and Selecting Multiple Tools
    
    
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key="your-api-key")
    
    # Define multiple tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Execute web search to retrieve latest information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
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
                "description": "Execute mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression (e.g., 2 + 2, sqrt(16))"
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
                "description": "Get stock price information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]
    
    # LLM selects appropriate tool
    def run_agent_with_tools(user_query):
        """Execute agent with tools"""
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
    
                print(f"Selected tool: {function_name}")
                print(f"Arguments: {function_args}")
    
        return message
    
    # Try with different queries
    queries = [
        "Please calculate 123 + 456",
        "Tell me Apple's stock price",
        "Tell me the 2024 Nobel Prize winners"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        run_agent_with_tools(query)
    
    # Example output:
    # Query: Please calculate 123 + 456
    # Selected tool: calculate
    # Arguments: {'expression': '123 + 456'}
    #
    # Query: Tell me Apple's stock price
    # Selected tool: get_stock_price
    # Arguments: {'symbol': 'AAPL'}
    #
    # Query: Tell me the 2024 Nobel Prize winners
    # Selected tool: search_web
    # Arguments: {'query': '2024 Nobel Prize winners'}
    

## Anthropic Tool Use

### Tool Use with Claude
    
    
    import anthropic
    import json
    
    client = anthropic.Anthropic(api_key="your-api-key")
    
    # Tool definition
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information for a specified location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    # Agent execution
    def run_claude_agent(user_message):
        """Execute Claude agent"""
        messages = [{"role": "user", "content": user_message}]
    
        while True:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                tools=tools,
                messages=messages
            )
    
            # Process response
            if response.stop_reason == "tool_use":
                # Process tool call
                tool_use_block = next(
                    block for block in response.content
                    if block.type == "tool_use"
                )
    
                tool_name = tool_use_block.name
                tool_input = tool_use_block.input
    
                print(f"Tool use: {tool_name}")
                print(f"Input: {tool_input}")
    
                # Execute tool
                if tool_name == "get_weather":
                    result = get_weather(**tool_input)
                else:
                    result = "Unknown tool"
    
                # Return result to Claude
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
                # Final answer
                final_answer = next(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
                return final_answer
    
    def get_weather(location):
        """Weather retrieval function"""
        return json.dumps({
            "location": location,
            "temperature": 22,
            "condition": "sunny"
        }, ensure_ascii=False)
    
    # Execute
    answer = run_claude_agent("Please tell me the weather in Tokyo")
    print(f"Answer: {answer}")
    

## Tool Definition and Schema Design

### Design Principles for Effective Tool Schemas

#### 1\. Clear Descriptions
    
    
    # Bad example
    {
        "name": "search",
        "description": "Search"
    }
    
    # Good example
    {
        "name": "search_products",
        "description": """Search product database to retrieve related products.
        Can search by product name, category, and price range.
        Returns up to 10 product information items."""
    }
    

#### 2\. Appropriate Parameter Design
    
    
    search_products_tool = {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search product database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword (product name or category)"
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price (yen)",
                        "minimum": 0
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price (yen)",
                        "minimum": 0
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "books", "food"],
                        "description": "Product category"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_asc", "price_desc", "popularity"],
                        "description": "Sort order",
                        "default": "popularity"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of items to retrieve",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
    

#### 3\. Explicit Error Cases
    
    
    def search_products(query, min_price=None, max_price=None,
                             category=None, sort_by="popularity", limit=10):
        """
        Search products.
    
        Returns:
            dict: Product list on success, error information on failure
            {
                "success": bool,
                "data": [...],  # On success
                "error": str,   # On error
                "error_code": str  # On error
            }
        """
        try:
            # Validation
            if not query or len(query) < 2:
                return {
                    "success": False,
                    "error": "Search query must be at least 2 characters",
                    "error_code": "INVALID_QUERY"
                }
    
            if min_price and max_price and min_price > max_price:
                return {
                    "success": False,
                    "error": "Minimum price exceeds maximum price",
                    "error_code": "INVALID_PRICE_RANGE"
                }
    
            # Execute search (database access, etc.)
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
    

## Error Handling and Retry

### Robust Tool Execution
    
    
    import time
    import logging
    from typing import Any, Dict, Callable
    
    class ToolExecutor:
        """Wrapper for safe tool execution"""
    
        def __init__(self, max_retries=3, timeout=30):
            self.max_retries = max_retries
            self.timeout = timeout
            self.logger = logging.getLogger(__name__)
    
        def execute(self, tool_func: Callable, args: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute tool safely
    
            Args:
                tool_func: Function to execute
                args: Function arguments
    
            Returns:
                Execution result or error information
            """
            for attempt in range(self.max_retries):
                try:
                    # Execute with timeout
                    result = self._execute_with_timeout(tool_func, args)
    
                    return {
                        "success": True,
                        "result": result,
                        "attempt": attempt + 1
                    }
    
                except TimeoutError:
                    self.logger.warning(f"Tool execution timed out (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return {
                            "success": False,
                            "error": "Timed out",
                            "error_type": "timeout"
                        }
    
                except ValueError as e:
                    # Don't retry validation errors
                    self.logger.error(f"Validation error: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "error_type": "validation"
                    }
    
                except Exception as e:
                    # Other errors
                    self.logger.error(f"Tool execution error: {str(e)} (attempt {attempt + 1})")
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
                "error": "Maximum retry count reached",
                "error_type": "max_retries"
            }
    
        def _execute_with_timeout(self, func: Callable, args: Dict[str, Any]) -> Any:
            """Execute function with timeout"""
            import signal
    
            def timeout_handler(signum, frame):
                raise TimeoutError()
    
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
    
            try:
                result = func(**args)
                signal.alarm(0)  # Cancel timeout
                return result
            except:
                signal.alarm(0)
                raise
    
    # Usage example
    executor = ToolExecutor(max_retries=3, timeout=10)
    
    def risky_api_call(param):
        """Unstable API call"""
        import random
        if random.random() < 0.3:
            raise ConnectionError("API connection error")
        return {"data": f"Result: {param}"}
    
    result = executor.execute(risky_api_call, {"param": "test"})
    print(result)
    

## External API Integration

### Practical API Integration Examples

#### 1\. Weather API Integration (OpenWeatherMap)
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    import requests
    from typing import Dict, Optional
    
    class WeatherTool:
        """Weather information retrieval tool"""
    
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
        def get_weather(self, location: str, unit: str = "metric") -> Dict:
            """
            Get weather information for specified location
    
            Args:
                location: City name
                unit: Temperature unit (metric: Celsius, imperial: Fahrenheit)
    
            Returns:
                Dictionary with weather information
            """
            try:
                params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": unit,
                    "lang": "en"
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
                        "error": f"City '{location}' not found"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API error: {str(e)}"
                    }
    
            except requests.exceptions.Timeout:
                return {
                    "success": False,
                    "error": "API request timed out"
                }
    
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Unexpected error: {str(e)}"
                }
    
    # Integration with agent
    weather_tool = WeatherTool(api_key="your-openweathermap-api-key")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a specified location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name (e.g., Tokyo, Osaka)"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # During tool execution
    # result = weather_tool.get_weather("Tokyo")
    

## Tool Chains and Collaboration

### Coordinated Execution of Multiple Tools
    
    
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key="your-api-key")
    
    class AgentWithToolChain:
        """Agent with tool chain"""
    
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
                        "description": "Search stock symbol from company name",
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
                        "description": "Get current price from stock symbol",
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
                        "description": "Calculate price change rate",
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
            """Search stock symbol from company name (mock)"""
            mapping = {
                "Apple": "AAPL",
                "Microsoft": "MSFT",
                "Google": "GOOGL"
            }
            symbol = mapping.get(company_name, "UNKNOWN")
            return json.dumps({"symbol": symbol})
    
        def get_stock_price(self, symbol: str) -> str:
            """Get stock price (mock)"""
            prices = {
                "AAPL": {"current": 150.25, "previous": 148.50},
                "MSFT": {"current": 380.75, "previous": 375.00},
                "GOOGL": {"current": 140.50, "previous": 142.00}
            }
            data = prices.get(symbol, {"current": 0, "previous": 0})
            return json.dumps(data)
    
        def calculate_change(self, current_price: float, previous_price: float) -> str:
            """Calculate price change rate"""
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            return json.dumps({
                "change": round(change, 2),
                "change_percent": round(change_percent, 2)
            })
    
        def run(self, user_query: str, max_iterations: int = 10) -> str:
            """Execute agent (tool chain support)"""
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
                    # Final answer
                    return message.content
    
                # Process tool calls
                messages.append(message)
    
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
    
                    # Execute tool
                    if function_name in self.tools:
                        result = self.tools[function_name](**function_args)
                    else:
                        result = json.dumps({"error": "Unknown tool"})
    
                    # Add result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result
                    })
    
            return "Maximum iteration count reached"
    
    # Usage example
    agent = AgentWithToolChain()
    answer = agent.run("How has Apple's stock price changed compared to yesterday?")
    print(answer)
    # LLM automatically coordinates: search_company → get_stock_price → calculate_change
    

## Security and Rate Limiting

### Secure Agent Design

#### 1\. Tool Execution Approval Flow
    
    
    class SecureAgent:
        """Secure agent with approval flow"""
    
        def __init__(self, require_approval_for=None):
            self.require_approval_for = require_approval_for or [
                "delete_*",
                "send_email",
                "make_payment"
            ]
    
        def requires_approval(self, tool_name: str) -> bool:
            """Determine if tool requires approval"""
            import fnmatch
            for pattern in self.require_approval_for:
                if fnmatch.fnmatch(tool_name, pattern):
                    return True
            return False
    
        def request_approval(self, tool_name: str, args: dict) -> bool:
            """Request approval from user"""
            print(f"\n⚠️  Approval required")
            print(f"Tool: {tool_name}")
            print(f"Arguments: {args}")
            response = input("Execute? (yes/no): ")
            return response.lower() == "yes"
    
        def execute_tool(self, tool_name: str, args: dict):
            """Execute tool with approval flow"""
            if self.requires_approval(tool_name):
                if not self.request_approval(tool_name, args):
                    return {"success": False, "error": "User denied"}
    
            # Execute tool
            return self.tools[tool_name](**args)
    

#### 2\. Rate Limiting Implementation
    
    
    import time
    from collections import defaultdict
    
    class RateLimiter:
        """Rate limiting for tool execution"""
    
        def __init__(self, max_calls_per_minute=10):
            self.max_calls = max_calls_per_minute
            self.calls = defaultdict(list)
    
        def allow_call(self, tool_name: str) -> bool:
            """Determine if call is allowed"""
            now = time.time()
            one_minute_ago = now - 60
    
            # Filter calls within 1 minute
            self.calls[tool_name] = [
                t for t in self.calls[tool_name]
                if t > one_minute_ago
            ]
    
            if len(self.calls[tool_name]) >= self.max_calls:
                return False
    
            self.calls[tool_name].append(now)
            return True
    
    # Usage example
    limiter = RateLimiter(max_calls_per_minute=5)
    
    if limiter.allow_call("expensive_api"):
        result = call_expensive_api()
    else:
        print("Rate limit reached. Please wait.")
    

## Summary

### What We Learned in This Chapter

  * ✅ **Function Calling** : OpenAI/Anthropic Function Calling API
  * ✅ **Tool Schema** : Type-safe definitions with JSON schema
  * ✅ **Error Handling** : Retry, timeout, error processing
  * ✅ **External API Integration** : Implementation of weather API, stock price API, etc.
  * ✅ **Tool Chains** : Coordinated execution of multiple tools
  * ✅ **Security** : Approval flow, rate limiting

> **Important Design Principles** : Tools should have clear responsibilities, handle errors appropriately, and be designed with security in mind

[← Chapter 1: Agent Basics](<./chapter1-agent-basics.html>) [Chapter 3: Multi-Agent →](<./chapter3-multi-agent.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
