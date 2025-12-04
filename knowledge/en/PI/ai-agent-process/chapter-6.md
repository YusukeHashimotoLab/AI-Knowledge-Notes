---
title: "Chapter 6: Integration with Large Language Models"
chapter_title: "Chapter 6: Integration with Large Language Models"
subtitle: Advanced Collaborative Control and Process Diagnosis with LLM Agents
---

This chapter covers Integration with Large Language Models. You will learn mechanism of Tool Use (Function Calling), Know the components of LangChain's agent framework, and List the advantages of hybrid architectures.

[← Back to Series Index](<./index.html>)

## Chapter Overview

This chapter teaches how to integrate large language models (LLMs) with reinforcement learning agents to build advanced process control systems. By leveraging LLMs' natural language understanding and reasoning capabilities, we can enable flexible decision-making, anomaly diagnosis, and human collaboration that would be difficult with traditional RL agents alone.

**💡 What You'll Learn**

  * Building LLM agents using Claude API / OpenAI API
  * Implementing agent frameworks with LangChain
  * Designing hybrid systems combining LLM and RL agents
  * External system integration via Tool Use (Function Calling)
  * Process anomaly diagnosis and LLM explanation generation
  * Best practices for real plant deployment

## 6.1 Fundamentals of LLM Agents

Large Language Models (LLMs) are AI systems capable of understanding and generating natural language. In process control applications, they can perform tasks difficult for traditional rule-based systems, such as sensor data interpretation, anomaly diagnosis, and control strategy proposals.

### 6.1.1 Characteristics of LLM Agents

Feature | RL Agent | LLM Agent | Hybrid  
---|---|---|---  
Learning Method | Trial-and-error (reward maximization) | Pre-trained (inference only) | RL for optimization, LLM for reasoning  
Inference Speed | ⚡⚡⚡ Fast (ms) | ⚡ Medium (1-5 sec) | ⚡⚡ Optimized by role division  
Explainability | ⭐ Low | ⭐⭐⭐ High (natural language) | ⭐⭐⭐ LLM provides explanation  
Flexibility | ⭐⭐ Within training scope | ⭐⭐⭐ General reasoning capable | ⭐⭐⭐ Best of both  
Cost | High initial training cost | API call cost | Balanced  
  
### Example 1: Process State Analysis with Claude API

We implement an agent that uses Claude API (Anthropic) to analyze the state of a CSTR (Continuous Stirred Tank Reactor) and explain the operational status in natural language.
    
    
    import anthropic
    import os
    import json
    from typing import Dict, List
    
    # ===================================
    # Example 1: CSTR State Analysis with Claude API
    # ===================================
    
    class ClaudeProcessAnalyzer:
        """Process analysis agent using Claude API"""
    
        def __init__(self, api_key: str = None):
            """
            Args:
                api_key: Anthropic API Key (auto-retrieved from ANTHROPIC_API_KEY environment variable)
            """
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("API key required. Set ANTHROPIC_API_KEY environment variable.")
    
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = "claude-3-5-sonnet-20241022"
    
        def analyze_cstr_state(self, state: Dict, target: Dict) -> Dict:
            """Analyze CSTR state and explain operational status
    
            Args:
                state: Current state {'temperature': float, 'concentration': float, ...}
                target: Target values {'temperature': float, 'concentration': float, ...}
    
            Returns:
                {'status': str, 'analysis': str, 'recommendations': List[str]}
            """
            # Build prompt
            prompt = f"""You are a chemical process engineer. Please analyze the following CSTR (Continuous Stirred Tank Reactor) state.
    
    **Current State:**
    - Temperature: {state['temperature']:.1f} K
    - Concentration: {state['concentration']:.3f} mol/L
    - Flow rate: {state['flow_rate']:.2f} L/min
    - Heating power: {state['heating_power']:.2f} kW
    
    **Target Values:**
    - Target temperature: {target['temperature']:.1f} K
    - Target concentration: {target['concentration']:.3f} mol/L
    
    Please respond in JSON format as follows:
    {{
        "status": "Normal or Caution or Abnormal",
        "analysis": "Explanation of current operational status (within 100 characters)",
        "recommendations": ["Recommended action 1", "Recommended action 2", ...]
    }}"""
    
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
    
            # Parse response
            response_text = message.content[0].text
    
            # Parse JSON (handle cases wrapped in ```json```)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text
    
            result = json.loads(json_str)
            return result
    
    
    # Usage example
    if __name__ == "__main__":
        # CSTR state data
        current_state = {
            'temperature': 365.0,  # K (15K above target of 350K)
            'concentration': 0.32,  # mol/L
            'flow_rate': 1.0,
            'heating_power': 5.2
        }
    
        target_state = {
            'temperature': 350.0,
            'concentration': 0.30
        }
    
        # Execute analysis
        analyzer = ClaudeProcessAnalyzer()
        result = analyzer.analyze_cstr_state(current_state, target_state)
    
        print("=== CSTR State Analysis Result ===")
        print(f"Status: {result['status']}")
        print(f"\nAnalysis: {result['analysis']}")
        print(f"\nRecommended Actions:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Expected output example:
    # === CSTR State Analysis Result ===
    # Status: Caution
    #
    # Analysis: Temperature is 15K above target, indicating overshoot. Concentration is also slightly high but within acceptable range.
    #
    # Recommended Actions:
    #   1. Reduce heating power to approximately 3.5kW
    #   2. Increase cooling water flow by 10%
    #   3. Continue monitoring to prevent temperature from dropping below 340K
    

**💡 Pro Tip: API Key Management**

In production environments, store API Keys in environment variables or Secret Manager, and never embed them directly in code. Also be aware of API call rate limits (Anthropic: 50 requests/min).

## 6.2 Implementing Tool Use (Function Calling)

Using Claude API's Tool Use feature, LLMs can call external functions to retrieve real-time data or execute control actions. This enables LLM agents to interact directly with actual processes.
    
    
    ```mermaid
    flowchart LR
        A[User/Sensors] -->|State data| B[LLM Agent]
        B -->|Tool Call| C[get_sensor_data]
        C -->|Sensor values| B
        B -->|Tool Call| D[execute_control_action]
        D -->|Control signal| E[CSTR]
        E -->|Feedback| A
    
        style B fill:#e3f2fd
        style C fill:#e8f5e9
        style D fill:#fff3e0
    ```

### Example 2: Process Control with Tool Use

We enable LLMs to call actual process control functions.
    
    
    # ===================================
    # Example 2: Process Control with Tool Use
    # ===================================
    
    class ClaudeProcessController:
        """Process control agent using Tool Use"""
    
        def __init__(self, api_key: str = None):
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022"
    
            # CSTR being controlled (simulation)
            self.cstr_state = {
                'temperature': 355.0,
                'concentration': 0.35,
                'heating_power': 4.5,
                'flow_rate': 1.0
            }
    
        def get_sensor_data(self) -> Dict:
            """Get sensor data (Tool function)"""
            return self.cstr_state
    
        def set_heating_power(self, power: float) -> Dict:
            """Set heating power (Tool function)
    
            Args:
                power: Heating power [kW] (range 0-10)
            """
            power = max(0.0, min(10.0, power))  # Limit to safe range
            self.cstr_state['heating_power'] = power
            return {
                'success': True,
                'new_power': power,
                'message': f'Heating power set to {power:.1f}kW'
            }
    
        def set_flow_rate(self, flow: float) -> Dict:
            """Set flow rate (Tool function)
    
            Args:
                flow: Flow rate [L/min] (range 0.5-2.0)
            """
            flow = max(0.5, min(2.0, flow))
            self.cstr_state['flow_rate'] = flow
            return {
                'success': True,
                'new_flow': flow,
                'message': f'Flow rate set to {flow:.2f}L/min'
            }
    
        def define_tools(self) -> List[Dict]:
            """Define tools available to the LLM"""
            return [
                {
                    "name": "get_sensor_data",
                    "description": "Retrieves CSTR sensor data (temperature, concentration, heating power, flow rate).",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "set_heating_power",
                    "description": "Sets CSTR heating power. Specify in range 0-10kW.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "power": {
                                "type": "number",
                                "description": "Heating power to set [kW]"
                            }
                        },
                        "required": ["power"]
                    }
                },
                {
                    "name": "set_flow_rate",
                    "description": "Sets CSTR flow rate. Specify in range 0.5-2.0 L/min.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "flow": {
                                "type": "number",
                                "description": "Flow rate to set [L/min]"
                            }
                        },
                        "required": ["flow"]
                    }
                }
            ]
    
        def execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
            """Execute tool"""
            if tool_name == "get_sensor_data":
                return self.get_sensor_data()
            elif tool_name == "set_heating_power":
                return self.set_heating_power(tool_input['power'])
            elif tool_name == "set_flow_rate":
                return self.set_flow_rate(tool_input['flow'])
            else:
                return {'error': f'Unknown tool: {tool_name}'}
    
        def run_control_task(self, user_request: str, max_iterations: int = 5) -> str:
            """Execute control task
    
            Args:
                user_request: User request (e.g., "Lower temperature to 350K")
                max_iterations: Maximum Tool call iterations
    
            Returns:
                Final LLM response
            """
            messages = [
                {"role": "user", "content": user_request}
            ]
    
            for iteration in range(max_iterations):
                # Call Claude API
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    tools=self.define_tools(),
                    messages=messages
                )
    
                # Log response
                print(f"\n=== Iteration {iteration + 1} ===")
                print(f"Stop reason: {response.stop_reason}")
    
                # If there are Tool calls
                if response.stop_reason == "tool_use":
                    # Add Assistant's response to message history
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
    
                    # Execute each Tool call
                    tool_results = []
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input
    
                            print(f"Tool call: {tool_name}({tool_input})")
    
                            # Execute Tool
                            result = self.execute_tool(tool_name, tool_input)
                            print(f"Tool result: {result}")
    
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": json.dumps(result)
                            })
    
                    # Add Tool results to message history
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
    
                # Termination condition
                elif response.stop_reason == "end_turn":
                    # Extract final response
                    final_response = ""
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            final_response += content_block.text
    
                    return final_response
    
            return "Maximum iterations reached"
    
    
    # Usage example
    if __name__ == "__main__":
        controller = ClaudeProcessController()
    
        # Task: Lower temperature
        result = controller.run_control_task(
            "Current CSTR temperature is 355K and target is 350K. Please lower the temperature."
        )
    
        print("\n=== Final Result ===")
        print(result)
        print(f"\nCurrent state: {controller.cstr_state}")
    
    # Expected output example:
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
    # === Final Result ===
    # I have reduced the heating power from 4.5kW to 3.5kW. With this adjustment,
    # the reactor temperature should gradually approach the target value of 350K.
    # I recommend checking the temperature again after 10 minutes and making fine adjustments as needed.
    

**⚠️ Safety Precautions**

When controlling actual processes with Tool Use, the following safety measures are essential:

  * Implement input value range checks in all Tool functions
  * Require human approval for dangerous operations (emergency shutdown, etc.)
  * Validate LLM recommendations in simulation before execution
  * Provide fallback mechanisms (PID control, etc.) for API failures

## 6.3 Agent Framework with LangChain

LangChain is a framework for developing LLM applications, allowing concise implementation of agent construction, memory management, and tool integration.

### 6.3.1 Components of LangChain Agents
    
    
    ```mermaid
    flowchart TD
        A[User Input] --> B[Agent Executor]
        B --> C[LLMClaude/GPT-4]
        C --> D{Tool needed?}
        D -->|Yes| E[Toolget_sensor_dataset_control]
        E --> F[Tool Result]
        F --> C
        D -->|No| G[Final Response]
    
        H[MemoryConversation History] -.-> B
        B -.->|Update| H
    
        style C fill:#e3f2fd
        style E fill:#e8f5e9
        style H fill:#fff3e0
    ```

### Example 3: Process Diagnostic Agent with LangChain

We build an agent that performs process diagnosis while maintaining conversation history using LangChain.
    
    
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_anthropic import ChatAnthropic
    from langchain.tools import Tool
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    import numpy as np
    
    # ===================================
    # Example 3: LangChain Process Diagnostic Agent
    # ===================================
    
    class ProcessDiagnosticAgent:
        """Process diagnostic agent using LangChain"""
    
        def __init__(self, api_key: str = None):
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
            # Initialize LLM
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                anthropic_api_key=self.api_key,
                temperature=0.0
            )
    
            # Process state (simulation)
            self.process_history = []
            self.current_state = {
                'temperature': 348.0,
                'pressure': 2.1,  # bar
                'flow_rate': 1.0,
                'concentration': 0.28
            }
    
            # Memory (stores conversation history)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
    
            # Build agent
            self.agent_executor = self._build_agent()
    
        def _get_current_state(self, query: str = "") -> str:
            """Get current process state"""
            state_str = f"""Current CSTR state:
    - Temperature: {self.current_state['temperature']:.1f} K
    - Pressure: {self.current_state['pressure']:.2f} bar
    - Flow rate: {self.current_state['flow_rate']:.2f} L/min
    - Concentration: {self.current_state['concentration']:.3f} mol/L"""
            return state_str
    
        def _get_historical_trend(self, query: str = "") -> str:
            """Get historical trend data"""
            if len(self.process_history) == 0:
                return "No historical data available."
    
            temps = [h['temperature'] for h in self.process_history[-10:]]
            avg_temp = np.mean(temps)
            std_temp = np.std(temps)
    
            trend = "rising" if temps[-1] > temps[0] else "declining"
    
            return f"""Trend over last 10 steps:
    - Average temperature: {avg_temp:.1f} K
    - Temperature standard deviation: {std_temp:.2f} K
    - Trend: {trend} tendency
    - Variation range: {min(temps):.1f} - {max(temps):.1f} K"""
    
        def _diagnose_anomaly(self, query: str = "") -> str:
            """Diagnose anomalies"""
            state = self.current_state
            issues = []
    
            # Temperature anomaly check
            if state['temperature'] < 340 or state['temperature'] > 360:
                issues.append(f"Temperature anomaly: {state['temperature']:.1f} K (normal range: 340-360K)")
    
            # Pressure anomaly check
            if state['pressure'] < 1.8 or state['pressure'] > 2.5:
                issues.append(f"Pressure anomaly: {state['pressure']:.2f} bar (normal range: 1.8-2.5bar)")
    
            # Concentration anomaly check
            if state['concentration'] < 0.25 or state['concentration'] > 0.35:
                issues.append(f"Concentration anomaly: {state['concentration']:.3f} mol/L (normal range: 0.25-0.35)")
    
            if len(issues) == 0:
                return "No anomalies currently detected. All parameters are within normal range."
            else:
                return "Detected anomalies:\n" + "\n".join([f"- {issue}" for issue in issues])
    
        def _build_agent(self) -> AgentExecutor:
            """Build agent"""
    
            # Define tools
            tools = [
                Tool(
                    name="get_current_state",
                    func=self._get_current_state,
                    description="Retrieves current process state (temperature, pressure, flow rate, concentration)."
                ),
                Tool(
                    name="get_historical_trend",
                    func=self._get_historical_trend,
                    description="Analyzes historical process data trends. Returns average values, standard deviation, and trend direction."
                ),
                Tool(
                    name="diagnose_anomaly",
                    func=self._diagnose_anomaly,
                    description="Diagnoses process parameter anomalies. Detects values outside normal range."
                )
            ]
    
            # ReAct prompt template
            template = """You are an expert in chemical process diagnosis. Use the available tools to answer the user's questions.
    
    Available tools:
    {tools}
    
    Tool names: {tool_names}
    
    Respond in the following format:
    
    Question: Input question
    Thought: Think about what to do
    Action: Tool name to execute
    Action Input: Input to the tool
    Observation: Tool result
    ... (repeat this Thought/Action/Action Input/Observation as needed)
    Thought: I now know the final answer
    Final Answer: Final answer to user
    
    Question: {input}
    
    {agent_scratchpad}"""
    
            prompt = PromptTemplate.from_template(template)
    
            # Create agent
            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
    
            # Create Agent Executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
    
            return agent_executor
    
        def diagnose(self, query: str) -> str:
            """Execute diagnostic query
    
            Args:
                query: Question from user
    
            Returns:
                Diagnostic result
            """
            # Add current state to history
            self.process_history.append(self.current_state.copy())
    
            # Execute agent
            result = self.agent_executor.invoke({
                "input": query
            })
    
            return result['output']
    
    
    # Usage example
    if __name__ == "__main__":
        agent = ProcessDiagnosticAgent()
    
        # Query 1: Check current state
        print("=== Query 1 ===")
        result1 = agent.diagnose("Please tell me the current process state.")
        print(f"\nResponse: {result1}\n")
    
        # Change state (create anomaly)
        agent.current_state['temperature'] = 365.0  # Abnormal value
        agent.current_state['pressure'] = 2.7  # Abnormal value
    
        # Query 2: Anomaly diagnosis
        print("\n=== Query 2 ===")
        result2 = agent.diagnose("Are there any anomalies? If so, please explain the cause and countermeasures.")
        print(f"\nResponse: {result2}\n")
    
    # Expected output example:
    # === Query 1 ===
    #
    # > Entering new AgentExecutor chain...
    # Thought: I need to check the process state
    # Action: get_current_state
    # Action Input:
    # Observation: Current CSTR state:
    # - Temperature: 348.0 K
    # - Pressure: 2.10 bar
    # ...
    # Thought: I now know the final answer
    # Final Answer: The current process is operating normally. Temperature is 348K, pressure 2.1bar,
    # flow rate 1.0L/min, concentration 0.28mol/L, with all parameters within normal range.
    #
    # === Query 2 ===
    #
    # > Entering new AgentExecutor chain...
    # Thought: I should use the anomaly diagnosis tool
    # Action: diagnose_anomaly
    # Action Input:
    # Observation: Detected anomalies:
    # - Temperature anomaly: 365.0 K (normal range: 340-360K)
    # - Pressure anomaly: 2.70 bar (normal range: 1.8-2.5bar)
    # Thought: I now know the final answer
    # Final Answer: Currently, two anomalies are detected.
    #
    # 1. Temperature anomaly (365K): Exceeds normal range (340-360K)
    #    Cause: Excessive heating power, insufficient cooling, or accelerated exothermic reaction
    #    Countermeasure: Reduce heating power by 20%, increase cooling water flow by 15%
    #
    # 2. Pressure anomaly (2.7bar): Exceeds normal range (1.8-2.5bar)
    #    Cause: Outlet valve clogging or increased gas phase products
    #    Countermeasure: Increase outlet valve opening by 10%, reduce gas phase generation by lowering temperature
    

**💡 Benefits of LangChain**

  * **Memory Management** : Automatically stores and references conversation history
  * **Tool Integration** : Any Python function can be converted to an LLM tool
  * **Error Handling** : Automatically retries on LLM output parsing failures
  * **Modularity** : LLM, Tools, and Memory can be changed independently

## 6.4 Building LLM-RL Hybrid Systems

By combining LLM reasoning capabilities with RL agent optimization capabilities, we can build more advanced process control systems. LLMs handle high-level strategic decisions while RL agents execute low-level control.

### 6.4.1 Hierarchical Architecture
    
    
    ```mermaid
    flowchart TD
        A[Human Operator] -->|Instructions/Inquiries| B[LLM CoordinatorClaude/GPT-4]
        B -->|Strategic decisionGoal setting| C[RL Agent 1Reactor control]
        B -->|Strategic decisionGoal setting| D[RL Agent 2Separator control]
        B -->|Strategic decisionGoal setting| E[RL Agent 3Heat exchanger control]
    
        C -->|Status report| B
        D -->|Status report| B
        E -->|Status report| B
    
        C -.->|Coordination| D
        D -.->|Coordination| E
    
        C --> F[CSTR]
        D --> G[Distillation Column]
        E --> H[Heat Exchanger]
    
        style B fill:#e3f2fd
        style C fill:#e8f5e9
        style D fill:#e8f5e9
        style E fill:#e8f5e9
    ```

### Example 4: LLM Coordinator and RL Workers

We implement a system where an LLM manages multiple RL agents to achieve overall optimization.
    
    
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from typing import Dict, List
    
    # ===================================
    # Example 4: LLM-RL Hybrid System
    # ===================================
    
    class RLWorkerAgent:
        """RL control worker agent"""
    
        def __init__(self, name: str, env: gym.Env, model_path: str = None):
            self.name = name
            self.env = env
    
            # Load pre-trained model (or dummy)
            if model_path:
                self.model = PPO.load(model_path)
            else:
                # For demo: random agent
                self.model = None
    
            self.current_state = None
            self.current_reward = 0.0
            self.cumulative_reward = 0.0
    
        def reset(self):
            """Reset environment"""
            self.current_state = self.env.reset()
            self.cumulative_reward = 0.0
            return self.current_state
    
        def step(self, target_setpoint: Dict = None) -> Dict:
            """Execute one step
    
            Args:
                target_setpoint: Target values instructed by LLM
    
            Returns:
                {'state': ..., 'reward': ..., 'done': ...}
            """
            # Select action with RL model
            if self.model:
                action, _ = self.model.predict(self.current_state, deterministic=True)
            else:
                # For demo: random action
                action = self.env.action_space.sample()
    
            # Execute in environment
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
            """Generate status report (for LLM reporting)"""
            if self.current_state is None:
                return f"{self.name}: Uninitialized"
    
            state_str = ", ".join([f"{k}={v:.2f}" for k, v in
                                   enumerate(self.current_state)])
    
            return f"""{self.name} state:
    - State: [{state_str}]
    - Recent reward: {self.current_reward:.3f}
    - Cumulative reward: {self.cumulative_reward:.3f}"""
    
    
    class LLMCoordinator:
        """Overall coordination agent with LLM"""
    
        def __init__(self, workers: List[RLWorkerAgent], api_key: str = None):
            self.workers = workers
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022"
            self.conversation_history = []
    
        def collect_worker_reports(self) -> str:
            """Collect status reports from all workers"""
            reports = []
            for worker in self.workers:
                reports.append(worker.get_status_report())
    
            return "\n\n".join(reports)
    
        def coordinate(self, user_goal: str = None) -> Dict:
            """Coordinate workers and determine strategy
    
            Args:
                user_goal: Goal from operator (e.g., "Increase production by 10%")
    
            Returns:
                Instructions for each worker
            """
            # Collect worker reports
            worker_reports = self.collect_worker_reports()
    
            # Build prompt
            if user_goal:
                prompt = f"""You are a supervisor managing the entire process.
    
    **Operator's goal:**
    {user_goal}
    
    **Current state of each unit:**
    {worker_reports}
    
    Please issue instructions to each worker agent (RL control) in the following JSON format:
    {{
        "strategy": "Overall strategy explanation",
        "worker_instructions": {{
            "worker_0": {{"action": "Specific instruction"}},
            "worker_1": {{"action": "Specific instruction"}},
            "worker_2": {{"action": "Specific instruction"}}
        }}
    }}"""
            else:
                prompt = f"""Please analyze the current state of each unit and suggest improvements:
    
    {worker_reports}
    
    Respond in JSON format."""
    
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
    
            response_text = message.content[0].text
    
            # Parse JSON
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
            """Execute coordinated control
    
            Args:
                user_goal: Overall goal
                n_steps: Number of execution steps
            """
            print(f"=== Coordinated Control Start ===")
            print(f"Goal: {user_goal}\n")
    
            # Initialize workers
            for worker in self.workers:
                worker.reset()
    
            for step in range(n_steps):
                print(f"\n--- Step {step + 1}/{n_steps} ---")
    
                # Coordinate with LLM every 5 steps
                if step % 5 == 0:
                    instructions = self.coordinate(user_goal)
                    print(f"\nLLM Strategy: {instructions.get('strategy', 'N/A')}")
                    print(f"Instructions: {instructions.get('worker_instructions', {})}")
    
                # Execute each worker for 1 step
                for i, worker in enumerate(self.workers):
                    result = worker.step()
                    print(f"{worker.name}: reward={result['reward']:.3f}, "
                          f"cumulative={result['cumulative_reward']:.3f}")
    
            print(f"\n=== Coordinated Control Complete ===")
            final_report = self.collect_worker_reports()
            print(f"\n{final_report}")
    
    
    # Usage example
    if __name__ == "__main__":
        # Simple demo environment
        class DummyEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,))
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
                self.state = np.array([0.0, 0.0, 0.0, 0.0])
    
            def reset(self):
                self.state = np.random.randn(4)
                return self.state
    
            def step(self, action):
                self.state += 0.1 * action[0]  # Simple dynamics
                reward = -np.sum(self.state**2)  # Higher reward closer to origin
                done = False
                return self.state, reward, done, {}
    
        # Create worker agents
        workers = [
            RLWorkerAgent("Worker_Reactor", DummyEnv()),
            RLWorkerAgent("Worker_Separator", DummyEnv()),
            RLWorkerAgent("Worker_HeatExchanger", DummyEnv())
        ]
    
        # Create LLM Coordinator
        coordinator = LLMCoordinator(workers)
    
        # Execute coordinated control
        coordinator.run_coordinated_control(
            user_goal="Improve stability of all units while reducing energy consumption by 10%.",
            n_steps=10
        )
    
    # Expected output example:
    # === Coordinated Control Start ===
    # Goal: Improve stability of all units while reducing energy consumption by 10%.
    #
    # --- Step 1/10 ---
    # LLM Strategy: To stabilize each unit, first suppress reactor temperature fluctuations,
    # optimize separator reflux ratio, and improve heat exchanger efficiency.
    # Instructions: {'worker_0': {'action': 'Maintain temperature within ±2K'},
    #        'worker_1': {'action': 'Adjust reflux ratio to 0.7'},
    #        'worker_2': {'action': 'Improve heat transfer coefficient by 5%'}}
    #
    # Worker_Reactor: reward=-2.134, cumulative=-2.134
    # Worker_Separator: reward=-1.876, cumulative=-1.876
    # Worker_HeatExchanger: reward=-3.021, cumulative=-3.021
    # ...
    

**🎯 Advantages of LLM-RL Hybrid**

  * **Separation of strategy and tactics** : LLM handles high-level strategy, RL handles low-level control
  * **Explainability** : LLM explains decision reasons in natural language
  * **Adaptability** : LLM flexibly responds to new goals and constraints
  * **Human collaboration** : Natural language dialogue with operators possible

## 6.5 Best Practices for Production Deployment

When deploying LLM agents to actual processes, even more careful design is required than for RL agents.

### 6.5.1 Safety and Fallback

Risk | Countermeasure | Implementation Method  
---|---|---  
API failure | Fallback control | Automatic switch to PID/MPC  
Rate limiting | Caching | Reuse identical queries (TTL: 5 min)  
Inappropriate recommendations | Range checking | Min/max constraints on all control values  
Latency | Asynchronous execution | Run LLM in separate thread  
Cost overrun | Budget management | $1000 monthly cap, alert on excess  
  
### Example 5: Production-Ready Integrated System
    
    
    # ===================================
    # Example 5: Production-Ready LLM-RL Integrated System
    # ===================================
    
    import time
    import logging
    from functools import lru_cache
    import hashlib
    
    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
    class ProductionLLMController:
        """Production-ready LLM control system"""
    
        def __init__(self, api_key: str = None):
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022"
    
            # Statistics
            self.api_call_count = 0
            self.cache_hit_count = 0
            self.fallback_count = 0
            self.total_cost = 0.0  # USD
    
            # Constraints
            self.monthly_budget = 1000.0  # USD
            self.max_retries = 3
            self.timeout = 10.0  # seconds
    
            # Fallback controller (PID)
            self.fallback_controller = SimplePIDController()
    
        @lru_cache(maxsize=100)
        def _cached_llm_call(self, prompt_hash: str, prompt: str) -> str:
            """Cached LLM call"""
            logger.info(f"Cache miss for hash {prompt_hash[:8]}. Calling API...")
    
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout
                )
    
                self.api_call_count += 1
    
                # Cost calculation (approx: $3/1M input tokens, $15/1M output tokens)
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
                self.total_cost += cost
    
                logger.info(f"API call #{self.api_call_count}, Cost: ${cost:.4f}, "
                           f"Total: ${self.total_cost:.2f}")
    
                # Budget check
                if self.total_cost > self.monthly_budget:
                    logger.error(f"Monthly budget ${self.monthly_budget} exceeded!")
                    raise ValueError("Budget exceeded")
    
                return message.content[0].text
    
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise
    
        def llm_control_decision(self, state: Dict, target: Dict) -> Dict:
            """LLM control decision (with caching and fallback)
    
            Args:
                state: Current state
                target: Target values
    
            Returns:
                {'action': ..., 'reasoning': ..., 'fallback_used': bool}
            """
            # Build prompt
            prompt = f"""CSTR control decision:
    Current: temperature {state['temperature']:.1f}K, concentration {state['concentration']:.3f}
    Target: temperature {target['temperature']:.1f}K, concentration {target['concentration']:.3f}
    
    Please recommend heating power (0-10kW). Respond in JSON format:
    {{"heating_power": float, "reasoning": "reason"}}"""
    
            # Hash prompt (cache key)
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    
            try:
                # Retrieve from cache or call API
                response = self._cached_llm_call(prompt_hash, prompt)
    
                # Parse JSON
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                else:
                    json_str = response
    
                result = json.loads(json_str)
    
                # Safety range check
                power = max(0.0, min(10.0, result['heating_power']))
    
                if power != result['heating_power']:
                    logger.warning(f"Clamped heating power from {result['heating_power']} to {power}")
    
                return {
                    'action': {'heating_power': power},
                    'reasoning': result.get('reasoning', 'N/A'),
                    'fallback_used': False
                }
    
            except Exception as e:
                # Fallback: PID control
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
            """Get statistics"""
            return {
                'api_calls': self.api_call_count,
                'cache_hits': self.cache_hit_count,
                'fallback_uses': self.fallback_count,
                'total_cost_usd': self.total_cost,
                'budget_remaining_usd': self.monthly_budget - self.total_cost
            }
    
    
    class SimplePIDController:
        """PID controller for fallback"""
    
        def __init__(self, Kp: float = 0.5, Ki: float = 0.1, Kd: float = 0.05):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.integral = 0.0
            self.prev_error = 0.0
    
        def compute(self, current: float, setpoint: float) -> float:
            """PID control calculation"""
            error = setpoint - current
            self.integral += error
            derivative = error - self.prev_error
            self.prev_error = error
    
            output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    
            # Heating power range limit
            return max(0.0, min(10.0, output))
    
    
    # Usage example
    if __name__ == "__main__":
        controller = ProductionLLMController()
    
        # Simulation: execute 20 steps
        state = {'temperature': 348.0, 'concentration': 0.30}
        target = {'temperature': 350.0, 'concentration': 0.30}
    
        for step in range(20):
            print(f"\n=== Step {step + 1} ===")
    
            # Control decision
            decision = controller.llm_control_decision(state, target)
    
            print(f"Action: {decision['action']}")
            print(f"Reasoning: {decision['reasoning']}")
            print(f"Fallback: {decision['fallback_used']}")
    
            # State update (simple model)
            heating_power = decision['action']['heating_power']
            state['temperature'] += (heating_power - 5.0) * 0.5  # Simple dynamics
    
            # Display statistics every 10 steps
            if (step + 1) % 10 == 0:
                stats = controller.get_statistics()
                print(f"\n=== Statistics ===")
                print(f"API Calls: {stats['api_calls']}")
                print(f"Fallback Uses: {stats['fallback_uses']}")
                print(f"Total Cost: ${stats['total_cost_usd']:.2f}")
                print(f"Budget Remaining: ${stats['budget_remaining_usd']:.2f}")
    
        # Final statistics
        final_stats = controller.get_statistics()
        print(f"\n=== Final Statistics ===")
        for key, value in final_stats.items():
            print(f"{key}: {value}")
    
    # Expected output example:
    # === Step 1 ===
    # Cache miss for hash a3b5f21c. Calling API...
    # API call #1, Cost: $0.0032, Total: $0.00
    # Action: {'heating_power': 5.5}
    # Reasoning: Temperature is 2K below target, increasing heating power moderately
    # Fallback: False
    #
    # === Step 2 ===
    # Action: {'heating_power': 5.5}  # Cache hit
    # Reasoning: Temperature is 2K below target, increasing heating power moderately
    # Fallback: False
    # ...
    #
    # === Statistics ===
    # API Calls: 5
    # Fallback Uses: 0
    # Total Cost: $0.02
    # Budget Remaining: $999.98
    

**⚠️ Critical Points for Production Deployment**

  1. **Always implement fallback mechanisms** : Process must continue operating safely during API failures
  2. **Budget management** : Set monthly cost caps and alert on excess
  3. **Latency countermeasures** : Make LLM calls asynchronous, leverage caching
  4. **Audit logs** : Record all LLM decisions and reasoning (regulatory compliance)
  5. **Staged deployment** : Start with advisory mode (suggestions only) → supervised automatic control → fully automatic

## Learning Objectives Review

Upon completing this chapter, you should be able to explain and implement the following:

### Basic Understanding

  * ✅ Explain the differences and complementarity between LLM and RL agents
  * ✅ Understand the mechanism of Tool Use (Function Calling)
  * ✅ Know the components of LangChain's agent framework
  * ✅ List the advantages of hybrid architectures

### Practical Skills

  * ✅ Implement LLM agents using Claude API / OpenAI API
  * ✅ Integrate external functions (sensor acquisition, control execution) with Tool Use
  * ✅ Build agents with memory using LangChain
  * ✅ Implement hybrid systems with LLM Coordinator and RL Workers
  * ✅ Implement production-ready safety mechanisms (fallback, budget management)

### Application Ability

  * ✅ Apply LLMs to process anomaly diagnosis
  * ✅ Design collaborative work systems with human operators
  * ✅ Select system architecture considering LLM cost and latency
  * ✅ Ensure safety and robustness for real plant deployment

## Exercises

### Easy (Fundamentals)

**Q1** : What are the main differences between LLM agents and RL agents?

View Answer

**Correct Answer** :

  * **Learning method** : RL learns through trial-and-error, LLM is pre-trained (inference only)
  * **Inference speed** : RL is fast (ms order), LLM is medium (1-5 seconds)
  * **Explainability** : RL is a black box, LLM can explain in natural language
  * **Flexibility** : RL is within training scope, LLM capable of general reasoning

**Explanation** : A hybrid architecture that leverages each's strengths, with LLM handling high-level strategy and RL handling low-level control, is effective.

**Q2** : What is Tool Use (Function Calling)?

View Answer

**Correct Answer** : A feature that allows LLMs to call external functions (APIs, databases, control systems, etc.) to retrieve real-time information or execute actions.

**Explanation** : For example, to answer "What's the current temperature?", an LLM calls the get_sensor_data() function to retrieve sensor values. This allows LLMs to reason based not only on static knowledge but also on dynamic environmental information.

### Medium (Application)

**Q3** : List three safety mechanisms implemented in Example 5's production-ready system and explain the purpose of each.

View Answer

**Correct Answer** :

  1. **Fallback control (PID)** : Safely continues process operation during API failures
  2. **Budget management** : Sets monthly cost cap with alerts/stops on excess
  3. **Range checking** : Limits LLM recommendations to physically safe ranges (0-10kW)

**Explanation** : These multi-layered defenses ensure process safety while accepting LLM uncertainty.

**Q4** : Give two reasons why the LLM Coordinator in the LLM-RL Hybrid system executes only every 5 steps.

View Answer

**Correct Answer** :

  1. **Cost reduction** : LLM API calls are expensive (~$0.003/call), so reduce frequency
  2. **Latency avoidance** : LLM inference takes 1-5 seconds, unsuitable for real-time control (100ms cycle)

**Explanation** : By specializing LLM for strategic decisions (long-term goal setting) and having RL agents handle high-frequency control loops, we leverage both's strengths.

### Hard (Advanced)

**Q5** : For a 3-unit process (reactor, separator, heat exchanger), design a strategy for the LLM Coordinator to achieve "10% energy reduction while maintaining production volume", considering the following constraints.

  * Reactor: Temperature↓ → Reaction rate↓ → Production volume↓
  * Separator: Reflux ratio↓ → Energy↓, Purity↓
  * Heat exchanger: Heat transfer coefficient↑ → Energy efficiency↑

View Answer

**Recommended Strategy** :

  1. **Reactor** : Lower temperature by 2K (340K → 338K) 
     * Reason: Slight reaction rate decrease (~5%) for energy savings
     * Countermeasure: Increase catalyst concentration by 5% to compensate reaction rate
  2. **Separator** : Reduce reflux ratio from 0.8 to 0.75 
     * Reason: Reduce reboiler load by 6%
     * Countermeasure: Purity decrease (98% → 97%) within acceptable range for downstream process
  3. **Heat exchanger** : Increase heat transfer area by 10% (additional investment) 
     * Reason: Improve waste heat recovery efficiency by 15%
     * Long-term energy reduction effect

**Overall Effect** :

  * Energy reduction: 5% + 6% + (-1%) = 10% achieved
  * Production volume: Compensated by catalyst increase, maintainable

**LLM Role** : Reason through this complex tradeoff analysis in natural language and instruct each RL agent with specific setpoints.

**Q6 (Coding)** : Extend the ClaudeProcessController in Example 2 to add the following features:

  * New Tool: check_safety_constraints() - Detects safety constraint violations
  * Upon safety constraint violation, LLM automatically proposes emergency shutdown procedure

Hint

Add the following to Tool definition:
    
    
    {
        "name": "check_safety_constraints",
        "description": "Checks for violations of safety constraints (temperature, pressure, concentration).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

Add implementation to execute_tool(), and return a message including emergency shutdown procedure if violations exist.

## References

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

## Next Steps

Congratulations! You have completed all 6 chapters of this series.

**🎓 Skills Acquired**

  * ✅ Agent architectures (Reactive, Deliberative, Hybrid)
  * ✅ Process environment modeling (OpenAI Gym compliant)
  * ✅ Reward design and optimization objectives
  * ✅ Multi-agent collaborative control (QMIX, Communication)
  * ✅ Real plant deployment and safety (CBF, CQL)
  * ✅ LLM-RL integration (Tool Use, LangChain, Hybrid Architecture)

### Recommended Next Actions

**Short-term (1-2 weeks):**

  * ✅ Build simulation environment for your own processes
  * ✅ Execute Chapter 1-6 code examples with actual data
  * ✅ Auto-generate process diagnostic reports with LLM agents

**Mid-term (1-3 months):**

  * ✅ Train RL agents in simulation environment
  * ✅ Build hybrid system integrated with LLM Coordinator
  * ✅ Develop staged deployment plan (advisory mode → automatic control)

**Long-term (6+ months):**

  * ✅ Demonstration tests on pilot plant
  * ✅ Real plant deployment and operational data collection
  * ✅ Conference presentations and paper writing

[← Chapter 5: Real Plant Deployment](<chapter-5.html>) [Back to Series Index](<./index.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
