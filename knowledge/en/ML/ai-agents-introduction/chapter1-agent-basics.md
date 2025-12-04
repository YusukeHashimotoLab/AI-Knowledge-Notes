---
title: "Chapter 1: AI Agent Fundamentals"
chapter_title: "Chapter 1: AI Agent Fundamentals"
---

This chapter covers the fundamentals of AI Agent Fundamentals, which what is an ai agent?. You will learn essential concepts and techniques.

## What is an AI Agent?

### Definition and Characteristics

An **AI Agent** is an AI system that perceives its environment, makes autonomous decisions, and takes actions to achieve goals. Unlike traditional static AI models, agents dynamically assess situations and complete complex tasks through multiple steps.

**Key Characteristics of AI Agents** :

  * **Autonomy** : Acts independently without human instructions
  * **Reactivity** : Recognizes environmental changes and responds appropriately
  * **Goal-oriented** : Plans actions toward clear objectives
  * **Learning** : Learns and improves from experience
  * **Tool Use** : Leverages external tools and APIs

### Differences from Traditional AI

Aspect | Traditional AI | AI Agent  
---|---|---  
Input/Output | Single input → Single output | Multi-step dialogue and actions  
Decision Making | Immediate response generation | Reasoning → Action → Observation loop  
External Integration | Limited or impossible | Utilizes tools, APIs, and search  
Task Complexity | Simple question answering | Multi-stage complex tasks  
Adaptability | Fixed behavior | Changes strategy based on situation  
  
## Agent Architecture

### Basic Loop: Perception, Reasoning, Action

AI agents achieve goals by repeating the following cycle:
    
    
    ```mermaid
    graph LR
        A[Perception] --> B[Reasoning]
        B --> C[Action]
        C --> D[Environment]
        D --> A
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
    ```

  1. **Perception** : Observe environmental state, user input, and previous action results
  2. **Reasoning** : Plan next action based on observed information
  3. **Action** : Execute tools, generate responses, perform tasks
  4. **Environment** : Action results reflected in environment, leading to next perception

### Key Agent Components
    
    
    # Basic agent structure
    class Agent:
        def __init__(self, llm, tools, memory):
            self.llm = llm              # Large language model (reasoning engine)
            self.tools = tools          # Available tool set
            self.memory = memory        # Conversation history and state
            self.max_iterations = 10    # Maximum execution count
    
        def run(self, task):
            """Agent execution loop"""
            self.memory.add_message("user", task)
    
            for i in range(self.max_iterations):
                # 1. Reasoning: Determine next action
                thought = self.think()
    
                # 2. Action: Execute tool or provide answer
                if thought.action:
                    observation = self.act(thought.action)
                    self.memory.add_observation(observation)
                else:
                    return thought.answer
    
            return "Could not complete task"
    
        def think(self):
            """Reason about next action using LLM"""
            prompt = self.build_prompt()
            response = self.llm.generate(prompt)
            return self.parse_response(response)
    
        def act(self, action):
            """Execute tool and get result"""
            tool = self.tools[action.tool_name]
            result = tool.execute(action.parameters)
            return result
    

## ReAct Pattern

### Integration of Reasoning and Acting

**ReAct** (Reasoning and Acting) is an agent pattern proposed by Yao et al. (2022) that determines actions while verbalizing the reasoning process.

**ReAct Steps** :

  1. **Thought** : Analyze current situation and consider next action
  2. **Action** : Select tool and determine parameters
  3. **Observation** : Confirm tool execution results
  4. Repeat → Eventually provide Answer

### Example ReAct Prompt
    
    
    REACT_PROMPT = """You are an assistant that answers questions. Please repeat thinking and acting in the following format.
    
    Available tools:
    - search: Execute web search (input: search query)
    - calculator: Calculate mathematical expressions (input: expression)
    - finish: Return final answer (input: answer text)
    
    Format:
    Question: User's question
    Thought: What you are thinking
    Action: tool_name[input]
    Observation: Tool execution result
    ... (repeat as needed)
    Thought: Final conclusion
    Action: finish[final answer]
    
    Question: {question}
    Thought:"""
    
    # Execution example
    question = "Who won the 2024 Nobel Prize in Physics?"
    response = """
    Thought: I need to search the web for the latest Nobel Prize information
    Action: search[2024 Nobel Prize in Physics winner]
    Observation: The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton. Their foundational research in machine learning was recognized.
    Thought: The winner has been identified from the search results
    Action: finish[The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton. They were recognized for their contributions to the theoretical foundation of machine learning, particularly neural networks.]
    """
    

### ReAct Agent Implementation
    
    
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
            """Mock search (use SerpAPI etc. in practice)"""
            # In actual implementation, call web search API
            return f"Search results: Information about {query}"
    
        def calculator(self, expression):
            """Calculator tool"""
            try:
                result = eval(expression)
                return f"Calculation result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
    
        def run(self, question, max_steps=5):
            """Execute ReAct loop"""
            prompt = REACT_PROMPT.format(question=question)
    
            for step in range(max_steps):
                # Generate next action with LLM
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
    
                text = response.choices[0].message.content
                prompt += text
    
                # Extract Action
                action_match = re.search(r'Action: (\w+)\[(.*?)\]', text)
                if not action_match:
                    continue
    
                tool_name = action_match.group(1)
                tool_input = action_match.group(2)
    
                # Check for termination
                if tool_name == "finish":
                    return tool_input
    
                # Execute tool
                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_input)
                    prompt += f"\nObservation: {observation}\nThought:"
                else:
                    prompt += f"\nObservation: Error - tool {tool_name} does not exist\nThought:"
    
            return "Reached maximum steps"
    
    # Usage example
    agent = ReActAgent(api_key="your-api-key")
    answer = agent.run("Please calculate 123 + 456")
    print(answer)  # Output: 579
    

## Chain-of-Thought

### Step-by-Step Reasoning Process

**Chain-of-Thought (CoT)** is a technique that breaks down complex problems into steps for reasoning. Proposed by Wei et al. (2022), it is especially effective for tasks requiring mathematical reasoning and logical thinking.

**Benefits of CoT** :

  * ✅ **Improved Accuracy** : Higher correct answer rate for complex problems
  * ✅ **Interpretability** : Reasoning process is visualized
  * ✅ **Error Detection** : Thinking process can be reviewed
  * ✅ **Easy Debugging** : Can identify where mistakes occurred

### Few-shot CoT Prompt
    
    
    COT_PROMPT = """Please solve problems step by step as in the examples below.
    
    Q: At a cafe, coffee is 300 yen per cup and cake is 450 yen per piece. How much does it cost to buy 2 cups of coffee and 3 pieces of cake?
    A: First, calculate the total for coffee: 300 yen × 2 cups = 600 yen
    Next, calculate the total for cake: 450 yen × 3 pieces = 1,350 yen
    Finally, add them together: 600 yen + 1,350 yen = 1,950 yen
    Answer: 1,950 yen
    
    Q: {question}
    A: Let's think step by step."""
    
    # Implementation example
    from openai import OpenAI
    
    def chain_of_thought(question, api_key):
        """Reasoning using CoT"""
        client = OpenAI(api_key=api_key)
    
        prompt = COT_PROMPT.format(question=question)
    
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    
        return response.choices[0].message.content
    
    # Usage example
    question = "There are 12 apples. You gave 3 to a friend and bought 8 more. How many apples are there now?"
    answer = chain_of_thought(question, "your-api-key")
    print(answer)
    # Output:
    # First, calculate apples after giving some away: 12 - 3 = 9
    # Next, add the newly bought apples: 9 + 8 = 17
    # Answer: 17
    

### Zero-shot CoT ("Let's think step by step")

Kojima et al. (2022) discovered that the magic phrase "Let's think step by step" can elicit step-by-step reasoning without providing examples.
    
    
    def zero_shot_cot(question, api_key):
        """Zero-shot CoT: Step-by-step reasoning without examples"""
        client = OpenAI(api_key=api_key)
    
        # Step 1: Generate reasoning process
        prompt1 = f"{question}\n\nLet's think step by step."
    
        response1 = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt1}],
            temperature=0
        )
    
        reasoning = response1.choices[0].message.content
    
        # Step 2: Extract final answer from reasoning
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
    
    # Usage example
    result = zero_shot_cot(
        "When 5 is added to 3 times a certain number, the result is 23. What is the number?",
        "your-api-key"
    )
    print(f"Reasoning: {result['reasoning']}")
    print(f"Answer: {result['answer']}")
    

## Basic Agent Implementation

### Simple Agent Loop
    
    
    import json
    from openai import OpenAI
    from typing import List, Dict, Any
    
    class SimpleAgent:
        """Simple AI agent implementation"""
    
        def __init__(self, api_key: str, tools: Dict[str, callable]):
            self.client = OpenAI(api_key=api_key)
            self.tools = tools
            self.conversation_history = []
            self.system_prompt = """You are a capable AI assistant.
    Use tools as needed to answer user questions.
    
    Available tools:
    {tool_descriptions}
    
    Thinking process:
    1. Understand the question
    2. Select necessary tools
    3. Execute tools
    4. Integrate results and respond"""
    
        def get_tool_descriptions(self) -> str:
            """Generate tool descriptions"""
            descriptions = []
            for name, func in self.tools.items():
                desc = func.__doc__ or "No description"
                descriptions.append(f"- {name}: {desc}")
            return "\n".join(descriptions)
    
        def run(self, user_input: str, max_iterations: int = 5) -> str:
            """Execute agent"""
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
    
            for iteration in range(max_iterations):
                # Query LLM
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
    
                # Parse tool call
                tool_call = self.parse_tool_call(assistant_message)
    
                if tool_call:
                    # Execute tool
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
    
                    if tool_name in self.tools:
                        result = self.tools[tool_name](**tool_args)
    
                        # Add result to conversation history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": f"Executed tool {tool_name}"
                        })
                        self.conversation_history.append({
                            "role": "user",
                            "content": f"Result: {result}"
                        })
                    else:
                        # Unknown tool
                        self.conversation_history.append({
                            "role": "user",
                            "content": f"Error: Tool {tool_name} does not exist"
                        })
                else:
                    # No tool call = final answer
                    return assistant_message
    
            return "Reached maximum iterations"
    
        def parse_tool_call(self, message: str) -> Dict[str, Any]:
            """Extract tool call from message (simplified version)"""
            # In practice, a more robust parser is needed
            import re
            match = re.search(r'TOOL: (\w+)\((.*?)\)', message)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                # Simple argument parsing
                args = {}
                if args_str:
                    for arg in args_str.split(','):
                        key, value = arg.split('=')
                        args[key.strip()] = value.strip().strip('"\'')
                return {"name": tool_name, "args": args}
            return None
    
    # Tool definitions
    def get_weather(location: str) -> str:
        """Get weather for specified location"""
        # In practice, call API
        weather_data = {
            "Tokyo": "Sunny, 22°C",
            "Osaka": "Cloudy, 20°C",
            "Sapporo": "Rainy, 15°C"
        }
        return weather_data.get(location, "No data")
    
    def calculate(expression: str) -> float:
        """Calculate mathematical expression"""
        try:
            result = eval(expression)
            return result
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    # Execute agent
    agent = SimpleAgent(
        api_key="your-api-key",
        tools={
            "get_weather": get_weather,
            "calculate": calculate
        }
    )
    
    response = agent.run("Tell me the weather in Tokyo")
    print(response)
    

## Prompt Engineering for Agents

### Designing Effective System Prompts

Agent behavior is greatly influenced by the system prompt. Here are best practices for effective prompt design.

#### 1\. Clear Role Definition
    
    
    SYSTEM_PROMPT = """You are a customer support agent.
    
    Role:
    - Accurately understand user problems
    - Collect information using appropriate tools
    - Provide friendly and professional responses
    
    Important constraints:
    - Do not speculate on uncertain information
    - Handle personal information carefully
    - Escalate appropriately when errors occur
    """
    

#### 2\. Tool Usage Guidelines
    
    
    TOOL_USAGE_GUIDE = """
    Available tools:
    
    1. search_database(query: str) -> List[Dict]
       - Search database for relevant information
       - Use cases: Product information, order history search
    
    2. send_email(to: str, subject: str, body: str) -> bool
       - Send email
       - Use cases: Confirmation emails, notifications
    
    3. escalate_to_human(reason: str) -> None
       - Escalate to human operator
       - Use cases: Complex issues, complaint handling
    
    Tool selection principles:
    - First collect necessary information with search_database
    - Generate response if automatic handling is possible
    - Use escalate_to_human for complex or important cases
    """
    

#### 3\. Few-shot Examples
    
    
    FEW_SHOT_EXAMPLES = """
    Example 1:
    User: Please tell me the delivery status of order number 12345
    Thought: Need to search order information from database
    Action: search_database(query="order_number:12345")
    Observation: {order_id: 12345, status: "in_transit", tracking: "ABC123"}
    Response: Your order 12345 is currently in transit. The tracking number is ABC123.
    
    Example 2:
    User: I would like a refund
    Thought: Refunds involve important financial processing, should hand over to human
    Action: escalate_to_human(reason="refund request")
    Response: I understand you'd like a refund. Let me connect you with a representative.
    """
    

## Summary

### What We Learned in This Chapter

  * ✅ **AI Agent Definition** : Autonomy, reactivity, goal-orientation, tool use
  * ✅ **Agent Architecture** : Perception → Reasoning → Action loop
  * ✅ **ReAct Pattern** : Agent design integrating reasoning and action
  * ✅ **Chain-of-Thought** : Improved accuracy through step-by-step reasoning
  * ✅ **Basic Implementation** : Building simple agent loops
  * ✅ **Prompt Design** : Effective prompts for agents

### Key Concepts

> **Agent = LLM + Tools + Reasoning Loop**
> 
> AI agents autonomously solve complex tasks by combining the reasoning capabilities of large language models with external tools, repeating multi-step thinking and action.

### Next Steps

In Chapter 2, we will learn in detail about Function Calling and Tool Use:

  * OpenAI/Anthropic Function Calling API
  * Tool schema definition
  * External API integration and error handling
  * Practical tool implementation patterns

[← Series Overview](<./index.html>) [Chapter 2: Tool Use →](<./chapter2-tool-use.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
