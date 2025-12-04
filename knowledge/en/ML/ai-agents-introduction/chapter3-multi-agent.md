---
title: "Chapter 3: Multi-Agent Systems"
chapter_title: "Chapter 3: Multi-Agent Systems"
---

This chapter covers Multi. You will learn essential concepts and techniques.

## What are Multi-Agent Systems?

### Why Multiple Agents are Needed

Complex tasks that are difficult to solve with a single agent can be processed more efficiently and with higher quality when multiple specialized agents collaborate.

**Advantages of Multi-Agent Systems** :

  * ✅ **Specialization** : Each agent specializes in a specific role
  * ✅ **Parallel Processing** : Execute independent tasks simultaneously
  * ✅ **Scalability** : Dynamically adjust the number of agents
  * ✅ **Fault Tolerance** : Others can cover for partial failures
  * ✅ **Modularity** : Easy to add or remove agents

### Types of Multi-Agent Architectures

Architecture | Characteristics | Application Scenarios  
---|---|---  
**Parallel** | Agents execute independently in parallel | Data collection, multi-perspective analysis  
**Sequential** | Agents hand over processing in sequence | Pipeline processing, incremental refinement  
**Hierarchical** | Manager controls subordinate workers | Separation of complex planning and execution  
**Interactive** | Agents discuss and negotiate among themselves | Decision making, consensus building  
  
## Multi-Agent Design

### Agent Role Distribution
    
    
    ```mermaid
    graph TD
        M[Manager AgentTask decomposition and coordination] --> R[Researcher AgentInformation gathering]
        M --> W[Writer AgentDocument creation]
        M --> C[Critic AgentReview and evaluation]
    
        R --> M
        W --> M
        C --> M
    
        style M fill:#e3f2fd
        style R fill:#fff3e0
        style W fill:#f3e5f5
        style C fill:#e8f5e9
    ```

### Example of Role-Based Design
    
    
    from typing import List, Dict, Any
    from dataclasses import dataclass
    
    @dataclass
    class AgentRole:
        """Agent role definition"""
        name: str
        description: str
        capabilities: List[str]
        system_prompt: str
    
    # Agent role definitions
    RESEARCHER_ROLE = AgentRole(
        name="Researcher",
        description="Expert in information gathering and analysis",
        capabilities=["web_search", "database_query", "data_analysis"],
        system_prompt="""You are an excellent researcher.
    
    Role:
    - Gather related information from web searches and databases
    - Evaluate the credibility of collected information
    - Summarize key points and report to the team
    
    Important points:
    - Clearly cite information sources
    - Cross-check multiple information sources
    - Explicitly indicate uncertain information"""
    )
    
    WRITER_ROLE = AgentRole(
        name="Writer",
        description="Expert in document creation",
        capabilities=["content_generation", "formatting", "editing"],
        system_prompt="""You are an excellent writer.
    
    Role:
    - Create high-quality documents based on researcher's information
    - Readable and logical structure
    - Writing style and tone appropriate for the target audience
    
    Important points:
    - Clear and concise expression
    - Appropriate heading and paragraph structure
    - Proper use of citations and references"""
    )
    
    CRITIC_ROLE = AgentRole(
        name="Critic",
        description="Expert in quality review",
        capabilities=["quality_check", "fact_verification", "feedback"],
        system_prompt="""You are a reviewer with critical thinking skills.
    
    Role:
    - Critically review created documents
    - Verify factual accuracy
    - Provide specific improvement points
    
    Important points:
    - Constructive feedback
    - Specific improvement suggestions
    - Clearly point out serious issues"""
    )
    

## Communication Protocols

### Message Passing

Communication between agents is conducted through structured messages.
    
    
    from dataclasses import dataclass
    from typing import Optional, Dict, Any
    from datetime import datetime
    from enum import Enum
    
    class MessageType(Enum):
        """Message types"""
        TASK = "task"              # Task instruction
        RESULT = "result"          # Task result
        QUERY = "query"            # Information request
        RESPONSE = "response"      # Information response
        ERROR = "error"            # Error notification
        STATUS = "status"          # Status update
    
    @dataclass
    class Message:
        """Inter-agent message"""
        type: MessageType
        sender: str
        receiver: str
        content: Dict[str, Any]
        timestamp: datetime
        message_id: str
        reply_to: Optional[str] = None
    
    class MessageBus:
        """Message bus (inter-agent communication)"""
    
        def __init__(self):
            self.messages: List[Message] = []
            self.subscribers: Dict[str, List[callable]] = {}
    
        def subscribe(self, agent_name: str, callback: callable):
            """Register agent for message reception"""
            if agent_name not in self.subscribers:
                self.subscribers[agent_name] = []
            self.subscribers[agent_name].append(callback)
    
        def publish(self, message: Message):
            """Deliver message"""
            self.messages.append(message)
    
            # Deliver message to receiver
            if message.receiver in self.subscribers:
                for callback in self.subscribers[message.receiver]:
                    callback(message)
    
        def broadcast(self, message: Message):
            """Broadcast to all agents"""
            self.messages.append(message)
    
            for agent_name, callbacks in self.subscribers.items():
                if agent_name != message.sender:
                    for callback in callbacks:
                        callback(message)
    
    # Usage example
    import uuid
    
    bus = MessageBus()
    
    def researcher_receive(message: Message):
        print(f"Researcher received: {message.type.value} from {message.sender}")
    
    def writer_receive(message: Message):
        print(f"Writer received: {message.type.value} from {message.sender}")
    
    # Register agents
    bus.subscribe("researcher", researcher_receive)
    bus.subscribe("writer", writer_receive)
    
    # Send message
    task_message = Message(
        type=MessageType.TASK,
        sender="manager",
        receiver="researcher",
        content={"task": "Research AI trends in 2024"},
        timestamp=datetime.now(),
        message_id=str(uuid.uuid4())
    )
    
    bus.publish(task_message)
    

### Shared Memory Approach
    
    
    from typing import Dict, Any, Optional
    import threading
    
    class SharedMemory:
        """Shared memory between agents"""
    
        def __init__(self):
            self.data: Dict[str, Any] = {}
            self.lock = threading.Lock()
            self.subscribers: Dict[str, List[callable]] = {}
    
        def write(self, key: str, value: Any, agent_name: str):
            """Write data"""
            with self.lock:
                self.data[key] = {
                    "value": value,
                    "author": agent_name,
                    "timestamp": datetime.now()
                }
    
                # Notify changes
                self._notify_subscribers(key, value, agent_name)
    
        def read(self, key: str) -> Optional[Any]:
            """Read data"""
            with self.lock:
                if key in self.data:
                    return self.data[key]["value"]
                return None
    
        def subscribe_to_key(self, key: str, callback: callable):
            """Watch for changes to a specific key"""
            if key not in self.subscribers:
                self.subscribers[key] = []
            self.subscribers[key].append(callback)
    
        def _notify_subscribers(self, key: str, value: Any, agent_name: str):
            """Notify subscribers"""
            if key in self.subscribers:
                for callback in self.subscribers[key]:
                    callback(key, value, agent_name)
    
    # Usage example
    memory = SharedMemory()
    
    def on_research_complete(key, value, agent_name):
        print(f"Research completed by {agent_name}: {value}")
    
    memory.subscribe_to_key("research_result", on_research_complete)
    
    # Researcher writes result
    memory.write("research_result", "Major AI trends in 2024...", "researcher")
    

## Collaboration Patterns

### 1\. Parallel Execution Pattern
    
    
    import asyncio
    from typing import List, Dict, Any
    
    class ParallelAgentSystem:
        """Parallel execution agent system"""
    
        def __init__(self, agents: List[Any]):
            self.agents = agents
    
        async def execute_parallel(self, task: str) -> List[Dict[str, Any]]:
            """Execute all agents in parallel"""
            tasks = [
                agent.process(task)
                for agent in self.agents
            ]
    
            results = await asyncio.gather(*tasks, return_exceptions=True)
    
            # Aggregate results
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
    
    # Usage example (pseudo-code)
    class ResearchAgent:
        def __init__(self, name: str, specialty: str):
            self.name = name
            self.specialty = specialty
    
        async def process(self, query: str) -> Dict[str, Any]:
            # Execute research asynchronously
            await asyncio.sleep(1)  # Simulate API call
            return {
                "specialty": self.specialty,
                "findings": f"Research results on {query} regarding {self.specialty}"
            }
    
    # Execute multiple specialized agents in parallel
    agents = [
        ResearchAgent("Tech Researcher", "Technology trends"),
        ResearchAgent("Market Researcher", "Market analysis"),
        ResearchAgent("Academic Researcher", "Academic research")
    ]
    
    system = ParallelAgentSystem(agents)
    results = asyncio.run(system.execute_parallel("AI in 2024"))
    print(results)
    

### 2\. Sequential Execution (Pipeline) Pattern
    
    
    from typing import Any, List, Callable
    
    class SequentialAgentSystem:
        """Sequential execution agent system (pipeline)"""
    
        def __init__(self):
            self.pipeline: List[Callable] = []
    
        def add_stage(self, agent: Callable):
            """Add agent to pipeline"""
            self.pipeline.append(agent)
    
        def execute(self, initial_input: Any) -> Any:
            """Execute pipeline"""
            current_data = initial_input
    
            for i, agent in enumerate(self.pipeline):
                print(f"Stage {i+1}: {agent.__name__}")
                current_data = agent(current_data)
                print(f"  Output: {current_data}\n")
    
            return current_data
    
    # Agents for each pipeline stage
    def data_collector(query: str) -> Dict[str, Any]:
        """Stage 1: Data collection"""
        return {
            "query": query,
            "raw_data": f"Raw data regarding {query}...",
            "sources": ["source1", "source2"]
        }
    
    def data_analyzer(data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Data analysis"""
        data["analysis"] = "Analysis results: Main trends are..."
        data["insights"] = ["Insight 1", "Insight 2"]
        return data
    
    def report_generator(data: Dict[str, Any]) -> str:
        """Stage 3: Report generation"""
        report = f"""
    Research Report: {data['query']}
    
    Analysis Results:
    {data['analysis']}
    
    Key Insights:
    - {data['insights'][0]}
    - {data['insights'][1]}
    
    Sources: {', '.join(data['sources'])}
        """
        return report.strip()
    
    # Build and execute pipeline
    pipeline = SequentialAgentSystem()
    pipeline.add_stage(data_collector)
    pipeline.add_stage(data_analyzer)
    pipeline.add_stage(report_generator)
    
    final_report = pipeline.execute("Latest AI agent trends")
    print("=== Final Report ===")
    print(final_report)
    

### 3\. Hierarchical (Manager-Worker) Pattern
    
    
    from typing import List, Dict, Any
    from openai import OpenAI
    
    class ManagerAgent:
        """Manager agent (task decomposition and coordination)"""
    
        def __init__(self, api_key: str, workers: List[Any]):
            self.client = OpenAI(api_key=api_key)
            self.workers = workers
            self.task_history = []
    
        def execute(self, user_request: str) -> str:
            """Process user request"""
            # Step 1: Decompose task
            subtasks = self.decompose_task(user_request)
    
            # Step 2: Delegate to workers
            results = self.delegate_to_workers(subtasks)
    
            # Step 3: Synthesize results
            final_result = self.synthesize_results(user_request, results)
    
            return final_result
    
        def decompose_task(self, request: str) -> List[Dict[str, Any]]:
            """Decompose task into subtasks"""
            worker_capabilities = "\n".join([
                f"- {w.name}: {w.capabilities}"
                for w in self.workers
            ])
    
            prompt = f"""Please decompose the following request into subtasks to be assigned to available workers.
    
    Request: {request}
    
    Available workers:
    {worker_capabilities}
    
    Output each subtask in the following format:
    1. [Worker name] Task description
    2. [Worker name] Task description
    ..."""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
    
            # Parse subtasks (simplified version)
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
            """Delegate tasks to workers"""
            results = []
    
            for subtask in subtasks:
                worker_name = subtask["worker"]
                task = subtask["task"]
    
                # Find the corresponding worker
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
                        "result": f"Error: Worker {worker_name} not found"
                    })
    
            return results
    
        def synthesize_results(self, original_request: str, results: List[Dict[str, Any]]) -> str:
            """Synthesize results and generate final answer"""
            results_text = "\n\n".join([
                f"{r['worker']}'s results:\n{r['result']}"
                for r in results
            ])
    
            prompt = f"""Please synthesize the results from each worker and generate a final answer for the following request.
    
    Original request: {original_request}
    
    Worker results:
    {results_text}
    
    Generate the synthesized answer:"""
    
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
    class WorkerAgent:
        """Worker agent"""
    
        def __init__(self, name: str, capabilities: str, system_prompt: str, api_key: str):
            self.name = name
            self.capabilities = capabilities
            self.system_prompt = system_prompt
            self.client = OpenAI(api_key=api_key)
    
        def execute(self, task: str) -> str:
            """Execute task"""
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task}
                ],
                temperature=0.7
            )
    
            return response.choices[0].message.content
    
    # Usage example
    researcher = WorkerAgent(
        name="Researcher",
        capabilities="Web search, data collection",
        system_prompt="You are a research expert.",
        api_key="your-api-key"
    )
    
    writer = WorkerAgent(
        name="Writer",
        capabilities="Document creation, editing",
        system_prompt="You are a document creation expert.",
        api_key="your-api-key"
    )
    
    manager = ManagerAgent(
        api_key="your-api-key",
        workers=[researcher, writer]
    )
    
    result = manager.execute("Please create a 1000-word report on AI trends in 2024")
    print(result)
    

## Orchestration Strategies

### Dynamic Task Assignment
    
    
    from typing import List, Dict, Any
    import time
    
    class TaskOrchestrator:
        """Task orchestrator"""
    
        def __init__(self, agents: List[Any]):
            self.agents = agents
            self.task_queue = []
            self.agent_status = {agent.name: "idle" for agent in agents}
    
        def add_task(self, task: Dict[str, Any]):
            """Add task to queue"""
            self.task_queue.append(task)
    
        def get_available_agent(self, required_capability: str = None):
            """Get available agent"""
            for agent in self.agents:
                if self.agent_status[agent.name] == "idle":
                    if required_capability is None or required_capability in agent.capabilities:
                        return agent
            return None
    
        def execute_tasks(self):
            """Process task queue"""
            while self.task_queue:
                task = self.task_queue.pop(0)
    
                # Find appropriate agent
                agent = self.get_available_agent(task.get("required_capability"))
    
                if agent:
                    print(f"Assigning task '{task['name']}' to {agent.name}")
                    self.agent_status[agent.name] = "busy"
    
                    # Execute task (assuming asynchronous)
                    result = agent.execute(task)
    
                    self.agent_status[agent.name] = "idle"
                    print(f"{agent.name} completed task '{task['name']}'")
                else:
                    # If agent unavailable, return to queue
                    self.task_queue.append(task)
                    time.sleep(1)
    

## State Management and Conflict Resolution

### Distributed State Synchronization
    
    
    from typing import Dict, Any, Optional
    from datetime import datetime
    import json
    
    class StateManager:
        """State management between agents"""
    
        def __init__(self):
            self.state: Dict[str, Any] = {}
            self.version: Dict[str, int] = {}
            self.history: List[Dict[str, Any]] = []
    
        def update_state(self, key: str, value: Any, agent_name: str) -> bool:
            """Update state (with versioning)"""
            current_version = self.version.get(key, 0)
    
            # Record update
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
            """Get state (with specific version support)"""
            if version is None:
                return self.state.get(key)
    
            # Search for specific version in history
            for record in reversed(self.history):
                if record["key"] == key and record["version"] == version:
                    return record["value"]
    
            return None
    
        def resolve_conflict(self, key: str, conflicting_values: List[Dict[str, Any]]) -> Any:
            """Resolve conflict"""
            # Adopt value with latest timestamp (Last-Write-Wins)
            latest = max(conflicting_values, key=lambda x: x["timestamp"])
            return latest["value"]
    
    # Usage example
    state_manager = StateManager()
    
    # Multiple agents update same key
    state_manager.update_state("document_title", "Introduction to AI Agents", "agent1")
    state_manager.update_state("document_title", "Complete Guide to AI Agents", "agent2")
    
    # Get latest value
    current_title = state_manager.get_state("document_title")
    print(f"Current title: {current_title}")
    
    # Check history
    print("\nUpdate history:")
    for record in state_manager.history:
        print(f"  v{record['version']}: {record['value']} (by {record['agent']})")
    

## Summary

### What We Learned in This Chapter

  * ✅ **Multi-Agent Design** : Role distribution and specialization
  * ✅ **Communication Protocols** : Message passing and shared memory
  * ✅ **Collaboration Patterns** : Implementation of parallel, sequential, and hierarchical approaches
  * ✅ **Orchestration** : Task assignment and coordination
  * ✅ **State Management** : Distributed state synchronization and conflict resolution

### Design Principles

> **Effective multi-agent systems** are realized through clear role distribution, efficient communication, appropriate orchestration, and robust state management

[← Chapter 2: Tool Use](<./chapter2-tool-use.html>) [Chapter 4: Practical Applications →](<./chapter4-applications.html>)

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
