import asyncio
from typing import Dict
from swarm import Agent

class ReasoningNetwork:
    def __init__(self, processing_factor: int, reasoning_size: int, client):
        self.processing_factor = processing_factor
        self.reasoning_size = reasoning_size
        self.client = client
        self.agents: Dict[str, Agent] = {}

    async def create_agent(self, name: str, is_reasoning: bool) -> Agent:
        instructions = ("Analyze and synthesize information to form a coherent thought. Use reasoning ablilites, using the PUREST form of logical thinking."
                        if is_reasoning else
                        "Integrate inputs from child neurons to develop a reasoning process.")
        agent = Agent(name=name, instructions=instructions)
        self.agents[name] = agent
        return agent

    async def init_network(self) -> None:
        queue = asyncio.Queue()
        await queue.put(("root", 0))
        
        while not queue.empty():
            name, depth = await queue.get()
            is_reasoning = self.reasoning_size == self.processing_factor ** depth
            await self.create_agent(name, is_reasoning)
            
            if not is_reasoning:
                for i in range(self.processing_factor):
                    child_name = f"{name}_child_{i}"
                    await queue.put((child_name, depth + 1))

    async def run_agent(self, agent: Agent, input_str: str) -> str:
        return await asyncio.to_thread(self.client.run, agent=agent, messages=[{"role": "user", "content": input_str}])

    async def process_agent(self, name: str, input_str: str) -> str:
        agent = self.agents[name]
        children = [f"{name}_child_{i}" for i in range(self.processing_factor)]
        
        if children[0] not in self.agents:  # Leaf node
            response = await self.run_agent(agent, input_str)
            return response.messages[-1]['content']
        
        tasks = [self.process_agent(child, input_str) for child in children]
        child_outputs = await asyncio.gather(*tasks)
        child_context = "\n".join(f"{i+1}. {output}" for i, output in enumerate(child_outputs))
        
        response = await self.run_agent(agent, f"Integrate child outputs for reasoning:\n{child_context}")
        return response.messages[-1]['content']

    async def execute(self, input_str: str) -> str:
        await self.init_network()
        return await self.process_agent("root", input_str)
