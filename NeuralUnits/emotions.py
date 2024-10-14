import asyncio
from typing import Dict
from swarm import Agent

class EmotionNetwork:
    def __init__(self, processing_factor: int, emotion_size: int, client):
        self.processing_factor = processing_factor
        self.emotion_size = emotion_size
        self.client = client
        self.agents: Dict[str, Agent] = {}

    async def create_agent(self, name: str, is_emotion: bool) -> Agent:
        instructions = ("Think about the emotions you feel about the world around you. Use the most PURE form of emotional thinking."
                        if is_emotion else
                        "Process the output of child neurons and tie them together to develop an emotional response.")
        agent = Agent(name=name, instructions=instructions)
        self.agents[name] = agent
        return agent

    async def init_network(self) -> None:
        queue = asyncio.Queue()
        await queue.put(("root", 0))
        
        while not queue.empty():
            name, depth = await queue.get()
            is_emotion = self.emotion_size == self.processing_factor ** depth
            await self.create_agent(name, is_emotion)
            
            if not is_emotion:
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
        
        response = await self.run_agent(agent, f"Process child outputs:\n{child_context}")
        return response.messages[-1]['content']

    async def execute(self, input_str: str) -> str:
        await self.init_network()
        return await self.process_agent("root", input_str)