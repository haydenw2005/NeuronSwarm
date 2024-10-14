from swarm import Agent, Swarm
from dotenv import load_dotenv
from NeuralUnits.emotions import EmotionNetwork
from NeuralUnits.reasoning import ReasoningNetwork
import asyncio


load_dotenv()

client = Swarm()

PROCESSING_FACTOR = 5

NETWORK_SIZE = 625

question = "Are there any bad actors at OpenAI? Who else threatens the future of AI?"

async def main():
    print("Starting network...")
    emotion_network = EmotionNetwork(PROCESSING_FACTOR, NETWORK_SIZE, client)
    reasoning_network = ReasoningNetwork(PROCESSING_FACTOR, NETWORK_SIZE, client)
    
    emotion_task = asyncio.create_task(emotion_network.execute(question))
    reasoning_task = asyncio.create_task(reasoning_network.execute(question))
    
    emotion_response, reasoning_response = await asyncio.gather(emotion_task, reasoning_task)
    #print(f"Emotional response: {emotion_response}")
    #print(f"Reasoning response: {reasoning_response}")
    print(f"Question: {question}\n")
    agent = Agent(
        name="Agent",
        instructions="You are the part of the brain that ties together different ways of thinking. You will receive an emotion response and a reasoning response. Combine them to form a final response.",
    )
    
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": f"Emotional response: {emotion_response}"}, {"role": "user", "content": f"Logical reasoning response: {reasoning_response}"}],
    )
    
    print("----- RESPONSE ---------------------------------------------")
    print(response.messages[-1]['content'])
    
if __name__ == "__main__":
    asyncio.run(main())
