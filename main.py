from config.env_loader import *
from agent.stock_agent import agent_executor


response = agent_executor.invoke(
    {"input": "What is the current stock price of Tesla (TSLA)?"}
)

print(response["output"])


response = agent_executor.invoke(
    {"input": "Predict the next closing price of (TSLA)."}
)

print(response["output"])


response = agent_executor.invoke(
    {"input": "Predict the next 7 days closing prices for (TSLA)"}
)

print(response["output"])


response = agent_executor.invoke(
    {"input": "Train RL agent for (TSLA) for 5000 timesteps"}
)

print(response["output"])


response = agent_executor.invoke(
    {"input": "Run trading simulation on (TSLA) for the next 30 days"}
)

print(response["output"])