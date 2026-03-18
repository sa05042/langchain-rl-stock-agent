# **LangChain Reinforcement Learning Stock Prediction Agent**

An AI-powered financial analysis system that combines:
- LSTM deep learning models for stock price forecasting
- Reinforcement Learning (PPO) for trading strategy optimization
- LangChain agents for natural-language interaction

The system allows users to ask questions such as:
- "What is the current stock price of Tesla?"
- "Predict the next closing price of TSLA."
- "Predict the next 7 days closing prices."
- "Train a reinforcement learning trading agent."
- "Run a trading simulation."

The AI agent automatically decides which tools to call to perform the analysis.

## **Project Architecture**

The system integrates three AI components:
### **1. Time Series Forecasting**
- LSTM neural networks predict future stock prices.
### **2. Reinforcement Learning**
- A PPO agent learns trading strategies in a simulated environment.
### **3. LLM Agent (LangChain)**
- Natural language interface for financial analysis.

### **Pipeline**
		User Query
			 │
			 ▼
		LangChain Agent
			 │
			 ▼
		Tool Selection
			 │
			 ├── Stock Data Retrieval
			 ├── LSTM Price Forecasting
			 ├── RL Agent Training
			 └── Trading Simulation

## **Repository Structure**
		langchain-rl-stock-agent/
					│
					├── main.py
					├── requirements.txt
					├── README.md
					├── .env.example
					│
					├── config/
					│   └── env_loader.py
					│
					├── cache/
					│   └── model_cache.py
					│
					├── data/
					│   └── stock_data.py
					│
					├── models/
					│   └── lstm_model.py
					│
					├── rl/
					│   └── trading_env.py
					│
					├── tools/
					│   └── stock_tools.py
					│
					└── agent/
						└── stock_agent.py
### **Root Files**
- **`main.py`** – Entry point of the project. Runs the agent and example simulations.  
- **`requirements.txt`** – Lists all Python dependencies required to run the project.  
- **`README.md`** – Project documentation file.  
- **`.env.example`** – Template for environment variables (e.g., API keys). Copy to `.env` and add your credentials.  

### **Folders**

- **`config/`** – Configuration files.  
  - `env_loader.py` – Loads environment variables from `.env`.  

- **`cache/`** – Caching system for models and agents.  
  - `model_cache.py` – Stores trained LSTM models and RL agents to avoid retraining.  

- **`data/`** – Handles stock data.  
  - `stock_data.py` – Fetches historical stock data using `yfinance`.  

- **`models/`** – AI models for stock prediction.  
  - `lstm_model.py` – Defines and trains LSTM models for forecasting stock prices.  

- **`rl/`** – Reinforcement Learning environments.  
  - `trading_env.py` – Custom Gym environment for stock trading simulations.  

- **`tools/`** – LangChain tools.  
  - `stock_tools.py` – Implements functions for training agents, simulating trading, and predicting stock prices.  

- **`agent/`** – LangChain agent logic.  
  - `stock_agent.py` – Integrates the LLM (ChatOpenAI), tools, and executes queries for analysis and trading.
						

## **Features**
### **Real-time Stock Data**
Uses Yahoo Finance API to retrieve historical and latest stock prices.
### **Deep Learning Forecasting**
LSTM model predicts stock prices using historical data.
### **Monte Carlo Forecasting**
Future price predictions include stochastic noise to simulate realistic volatility.
### **Reinforcement Learning Trading**
A PPO (Proximal Policy Optimization) agent learns optimal trading policies.
### **Trading Simulation**
Evaluates the RL agent on unseen test data.
### **Natural Language AI Agent**
LangChain agent interprets user queries and executes the correct tools.

## **Installation**
Clone the repository:
			
	git clone https://github.com/sa05042/langchain-rl-stock-agent.git
					
	cd langchain-rl-stock-agent
					
Install dependencies:
					
	pip install -r requirements.txt

Environment Variables:

Create environment file
	
	cp .env.example .env
	
Add OpenAI key in .env.
	
	OPENAI_API_KEY=your_openai_api_key

## **Requirements**
Main dependencies:
- Python 3.9+
- TensorFlow
- Stable-Baselines3
- Gymnasium
- LangChain
- OpenAI API
- yfinance
- scikit-learn
- numpy
- pandas
				
Install using:
	
	pip install -r requirements.txt

## **Running the Project**

Run the main script:
	
	python main.py
					
Example output:
				
	The current stock price for TSLA is 248.31 USD
	Predicted next closing price for TSLA: 250.41
	RL agent trained on TSLA
	Trading simulation completed
			
## **Example Queries**
The AI agent supports natural language queries:
### **Get current stock price**
	What is the current stock price of Tesla (TSLA)?
### **Predict next day price**
	Predict the next closing price of TSLA
### **Predict next 7 days**
	Predict the next 7 days closing prices for TSLA
### **Train RL agent**
	Train RL agent for TSLA for 5000 timesteps
### **Simulate trading**
	Run trading simulation on TSLA for the next 30 days
			
## **Model Pipeline**
				Historical Stock Data
						│
						▼
				LSTM Forecast Model
						│
						▼
				Synthetic Future Prices
						│
						▼
				Trading Environment
						│
						▼
				PPO Reinforcement Learning Agent
						│
						▼
				Trading Strategy

## **Example Trading Simulation Output**
	{
	 "ticker": "TSLA",
	 "days": 30,
	 "initial_portfolio_value": 10000,
	 "final_portfolio_value": 11235,
	 "total_gain": 1235
	}

## **Future Improvements**
Possible extensions:
- Multi-stock portfolio optimization
- Transformer-based time series models
- Risk-aware reward functions
- GPU acceleration
- Real-time trading dashboards
- Backtesting frameworks
				
## **Disclaimer**
- This project is intended for educational and research purposes only.
- It does not constitute financial advice and should not be used for real-world trading without further validation.

## **Author**
Sabbir

Research interests: Machine Learning, Deep Learning, Privacy, and Security.
