import numpy as np
import yfinance as yf
import tensorflow as tf

from langchain.tools import tool
from stable_baselines3 import PPO

from cache.model_cache import trained_models, trained_rl_agents, global_stock_data
from data.stock_data import get_stock_data
from models.lstm_model import build_and_train_model
from rl.trading_env import TradingEnv


@tool
def train_rl_agent(ticker="AAPL", timesteps=500):

    df = get_stock_data(ticker, period="2y")

    prices = df["Close"].values.astype(float)

    split_idx = int(len(prices)*0.8)

    train_prices = prices[:split_idx]

    test_prices = prices[split_idx:]

    model, scaler, seq_len = build_and_train_model(train_prices.reshape(-1,1))

    trained_models[ticker] = (model, scaler, seq_len)

    env = TradingEnv(
        train_prices,
        lstm_model=model,
        scaler=scaler,
        seq_len=seq_len,
        use_lstm_forecast=True,
        forecast_days=timesteps
    )

    rl_agent = PPO("MlpPolicy", env, verbose=0)

    rl_agent.learn(total_timesteps=timesteps)

    trained_rl_agents[ticker] = {
        "agent": rl_agent,
        "test_prices": test_prices,
        "scaler": scaler,
        "seq_len": seq_len,
        "lstm_model": model
    }

    return f"RL agent trained on {ticker} with {len(train_prices)} steps of training data."


@tool
def simulate_trading(ticker="AAPL", days=30):
    """Simulates trading with a trained RL agent on the test set (unseen data).
    Returns portfolio values, total gain, and per-step rewards.
    """

    if ticker not in trained_rl_agents:
        return f"No RL agent trained for {ticker}. Train it first."

    bundle = trained_rl_agents[ticker]
    rl_agent = bundle["agent"]
    test_prices = bundle["test_prices"]
    scaler = bundle["scaler"]
    seq_len = bundle["seq_len"]
    model = bundle["lstm_model"]

    # Use only unseen test prices
    env = TradingEnv(
        test_prices,
        lstm_model=model,
        scaler=scaler,
        seq_len=seq_len,
        use_lstm_forecast=True,
        forecast_days=days
    )

    obs, _ = env.reset()
    portfolio_history = []
    reward_history = []

    for _ in range(days):
        action, _ = rl_agent.predict(obs, deterministic=True)
        action = int(np.asarray(action).squeeze())  # safe convert

        obs, reward, done, _, info = env.step(action)
        portfolio_history.append(float(info["portfolio_value"]))
        reward_history.append(float(reward))

        if done:
            break

    final_portfolio_value = portfolio_history[-1]
    initial_balance = env.initial_balance
    total_gain = final_portfolio_value - initial_balance

    return {
        "ticker": ticker,
        "days": len(portfolio_history),
        "initial_portfolio_value": round(initial_balance, 2),
        "final_portfolio_value": round(final_portfolio_value, 2),
        "total_gain": round(total_gain, 2),
        "portfolio_trend": [round(v, 2) for v in portfolio_history],
        "rewards": [round(r, 6) for r in reward_history]  # high precision since they can be small
    }


@tool
def predict_stock_price_lstm(ticker="AAPL"):
    """Forecasts the next-day closing price using a cached LSTM model with optimized TensorFlow graph."""

    if ticker not in trained_models:
        df = get_stock_data(ticker)
        model, scaler, seq_len = build_and_train_model(df.values)
        trained_models[ticker] = (model, scaler, seq_len)
    else:
        model, scaler, seq_len = trained_models[ticker]

    df = global_stock_data[ticker]
    prices = df["Close"].values.astype(float).reshape(-1, 1)
    scaled = scaler.transform(prices)

    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)  # shape (1, seq_len, 1)

    # Precompile tf.function once
    import tensorflow as tf
    @tf.function
    def predict_step(seq_tensor):
        return model(seq_tensor, training=False)

    seq_tensor = tf.convert_to_tensor(last_seq, dtype=tf.float32)
    pred_scaled = predict_step(seq_tensor).numpy()

    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    return f"Predicted next closing price for {ticker}: {pred_price:.2f}"

@tool
def predict_future_prices_lstm(ticker: str = "AAPL", days: int = 7, mc_simulations: int = 30) -> dict:
    """Forecast the next N days of closing prices with realistic fluctuations using Monte Carlo dropout."""

    if ticker not in trained_models:
        df = get_stock_data(ticker)
        model, scaler, seq_len = build_and_train_model(df.values)
        trained_models[ticker] = (model, scaler, seq_len)
    else:
        model, scaler, seq_len = trained_models[ticker]

    df = global_stock_data[ticker]
    prices = df["Close"].values.astype(float).reshape(-1, 1)
    scaled = scaler.transform(prices)

    last_seq = scaled[-seq_len:]  # shape (seq_len, 1)

    predictions_mc = []

    # Compute recent historical volatility (percent change)
    returns = np.diff(prices.flatten()) / prices.flatten()[:-1]
    volatility = np.std(returns[-50:])  # use last 50 days for realistic volatility

    import tensorflow as tf

    @tf.function
    def predict_step(seq_tensor):
        return model(seq_tensor, training=True)  # keep dropout active

    for day in range(days):
        day_preds = []

        for _ in range(mc_simulations):
            seq_tensor = tf.convert_to_tensor(last_seq.reshape(1, seq_len, last_seq.shape[1]), dtype=tf.float32)
            pred_scaled = predict_step(seq_tensor).numpy()
            
            # Add random noise scaled to volatility
            noise = np.random.normal(0, volatility, size=pred_scaled.shape)
            pred_scaled_noisy = pred_scaled + noise

            # Convert back to price
            pred_price = scaler.inverse_transform(pred_scaled_noisy)[0][0]
            day_preds.append(pred_price)

        # Average prediction of MC simulations
        avg_pred = np.mean(day_preds)
        predictions_mc.append(round(float(avg_pred), 2))

        # Update sequence with one randomly selected MC path for next step
        pred_scaled_for_seq = scaler.transform([[np.random.choice(day_preds)]])
        last_seq = np.vstack([last_seq[1:], pred_scaled_for_seq])

    return {
        "ticker": ticker,
        "days": int(days),
        "predictions": predictions_mc
    }

@tool
def get_latest_stock_price(ticker: str) -> str:
    """Gets the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period='1d')
    if not todays_data.empty:
        return f"The current stock price for {ticker} is {todays_data['Close'].iloc[0]:.2f} USD."
    return f"Could not retrieve stock price for {ticker}."