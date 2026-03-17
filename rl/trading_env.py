import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf


class TradingEnv(gym.Env):

    def __init__(self, prices, lstm_model, scaler, seq_len=10, initial_balance=10000,
                 use_lstm_forecast=True, forecast_days=5):

        super().__init__()

        self.prices = np.asarray(prices, dtype=float).flatten()

        self.lstm_model = lstm_model
        self.scaler = scaler
        self.seq_len = seq_len
        self.initial_balance = float(initial_balance)

        self.use_lstm_forecast = use_lstm_forecast
        self.forecast_days = forecast_days

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(seq_len, 1),
            dtype=np.float32
        )

        @tf.function
        def predict_step(seq_tensor):
            return self.lstm_model(seq_tensor, training=False)

        self.predict_step = predict_step


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.balance = float(self.initial_balance)
        self.shares = 0
        self.current_step = self.seq_len
        self.prev_value = self.initial_balance

        if self.use_lstm_forecast:
            self.forecast_prices = self._generate_lstm_forecast(self.forecast_days)
        else:
            self.forecast_prices = self.prices

        return self._get_obs(), {}


    def _get_obs(self):

        prices_slice = np.array(
            self.forecast_prices[self.current_step-self.seq_len:self.current_step],
            dtype=float
        ).reshape(-1,1)

        scaled = self.scaler.transform(prices_slice)

        return scaled.astype(np.float32)


    def _generate_lstm_forecast(self, days):

        last_seq = self.scaler.transform(
            self.prices[-self.seq_len:].reshape(-1,1)
        )

        predictions = []

        returns = np.diff(self.prices.flatten()) / self.prices.flatten()[:-1]

        volatility = np.std(returns)

        for _ in range(days):

            seq_tensor = tf.convert_to_tensor(
                last_seq.reshape(1,self.seq_len,1),
                dtype=tf.float32
            )

            pred_scaled = self.predict_step(seq_tensor).numpy()

            noise = np.random.normal(0, volatility, size=pred_scaled.shape)

            pred_scaled_noisy = pred_scaled + noise

            pred_price = self.scaler.inverse_transform(pred_scaled_noisy)[0][0]

            predictions.append(pred_price)

            pred_scaled_for_seq = self.scaler.transform([[pred_price]])

            last_seq = np.vstack([last_seq[1:], pred_scaled_for_seq])

        return np.concatenate([self.prices, np.array(predictions)])


    def step(self, action):

        price = float(self.forecast_prices[self.current_step])

        fee_pct = 0.001

        if action == 1:

            budget = float(self.balance * 0.5)

            shares_to_buy = int(budget // price)

            cost = shares_to_buy * price

            fee = cost * fee_pct

            if shares_to_buy > 0:

                self.shares += shares_to_buy

                self.balance -= (cost + fee)

        elif action == 2 and self.shares > 0:

            shares_to_sell = int(self.shares * 0.5)

            revenue = shares_to_sell * price

            fee = revenue * fee_pct

            if shares_to_sell > 0:

                self.shares -= shares_to_sell

                self.balance += (revenue - fee)

        self.current_step += 1

        done = self.current_step >= len(self.forecast_prices)-1

        total_value = float(self.balance + self.shares * price)

        reward = (total_value - self.prev_value)/self.prev_value

        self.prev_value = total_value

        if self.balance <=0 and self.shares ==0:

            done=True

            reward=-1.0

        info = {

            "step": int(self.current_step),
            "price": price,
            "balance": float(self.balance),
            "shares": int(self.shares),
            "portfolio_value": total_value
        }

        return self._get_obs(), reward, done, False, info