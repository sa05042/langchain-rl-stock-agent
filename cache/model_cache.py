# ---------------------------
# Cache for trained models
# ---------------------------

trained_models = {}  # {ticker: (model, scaler, seq_len)}
trained_rl_agents = {}  # cache {ticker: RL agent}
global_stock_data = {}