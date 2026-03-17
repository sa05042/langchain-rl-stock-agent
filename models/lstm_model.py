import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def build_and_train_model(data, seq_len=10, epochs=50, batch_size=16):

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    X, y = [], []

    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler, seq_len