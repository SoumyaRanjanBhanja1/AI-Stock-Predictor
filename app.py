import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def build_predictor(ticker_symbol):
    # 1. Fetch Live Data
    data = yf.Ticker(ticker_symbol)
    df = data.history(period="max")

    # 2. Data Cleaning
    del df["Dividends"]
    del df["Stock Splits"]

    # 3. Setting Target (Kal price badhega ya nahi?)
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    # 4. Feature Engineering (Rolling Averages)
    horizons = [2, 5, 60, 250]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors += [ratio_column, trend_column]

    df = df.dropna()

    # 5. MAANG-Level Model (Random Forest)
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    train = df.iloc[:-100]
    test = df.iloc[-100:]

    model.fit(train[new_predictors], train["Target"])
    
    # Prediction
    preds = model.predict(test[new_predictors])
    score = precision_score(test["Target"], preds)
    
    print(f"Model Precision Score: {score * 100:.2f}%")
    return model


