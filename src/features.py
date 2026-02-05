# 1. Prepare Swing Trading Data (5-day window)
target_window = 5
profit_threshold = 0.03 # Looking for at least 3% growth in 5 days

def prepare_pro_ml_data(df, ticker_name):
    data = df.xs(ticker_name, level='ticker').copy()
    
    # --- Technical Indicators ---
    # 1. RSI (Simple version)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. Trend (Price vs 20-day Average)
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['Trend_Signal'] = data['close'] / data['MA20']
    
    # 3. Momentum & Volatility (From before)
    data['lag_1'] = data['return'].shift(1)
    data['volatility_5d'] = data['return'].rolling(window=5).std().shift(1)
    data['vol_velocity'] = (data['volume'] / data['volume'].rolling(window=10).mean()).shift(1)

    # 4. time effect
    data['shamsi_month'] = data.index.map(lambda x: jdatetime.date.fromgregorian(date=x.date()).month)
    data['is_dividend_season'] = data['shamsi_month'].isin([2, 3, 4]).astype(int) # فصل مجامع
    
    # 5. Target
    data['target'] = (data['return'].shift(-1) > 0).astype(int)
    
    return data.dropna()

def prepare_swing_data(df, ticker):
    data = prepare_pro_ml_data(df, ticker) # Use your existing feature engine
    
    # New Target: Is price 5 days from now > 3% higher than today?
    data['future_close'] = data['close'].shift(-target_window)
    data['swing_target'] = (data['future_close'] > data['close'] * (1 + profit_threshold)).astype(int)
    
    # Drop rows where we don't have future data
    return data.dropna()
