import pandas as pd
import xgboost as xgb
import os
import glob # For handling wildcards like *.csv
from src.features import prepare_swing_data 

def run_trading_system(data_folder, golden_list):
    """
    Scans individual CSV files matching the tickers in the Golden List.
    """
    # 1. Get tickers from your Golden List index
    target_tickers = golden_list.index.unique().tolist()
    
    recommendations = []
    print(f"🔍 Searching for {len(target_tickers)} tickers in: {data_folder}")

    for ticker in target_tickers:
        try:
            # 2. Smart Search for the ticker's file
            # This looks for any file that starts with the ticker name in the folder
            search_pattern = os.path.join(data_folder, f"{ticker}*.csv")
            matching_files = glob.glob(search_pattern)

            if not matching_files:
                continue # Skip if file not found

            # Load the first matching file
            file_path = matching_files[0]
            df_ticker = pd.read_csv(file_path)
            
            # Basic cleanup
            df_ticker.columns = [c.lower() for c in df_ticker.columns]
            df_ticker['date'] = pd.to_datetime(df_ticker['date'])
            
            # 3. Feature Engineering
            data = prepare_swing_data(df_ticker, ticker)
            
            # 4. XGBoost Prediction
            features = ['rsi', 'vol_velocity', 'sma_ratio', 'volatility']
            X = data[features]
            
            # Simple model training on available history
            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
            model.fit(X.iloc[:-1], data['swing_target'].iloc[:-1])
            
            prob = model.predict_proba(X.tail(1))[:, 1][0]
            
            # 5. Logical Filter
            if prob > 0.70 and data['rsi'].iloc[-1] < 70:
                recommendations.append({
                    'Ticker': ticker,
                    'AI_Confidence': f"{prob*100:.1f}%",
                    'RSI': round(data['rsi'].iloc[-1], 2),
                    'Status': '🔥 STRONG BUY'
                })
                
        except Exception as e:
            # print(f"⚠️ Skipping {ticker}: {e}") # Debug only
            continue

    return pd.DataFrame(recommendations)

if __name__ == "__main__":
    print("\n" + "="*40)
    print("🚀 TSETMC MULTI-FILE AI SCANNER")
    print("="*40)
    
    # --- CONFIGURATION (Check these paths!) ---
    # Use forward slashes '/' to avoid Windows path errors
    DATA_PATH = "data\raw" 
    GOLDEN_LIST_PATH = "data\cleaned\EDA_golden_list.csv"

    # Verify if Golden List exists
    if os.path.exists(GOLDEN_LIST_PATH):
        g_list = pd.read_csv(GOLDEN_LIST_PATH)
        
        # Set ticker as index
        if 'ticker' in g_list.columns:
            g_list.set_index('ticker', inplace=True)
        
        results = run_trading_system(DATA_PATH, g_list)
        
        if not results.empty:
            print("\n🎯 TOP AI OPPORTUNITIES FOUND:")
            print(results.sort_values(by='AI_Confidence', ascending=False).to_string(index=False))
        else:
            print("\n✅ Scan complete. No high-confidence signals today.")
    else:
        print(f"❌ ERROR: Golden List not found at: {os.path.abspath(GOLDEN_LIST_PATH)}")
        print("💡 Make sure you exported the CSV from your notebook first!")
    
    print("="*40)