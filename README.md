# Data-Driven Stock Market Scanner (TSETMC)

An end-to-end quantitative trading system for the Tehran Stock Exchange. This project leverages **Exploratory Data Analysis (EDA)** and **Gradient Boosted Decision Trees (XGBoost)** to identify high-probability 5-day swing trading opportunities.

## 📊 Dataset
The historical data used in this project is sourced from the following Kaggle dataset:
**[Tehran Stock Exchange Dataset by Mehrad Aria](https://www.kaggle.com/datasets/mehradaria/tehran-stock-exchange)**

It includes daily OHLCV (Open, High, Low, Close, Volume) data for hundreds of tickers in the TSETMC market.

## 🛠 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- **Efficiency Scoring:** Calculated Risk-Adjusted Returns to filter the "Golden List" of top-performing tickers.
- **Drawdown Analysis:** Filtered out stocks with historical extreme volatility or long recovery periods (Max Drawdown).
- **Liquidity Check:** Implemented volume filters to ensure assets are tradable and avoid "liquidity traps."

### 2. Feature Engineering
- **Technical Indicators:** Integration of RSI, Moving Averages, and Volatility measures.
- **Volume Dynamics:** Captured "Smart Money" entry signs through Volume Velocity and Relative Volume.
- **Seasonality:** Incorporated calendar-based features specific to the Iranian market fiscal cycles.

### 3. Machine Learning Modeling
- **Algorithm:** XGBoost Classifier (Champion Model) vs. SVM.
- **Target:** Predicting a >3% price increase within a 5-day window (Swing Trading).
- **Optimization:** Hyperparameter tuning to maximize Precision (minimizing false buy signals).

### 4. Realistic Backtesting
- **Commission Management:** All backtests include a **1.25% transaction fee**, ensuring realistic net profit calculations.
- **Confidence Thresholding:** Implemented a >70% probability filter to reduce over-trading and overcome fee erosion.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure your data follows the structure of the Kaggle dataset.
4. Run the scanner: `python main.py`.
