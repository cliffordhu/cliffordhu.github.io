---
title: '2025-02-07 Random Forest Model Prediction Result'
date: 2025-02-07
permalink: /posts/2025/02/random-forest-model-prediction-result_2025-02-07_10/
tags:
  - rf reg result
  - rf clf result
  - updated on 2025-02-07
---
## Ticker Rank List: Uptrend to Downtrend This document presents a rank list of tickers based on their predicted trend, from uptrend to downtrend. The predictions are generated by three different models:
 CSV Result can be download [ Here ](https://cliffordhu.github.io/images/2025-02-07-random-forest-model-prediction-result_2025-02-07_10.csv) 

* **0- LSTM Model (15-Day Lookback):** This model uses Long Short-Term Memory (LSTM) networks to predict the next 15 days' price slope (K-slope). 
* **1- RF Model (15-Day Lookback):** This random forest (RF) classification model predicts whether the future 15-day price slope is positive based on 300 decision trees trained on 260 features. These features encompass: 
     * a. Price action- EMA,Passed Gain, Vol. ATR?...  
     * b. Momentum indicators  RSI, CCI,...  
     * c. Oscillation indicators  BB, MACD, KDJ... 
     * d. Macroeconomic indicators Interest Rate, VIX, Exchange Rate, gloal market, CPI? GDP? Fed decision? Employeement rate change? 
 * **2- RF Model (39-Day Lookback):** Similar to model 1, but with a 39-day lookback window for its predictions. 

 **Model Training Date:** 2025-02-07-10 
 
 **Important Considerations:** 
 
 * **Bias:** The model is biased towards long positions and may not be accurate for short positions. It's recommended for portfolio sizing, ticker screening, risk management, not for day trading.
 * **Unforeseen Events:** Avoid using the model near major events like earnings announcements, Fed decisions, etc., as it may not capture these impacts. It's best suited for stable instruments like indices and ETFs.
 * **Overfitting:** Due to the inherent overfitting problem, this model is best used as a directional guidance tool within your investment strategy. For example, consider a put-shorting scenario:
     * Select candidates only when: 
         * The classification RF model (CLF) votes indicate a high probability of a positive price move.
         * The LSTM and regression RF (REG) K-means are positive. 
         * The past 5 days' trend for all indicators is positive. 
 
 **Disclaimer:** This model is provided for informational purposes only and should not be considered financial advice. Always conduct your own research and due diligence before making investment decisions.



** Result Table **

</details>

|    | Symbol                                                  |   0 LSTM Prediction |   0 5-day Trend of LSTM | 1 Vote for Positve %   | 1 5-day Trend of Vote                        | 1 K mean predicted by reggresion             | 1 5-day Trend of K mean                      | 2 Vote for Positve %   | 2 5-day Trend of Vote                        | 2 K mean predicted by reggresion             | 2 5-day Trend of K mean                      |   3 LDA Gain Loss dB |   Total | Sector                 |   Rank |   Rank Percent |
|---:|:--------------------------------------------------------|--------------------:|------------------------:|:-----------------------|:---------------------------------------------|:---------------------------------------------|:---------------------------------------------|:-----------------------|:---------------------------------------------|:---------------------------------------------|:---------------------------------------------|---------------------:|--------:|:-----------------------|-------:|---------------:|
|  0 | [META](https://finance.yahoo.com/quote/META/financials) |                 nan |                     nan | 59.0%                  | <span style="color: green;"> 0.24806 </span> | <span style="color: green;"> 0.04248 </span> | <span style="color: green;"> 0.03174 </span> | 63.0%                  | <span style="color: green;"> 0.22981 </span> | <span style="color: red;"> -0.02195 </span>  | <span style="color: green;"> 0.00832 </span> |             14.5631  | 16.7286 | Communication Services |      6 |           0.97 |
|  1 | [GOOG](https://finance.yahoo.com/quote/GOOG/financials) |                 nan |                     nan | 68.0%                  | <span style="color: green;"> 0.31686 </span> | <span style="color: green;"> 0.09576 </span> | <span style="color: green;"> 0.0292 </span>  | 66.0%                  | <span style="color: green;"> 0.20267 </span> | <span style="color: green;"> 0.05567 </span> | <span style="color: green;"> 0.02343 </span> |             10.4333  | 13.9096 | Communication Services |     15 |           0.94 |
|  2 | [TSLA](https://finance.yahoo.com/quote/TSLA/financials) |                 nan |                     nan | 59.0%                  | <span style="color: green;"> 0.03077 </span> | <span style="color: green;"> 0.11551 </span> | <span style="color: green;"> 0.02044 </span> | 58.0%                  | <span style="color: red;"> -0.00821 </span>  | <span style="color: green;"> 0.17128 </span> | <span style="color: green;"> 0.01687 </span> |             14.4533  | 16.1725 | Consumer Discretionary |      8 |           0.97 |
|  3 | [TJX](https://finance.yahoo.com/quote/TJX/financials)   |                 nan |                     nan | 63.0%                  | <span style="color: green;"> 0.07871 </span> | <span style="color: green;"> 0.06628 </span> | <span style="color: green;"> 0.00847 </span> | 62.0%                  | <span style="color: green;"> 0.13899 </span> | <span style="color: green;"> 0.06219 </span> | <span style="color: green;"> 0.00821 </span> |             12.5078  | 15.0123 | Consumer Discretionary |     11 |           0.95 |
|  4 | [COST](https://finance.yahoo.com/quote/COST/financials) |                 nan |                     nan | 70.0%                  | <span style="color: green;"> 0.2444 </span>  | <span style="color: green;"> 0.49497 </span> | <span style="color: red;"> -0.04683 </span>  | 72.0%                  | <span style="color: green;"> 0.15181 </span> | <span style="color: green;"> 0.48491 </span> | <span style="color: red;"> -0.02184 </span>  |             13.8655  | 18.0424 | Consumer Staples       |      3 |           0.99 |
|  5 | [VOX](https://finance.yahoo.com/quote/VOX/financials)   |                 nan |                     nan | 64.0%                  | <span style="color: green;"> 0.26729 </span> | <span style="color: red;"> -0.10851 </span>  | <span style="color: red;"> -0.00289 </span>  | 60.0%                  | <span style="color: green;"> 0.11844 </span> | <span style="color: red;"> -0.0933 </span>   | <span style="color: red;"> -0.00208 </span>  |             12.9254  | 15.3121 | ETF                    |     10 |           0.96 |
|  6 | [XLC](https://finance.yahoo.com/quote/XLC/financials)   |                 nan |                     nan | 63.0%                  | <span style="color: green;"> 0.0509 </span>  | <span style="color: red;"> -0.13858 </span>  | <span style="color: green;"> 0.00579 </span> | 64.0%                  | <span style="color: green;"> 0.0977 </span>  | <span style="color: red;"> -0.11431 </span>  | <span style="color: green;"> 0.01496 </span> |              9.68516 | 12.3615 | ETF                    |     21 |           0.91 |
|  7 | [BK](https://finance.yahoo.com/quote/BK/financials)     |                 nan |                     nan | 74.0%                  | <span style="color: green;"> 0.11774 </span> | <span style="color: red;"> -0.06866 </span>  | <span style="color: red;"> -0.01838 </span>  | 70.0%                  | <span style="color: green;"> 0.14387 </span> | <span style="color: red;"> -0.06885 </span>  | <span style="color: red;"> -0.02078 </span>  |             13.2458  | 17.6237 | Financials             |      4 |           0.98 |
|  8 | [TROW](https://finance.yahoo.com/quote/TROW/financials) |                 nan |                     nan | 65.0%                  | <span style="color: green;"> 0.05294 </span> | <span style="color: red;"> -0.01725 </span>  | <span style="color: green;"> 4e-05 </span>   | 62.0%                  | <span style="color: green;"> 0.0224 </span>  | <span style="color: red;"> -0.04537 </span>  | <span style="color: red;"> -0.00852 </span>  |             14.2033  | 16.9265 | Financials             |      5 |           0.98 |
|  9 | [ISRG](https://finance.yahoo.com/quote/ISRG/financials) |                 nan |                     nan | 59.0%                  | <span style="color: green;"> 0.07136 </span> | <span style="color: green;"> 0.12561 </span> | <span style="color: green;"> 0.01297 </span> | 59.0%                  | <span style="color: green;"> 0.09799 </span> | <span style="color: green;"> 0.1248 </span>  | <span style="color: green;"> 0.00351 </span> |             10.1562  | 11.9537 | Health Care            |     22 |           0.91 |
| 10 | [DDOG](https://finance.yahoo.com/quote/DDOG/financials) |                 nan |                     nan | 52.0%                  | <span style="color: green;"> 0.09402 </span> | <span style="color: red;"> -0.16204 </span>  | <span style="color: red;"> -0.00261 </span>  | 52.0%                  | <span style="color: red;"> -0.00795 </span>  | <span style="color: red;"> -0.20476 </span>  | <span style="color: red;"> -0.03746 </span>  |             15.9012  | 16.3414 | Information Technology |      7 |           0.97 |
| 11 | [CDNS](https://finance.yahoo.com/quote/CDNS/financials) |                 nan |                     nan | 75.0%                  | <span style="color: green;"> 0.35596 </span> | <span style="color: green;"> 0.08001 </span> | <span style="color: red;"> -0.00953 </span>  | 76.0%                  | <span style="color: green;"> 0.36677 </span> | <span style="color: green;"> 0.07136 </span> | <span style="color: red;"> -0.00534 </span>  |              9.04135 | 14.0963 | Information Technology |     13 |           0.94 |
 </details>

