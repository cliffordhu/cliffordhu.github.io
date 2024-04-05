---
title: '2024-03-31 LSTM, Random Forest Model Prediction Result'
date: 2024-03-31
permalink: /posts/2024/03/random-forest-model-prediction-result-0331/
tags:
  - rf reg result
  - rf clf result
  - updated on 2024-03-31
---
** This is the rank list for the Tickers from uptrend to downtrend Vote by 300 decision tree trained on 260 features run on 2024-03-31.      CLF classifier vote is counted only if the tree predicts 15 future days price slope is positive, Slope mean value from 300 decision tree is caculated by Regression model.      The trend of past 5 days are also given. It would be best candiate for long if the volte for postive % is high, K mean is high, 5 days trend is going up.  ** 



** Result Table **

</details>

|    | Symbol                                                | 0 LSTM Prediction                            | 0 5-day Trend of LSTM                        | 1 Vote for Positve %   | 1 5-day Trend of Vote                        | 1 K mean predicted by reggresion            | 1 5-day Trend of K mean                     | 2 Vote for Positve %   | 2 5-day Trend of Vote                       | 2 K mean predicted by reggresion            | 2 5-day Trend of K mean                     |     Total |   Rank |   Rank Percent |
|---:|:------------------------------------------------------|:---------------------------------------------|:---------------------------------------------|:-----------------------|:---------------------------------------------|:--------------------------------------------|:--------------------------------------------|:-----------------------|:--------------------------------------------|:--------------------------------------------|:--------------------------------------------|----------:|-------:|---------------:|
|  0 | [SPY](https://finance.yahoo.com/quote/SPY/financials) | <span style="color: green;"> 0.10661 </span> | <span style="color: green;"> 0.00698 </span> | 76.0%                  | <span style="color: green;"> 0.00919 </span> | <span style="color: red;"> -0.11482 </span> | <span style="color: red;"> -0.07303 </span> | 75.0%                  | <span style="color: red;"> -0.1348 </span>  | <span style="color: red;"> -0.13869 </span> | <span style="color: red;"> -0.07896 </span> |  5.17338  |      1 |            0.5 |
|  1 | [IWM](https://finance.yahoo.com/quote/IWM/financials) | <span style="color: green;"> 0.07402 </span> | <span style="color: red;"> -0.03114 </span>  | 50.0%                  | <span style="color: red;"> -0.21237 </span>  | <span style="color: red;"> -0.27366 </span> | <span style="color: red;"> -0.05314 </span> | 44.0%                  | <span style="color: red;"> -0.39365 </span> | <span style="color: red;"> -0.27717 </span> | <span style="color: red;"> -0.05615 </span> | -0.499718 |      2 |            0   |
 </details>

