---
title: 'Time Series Prediction using Random Forest model'
date: 2024-03-16
permalink: /posts/2024/03/RandomForest_Model/
tags:
  - Random Forest Stock Prediction
  - Machine Learning
  - Time Series Prediction
---

**Time Series Prediction using Random Forest model**

# Model Selection

</details>
Time series prediction models come in many flavors, each with strengths and weaknesses. Most commonly used models are

- **ARIMA (Autoregressive Integrated Moving Average):** This classic model excels at capturing trends and seasonality in data. It's relatively simple to understand and implement, but it struggles with non-linear patterns and external factors. Also it has a single variable input.
- **LSTM (Long Short-Term Memory)/Transformer Model:** A powerful deep learning model, LSTMs can handle complex non-linear relationships and long-term dependencies within the data. However, they require a significant amount of data for training and can be computationally expensive. It also has gradient vanishing and exploding challenge in model training.
- **[Random Forest:](https://williamkoehrsen.medium.com/random-forest-simple-explanation-377895a60d2d)** This is a powerful machine learning algorithm that works by combining multiple decision trees. Each tree makes a prediction, and the final output is the average (for regression) or most frequent vote (for classification) from all the trees. These tree-like structures use a series of yes/no questions to classify data or predict a continuous value. This ensemble method is best known for its versatility and robustness. It creates multiple decision trees, each with some randomness injected during training. This helps prevent overfitting and improves the overall accuracy. It can also capture various patterns but might not be the most accurate for pure time series forecasting, especially for long-term predictions. It has several advantages in stock market prediction:
  - **MultiVariant inputs.** Stock price can be influenced by many factors, this model can handle multiple features easily.
  - **Capture short term pattern.** Stock price action may have certain pattern in short term. Random forest model randomly look at a piece of the panel data to effectively capture the possible short-term pattern.
  - **Data Efficiency.** Unlike LSTM model, the required data set can be as small as 1000+points. The model can be small enough to run live trading analysis.
  - **Result Presentation:** It can provide the confidence level (by counting the votes) which other model cannot give.
  - **Result format:** It can give two types of results. For regression format, the individual trees predict a continuous value, and the final output is the average of those predictions. For classification format, each tree predicts a class, and the final output is the most voted-for class by the trees. In stock analysis, the class can be defined as the different range of gains.
  
Inspired by [Dr. Earnest Chan's](https://epchan.blogspot.com/) talk on youtube, I start the journey of using Random Forest Model to simulate stock market gain action.  

# Training Data Preparation

## Data Type in statistics

There are three types of data in the traditional statistics study. Sequential, cross sectional and panel data. key differences are shown below:

- **Cross-sectional data:** This type of data captures information from multiple subjects at a single point in time. Imagine a snapshot of a stock price. It's useful for understanding current price’s relationships to other observations but has no memory of the trend or history. Every observation is independent to each other. Random Tree Model use Cross sectional data as basis to create decision trees. But the stock price is sequential data in nature. Therefore it cannot be used directly.
- **Sequential data:** This data captures information from the same subjects over multiple points in time. It's like a stock price, showing how things change. Sequential data allows researchers to see how variables evolve and potentially influence each other over time. ARIMA model and LSTM model are best used for sequential data modeling. However, they have the limitations mentioned above.
- **Panel data:** This data type is created by combining the sequential data with cross sectional data by adding the related historical observation from sequential data to cross sectional data as features. It finds the best use in RF model since RF model use cross sectional data format as input to process sequential data.

## Conversion from sequential data to Panel data

### **DataX Preparation:**  

When dealing with time series data like stock prices, which often exhibit trends and seasonality, converting them to stationary data can be crucial for training a Random Forest (RF) model to extract underlying patterns. If the data isn't stationary, those patterns and trends are buried in the random walk noise. The signal to noise ratio is too small to make the model training failed. In statistics the definition of stationary means the mean (average), variance (spread), and autocorrelation (correlation between observations at different time lags) should remain consistent throughout the entire data set. There are several ways to remove trend, bias, seasonality,

 **1. Differentiate the price to get linear gain, take log operation to get log scale gain so that the distribution is normal.**
```python
gain=np.log(1+self.df1\['X'\].diff()/self.df1\['X'\])\*100
```
 **2. Add Lag features to identify those seasonality.**
```python
case "weekdays":
               tmp=pd.DataFrame(index=self.df1.index)
               tmp['weekdays']=pd.to_datetime(tmp.index).weekday 
               self.df1=pd.concat([self.df1,tmp],axis=1)
 case "weeks":
                tmp=pd.DataFrame(index=self.df1.index)
                tmp['weeks']=pd.to_datetime(tmp.index).isocalendar()['week']
                self.df1=pd.concat([self.df1,tmp],axis=1)

```
 **3. Volatility features: Include measures like standard deviation of past returns to capture risk.**
 
 **4. Technical indicators: Calculate technical indicators like moving averages, Relative Strength Index (RSI), or Bollinger Bands to incorporate technical analysis insights.**
```python
# add more indicators
         #momentum indicator
         for i in np.arange(5,35,5):
             self.add_x("ema",i)
             self.add_x("rsi",i)
             self.add_x("cci",i) 
             self.add_x("roc",i)
             self.add_x("rocp",i)
             self.add_x("atr",i)
         self.add_x("will",14)
         self.add_x("will",28)
         self.add_x("macd",9)
         self.add_x("kdj",9)
         self.add_x("bb",14)
         self.add_x("bb",28)
         self.add_x("obv",5)
         self.add_x("vol",5)
         self.add_x("HLrange",5)
```
 **5. Detrend by normalizing the data**
```python
self.df1 = self.df1.rename(columns={'Adj Close': 'X'})

tmp=self.df1.X.mean()

self.df1\['scale_factor'\]=tmp/self.df1.X
```
 **Other Things to consider:**

- ***Feature selection:*** Choose historical observations that are relevant and informative for your target variable. Too many features can lead to underfitting or overfitting. Current features include (beyond the above indicators):
```pyhton
         ndate=5
         # #Total market
         self.add_x("QQQ",ndate)
         self.add_x("SPY",ndate)
         self.add_x("IWM",ndate)
         self.add_x("TLT",ndate)

         # market index
         self.add_x("NYSE",ndate)
         self.add_x("SP500",ndate)
         self.add_x("RUT",ndate)
         # Commodity
         #self.add_x("CrudeOil",ndate)
         self.add_x("GLD",ndate)
         #self.add_x("BTC",ndate)
         #World Market
         self.add_x("EU",ndate)
         self.add_x("JPN",ndate)
         self.add_x("CHN",ndate)
         
         # volatility 
         self.add_x("VIX",ndate)

         # Money Supply
         #self.add_x("WeeKT",16) 
         self.add_x("Year5T",ndate)
         self.add_x("Year10T",ndate)
         self.add_x("Year30T",ndate)

         # exchange rate
         self.add_x("EURUSD",ndate) 
         self.add_x("GBPUSD",ndate)
         self.add_x("JPY",ndate)
                  
         # week days, weeks and encoded  ticker symbol 
         self.add_x('tk_encode',1)
         self.add_x('weekdays',1)
         self.add_x('weeks',1)
```

- **Window size (for rolling statistics):** The appropriate window size for rolling statistics depends on the nature of your data and the time-scale of interest. Currently, short term behavior is selected to 5, long term behavior is selected upto 21 to 35.
- **Data availability:** Ensure you have enough historical data to calculate meaningful features. The stock price daily historical data has about 3500 observations. Just good enough for random forest model. In case more data is needed, suggest to
  - - check Markov Chain model to learn from the historical data and generate simulated training set.
      - Use 15m or 1m data to increase the observation.
      - Use All in one Model, namely train all ticker using one model instead of one ticker for one model.

### **DataY Preparation:**   

1. For the target Y, instead of predicting the gain of the next day, I try to predict the next nday’s regression slop k, bias b and variation v instead.
2. There are two RF models. To prepare the training goal Y for regression model:
```python
for i in np.arange(len(self.df1['X'])-ndate):
                 y=self.df1['X'][i:i+ndate+1].to_numpy()
                 stds.append(np.std(y))
                 # Calculate the slope and offset using linear regression
                 k, b = np.linalg.lstsq(np.vstack([x,np.ones(len(x))]).T, y, rcond=None)[0]
                 ks.append(k)
                 bs.append(b)
```
To prepare the training goal Y for the classifier model:  We assign [-4:4] to represent the gain from -3 standard deviation to +3 standard deviation. 
```python
             rng=np.mean(abs(dataY))
             for i in dataY.to_numpy():
                if np.isnan(i):
                    metaclass.append(np.nan)
                if  i>=3*rng: 
                    metaclass.append(4)
                if  i<3*rng and i>=2*rng: 
                    metaclass.append(3)
                if  i<2*rng and i>=rng: 
                    metaclass.append(2)
                if  i<rng and i>=0: 
                    metaclass.append(1)
                if  i<0 and i>=-rng: 
                    metaclass.append(-1)
                if  i<-rng and i>=-2*rng:
                    metaclass.append(-2)
                if  i<-2*rng and i>=-3*rng:
                    metaclass.append(-3)
                if i<-3*rng: 
                    metaclass.append(-4)
             self.df1[f'Y_metaclass_{ndate}']=metaclass

```

After several tries, I found the n_estimator from 1000 to 3000 make no significant improvement. To avoid over fitting the max_feature is selected to default “sqrt”, which means to train each decision tree only sqrt of 230 features is selected. Minimum leaf is set to 50 for now. Need to fine tune to find a better result. Oob_score
```python
self.classifier=RandomForestClassifier(n_estimators=self.N, random_state=0,max_features="sqrt", min_samples_leaf=self.M,oob_score=True,n_jobs=10,warm_start=False)

self.classifier.fit(dataX.to_numpy(),dataY\[f'Y_metaclass_{SelectedSetN}'\].to_numpy())
```
# **Model Training and HyperParameter Tuning:**   
to be continued....
</details>
