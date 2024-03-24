---
title: '2024-03-24 ETF Scrapper'
date: 2024-03-21
permalink: /posts/2024/03/ETF-Scrapper/
tags:
  - ETFDB scrapper
  - How to select the pool of tickers
  - updated on 2024-03-24
---
# ETFpy is a Python library that allows users to scrape data from etfdb.com, a website that provides comprehensive information on ETFs, including trading data, performance metrics, assets allocations end more.

## Installation
Install with pip as a package pip
```python
pip install pyetfdb-scraper
```
## usage

```python
import pyetfdb_scraper
from pyetfdb_scraper import etf
from pyetfdb_scraper.etf import ETF, load_etfs
etfs=load_etfs()
result=[]
import time
```
## Parse the top 10 holders of each ETF and histogram the ticker held by ETFS. The top 150 tickers are selected on the watch list. 
There are a lot of N/As because they are bond tickers, not available on the stock market. Also, some tickers are from Europe, not tradable. 
Finally, if the API is called too fast for too long. the  system will give a timeout. like this 

- Exception raised. Retrying for 2 time. Error code is HTTPSConnectionPool(host='etfdb.com', port=443): Max retries exceeded with url: /etf/SRLN (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x000002AB9A5DBFD0>, 'Connection to etfdb.com timed out. (connect timeout=None)'))

- Reduced retries. Sleeping for 15 mins. The current symbol is SRLN
In this case we have to wait for 15 mins to rest the link. 

All the tickers are saved to Stocklist and to csv file. 

```Python
i=0
etfs1=etfs[i:]
while (1):
    try: 
        for etf in etfs1:
         
            tk=ETF(etf)
            info=tk.to_dict()
            i=i+1
            for item in info['holdings']['top_holdings']:
               print(item['symbol'])
               result.append(item['symbol']) 
    except:
        i=i+1
        if i<=len(etfs):
            etfs1=etfs[i:]
            time.sleep(60)
        else:
            break

StockList=pd.DataFrame(result)
StockList.to_csv('StockList.csv')


```

## post data analysis- Histogram. 
After all tickers are saved. it is plotted as a histogram plot. and the ticker is ranked by the appearance count and saved to TickerList.csv


```Python
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

word_list = result

counts = Counter(word_list)

labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

bar_width = 0.35

plt.bar(indexes, values)

lists=pd.DataFrame({'Symbol':labels,'Counts':values})
lists.to_csv('TickerList.csv')

# add labels
plt.xticks(indexes + bar_width, labels)
plt.show()


```


** Result Table **
