#!/usr/bin/env python
# coding: utf-8

# ## Web Scrapping

# In[54]:


import json
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd


# In[55]:


url = "https://www.coingecko.com/"
r = requests.get(url)
soup = bs(r.content, 'html.parser')

results = soup.find_all('a')


# In[56]:


#create empty list
tags_raw=[]

#traverse the result and store the hyperlink data into tags list
for mtag in results:
    tags_raw.append(str(mtag))


# In[57]:


baseurl='https://www.coingecko.com'

historytag=[]

for htag in tags_raw:
    if 'd-lg-none font-bold tw-w-12' in htag and 'img class' not in htag:
        mytag=htag.replace('<a class="d-lg-none font-bold tw-w-12" href="', '')
         
        instanceofquotes = mytag.find('"')
        #cleanurl = lenofstring - instanceofquotes 
        mytag_cleaned = (mytag[0:instanceofquotes])
        historytag.append(baseurl+mytag_cleaned+'/historical_data?end_date=2022-05-17&start_date=2016-05-01&page=')
        #historytag.append(baseurl+mytag)

#print(historytag)

res = []
i=0
for each in historytag:
    i+=1
    print('Currency_no:',i)
    for r in range(1,38):
        
        r1 = requests.get(each.lower()+str(r))
        soup1 = bs(r1.content, 'html.parser')
        tabledata = soup1.find('table', {'class' : 'table table-striped text-sm text-lg-normal'})
        tablebody = tabledata.find('tbody')
        tablerow = tablebody.find_all('tr')
        currecny = each.split("/")[5]
        #print(currecny)
        for each1 in tablerow:

            tableth = each1.find('th')
            #print(tableth.text.strip())
            tabletd = each1.find_all('td')
            #print(tabletd[0].text.strip(),tabletd[1].text.strip(),tabletd[2].text.strip(),tabletd[3].text.strip())
            response = dict()
            response['Date'] = tableth.text.strip()
            response['Currency'] = currecny
            response['MarketCap'] = tabletd[0].text.strip()
            response['Volume'] = tabletd[1].text.strip()
            response['Open'] = tabletd[2].text.strip()
            response['Close'] = tabletd[3].text.strip()
            res.append(response)


# In[ ]:


len(res)


# In[ ]:


df = pd.DataFrame(res)
df


# ## Data Cleaning

# In[16]:


import pandas as pd
import json
import string as st
import numpy as np
import seaborn as sns

sns.set()

# Make charts a bit bolder
#sns.set_context("talk")

#%matplotlib inline
#%config Completer.use_jedi = False

# Default figure size
sns.set(rc={"figure.figsize": (12,10)})
sns.set_style('whitegrid')


# In[17]:


df = df.replace('N/A',np.nan).replace(np.nan,0)     #Removing all NaNs
df


# In[18]:


#Removing Special characters from MarketCap, Volume, Open and Close columns and converting to float datatype. 

df['MarketCap'] = df['MarketCap'].str.replace('$','').str.replace(',','')
df['MarketCap'] = df['MarketCap'].astype('float')

df['Close'] = df['Close'].str.replace('$','').str.replace(',','')
df['Close'] = df['Close'].astype('float')

df['Volume'] = df['Volume'].str.replace('$','').str.replace(',','')
df['Volume'] = df['Volume'].astype('float')

df['Open'] = df['Open'].str.replace('$','').str.replace(',','')
df['Open'] = df['Open'].astype('float')


# In[19]:


#Removing rows with missing values of Close price

df = df.replace(np.nan,0)
df = df[df.Close != 0]

df.shape


# In[27]:


df.info()


# ## Adding New Columns

# In[20]:


#Sorting the Dates grouped by Currency in ascending order.

df2 = df
df2=df2.groupby(['Currency']).apply(lambda x: x.sort_values(['Date'], ascending=True).drop('Currency', axis=1))
df2.reset_index(level=0, inplace=True)
df = df2


# In[21]:


#ADDING COLUMNS WITH ROLLING AVERAGE FOR BUY SELL ANALYSIS:

df['7D_SMA'] = df.groupby('Currency').rolling(7)['Close'].mean().droplevel(level=0)
df['20D_SMA'] = df.groupby('Currency').rolling(20)['Close'].mean().droplevel(level=0)
df['50D_SMA'] = df.groupby('Currency').rolling(50)['Close'].mean().droplevel(level=0)
df['100D_SMA'] = df.groupby('Currency').rolling(100)['Close'].mean().droplevel(level=0)
df['200D_SMA'] = df.groupby('Currency').rolling(200)['Close'].mean().droplevel(level=0)


# In[22]:


#Adding columns with percentage change in Close price in a day , a week and a month.

df['24h_%Change'] = df.groupby('Currency', sort=False)['Close'].apply(lambda x: x.pct_change(1)).to_numpy()
df['7D_%Change'] = df.groupby('Currency', sort=False)['Close'].apply(lambda x: x.pct_change(7)).to_numpy() 
df['30D_%Change'] = df.groupby('Currency', sort=False)['Close'].apply(lambda x: x.pct_change(30)).to_numpy()


# In[23]:


df = df.fillna(0)
df[df['Currency'] == 'bitcoin'].tail(10)


# In[24]:


# Converting close prices to various currencies

import requests
class RealTimeCurrencyConverter():
    def __init__(self,url):
        self.data= requests.get(url).json()
        self.currencies = self.data['rates']

    def convert(self, from_currency, to_currency, amount): 
        initial_amount = amount 
        #first convert it into USD if it is not in USD.
        # because our base currency is USD
        if from_currency != 'USD' : 
            amount = amount / self.currencies[from_currency] 

        # limiting the precision to 4 decimal places 
        amount = round(amount * self.currencies[to_currency], 4) 
        return amount

url = 'https://api.exchangerate-api.com/v4/latest/USD'
converter = RealTimeCurrencyConverter(url)
df['Close_INR']=(converter.convert('USD','INR',df.Close))
df['Close_EUR']=(converter.convert('USD','EUR',df.Close))
df['Close_JPY']=(converter.convert('USD','JPY',df.Close))
df['Close_GBP']=(converter.convert('USD','GBP',df.Close))
df['Close_CAD']=(converter.convert('USD','CAD',df.Close))
df['Close_DKK']=(converter.convert('USD','DKK',df.Close))


# In[28]:


#Converting date to datetime 
df['Date'] = pd.to_datetime(df['Date'])


# In[29]:


#Saving the cleansed data into JSON format

df.to_json('crypto.json', orient = 'records')


# In[59]:


df.info()


# ## Importing the cleansed structured data

# In[58]:


#Importing the cleaned dataset to do analysis

df = pd.read_json('crypto.json')
df.tail(10)


# ## EDA on structured dataset

# In[32]:


df['Currency'] = df['Currency'].str.upper()
df['Market_Billion'] = df['MarketCap'] / 1000000000
df['Volume_Million'] = df['MarketCap'] / 1000000
df['Volume_Billion'] = df['Volume'] / 1000000000


# In[33]:


wide_format = df.groupby(['Date', 'Currency'])['Close'].last().unstack()
wide_format.head(3)


# ## Top 10 crypto-currencies by MarketCap in USD

# In[34]:


import matplotlib.pyplot as plt

col = ['blue','orange','green','red','purple']

ax = df.groupby(['Currency'])['Market_Billion'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh', color=col);
ax.set_xlabel("Market cap (in billion USD)");
plt.title("Top 10 Currencies by Market Cap");


# ## Top 10 crypto-currencies by Volume(in Billion USD)

# In[35]:



bx = df.groupby(['Currency'])['Volume_Billion'].last().sort_values(ascending=False).head(10).sort_values()
bx = bx.plot(kind='barh' , color=col);
bx.set_xlabel("Volume (in billion USD)");
plt.title("Top 10 Currencies by Volume");


# ## Top 10 Crypto-currencies by Volume (in Million USD)

# In[36]:


cx = df.groupby(['Currency'])['Volume_Million'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh', color=col);
cx.set_xlabel("Volume (in million USD)");
plt.title("Top 10 Currencies by Volume");


# # Defining Top-5 & Top-10 currencies for analysis

# In[37]:


#Defining the top 10 currencies
top_10_currency_names = df.groupby(['Currency'])['MarketCap'].last().sort_values(ascending=False).head(10).index
data_top_10_currencies = df[df['Currency'].isin(top_10_currency_names)]
#data_top_10_currencies.head(5)

#Defining the top 5 currencies
top_5_currency_names = df.groupby(['Currency'])['MarketCap'].last().sort_values(ascending=False).head(5).index
data_top_5_currencies = df[df['Currency'].isin(top_5_currency_names)]
#data_top_5_currencies.head(5)


# ## Close Price Trend for Top 10 Currencies in INR

# In[38]:


dx = data_top_10_currencies.groupby(['Date', 'Currency'])['Close_INR'].mean().unstack().plot()
dx.set_ylabel("Price per 1 unit (in INR)")
plt.title("Price per unit of currency")


# ## Trend Charts from 2017 to 2022 for Top 10 currencies

# In[39]:


fx = data_top_10_currencies.groupby(['Date', 'Currency'])['Market_Billion'].mean().unstack().plot()
fx.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency");


# In[40]:


gx = data_top_10_currencies.groupby(['Date', 'Currency'])['Volume_Billion'].mean().unstack().plot()
gx.set_ylabel("Transaction Volume (in billion)");
plt.title("Transaction Volume per Currency");


# In[41]:


gx = data_top_10_currencies.groupby(['Date', 'Currency'])['Volume_Million'].mean().unstack().plot()
gx.set_ylabel("Transaction Volume (in million)");
plt.title("Transaction Volume per Currency");


# In[42]:


#Close_price trends for Top 5 crypto-currencies

ix = data_top_5_currencies[data_top_5_currencies.Date.dt.year >= 2020].groupby(['Date', 'Currency'])['Close_INR'].mean().unstack().plot();
ix.set_ylabel("Price per 1 unit (in INR)");
plt.title("Price per unit of currency (in 2022)");


# In[43]:


#MarketCap trends for Top 5 crypto-currencies

ix = data_top_5_currencies[data_top_5_currencies.Date.dt.year >= 2020].groupby(['Date', 'Currency'])['Market_Billion'].mean().unstack().plot();
ix.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency (in 2022)");


# In[44]:


#Transaction Volume trends for Top 5 crypto-currencies

ix = data_top_5_currencies[data_top_5_currencies.Date.dt.year >= 2020].groupby(['Date', 'Currency'])['Volume_Billion'].mean().unstack().plot();
ix.set_ylabel("Transaction Volume (in billion)");
plt.title("Transaction Volume per Currency (from 2020)");


# ## Correlation of Top 5 Currencies based on Marketcap

# In[45]:


plt.figure(figsize=(14,8))
sns.heatmap(wide_format[top_5_currency_names].corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True);


# In[46]:


is_bitcoin = df['Currency'] == 'BITCOIN'
is_ethereum = df['Currency'] == 'ETHEREUM'
is_ripple  = df['Currency'] == 'TETHER'


# In[47]:


# Pull out a part of dataset that only has the most interesting currencies
data_top_currencies = df[is_bitcoin | is_ethereum | is_ripple]


# In[48]:


ax = data_top_currencies.groupby(['Date', 'Currency'])['Close'].mean().unstack().plot()
ax.set_ylabel("Price per 1 unit (in USD)")


# In[49]:


ax = data_top_currencies[data_top_currencies.Date.dt.year >= 2018].groupby(['Date', 'Currency'])['Volume_Billion'].mean().unstack().plot()
ax.set_ylabel("Trading volume (in billion)");


# ## Buy or Sell Trends Using moving Averages for Bitcoin, Ethereum and Tether

# In[50]:


df_btc = df[df['Currency'] == 'BITCOIN']

df_btc = df_btc[df_btc['Date'] > '2022-01-01']

df_btc = df_btc.replace(0, np.nan)
df_btc = df_btc.dropna()
df_btc = df_btc.set_index('Date')
df_btc = df_btc[['Close','20D_SMA','50D_SMA']]


Buy = []
Sell = []

for i in range(len(df_btc)):
    
    if (df_btc['20D_SMA'].iloc[i] > df_btc['50D_SMA'].iloc[i]) and (df_btc['20D_SMA'].iloc[i-1] < df_btc['50D_SMA'].iloc[i-1]):
        Buy.append(i)
    elif (df_btc['20D_SMA'].iloc[i] < df_btc['50D_SMA'].iloc[i]) and (df_btc['20D_SMA'].iloc[i-1] > df_btc['50D_SMA'].iloc[i-1]):
        Sell.append(i)
        
        
plt.plot(df_btc['Close'], label = 'BTC Close Price', c = 'green', alpha =0.5)
plt.plot(df_btc['20D_SMA'], label = 'BTC_20D_Moving_Average', c = 'blue', alpha =0.9)
plt.plot(df_btc['50D_SMA'], label = 'BTC_50D_Moving_Average', c = 'maroon', alpha =0.9)
plt.scatter(df_btc.iloc[Buy].index, df_btc.iloc[Buy]['Close'], marker = '^',color = 'green', s=500)
plt.scatter(df_btc.iloc[Sell].index, df_btc.iloc[Sell]['Close'], marker = 'v',color = 'red', s=500)
plt.legend()
plt.show()


# In[51]:


#Filtering out ethereum records

df_eth = df[df['Currency'] == 'ETHEREUM']

df_eth = df_eth[df_eth['Date'] > '2022-01-01']

df_eth = df_eth.replace(0, np.nan)
df_eth = df_eth.dropna()
df_eth = df_eth.set_index('Date')
     
df_eth = df_eth[['Close','20D_SMA','50D_SMA']]

#Selecting the buying and selling poinnts
Buy = []
Sell = []

for i in range(len(df_eth)):
    
    if (df_eth['20D_SMA'].iloc[i] > df_eth['50D_SMA'].iloc[i]) and (df_eth['20D_SMA'].iloc[i-1] < df_eth['50D_SMA'].iloc[i-1]):
        Buy.append(i)
    elif (df_eth['20D_SMA'].iloc[i] < df_eth['50D_SMA'].iloc[i]) and (df_eth['20D_SMA'].iloc[i-1] > df_eth['50D_SMA'].iloc[i-1]):
        Sell.append(i)

        
#Plotting the moving averages

plt.plot(df_eth['Close'], label = 'ETH Close Price', c = 'black', alpha =0.5)
plt.plot(df_eth['20D_SMA'], label = 'ETH_20D_Moving_Average', c = 'blue', alpha =0.9)
plt.plot(df_eth['50D_SMA'], label = 'ETH_50D_Moving_Average', c = 'maroon', alpha =0.9)
plt.scatter(df_eth.iloc[Buy].index, df_eth.iloc[Buy]['Close'], marker = '^',color = 'green', s=500)
plt.scatter(df_eth.iloc[Sell].index, df_eth.iloc[Sell]['Close'], marker = 'v',color = 'red', s=500)
plt.legend()
plt.show() 


# In[52]:


#Filtering out Tether records

df_bnb = df[df['Currency'] == 'TETHER']
df_bnb = df_bnb[df_bnb['Date'] > '2022-01-01']

df_bnb = df_bnb.replace(0, np.nan)
df_bnb = df_bnb.dropna()
df_bnb = df_bnb.set_index('Date')
df_bnb = df_bnb[['Close','20D_SMA','50D_SMA']]

#Selecting the buying and selling poinnts
Buy = []
Sell = []

for i in range(len(df_bnb)):
    
    if (df_bnb['20D_SMA'].iloc[i] > df_bnb['50D_SMA'].iloc[i]) and (df_bnb['20D_SMA'].iloc[i-1] < df_bnb['50D_SMA'].iloc[i-1]):
        Buy.append(i)
    elif (df_bnb['20D_SMA'].iloc[i] < df_bnb['50D_SMA'].iloc[i]) and (df_bnb['20D_SMA'].iloc[i-1] > df_bnb['50D_SMA'].iloc[i-1]):
        Sell.append(i)

        
#Plotting the moving averages

plt.plot(df_bnb['Close'], label = 'TETHER Close Price', c = 'green', alpha =0.5)
plt.plot(df_bnb['20D_SMA'], label = 'TETHER_20D_Moving_Average', c = 'blue', alpha =0.9)
plt.plot(df_bnb['50D_SMA'], label = 'TETHER_50D_Moving_Average', c = 'maroon', alpha =0.9)
plt.scatter(df_bnb.iloc[Buy].index, df_bnb.iloc[Buy]['Close'], marker = '^',color = 'green', s=500)
plt.scatter(df_bnb.iloc[Sell].index, df_bnb.iloc[Sell]['Close'], marker = 'v',color = 'red', s=500)
plt.legend()
plt.show()      


# ## Predicting Bitcoin and Ethereum Close Prices based on 2022-05-01

# In[53]:


# Defining CAGR function which will help in calculating the CAGR

def CAGR(start, end, time):
    Growth_Rate = (end/start)**(1/time)-1
    return Growth_Rate

#Saving bitcoin data into one dataframe
df_bitcoin = df[df['Currency'] == 'BITCOIN']


#Saving ethereum data into one dataframe
df_ethereum = df[df['Currency'] == 'ETHEREUM']



#Extracting Close price data for Start date

start_btc = df_bitcoin['Close'][df_bitcoin['Date'] == '2020-04-30'].iloc[0]
start_eth = df_ethereum['Close'][df_ethereum['Date'] == '2020-04-30'].iloc[0]
print('Price of Ethereum on 2020-04-30=',start_eth,'$')
print('Price of Bitcoin on 2020-04-30=',start_btc,'$')

print('--------------------------------------------------------------------------')

#Extracting Close price data for End date
end_btc = df_bitcoin['Close'][df_bitcoin['Date'] == '2022-05-01'].iloc[0]
end_eth = df_ethereum['Close'][df_ethereum['Date'] == '2022-05-01'].iloc[0]
print('Price of Ethereum on 2022-05-01=',end_eth,'$')
print('Price of Bitcoin on 2022-05-01=',end_btc,'$')

print('--------------------------------------------------------------------------')
#Calculating CAGR for ethereum in the last 2 years
eth_CAGR = CAGR(start_eth, end_eth, 2)
btc_CAGR = CAGR(start_btc, end_btc, 2)

print('Etherium grows with a CAGR of',round(eth_CAGR,2),'%')
print('Bitcoin grows with a CAGR of',round(btc_CAGR,2),'%')

# Defining forcast function 
def forecast(end, CAGRr, years):
    forc = end + (1+CAGRr)**years
    return forc
print('--------------------------------------------------------------------------')
# forcasting ethereum price in next 6 years based on CAGR calculated
years = 6
print('Forecast for ethereum in 6 years time', round(forecast(end_eth, eth_CAGR, years)))
print('Forecast for bitcoin in 6 years time', round(forecast(end_btc, btc_CAGR, years)))


# In[ ]:





# In[ ]:




