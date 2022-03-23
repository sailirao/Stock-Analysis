import datetime
import pandas as pd
import pandas_datareader as data
from datetime import date
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np 
import seaborn as sns
import warnings

st.header('Time Series Analysis of Amubuja Cements Ltd. Stock Data During Covid-19 (March 2020 - March 2022)')

st.markdown('The stocks we selected are from different sectors and market cap. You’ll see how it is advantageous as we analyse the stocks further. For the start, we’ll investigate the Ambuja Cements Ltd. stock individually and then move on to the combined analysis. For this project, we have extracted csv file of historical data of all the above stocks from Yahoo finance.')

st.header('1. Understanding Data & General Statistics')


st.markdown('Now we plot the closing price (adjusted) of the stock over the period of 2 year to get a general idea of how the stock performed in the given period')

user_input='AMBUJACEM.NS'
start = '2020-03-01'
end = '2022-03-22'
df = data.DataReader(user_input, 'yahoo', start, end)
if st.checkbox("Show raw data", False):
    st.write(df.head())

Ambuja_df = df.round(2)
# Ambuja_df.head(2)

Ambuja_df.shape

Ambuja_df.isnull().sum()

Ambuja_df.dropna(inplace = True, axis = 0)

# Ambuja_df.dtypes

Ambuja_df.index = pd.to_datetime(Ambuja_df.index)
# Ambuja_df.head(2)

Ambuja_df.index.max() - Ambuja_df.index.min()

Ambuja_df.iloc[-90:].describe().astype(int)

Ambuja_df.index = Ambuja_df.index


st.header('2. General variation in the stock price')

st.markdown('We’ll set the ‘Date’ column as the index of the dataframe. It makes plotting easy')

st.markdown('Now we plot the closing price (adjusted) of the stock over the period of 2 year to get a general idea of how the stock performed in the given period')
# Ambuja_df["Adj Close"].plot(figsize = (8,5))
fig1 = plt.figure(figsize=(12,6))
plt.xlabel("Amubuja Cements Ltd. Stock- Adj Close")
plt.ylabel("Closing Prices")
plt.plot(Ambuja_df['Adj Close'])
plt.grid()
# plt.show()
st.pyplot(fig1)

st.markdown('In the above plot, if you notice, there is a drastic increase in the price of stock sometime around the month of July 2020')

st.header('3. Day-to-day percentage change(Daily returns)')

st.markdown('Daily percentage change in the price of the stock is calculated on the basis of percentage change between 2 consecutive days’ closing prices. Let’s say if the closing price of the stock yesterday was ₹500 and today the stock closed as ₹550. So, the percentage change is 10%. i.e. ((550–500) / 500)*100. No mystery here!')

st.markdown('Accordingly, we introduce a new column ‘Day_Perc_Change’ denoting the daily returns in the price of the stock. This can be done using in-built pct_change() function in python.')

Ambuja_df["Day_Perc_Change"] = Ambuja_df["Adj Close"].pct_change()*100
# Ambuja_df.head()


Ambuja_df.dropna(axis = 0, inplace = True)



fig2 = plt.figure(figsize = (12, 6))
plt.xlabel("Amubuja Cements Ltd. Stock-Daily Percent Change")
plt.ylabel("Daily Returns")
plt.plot(Ambuja_df["Day_Perc_Change"])
plt.grid()
st.pyplot(fig2)

st.markdown('It can be observed that for most of the days, the returns are between -2% to 2 % with few spikes in between crossing 10% mark on positive side and -9.0 on negative side.')

st.markdown('Plotting daily returns distribution histogram —')


# fig3 = Ambuja_df["Day_Perc_Change"].hist(bins = 50, figsize = (10,5)) 

fig3, ax = plt.subplots()
# ax.xlabel("Daily returns")
# ax.ylabel("Frequency")
ax.hist(Ambuja_df["Day_Perc_Change"], bins=30)
st.pyplot(fig3)
#satistics
Ambuja_df.Day_Perc_Change.describe()


st.markdown('The daily returns histogram is centered about origin. For the past 1 year, the mean daily returns has been about 0.185 and for most of the days the daily return was less than 2.5% implying that the Ambuja Cements Ltd stock has been less volatile over the period. During the period, the highest % change in positive direction was observed to be 9.84 % and was 9.41% in negative direction. Clearly, we didn’t had any instances of ‘bull run’ or ‘bear drop’!')

st.header('4. Trend Analysis')

st.markdown('Next we add a new column ‘Trend’ whose values are based on the day-to-day percentage change we calculated above. Trend is determined from below relationship')

from PIL import Image
image = Image.open('Trend.png')

st.image(image, caption='Trend')

st.markdown('We wish to see how the stock was trending in past 1 year. This can be visualized as a pie chart, with each sector representing the percentage of days each trend occurred. We’ll plot a pie chart for the ‘Trend’ column to visualize the relative frequency of each trend category.')

def trend(x):
    if x > -0.5 and x <= 0.5:
        return "Slight or No change"
    elif x > 0.5 and x <= 1:
        return "Slight Positive"
    elif x > -1 and x <= -0.5:
        return "Slight Negative"
    elif x > 1 and x <= 3:
        return "Positive"
    elif x > -3 and x <= -1:
        return "Negative"
    elif x > 3 and x <= 7:
        return "Among top gainers"
    elif x > -7 and x <= -3:
        return "Among top losers"
    elif x > 7:
        return "Bull run"
    elif x <= -7:
        return "Bear drop"
Ambuja_df["Trend"]= np.zeros(Ambuja_df["Day_Perc_Change"].count())
Ambuja_df["Trend"]= Ambuja_df["Day_Perc_Change"].apply(lambda x:trend(x))
# Ambuja_df.head()




Rel_pie_data = Ambuja_df.groupby("Trend")
pie_label = sorted([i for i in Ambuja_df.loc[:, "Trend"].unique()])

fig4, ax4 = plt.subplots()
ax4.pie(Rel_pie_data["Trend"].count(), labels = pie_label, 
        autopct = '%1.1f%%', radius = 3)

st.pyplot(fig4)

st.markdown('For the period under consideration from March 2020 -March 2021, the Ambuja Cements Ltd stock was among the top gainers for about 12.1 % of the time, and among the top losers for 7.3 %. For about 21.0 % of the time period, the stock has performed positively on a given day. Likewise, for most period of time (about 19.4 %) the stock showed a very slight change in the price. These observations are consistent with the daily return histogram we saw above.')


st.header('5. Daily Returns and Volume')


fig5, ax5 = plt.subplots()
ax5.stem(Ambuja_df.index, Ambuja_df["Day_Perc_Change"])
# ax5.ylabel("Daily Returns and Volume")
(Ambuja_df["Volume"]/1000000).plot(figsize = (15, 11), 
                                 color = "green", 
                                 alpha = 0.5)
ax4.grid()
st.pyplot(fig5)

st.markdown('By juxtaposing the daily trade volume(in green) with the daily returns(in blue), it was observed that whenever the volume of shares traded is high, there is comparatively high rise or fall in the price of the stock leading to the high returns. Thus, on a given day if unconventionally high volume of trading takes place, then one can expect a big change in the market in the either direction. Volume of shares traded when coupled with the rise or fall in Price of stock, in general, is an indicator of the confidence of the traders & investors in a particular company.')


st.header('6. Correlation Analysis Of Stocks with Pair plot and Joint plots')

st.markdown('Whenever we go for the diversification of the portfolio, we would NOT want the stocks to be related to each other. Mathematically, Pearson’s correlation coefficient (also called Pearson’s R value) between any pair of stocks should be close to 0. The idea behind is simple — suppose your portfolio comprises of the stocks that are highly correlated, then if one stock tumbles, the others might fall too and you’re at the risk of losing all your investment!')

st.markdown('We selected the aforementioned stocks to perform the correlation analysis. All these stocks are from different segments of Industry and Market cap. You are free to choose the stocks of your interest. the procedure remains the same.')

st.markdown('We’ll analyse the correlation between the different stocks in a pair-wise fashion with Seaborn pairplot.')



@st.cache(persist=True)
def allStock():
    Rel_df=df = data.DataReader('RELIANCE.NS', 'yahoo', start, end)
    Eicher_df=data.DataReader('EICHERMOT.NS', 'yahoo', start, end)
    Hindustan_df=data.DataReader('HINDUNILVR.NS', 'yahoo', start, end)
    TCS_df=data.DataReader('TCS.NS', 'yahoo', start, end)
    Ambuja_df=data.DataReader('CIPLA.NS', 'yahoo', start, end)
    
    return Rel_df,Eicher_df,Hindustan_df,TCS_df,Ambuja_df


# import package
import pandas_datareader.data as web
# set start and end dates 
start1 = datetime.datetime(2020, 3, 1)
end1 = datetime.datetime(2022, 3, 22)
# extract the closing price data
allStock()
combined_df = web.DataReader(["RELIANCE.NS", "CIPLA.NS","EICHERMOT.NS","HINDUNILVR.NS", "TCS.NS","AMBUJACEM.NS", "^NSEI"],
"yahoo", start = start1, end = end1)["Adj Close"]



combined_df.dropna(inplace = True, axis = 0)
# display first few rows
# combined_df.head()



# store daily returns of all above stocks in a new dataframe 
pct_chg_df = combined_df.pct_change()*100
pct_chg_df.dropna(inplace = True, how = "any", axis = 0)
# plotting pairplot  
import seaborn as sns
sns.set(style = "ticks", font_scale = 1.25)
fig6 = sns.pairplot(pct_chg_df)
st.pyplot(fig6)

st.markdown('Note that the correlation analysis is performed on the daily percentage change(daily returns) of the stock price and not on the stock price.')

st.markdown('If you observe carefully, the plots in the lower triangular area are the same as the plots in the upper triangular area with just axes interchanged. So, analyzing either set of plots would suffice. The diagonal represents the histograms, just like the one seen above for Ambuja Cements Ltd. stock.')

st.markdown('Takeaway--')

st.markdown('Reliance,Cipla, Eicher Motors, Hindustan Unilever and TCS stocks can be included in a portfolio as no two stocks show any significant correlation.')

st.markdown('Drawback--')

st.markdown('Although the pair plots provide very good visualization of all possible combinations between the bunch of stocks, it doesn’t provide any detailed information like Pearson’s R value or null-hypothesis p value to quantify the correlation. That’s where the joint plot comes into the picture!')

st.markdown('While Pair plot provides a visual insight into all possible correlations, Seaborn jointplot provides detailed information like Pearson’s R value (Pearson’s correlation coefficient) for each pair of stocks. Pearson’s R value ranges from -1 to 1. Negative value indicates a negative linear relation between the variables, while positive value indicates a positive relationship. Pearson’s R value closer to 1 (or -1) indicates strong correlation, while value closer to 0 indicates weak correlation.')

st.markdown('In addition to Pearson’s R value, joint plot also shows the respective histograms on the edges as well as null hypothesis p-value.')



import scipy.stats
from scipy.stats import stats

col1, col2, col3 = st.columns(3)
st.set_option('deprecation.showPyplotGlobalUse', False)
with col1:
    sns.jointplot(data=pct_chg_df, x="AMBUJACEM.NS", y= "RELIANCE.NS",kind="reg")
    a=stats.pearsonr(pct_chg_df["AMBUJACEM.NS"], pct_chg_df["RELIANCE.NS"])
    print(a)
    figA = plt.show()
    st.header("A")
    st.pyplot(figA)
    
    
with col2:
    sns.jointplot(data=pct_chg_df, x="AMBUJACEM.NS", y= "EICHERMOT.NS",kind="reg")
    b=stats.pearsonr(pct_chg_df["AMBUJACEM.NS"], pct_chg_df["EICHERMOT.NS"])
    print(b)
    figB = plt.show()
    st.header("B")
    st.pyplot(figB)
    
with col3:
    sns.jointplot(data=pct_chg_df, x="AMBUJACEM.NS", y= "HINDUNILVR.NS",kind="reg")
    c=stats.pearsonr(pct_chg_df["AMBUJACEM.NS"], pct_chg_df["HINDUNILVR.NS"])
    print(c)
    figC = plt.show()
    st.header("C")
    st.pyplot(figC)
    

col1, col2 = st.columns(2)

with col1:
    sns.jointplot(data=pct_chg_df, x="AMBUJACEM.NS", y= "TCS.NS",kind="reg")
    d=stats.pearsonr(pct_chg_df["AMBUJACEM.NS"], pct_chg_df["TCS.NS"])
    print(d)
    figD = plt.show()
    st.header("D")
    st.pyplot(figD)
    
with col2:
    sns.jointplot(data=pct_chg_df, x="AMBUJACEM.NS", y= "CIPLA.NS",kind="reg")
    e=stats.pearsonr(pct_chg_df["AMBUJACEM.NS"], pct_chg_df["CIPLA.NS"])
    print(e)
    figE = plt.show()
    st.header("E")
    st.pyplot(figE)

st.markdown('Having correlation is not the only parameter to determine which stocks to include in an portfolio and which to remove. There are several other factors at play. It’s best to seek advice of the experts and make an informed decision.')

st.header('8. Volatility analysis')

st.markdown('Volatility is one of the most important pillars in financial markets. A stock is said to have high volatility if its value can change dramatically within a short span of time. On other hand, lower volatility means that value of stock tends to be relatively steady over a period of time. These movements are due to several factors including demand and supply, sentiment, corporate actions, greed, and fear, etc. Mathematically, volatility is measured using a statistical measure called ‘standard deviation’, which measures an asset’s departure from its average value.')

st.markdown('We have already calculated the intraday returns (daily returns) of the Ambuja Cements Ltd. stock and several other stocks. Next, we will calculate the 7-day rolling mean(also called moving average) of the daily returns, then compute the standard deviation (which is square root of the variance) and plot the values. We will be using Pandas ‘rolling()’ function and ‘std()’ function .')

Rel_vol = pct_chg_df["AMBUJACEM.NS"].rolling(7).std()*np.sqrt(7)

fig7 = plt.figure(figsize = (12, 6))
Rel_vol.plot(figsize = (15, 7))
plt.ylabel(" Rolling Mean and Std. Deviation")
st.pyplot(fig7)


st.markdown('Next we’ll see the comparative volatility analysis of Ambuja Cements Ltd. stock with Reliance, Cipla,Eicher Motors, Hindustan Unilever, Tata Consultancy Services stock and NIFTY50 index. Just like above, we compute 7-day rolling mean, and standard deviation.')

fig8 = plt.figure(figsize = (15,7))
volatility_with_reliance = pct_chg_df[['AMBUJACEM.NS', 'RELIANCE.NS', '^NSEI']].rolling(7).std()*np.sqrt(7)
plt.plot(volatility_with_reliance)
plt.xlabel("Amubuja Cements Ltd.")
plt.ylabel("Reliance")
st.pyplot(fig8)


fig9 = plt.figure(figsize = (15,7))
volatility_with_eicher = pct_chg_df[['AMBUJACEM.NS', 'EICHERMOT.NS', '^NSEI']].rolling(7).std()*np.sqrt(7)
plt.plot(volatility_with_eicher)
plt.xlabel("Amubuja Cements Ltd.")
plt.ylabel("Eicher Motors")
st.pyplot(fig9)


fig10 = plt.figure(figsize = (15,7))
volatility_with_hul = pct_chg_df[['AMBUJACEM.NS', 'HINDUNILVR.NS', '^NSEI']].rolling(7).std()*np.sqrt(7)
plt.plot(volatility_with_hul)
plt.xlabel("Amubuja Cements Ltd.")
plt.ylabel("Hindustan Unilever Ltd.")
st.pyplot(fig10)

fig11 = plt.figure(figsize = (15,7))
volatility_with_tcs = pct_chg_df[['AMBUJACEM.NS', 'TCS.NS', '^NSEI']].rolling(7).std()*np.sqrt(7)
plt.plot(volatility_with_tcs)
plt.xlabel("Amubuja Cements Ltd.")
plt.ylabel("Tata Consultancy Services")
st.pyplot(fig11)

fig12 = plt.figure(figsize = (15,7))
volatility_with_ambuja = pct_chg_df[['AMBUJACEM.NS', 'CIPLA.NS', '^NSEI']].rolling(7).std()*np.sqrt(7)
plt.plot(volatility_with_ambuja)
plt.xlabel("Amubuja Cements Ltd.")
plt.ylabel("Cipla")
st.pyplot(fig12)

st.markdown('You can observe that Amubuja Cements Ltd. stock has higher volatility as compared to Eicher Motors, Hindustan Unilever Ltd, Tata Consultancy Services and Cipla except for Reliance where it has low volatility while the Nifty index has least volatility.')

st.markdown('Many traders and investors seek out higher volatility investments in order to make higher profits. If a stock does not move, not only it has low volatility, but also it has low gain potential. On the other hand, a stock or other security with a very high volatility level can have tremendous profit potential, but the risk is equally high.')

st.header('Conclusion')

st.markdown('To sum up, as stock market is an important sector, comparison between the time series models can be helpful in determining whether to buy a stock or sell it and this crucial purpose can be served with the help of this study of time series analysis of stock market prediction. We have tried our best to analyze all these stocks so if such pandemic occurs in the future one can get an idea on which type of sector to select to invest their funds in the stock market.')