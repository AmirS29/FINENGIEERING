import math
import numpy as np
import yfinance as yf
import matplotlib as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime as dt
from scipy import optimize

today = dt.datetime.today().strftime('%Y-%m-%d')
#approximately 5 years
start_date = (dt.datetime.today() - dt.timedelta(days=5*365)).strftime('%Y-%m-%d')  

#for Stocks, fixed income, commodities, real estate
#stock symbols for each company
stocks = ['AAPL', 'MSFT', 'AMZN', 'JNJ', 'XOM', 'TLT', 'GLD', 'VNQ']

#download stock data
data = yf.download(stocks, start=start_date, end=today)['Close']

# #display the first and last few rows that disclude holidays and weekends
# print(data.head())
# print(data.tail())

#plot the closing prices of the stocks
data.plot(figsize=(10, 6))
plt.title("Stock Prices of Selected Companies (5 Years)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend(stocks)
plt.grid(True)
plt.show()

# #additional information about each stock
# for ticker in stocks:
#     stock = yf.Ticker(ticker)
#     info = stock.info
#     print(f"Company: {info.get('longName', 'N/A')}")
#     print(f"Sector: {info.get('sector', 'N/A')}")
#     print(f"Market Cap: {info.get('marketCap', 'N/A')}")
#     print(f"PE Ratio: {info.get('trailingPE', 'N/A')}")
#     print('-' * 50)

#to use MVP
    #weights sum to 1 and find the optimal ones according to the minimized risk
    #Covariance matrix
    #expected return
    #risk or volatility (standard deviation)
    #investor fixes expected return and minimizes the risk
    #use log utility function for efficient frontier
    #optimization will be aided by scipy

#computes the percentage change of each stock from the previous day
#dropna() removes anything missing
returns = data.pct_change().dropna()
#calculates the average daily return of each stock
mean_returns = returns.mean()
#computes the covariance in a matrix between each stock
cov_matrix = returns.cov()

#number of assets
num_assets = len(stocks)

#assigns initial equal weights
weights = np.array([1/num_assets] * num_assets)

#weights need to be between 0 and 1 and sum up to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#this means each asset is between 0 and 1
bounds = tuple((0, 1) for _ in range(num_assets))

def get_ret_vol_sr(weights, risk_free_rate):
    weights = np.array(weights)
    ret = mean_returns.dot(weights)
    vol = np.sqrt(weights.T.dot(cov_matrix.dot(weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])

#negate Sharpe ratio as we need to max it but Scipy minimize the given function
def neg_sr(weights, risk_free_rate):
    return get_ret_vol_sr(weights, risk_free_rate)[-1] * -1

#check sum of weights 
def check_sum(weights):
    return np.sum(weights) - 1

#initializes each weight to be equal to one another while summing up to 1
init_guess = [1 / num_assets] * num_assets

#call minimizer and initialize the risk_free rate
risk_free_rate = 0.03 
opt_results = optimize.minimize(neg_sr, init_guess, args=(risk_free_rate,), constraints=constraints, bounds=bounds, method='SLSQP')
optimal_weights = opt_results.x
#optimal_weights for each stock
for st, i in zip(stocks,optimal_weights):
    print(f'Stock {st} has weight {np.round(i*100,2)} %')
    
# Generate random portfolios
num_portfolios = 50000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    #generate random weights
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  #normalize to sum to 1
    
    #calculate portfolio return, volatility, and Sharpe ratio
    portfolio_return, portfolio_volatility, portfolio_sr = get_ret_vol_sr(weights, risk_free_rate)
    
    #store results
    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = portfolio_sr

#plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier with Random Portfolios')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.grid(True)

#plot maximum sharpe ratio portfolio and plot it
max_sr_index = np.argmax(results[2])
max_sr_return = results[0,max_sr_index]
max_sr_volatility = results[1,max_sr_index]
plt.scatter(max_sr_volatility, max_sr_return, marker='*', color='r', s=200, label='Max Sharpe Ratio Portfolio')
plt.legend(loc='upper left')
plt.show()

#sr = mew - rf/risk


