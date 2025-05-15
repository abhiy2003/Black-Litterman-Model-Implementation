!pip install PyPortfolioOpt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion
from pypfopt import risk_models, expected_returns
from pypfopt import EfficientFrontier, objective_functions, CLA
from pypfopt.exceptions import OptimizationError

'''# Function to annualize daily returns
def annualize_daily_return(daily_return):
    return (1 + daily_return) ** 252 - 1

# Function to annualize standard deviation from variance
def annualize_std_dev(daily_variance):
    return np.sqrt(daily_variance * 252)

# Define the current 3-month Treasury bill rate (annualized risk-free rate)
risk_free_rate = 0.0509  # 5.09%

# Read the Excel file
file_path = 'etf_summary (1).xlsx'  # Replace with your file path
df = pd.read_excel(file_path, header=1)

# Ensure the column names match your Excel file
ticker_column = 'Ticker'
daily_return_column = 'Average Return (Daily Return)'
variance_column = 'Variance (Daily Return)'
sp500_daily_return_column = 'SP500 Average Return (Daily Return)'
sp500_variance_column = 'SP500 Variance (Daily Return)'
mcap_column = 'Mcap (Billions) Average'  # Market capitalization

# Annualize the average daily returns
df['Annualized Average Return'] = df[daily_return_column].apply(annualize_daily_return)
df['Annualized SP500 Average Return'] = df[sp500_daily_return_column].apply(annualize_daily_return)

# Calculate the annualized excess return
df['Annualized Excess Return'] = df['Annualized Average Return'] - risk_free_rate
df['Annualized SP500 Excess Return'] = df['Annualized SP500 Average Return'] - risk_free_rate
#print(df['Annualized Excess Return'])

# Annualize the standard deviation from the daily variance
df['Annualized Std Dev'] = df[variance_column].apply(annualize_std_dev)
df['Annualized SP500 Std Dev'] = df[sp500_variance_column].apply(annualize_std_dev)

# Calculate the Sharpe ratios
df['Sharpe Ratio (Annual)'] = df['Annualized Excess Return'] / df['Annualized Std Dev']
df['SP500 Sharpe Ratio (Annual)'] = df['Annualized SP500 Excess Return'] / df['Annualized SP500 Std Dev']

# Select relevant columns for the final result
results = df[[ticker_column,
              'Annualized Average Return',
              #'Annualized SP500 Average Return',
              'Annualized Excess Return',
              #'Annualized SP500 Excess Return',
              'Annualized Std Dev',
              #'Annualized SP500 Std Dev',
              'Sharpe Ratio (Annual)',
              'SP500 Sharpe Ratio (Annual)']]

# Display the result
#print(results)'''

'''# Read the historical prices file
prices_file_path = 'BL Model Data 7-24-24 (1).xlsx'  # Replace with your file path
prices_df = pd.read_excel(prices_file_path, header=1)

etf_prices = prices_df.copy()

# Convert 'Date' column to datetime and set as index
etf_prices['date'] = pd.to_datetime(etf_prices['date'])
#etf_prices.set_index('date', inplace=True)

current_date = pd.to_datetime("2024-06-28")

prior_start_date = current_date - pd.DateOffset(days=900)
prior_end_date = current_date - pd.DateOffset(days=1)

#print(prior_start_date, prior_end_date)

# Step 2: Filter the data for the prior period
prior_data = etf_prices[(etf_prices['date'] >= prior_start_date) & (etf_prices['date'] <= prior_end_date)]

# Convert 'Date' column to datetime and set as index
prior_data.set_index('date', inplace=True)

# Pivot the data to get adjusted closing prices
adj_closing_prices = prior_data.pivot(columns='Ticker', values='Adj Closing Price')

# Calculate the covariance matrix of excess returns using pypfopt.risk_models and covariance shrinkage
cov_matrix = risk_models.CovarianceShrinkage(adj_closing_prices).ledoit_wolf()
#print(cov_matrix)

# Filter for XLB rows to get corresponding SP500 prices
xlb_df = prior_data[prior_data['Ticker'] == 'XLB']

# Ensure the SP500 closing prices are under the correct column name
sp500_close_col = 'SP500 Closing Price'

# Filter data to include only the dates in the XLB rows
xlb_dates = xlb_df.index.unique()
filtered_prices_df = prior_data.loc[prior_data.index.isin(xlb_dates)]

# Extract SP500 closing prices corresponding to XLB dates
market_prices = filtered_prices_df[sp500_close_col]
#print(market_prices)

# Calculate the risk aversion parameter
delta = market_implied_risk_aversion(market_prices, frequency=252, risk_free_rate=0.0537)
#print("Risk aversion parameter (delta):", delta)

# Calculate market equilibrium returns (prior) based on market cap weights
market_caps = prior_data.groupby('Ticker')['Mcap (Billions)'].mean()
#print(market_caps)
prior_returns = market_implied_prior_returns(market_caps, delta, cov_matrix, risk_free_rate=0.0537)
#print(prior_returns)

# Specify investor views
viewdict = {
    "XLB": 0.10,
    "XLC": 0.21,
    "XLE": 0.15,
    "XLF": 0.05,
    "XLI": 0.08,
    "XLK": 0.30,
    "XLP": 0.04,
    "XLRE": 0.11,
    "XLU": 0.12,
    "XLV": 0.07,
    "XLY": 0.13
}

# Providing confidence levels
# Closer to 0.0 = Low confidence
# Closer to 1.0 = High confidence
confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

# Initialize the Black-Litterman model
bl = BlackLittermanModel(cov_matrix, # Covariance Matrix
                         pi = prior_returns, # Prior expected returns
                         absolute_views = viewdict, # Views
                         omega = 'idzorek', # Confidence levels in %
                         view_confidences = confidences) # Confidences

# Calculate the posterior returns
posterior_returns = bl.bl_returns()

# Create a DataFrame for the results
bl_results = pd.DataFrame({
    'Prior Return': prior_returns,
    'Posterior Return': posterior_returns,
    'Views': viewdict,
})

# Display the results
print(bl_results)
print()

# Plot the results
#bl_results.plot.bar(figsize=(12,8))

import pprint

# New covariance matrix
bl_cov_matrix = bl.bl_cov()

# Optimizing asset allocation
ef = EfficientFrontier(posterior_returns, bl_cov_matrix)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe(risk_free_rate=0.0537) # Optimizing weights for maximal Sharpe ratio
weights = ef.clean_weights() # Cleaning weights

# Display results
pprint.pprint(weights)
print()

# Display pie chart of results
#pd.Series(weights).plot.pie(figsize=(10,10))

# Expected portfolio performance with optimal weights
ef.portfolio_performance(verbose=True, risk_free_rate=0.0537)
print()'''

# Load the historical risk-free rates from the CSV file
rates_file_path = 'T-Bill Historical Rates.xlsx'  # Replace with your actual file path
rates_df = pd.read_excel(rates_file_path)

# Convert the 'Date' column to datetime, assuming it's in a non-standard format
rates_df['Date'] = pd.to_datetime(rates_df['Date'])

# Set the 'Dates' column as the index for easy lookup
rates_df.set_index('Date', inplace=True)

# Initialize investor views and confidences
'''viewdict = {
    "XLB": 0.10,
    "XLC": 0.21,
    "XLE": 0.15,
    "XLF": 0.05,
    "XLI": 0.08,
    "XLK": 0.30,
    "XLP": 0.04,
    "XLRE": 0.11,
    "XLU": 0.12,
    "XLV": 0.07,
    "XLY": 0.13
}

confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]'''

viewdict = {}

confidences = []

posterior_days = 900

# Load the Excel file
file_path = "BL Model Data 7-24-24 (1).xlsx"
data = pd.read_excel(file_path, header=1)

# Ensure the Date column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Proceed with extracting relevant columns as before
price_data = data[['date', 'Ticker', 'Adj Closing Price', 'Mcap (Billions)', 'Ticker Daily Return']]

# Extract the S&P 500 data
sp500_data = price_data[price_data['Ticker'] == 'SP500']

# Load the views file
views_file_path = 'BL Posterior Estimates-2.xlsx'
views_df = pd.read_excel(views_file_path)

# Check the column names of views_df
#print(views_df)

# Convert the 'Date' column in the views file to datetime
views_df['Date'] = pd.to_datetime(views_df['Date'])

# Set up a loop for each date from Jan 3, 2023, to June 28, 2024
start_date = pd.to_datetime("2023-01-03")
end_date = pd.to_datetime("2024-06-28")
loop_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Initialize the dictionary to store previous day's market weights
previous_weights = {}

# Initialize previous weights with 0 for all tickers
for ticker in data['Ticker'].unique():
    previous_weights[ticker] = 0

# Initialize final DataFrame
final_asset_weights = pd.DataFrame(columns=['Date',
                                            'Ticker',
                                            'Expected Asset Posterior Return',
                                            'Posterior Weight',
                                            'Market Weight',
                                            'Asset Closing Price',
                                            'Expected Posterior Return View',
                                            'Confidence Level',
                                            'Actual Asset Daily Return'])

# Initialize lists to store the results
final_portfolio_stats = []

# Loop through each date in the range
for current_date in loop_dates:
    # Check if current_date is in the DataFrame
    if current_date not in data['date'].values:
        continue  # Skip this date if it's not in the data
    # Step 1: Define the prior return period (900 days before the current date)
    prior_start_date = current_date - pd.DateOffset(days=posterior_days)
    prior_end_date = current_date - pd.DateOffset(days=1)

    #print(prior_start_date, prior_end_date)

    # Step 2: Filter the data for the prior period
    prior_data = data[(data['date'] >= prior_start_date) & (data['date'] <= prior_end_date)]

    # Convert 'Date' column to datetime and set as index
    prior_data.set_index('date', inplace=True)

    # Pivot the data to get adjusted closing prices
    adj_closing_prices = prior_data.pivot(columns='Ticker', values='Adj Closing Price')
    #print(adj_closing_prices)

    # Calculate the covariance matrix of excess returns using covariance shrinkage
    cov_matrix = risk_models.CovarianceShrinkage(adj_closing_prices).ledoit_wolf()

    # Filter for XLB rows to get corresponding SP500 prices
    xlb_df = prior_data[prior_data['Ticker'] == 'XLB']

    # Ensure the SP500 closing prices are under the correct column name
    sp500_close_col = 'SP500 Closing Price'

    # Filter data to include only the dates in the XLB rows
    xlb_dates = xlb_df.index.unique()
    filtered_prices_df = prior_data.loc[prior_data.index.isin(xlb_dates)]

    # Extract SP500 closing prices corresponding to XLB dates
    market_prices = filtered_prices_df[sp500_close_col]
    #print(market_prices)

    # Check if current_date exists in rates_df before accessing
    if current_date in rates_df.index:
        risk_free_rate = (rates_df.loc[current_date, 'Rate']) / 100
    else:
        # Handle the case where current_date is missing
        # For example, you can use the risk-free rate from the previous available date
        risk_free_rate = (rates_df.loc[:current_date, 'Rate'].iloc[-1]) / 100

    #print(risk_free_rate)

    # Calculate the risk aversion parameter
    delta = market_implied_risk_aversion(market_prices, frequency=252, risk_free_rate=risk_free_rate)

    # Calculate market equilibrium returns (prior) based on market cap weights
    #market_caps = prior_data.groupby('Ticker')['Mcap (Billions)'].mean()
    market_caps = data.loc[data['date'] == current_date, ['Ticker', 'Mcap (Billions)']].set_index('Ticker')['Mcap (Billions)']
    #print(market_caps)
    prior_returns = market_implied_prior_returns(market_caps, delta, cov_matrix, risk_free_rate=risk_free_rate)
    #print(prior_returns)

    # Load the views for the current date from the views file
    current_views = views_df[views_df['Date'] == current_date]
    viewdict = current_views.set_index('Sector')['Expected Posterior Return'].to_dict()
    '''if max(viewdict.values()) < 0:
        viewdict = {
        "XLB": 0.09,
        "XLC": 0.09,
        "XLE": 0.09,
        "XLF": 0.09,
        "XLI": 0.09,
        "XLK": 0.09,
        "XLP": 0.09,
        "XLRE": 0.09,
        "XLU": 0.09,
        "XLV": 0.09,
        "XLY": 0.09
    }'''
    '''for ticker, view in viewdict.items():
        if view > 0:
            viewdict[ticker] = 0.09
        elif view < 0:
            viewdict[ticker] = -0.09'''
    confidences = current_views.set_index('Sector')['Confidence Level'].to_list()
    #confidences = [1.0] * len(viewdict)

    #print(current_date, viewdict, confidences)

    # Initialize the Black-Litterman model
    bl = BlackLittermanModel(
        cov_matrix,  # Covariance Matrix
        pi=prior_returns,  # Prior expected returns
        absolute_views=viewdict,  # Views
        omega='idzorek',  # Confidence levels in %
        view_confidences=np.array(confidences)  # Confidences
    )

    # Calculate the posterior returns
    posterior_returns = bl.bl_returns()
    '''if max(posterior_returns) < risk_free_rate:
        posterior_returns = pd.Series({
                            "XLB": 0.09,
                            "XLC": 0.09,
                            "XLE": 0.09,
                            "XLF": 0.09,
                            "XLI": 0.09,
                            "XLK": 0.09,
                            "XLP": 0.09,
                            "XLRE": 0.09,
                            "XLU": 0.09,
                            "XLV": 0.09,
                            "XLY": 0.09
                        })'''
    #posterior_weights = bl.bl_weights()
    #print(posterior_returns, risk_free_rate)

    #viewdict = posterior_returns.to_dict()

    # Calculate daily returns for each ETF for the current date
    #daily_asset_returns = adj_closing_prices.pct_change().iloc[-1].mean()
    #daily_asset_returns = price_data[price_data['date'] == current_date].set_index('Ticker')['Ticker Daily Return'].mean()
    mean_daily_returns = prior_data.groupby('Ticker')['Ticker Daily Return'].mean()
    daily_returns = data.loc[(data['date'] == current_date), ['Ticker', 'Ticker Daily Return']].set_index('Ticker')['Ticker Daily Return']
    #print(current_date, daily_returns)

    # Annualize daily returns for each ETF
    #annualized_returns = daily_returns.apply(lambda x: (1 + x) ** 252 - 1)

    # Calculate market weights for market portfolio return
    market_weights = market_caps / market_caps.sum()

    # Calculate market return based on market weights
    market_return = mean_daily_returns.dot(market_weights) * 252

    min_weights = {}

    # Retrieve the previous day's market weights (for the constraint)
    if current_date != loop_dates[0] and not final_asset_weights.empty:  # Skip for the first date
        for ticker in market_weights.index:
            min_weights[ticker] = 0.2 * previous_weights[ticker]
    else:
        for ticker in market_weights.index:
            min_weights[ticker] = previous_weights[ticker]

    #print(previous_weights, min_weights)

    # New covariance matrix
    bl_cov_matrix = bl.bl_cov()

    # Calculate the Efficient Frontier to get metrics
    ef = EfficientFrontier(posterior_returns, bl_cov_matrix)
    ef.add_objective(objective_functions.L2_reg)
    ef.add_constraint(lambda x: x >= pd.Series(min_weights))  # Apply the min weight constraint
    #ef.max_sharpe(risk_free_rate=risk_free_rate)  # Optimize for Sharpe ratio

    try:
        if max(posterior_returns) > risk_free_rate:
            ef.max_sharpe(risk_free_rate=risk_free_rate)  # Optimize for Sharpe ratio if feasible
        else:
            ef.min_volatility()  # Optimize for minimum volatility if max_sharpe is not feasible
            #print(current_date, "Min Volatility")
    except OptimizationError:
        #print(current_date, "Min Volatility after OptError")
        #print(posterior_returns)
        '''print(f"Optimization failed for date: {current_date}")  # Print the date of failure
        print("Posterior Returns:", posterior_returns)  # Print posterior returns for inspection
        print("Min Weights:", min_weights)  # Print minimum weights for inspection'''
        ef = EfficientFrontier(posterior_returns, bl_cov_matrix)
        ef.add_objective(objective_functions.L2_reg)
        #ef.add_constraint(lambda x: x >= pd.Series(min_weights))  # Apply the min weight constraint
        ef.min_volatility()  # If max_sharpe fails, do min_volatility

    cleaned_weights = ef.clean_weights()

    #print(cleaned_weights)

    # Calculate the portfolio performance
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    previous_weights = market_weights.to_dict()

    # Append results to final DataFrame
    for ticker in cleaned_weights.keys():
        new_row = pd.DataFrame({
            'Date': [current_date],
            'Ticker': [ticker],
            'Expected Asset Posterior Return': [posterior_returns[ticker] if ticker in posterior_returns else None],
            'Posterior Weight': [cleaned_weights[ticker]],
            'Market Weight': [market_weights[ticker]],
            'Asset Closing Price': [data.loc[(data['date'] == current_date) & (data['Ticker'] == ticker), 'Adj Closing Price'].iloc[0]],
            'Expected Posterior Return View': [viewdict[ticker]],
            'Confidence Level': [confidences[list(viewdict.keys()).index(ticker)]],
            'Actual Asset Daily Return': [data.loc[(data['date'] == current_date) & (data['Ticker'] == ticker), 'Ticker Daily Return'].iloc[0]]
        })

        final_asset_weights = pd.concat([final_asset_weights, new_row], ignore_index=True)

    final_portfolio_stats.append({
        'Date': current_date,
        'Expected Annual Portfolio Return': expected_return,
        'Market Return': market_return,
        'Sharpe Ratio': sharpe_ratio,
        'Volatility': volatility
    })

# Convert results to DataFrames
final_asset_weights_df = pd.DataFrame(final_asset_weights)
final_portfolio_stats_df = pd.DataFrame(final_portfolio_stats)
print(final_asset_weights_df)
print(final_portfolio_stats_df)

# Save results to Excel or display
final_asset_weights_df.to_excel("Final_Asset_Weights.xlsx", index=False)
final_portfolio_stats_df.to_excel("Final_Portfolio_Stats.xlsx", index=False)
