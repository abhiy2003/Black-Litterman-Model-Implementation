!pip install PyPortfolioOpt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion
from pypfopt import risk_models, expected_returns
from pypfopt import EfficientFrontier, objective_functions, CLA
from pypfopt.exceptions import OptimizationError

# Load the historical risk-free rates from the CSV file
rates_file_path = 'T-Bill Historical Rates.xlsx'  # Replace with your actual file path
rates_df = pd.read_excel(rates_file_path)

# Convert the 'Date' column to datetime, assuming it's in a non-standard format
rates_df['Date'] = pd.to_datetime(rates_df['Date'])

# Set the 'Dates' column as the index for easy lookup
rates_df.set_index('Date', inplace=True)

# Initialize investor views and confidences (test data)
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
    market_caps = data.loc[data['date'] == current_date, ['Ticker', 'Mcap (Billions)']].set_index('Ticker')['Mcap (Billions)']
    #print(market_caps)
    prior_returns = market_implied_prior_returns(market_caps, delta, cov_matrix, risk_free_rate=risk_free_rate)
    #print(prior_returns)

    # Load the views for the current date from the views file
    current_views = views_df[views_df['Date'] == current_date]
    viewdict = current_views.set_index('Sector')['Expected Posterior Return'].to_dict()
    confidences = current_views.set_index('Sector')['Confidence Level'].to_list()

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

    # Calculate daily returns for each ETF for the current date
    mean_daily_returns = prior_data.groupby('Ticker')['Ticker Daily Return'].mean()
    daily_returns = data.loc[(data['date'] == current_date), ['Ticker', 'Ticker Daily Return']].set_index('Ticker')['Ticker Daily Return']
    #print(current_date, daily_returns)

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

    try:
        if max(posterior_returns) > risk_free_rate:
            ef.max_sharpe(risk_free_rate=risk_free_rate)  # Optimize for Sharpe ratio if feasible
        else:
            ef.min_volatility()  # Optimize for minimum volatility if max_sharpe is not feasible
            #print(current_date, "Min Volatility")
    except OptimizationError:
        #print(current_date, "Min Volatility after OptError")
        #print(posterior_returns)
        ef = EfficientFrontier(posterior_returns, bl_cov_matrix)
        ef.add_objective(objective_functions.L2_reg)
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
