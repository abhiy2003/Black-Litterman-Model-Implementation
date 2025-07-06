# Black-Litterman-Model-Implementation

This is an implementation of the Black-Litterman portfolio allocation model and its application to optimizing asset allocation among 11 sector-level ETFs with given historical market (S&P 500) data and sentiment-driven investor views and confidences. The code prints out the daily allocation weights vs. the market weights and their respective means for each ETF, and also the following daily portfolio statistics:
- Portfolio return
- Market return
- Sharpe ratio
- Volatility

The output also includes the following final portfolio statistics for our backtesting period (1/3/23 - 6/28/24): 
- Portfolo's total cumulative returns
- Market's total cumulative returns
- Sharpe ratio
- Volatility
- Alpha
- Beta

These portfolio statistics are saved to an Excel file. The code also plots the following:
- Mean of daily optimal portfolio allocation vs. daily market weight percentage as a side-by-side bar graph
- Monthly average of daily optimal portfolio allocation percentage for each ETF as a stacked bar graph
- Standard deviation of daily allocation weights by ETF as a bar graph
- Comparison of portfolio's and market's respective cumulative returns over the test period as a line graph
