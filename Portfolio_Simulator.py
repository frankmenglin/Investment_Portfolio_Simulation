import numpy as np
from TimeSeries import *

def analyze_portfolio(T, timestep, mu, sigma, correlation, weights, M=10000, risk_free_rate=0.00):
    if M < 100:
        raise ValueError("Number of Monte Carlo trials must be at least 100.")
    if not (np.allclose(correlation, correlation.T) and np.all(np.linalg.eigvals(correlation) >= 0)):
        raise ValueError("Correlation matrix must be symmetric and positive semi-definite.")
    weights = np.array(weights)
    weights /= weights.sum()  # Normalize weights

    # Container for metrics
    annual_returns = []
    max_drawdowns = []
    sharpe_ratios = []

    # Simulate M portfolios
    for i in range(M):
        if(i%100==1 and i>1):
            print( str(i-1) + " paths are simulated." + "\n")
        portfolio_values = np.sum([w * np.array(gbm.values) for w, gbm in zip(weights, generate_gbm_set(T, timestep, mu, sigma, correlation, S0=np.ones(len(mu))))], axis=0)
        portfolio_series = TimeSeries(timestep, portfolio_values)

        # Calculate metrics
        annual_return = portfolio_series.get_annualized_return()
        max_drawdown = portfolio_series.get_Max_Drawdown()
        sharpe_ratio = portfolio_series.get_Sharpe_Ratio(risk_free_rate)

        annual_returns.append(annual_return)
        max_drawdowns.append(max_drawdown)
        sharpe_ratios.append(sharpe_ratio)

    # Function to calculate and print percentiles
    def print_percentiles(data, label):
        percentiles = np.percentile(data, [99, 95, 75, 50, 25, 5, 1])
        print(f"{label} Percentiles:")
        print("99%: {:.4f}, 95%: {:.4f}, 75%: {:.4f}, 50%: {:.4f}, 25%: {:.4f}, 5%: {:.4f}, 1%: {:.4f}".format(*percentiles))

    # Displaying results
    print_percentiles(annual_returns, "Annualized Return")
    print_percentiles(max_drawdowns, "Max Drawdown")
    print_percentiles(sharpe_ratios, "Sharpe Ratio")

# Example usage
analyze_portfolio(T=2520, timestep=1/252, mu=np.array([0.05, 0.08]), sigma=np.array([0.1, 0.2]), correlation=np.array([[1, 0.5], [0.5, 1]]), weights=[0.6, 0.4])
