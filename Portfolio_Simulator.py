import numpy as np
import math

def simulate_portfolio_path(returns, volatilities, correlation_matrix, time_step_per_year, weights, total_steps):
    n_assets = len(returns)
    dt = 1.0 / time_step_per_year
    portfolio_path = []

    # Cholesky decomposition for correlated random walks
    L = np.linalg.cholesky(correlation_matrix)

    # Initial portfolio value (assuming it starts at 1)
    current_value = 1.0
    portfolio_path.append(current_value)

    for step in range(total_steps):
        # Generate correlated random variables
        Z = np.random.normal(size=n_assets)
        correlated_randoms = L.dot(Z)

        # Update each asset price using GBM formula
        asset_prices_change = (returns - 0.5 * volatilities ** 2) * dt + volatilities * np.sqrt(dt) * correlated_randoms
        asset_prices_change = np.exp(asset_prices_change)  # Convert to multiplicative factor

        # Update portfolio value
        current_value *= np.dot(weights, asset_prices_change)
        portfolio_path.append(current_value)

    return portfolio_path

def max_drawdown(portfolio_path):
    Current_Max = 0
    Current_max_drawdown = 0
    for value in portfolio_path:
        Current_Max = max(Current_Max, value)
        Current_drawdown = (Current_Max - value) / Current_Max
        Current_max_drawdown = max(Current_max_drawdown, Current_drawdown)
    return Current_max_drawdown

def analyze_portfolio(returns, volatilities, correlation_matrix, time_step_per_year, weights, total_steps, number_of_paths = 1000):
    print("We will start analyzing the portfolio base on your input.")
    print("Your portfolio consists of mixture of " + str(len(returns)) + " assets.")
    print("Return of each asset is given by the vector: ")
    print(returns)
    print("Volatility of each asset is given by the vector: ")
    print(volatilities)
    print("Correlations between assets are given by the following matrix: ")
    print(correlation_matrix)
    print("The investment time horizon, according to your input, is " + str(total_steps/time_step_per_year) + " years")
    
    Final_Value_Collection = []
    Max_Draw_Down_Collection = []
    for i in range(number_of_paths):
        if i%1000 == 999:
            print("Simulating " + str(i+1) + " paths now.")
        new_path = simulate_portfolio_path(returns, volatilities, correlation_matrix, time_step_per_year, weights, total_steps)
        Final_Value_Collection.append(new_path[-1])
        Max_Draw_Down_Collection.append(max_drawdown(new_path))
    Final_Value_Collection = sorted(Final_Value_Collection)
    Max_Draw_Down_Collection = sorted(Max_Draw_Down_Collection)
    ave = sum(Final_Value_Collection)/len(Final_Value_Collection)
    print("averaged final value/initial value is " + str(ave) + ", or annualized return of " + str(100*(ave**(time_step_per_year/total_steps)-1)) + " percent." )
    print("95 percentile final value is " + str(Final_Value_Collection[math.floor(0.95*number_of_paths)]))
    print("75 percentile final value is " + str(Final_Value_Collection[math.floor(0.75*number_of_paths)]))
    print("50 percentile final value is " + str(Final_Value_Collection[math.floor(0.50*number_of_paths)]))
    print("25 percentile final value is " + str(Final_Value_Collection[math.floor(0.25*number_of_paths)]))
    print("5 percentile final value is " + str(Final_Value_Collection[math.floor(0.05*number_of_paths)]))
    print("95 percentile max drawdown value is " + str(Max_Draw_Down_Collection[math.floor(0.95*number_of_paths)]))
    print("75 percentile max drawdown value is " + str(Max_Draw_Down_Collection[math.floor(0.75*number_of_paths)]))
    print("50 percentile max drawdown value is " + str(Max_Draw_Down_Collection[math.floor(0.50*number_of_paths)]))
    print("25 percentile max drawdown value is " + str(Max_Draw_Down_Collection[math.floor(0.25*number_of_paths)]))
    print("5 percentile max drawdown value is " + str(Max_Draw_Down_Collection[math.floor(0.05*number_of_paths)]))

# Example usage
annualized_returns = np.array([0.05, 0.07, 0.06])  # Annualized returns of 3 assets
annualized_volatility = np.array([0.1, 0.15, 0.12])  # Annualized volatilities
correlation_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])  # Correlation matrix
time_step_per_year = 500  # Time steps per year
weights = np.array([0.4, 0.4, 0.2])  # Weights of the assets in the portfolio
total_steps = 1000  # Total number of steps in the simulation

#portfolio_path = simulate_portfolio_path(annualized_returns, annualized_volatility, correlation_matrix, time_step_per_year, weights, total_steps)
analyze_portfolio(annualized_returns, annualized_volatility, correlation_matrix, time_step_per_year, weights, total_steps)
#print(portfolio_path[-1])