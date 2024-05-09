from typing import List
import numpy as np

def generate_gbm(T, timestep, mu, sigma, S0=1):
    """
    Generates a geometric Brownian motion time series.

    :param T: Total number of time periods (equivalent to number of steps - 1)
    :param timestep: The size of each time step in years
    :param mu: Annualized return (drift coefficient)
    :param sigma: Annualized volatility
    :param S0: Initial value of the time series (default 1)
    :return: TimeSeries object with generated GBM values
    """
    # Total number of steps
    N = T + 1
    times = np.linspace(0, T * timestep, N)
    steps = np.random.normal(loc=(mu - 0.5 * sigma**2) * timestep,
                         scale=sigma * np.sqrt(timestep),
                         size=T)
    # S0 * exp(cumulative sum of steps)
    values = S0 * np.exp(np.cumsum(steps))
    values = np.insert(values, 0, S0)  # Include the initial value at the start

    return TimeSeries(timestep, list(values))

def generate_brownian_motion(T, timestep, mu, sigma, S0=0):
    """
    Generates a Brownian motion time series.

    :param T: Total number of time periods (equivalent to number of steps - 1)
    :param timestep: The size of each time step in years
    :param mu: Mean of the increments (often zero in pure Brownian motion)
    :param sigma: Volatility or standard deviation of the increments
    :param S0: Initial value of the time series (default 0)
    :return: TimeSeries object with generated Brownian motion values
    """
    # Total number of steps
    N = T + 1
    times = np.linspace(0, T * timestep, N)
    increments = np.random.normal(loc=mu * timestep,
                                  scale=sigma * np.sqrt(timestep),
                                  size=T)
    # Cumulative sum of increments to get the Brownian path
    values = np.cumsum(increments)
    values = np.insert(values, 0, S0)  # Include the initial value at the start

    return TimeSeries(timestep, list(values))
    
def generate_gbm_set(T, timestep, mu, sigma, correlation, S0=None):
    n = len(mu)  # Number of series
    if S0 is None:
        S0 = np.ones(n)  # Default initial values
    # Validate dimensions
    if len(sigma) != n or correlation.shape != (n, n) or (S0 is not None and len(S0) != n):
        raise ValueError("Dimensions of inputs do not match or are incorrect.")
    # Check if correlation matrix is valid
    if not (np.allclose(correlation, correlation.T) and np.all(np.linalg.eigvals(correlation) >= 0)):
        raise ValueError("Correlation matrix must be symmetric and positive semi-definite.")

    # Cholesky decomposition of the correlation matrix
    L = np.linalg.cholesky(correlation)

    # Total number of steps
    N = T + 1

    # Initialize array to hold the GBM paths
    paths = np.zeros((N, n))
    paths[0, :] = S0  # Set initial values

    # Generate paths
    for t in range(1, N):
        # Independent normal random variables
        Z = np.random.normal(size=n)
        # Correlated random variables
        correlated_Z = L @ Z
        # Calculate GBM step for each series
        step = (mu - 0.5 * sigma**2) * timestep + sigma * np.sqrt(timestep) * correlated_Z
        paths[t, :] = paths[t-1, :] * np.exp(step)

    # Create TimeSeries objects
    time_series_list = [TimeSeries(timestep, list(paths[:, i])) for i in range(n)]

    return time_series_list

def generate_brownian_motion_set(T, timestep, mu, sigma, correlation, S0=None):
    n = len(mu)  # Number of series
    if S0 is None:
        S0 = np.ones(n)  # Default initial values
    # Validate dimensions
    if len(sigma) != n or correlation.shape != (n, n) or (S0 is not None and len(S0) != n):
        raise ValueError("Dimensions of inputs do not match or are incorrect.")
    # Check if correlation matrix is valid
    if not (np.allclose(correlation, correlation.T) and np.all(np.linalg.eigvals(correlation) >= 0)):
        raise ValueError("Correlation matrix must be symmetric and positive semi-definite.")

    # Cholesky decomposition of the correlation matrix
    L = np.linalg.cholesky(correlation)

    # Total number of steps
    N = T + 1

    # Initialize array to hold the Brownian motion paths
    paths = np.zeros((N, n))
    paths[0, :] = S0  # Set initial values

    # Generate paths
    for t in range(1, N):
        # Independent normal random variables
        Z = np.random.normal(size=n)
        # Correlated random variables
        correlated_Z = L @ Z
        # Calculate Brownian motion step for each series
        step = mu * timestep + sigma * np.sqrt(timestep) * correlated_Z
        paths[t, :] = paths[t-1, :] + step

    # Create TimeSeries objects
    time_series_list = [TimeSeries(timestep, list(paths[:, i])) for i in range(n)]

    return time_series_list

class TimeSeries:
    def __init__(self, time_step: float, values: List[float]):
        """
        Initializes a TimeSeries object.

        :param time_step: Time step between data points, as a double-precision float.
        :param values: A list of float values representing the time series data points.
        """
        self.time_step = time_step
        self.values = values

    def __repr__(self):
        """
        Provides a string representation of the TimeSeries object.
        """
        return f"TimeSeries(time_step={self.time_step}, values={self.values})"

    def add_value(self, value: float):
        """
        Adds a new value to the time series.

        :param value: A float value to be added to the series.
        """
        self.values.append(value)

    def get_values(self):
        """
        Returns the list of values of the time series.
        """
        return self.values

    def get_time_step(self):
        """
        Returns the time step between data points.
        """
        return self.time_step

    def __add__(self, other):
        """
        Overloads the + operator to add two time series with the same time step and length.
        Raises ValueError if the time steps or lengths do not match.
        """
        if not isinstance(other, TimeSeries):
            return NotImplemented
        if self.time_step != other.time_step or len(self.values) != len(other.values):
            raise ValueError("Time series must have the same time step and number of data points for addition.")
        return TimeSeries(self.time_step, [a + b for a, b in zip(self.values, other.values)])

    def __mul__(self, other):
        """
        Overloads the * operator to multiply all elements of the time series by a constant.
        """
        if not isinstance(other, (int, float)):  # Ensure the multiplier is a number
            return NotImplemented
        return TimeSeries(self.time_step, [a * other for a in self.values])

    def __rmul__(self, other):
        """
        Overloads the right multiplication so that multiplication remains commutative.
        """
        return self.__mul__(other)
    
    def get_total_steps(self):
        """
        Returns total number of time steps of the time series.
        """
        return len(self.values)
    def get_total_time_horizon(self):
        """
        Returns the total time horizon of the time series, calculated as the
        time step multiplied by the number of intervals between consecutive values.
    
        This calculates the time from the start to the last data point,
        assuming the first data point is at time = 0.
        """
        return self.time_step * (len(self.values) - 1)
    def get_annualized_return(self):
        """
        Returns the annualized return, viewing the time series as an investment price series.
        The time step is considered to be in units of years.
        """
        # Ensure there are at least two values for the calculation
        if len(self.values) < 2:
            raise ValueError("Time series must have at least two values to calculate an annualized return.")

        # Time horizon in years
        total_time_horizon = self.get_total_time_horizon()

        # Return on investment
        roi = self.values[-1] / self.values[0]

        # Annualized return
        annualized_return = roi ** (1 / total_time_horizon)

        return annualized_return - 1.0
    


    def get_annualized_volatility(self):
        """
        Returns the annualized volatility of the time series, viewing it as an investment price series.
        The time step is considered to be in units of years.
        """
        if len(self.values) < 2:
            raise ValueError("Time series must have at least two values to calculate annualized volatility.")

        # Logarithmic returns
        returns = np.diff(np.log(self.values))

        # Standard deviation of returns
        std_dev = np.std(returns)

        # Annualizing based on time step
        time_step = self.get_time_step()  # Assumes time_step is in years

        # Annualizing volatility
        annualized_volatility = std_dev * (1 / time_step)**0.5

        return annualized_volatility

    def get_Sharpe_Ratio(self, risk_free_rate=0):
        """
        Returns the Sharpe ratio of the time series, viewing it as an investment price series.
        The time step is considered to be in units of years.
        The risk-free rate is annualized, as is the return of the investment.
        """
        # Calculate annualized return
        annualized_return = self.get_annualized_return()

        # Calculate annualized volatility
        annualized_volatility = self.get_annualized_volatility()

        # Ensure volatility is not zero to avoid division by zero
        if annualized_volatility == 0:
            raise ValueError("Volatility is zero, Sharpe Ratio cannot be calculated.")

        # Calculate Sharpe Ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        return sharpe_ratio
    def get_Max_Drawdown(self):
        """
        Returns the Maximum Drawdown of the time series, viewing it as an investment price series.
        The time step is considered to be in units of years.
        """
        # Retrieve values for max drawdown calculation
        V = self.get_values()
        Current_Max = V[0]
        Current_Max_Drawdown = 0

        for price in V:
            # Update the current maximum if the current price is higher
            Current_Max = max(Current_Max, price)

            # Calculate drawdown from the current maximum
            drawdown = (Current_Max - price) / Current_Max  # Drawdown calculated as a ratio

            # Update the maximum drawdown if the current drawdown is larger
            Current_Max_Drawdown = max(Current_Max_Drawdown, drawdown)

        return Current_Max_Drawdown
  
  
