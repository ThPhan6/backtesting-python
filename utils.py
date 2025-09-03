import pandas as pd
import numpy as np
from typing import Union

def calculate_metrics(returns: pd.Series) -> dict:
    """
    Calculate various performance metrics from returns series.
    
    Args:
        returns: Series of periodic returns
        
    Returns:
        Dictionary of calculated metrics
    """
    if returns.empty or returns.std() == 0:
        return {
            'total_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0
        }
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    # Annualized volatility (assuming daily returns)
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 2% risk-free rate)
    excess_returns = returns - 0.02/252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, currency: str = "$") -> str:
    """
    Format a value as currency string.
    
    Args:
        value: Numeric value to format
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return "N/A"
    return f"{currency}{value:,.2f}"

def validate_date_range(start_date, end_date) -> bool:
    """
    Validate that the date range is valid for backtesting.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        True if valid, False otherwise
    """
    if start_date >= end_date:
        return False
    
    # Check if range is at least 30 days
    if (end_date - start_date).days < 30:
        return False
    
    return True

def calculate_position_size(capital: float, price: float, position_pct: float) -> int:
    """
    Calculate the number of shares to buy given capital and position size.
    
    Args:
        capital: Available capital
        price: Current price per share
        position_pct: Percentage of capital to use (0-1)
        
    Returns:
        Number of shares to buy
    """
    if price <= 0 or capital <= 0 or position_pct <= 0:
        return 0
    
    max_investment = capital * position_pct
    shares = int(max_investment / price)
    
    return shares

def detect_outliers(data: pd.Series, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers in a data series.
    
    Args:
        data: Data series to analyze
        method: Method to use ('iqr' or 'zscore')
        
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > 3
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Return series
        window: Rolling window size
        
    Returns:
        Rolling Sharpe ratio series
    """
    excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    return rolling_mean / rolling_std * np.sqrt(252)

def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR value
    """
    if returns.empty:
        return 0
    
    return returns.quantile(confidence_level)

def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level
        
    Returns:
        Expected Shortfall value
    """
    if returns.empty:
        return 0
    
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()
