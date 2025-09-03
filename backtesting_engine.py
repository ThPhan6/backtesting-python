import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class BacktestingEngine:
    """
    A comprehensive backtesting engine for trading strategies.
    """
    
    def __init__(self, initial_capital: float = 10000, position_size: float = 1.0, 
                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital amount
            position_size: Fraction of capital to use per trade (0-1)
            stop_loss: Stop loss percentage (0-1)
            take_profit: Take profit percentage (0-1)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the backtesting simulation.
        
        Args:
            data: OHLCV price data
            signals: Trading signals with Position column
            
        Returns:
            Dictionary containing backtest results
        """
        # Ensure data is properly indexed
        data = data.copy()
        signals = signals.copy()
        
        # Initialize tracking variables
        portfolio_value = [self.initial_capital]
        cash = self.initial_capital
        shares = 0
        trades = []
        
        # Track portfolio and benchmark
        benchmark_value = [self.initial_capital]
        benchmark_shares = self.initial_capital / data['Close'].iloc[0]
        
        # Current trade tracking
        current_trade = None
        
        for i in range(1, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Update benchmark (buy and hold)
            benchmark_value.append(benchmark_shares * current_price)
            
            # Check for signals
            if current_date in signals.index:
                signal = signals.loc[current_date, 'Position']
                
                # Buy signal
                if signal == 1 and shares == 0:
                    shares_to_buy = int((cash * self.position_size) / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        if cost <= cash:
                            shares = shares_to_buy
                            cash -= cost
                            current_trade = {
                                'Entry Date': current_date,
                                'Entry Price': current_price,
                                'Shares': shares,
                                'Type': 'Long'
                            }
                
                # Sell signal
                elif signal == -1 and shares > 0:
                    proceeds = shares * current_price
                    cash += proceeds
                    
                    if current_trade:
                        # Record completed trade
                        trade_return = (current_price - current_trade['Entry Price']) / current_trade['Entry Price']
                        pnl = proceeds - (current_trade['Shares'] * current_trade['Entry Price'])
                        
                        trades.append({
                            'Entry Date': current_trade['Entry Date'],
                            'Exit Date': current_date,
                            'Entry Price': current_trade['Entry Price'],
                            'Exit Price': current_price,
                            'Shares': current_trade['Shares'],
                            'Return': trade_return,
                            'P&L': pnl,
                            'Type': current_trade['Type']
                        })
                    
                    shares = 0
                    current_trade = None
            
            # Check stop loss and take profit
            if shares > 0 and current_trade:
                entry_price = current_trade['Entry Price']
                
                # Stop loss check
                if self.stop_loss and current_price <= entry_price * (1 - self.stop_loss):
                    proceeds = shares * current_price
                    cash += proceeds
                    
                    trade_return = (current_price - entry_price) / entry_price
                    pnl = proceeds - (shares * entry_price)
                    
                    trades.append({
                        'Entry Date': current_trade['Entry Date'],
                        'Exit Date': current_date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'Shares': shares,
                        'Return': trade_return,
                        'P&L': pnl,
                        'Type': 'Stop Loss'
                    })
                    
                    shares = 0
                    current_trade = None
                
                # Take profit check
                elif self.take_profit and current_price >= entry_price * (1 + self.take_profit):
                    proceeds = shares * current_price
                    cash += proceeds
                    
                    trade_return = (current_price - entry_price) / entry_price
                    pnl = proceeds - (shares * entry_price)
                    
                    trades.append({
                        'Entry Date': current_trade['Entry Date'],
                        'Exit Date': current_date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'Shares': shares,
                        'Return': trade_return,
                        'P&L': pnl,
                        'Type': 'Take Profit'
                    })
                    
                    shares = 0
                    current_trade = None
            
            # Calculate current portfolio value
            current_portfolio_value = cash + (shares * current_price)
            portfolio_value.append(current_portfolio_value)
        
        # Close any remaining position
        if shares > 0 and current_trade:
            final_price = data['Close'].iloc[-1]
            proceeds = shares * final_price
            
            trade_return = (final_price - current_trade['Entry Price']) / current_trade['Entry Price']
            pnl = proceeds - (shares * current_trade['Entry Price'])
            
            trades.append({
                'Entry Date': current_trade['Entry Date'],
                'Exit Date': data.index[-1],
                'Entry Price': current_trade['Entry Price'],
                'Exit Price': final_price,
                'Shares': shares,
                'Return': trade_return,
                'P&L': pnl,
                'Type': 'Final'
            })
        
        # Create results
        portfolio_series = pd.Series(portfolio_value, index=data.index)
        benchmark_series = pd.Series(benchmark_value, index=data.index)
        trade_history = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_series, benchmark_series, trade_history
        )
        
        return {
            'portfolio_value': portfolio_series,
            'benchmark_value': benchmark_series,
            'trade_history': trade_history,
            'performance_metrics': performance_metrics,
            'price_data': data,
            'signals': signals
        }
    
    def _calculate_performance_metrics(self, portfolio: pd.Series, benchmark: pd.Series, 
                                     trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        """
        # Basic returns
        total_return = (portfolio.iloc[-1] - portfolio.iloc[0]) / portfolio.iloc[0]
        benchmark_return = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0]
        
        # Daily returns
        portfolio_returns = portfolio.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days, 2% risk-free rate)
        excess_returns = portfolio_returns - 0.02/252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Maximum drawdown
        running_max = portfolio.expanding().max()
        drawdown = (portfolio - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Trade-based metrics
        if not trades.empty:
            win_rate = (trades['Return'] > 0).mean()
            avg_trade_return = trades['Return'].mean()
            total_trades = len(trades)
        else:
            win_rate = 0
            avg_trade_return = 0
            total_trades = 0
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'total_trades': total_trades
        }
