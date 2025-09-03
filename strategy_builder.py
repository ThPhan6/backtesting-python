import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any

class StrategyBuilder:
    """
    Build and generate trading signals for various strategies.
    """
    
    def __init__(self, strategy_type: str, parameters: Dict[str, Any]):
        """
        Initialize strategy builder.
        
        Args:
            strategy_type: Type of strategy to build
            parameters: Strategy-specific parameters
        """
        self.strategy_type = strategy_type
        self.parameters = parameters
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the configured strategy.
        
        Args:
            data: OHLCV price data
            
        Returns:
            DataFrame with Position column (1 for buy, -1 for sell, 0 for hold)
        """
        df = data.copy()
        
        if self.strategy_type == "Simple Moving Average Crossover":
            return self._sma_crossover_strategy(df)
        elif self.strategy_type == "RSI Mean Reversion":
            return self._rsi_mean_reversion_strategy(df)
        elif self.strategy_type == "EMA + RSI Combo":
            return self._ema_rsi_combo_strategy(df)
        elif self.strategy_type == "Custom":
            return self._custom_strategy(df)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
    
    def _sma_crossover_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple Moving Average Crossover Strategy.
        Buy when short SMA crosses above long SMA, sell when it crosses below.
        """
        short_window = self.parameters['short_window']
        long_window = self.parameters['long_window']
        
        # Calculate SMAs
        df['SMA_Short'] = ta.sma(df['Close'], length=short_window)
        df['SMA_Long'] = ta.sma(df['Close'], length=long_window)
        
        # Generate signals
        df['Signal'] = 0
        df['Position'] = 0
        
        # Buy signal: short SMA crosses above long SMA
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
        
        # Generate position changes
        df['Position'] = df['Signal'].diff()
        
        # Clean up signals (only keep actual crossovers)
        signals = df[df['Position'] != 0].copy()
        
        return signals[['Position']]
    
    def _rsi_mean_reversion_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RSI Mean Reversion Strategy.
        Buy when RSI is oversold, sell when overbought.
        """
        rsi_period = self.parameters['rsi_period']
        oversold = self.parameters['rsi_oversold']
        overbought = self.parameters['rsi_overbought']
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        
        # Generate signals
        df['Signal'] = 0
        df['Position'] = 0
        
        # Buy signal: RSI crosses below oversold level
        df.loc[(df['RSI'] < oversold) & (df['RSI'].shift(1) >= oversold), 'Position'] = 1
        
        # Sell signal: RSI crosses above overbought level
        df.loc[(df['RSI'] > overbought) & (df['RSI'].shift(1) <= overbought), 'Position'] = -1
        
        # Filter only actual signals
        signals = df[df['Position'] != 0].copy()
        
        return signals[['Position']]
    
    def _ema_rsi_combo_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EMA + RSI Combo Strategy.
        Buy when short EMA > long EMA and RSI > threshold, sell when opposite.
        """
        ema_short = self.parameters['ema_short']
        ema_long = self.parameters['ema_long']
        rsi_period = self.parameters['rsi_period']
        rsi_threshold = self.parameters['rsi_threshold']
        
        # Calculate indicators
        df['EMA_Short'] = ta.ema(df['Close'], length=ema_short)
        df['EMA_Long'] = ta.ema(df['Close'], length=ema_long)
        df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        
        # Generate signals
        df['Position'] = 0
        
        # Buy conditions: EMA crossover up and RSI above threshold
        buy_condition = (
            (df['EMA_Short'] > df['EMA_Long']) & 
            (df['EMA_Short'].shift(1) <= df['EMA_Long'].shift(1)) &
            (df['RSI'] > rsi_threshold)
        )
        
        # Sell conditions: EMA crossover down or RSI below threshold
        sell_condition = (
            (df['EMA_Short'] < df['EMA_Long']) & 
            (df['EMA_Short'].shift(1) >= df['EMA_Long'].shift(1))
        ) | (df['RSI'] < rsi_threshold)
        
        df.loc[buy_condition, 'Position'] = 1
        df.loc[sell_condition, 'Position'] = -1
        
        # Filter only actual signals
        signals = df[df['Position'] != 0].copy()
        
        return signals[['Position']]
    
    def _custom_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Custom strategy based on user-selected indicators.
        """
        df['Position'] = 0
        buy_conditions = []
        sell_conditions = []
        
        # Add SMA condition
        if self.parameters.get('use_sma', False):
            sma_period = self.parameters['sma_period']
            df['SMA'] = ta.sma(df['Close'], length=sma_period)
            buy_conditions.append(df['Close'] > df['SMA'])
            sell_conditions.append(df['Close'] < df['SMA'])
        
        # Add EMA condition
        if self.parameters.get('use_ema', False):
            ema_period = self.parameters['ema_period']
            df['EMA'] = ta.ema(df['Close'], length=ema_period)
            buy_conditions.append(df['Close'] > df['EMA'])
            sell_conditions.append(df['Close'] < df['EMA'])
        
        # Add RSI condition
        if self.parameters.get('use_rsi', False):
            rsi_period = self.parameters['rsi_period']
            rsi_buy = self.parameters['rsi_buy']
            rsi_sell = self.parameters['rsi_sell']
            df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
            buy_conditions.append(df['RSI'] < rsi_buy)
            sell_conditions.append(df['RSI'] > rsi_sell)
        
        # Combine conditions
        if buy_conditions:
            final_buy_condition = buy_conditions[0]
            for condition in buy_conditions[1:]:
                final_buy_condition = final_buy_condition & condition
            df.loc[final_buy_condition, 'Position'] = 1
        
        if sell_conditions:
            final_sell_condition = sell_conditions[0]
            for condition in sell_conditions[1:]:
                final_sell_condition = final_sell_condition & condition
            df.loc[final_sell_condition, 'Position'] = -1
        
        # Filter only actual signals
        signals = df[df['Position'] != 0].copy()
        
        return signals[['Position']]
    
    def get_indicator_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get calculated indicator data for visualization.
        """
        df = data.copy()
        
        if self.strategy_type == "Simple Moving Average Crossover":
            short_window = self.parameters['short_window']
            long_window = self.parameters['long_window']
            df['SMA_Short'] = ta.sma(df['Close'], length=short_window)
            df['SMA_Long'] = ta.sma(df['Close'], length=long_window)
            
        elif self.strategy_type == "RSI Mean Reversion":
            rsi_period = self.parameters['rsi_period']
            df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
            
        elif self.strategy_type == "EMA + RSI Combo":
            ema_short = self.parameters['ema_short']
            ema_long = self.parameters['ema_long']
            rsi_period = self.parameters['rsi_period']
            df['EMA_Short'] = ta.ema(df['Close'], length=ema_short)
            df['EMA_Long'] = ta.ema(df['Close'], length=ema_long)
            df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        
        return df
