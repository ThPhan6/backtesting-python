import pandas as pd
import numpy as np
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
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        
        # Generate signals with position tracking
        df['Position'] = 0
        current_position = 0
        
        for i in range(long_window, len(df)):
            sma_short = df['SMA_Short'].iloc[i]
            sma_long = df['SMA_Long'].iloc[i]
            sma_short_prev = df['SMA_Short'].iloc[i-1]
            sma_long_prev = df['SMA_Long'].iloc[i-1]
            
            # Buy signal: short SMA crosses above long SMA
            if (sma_short > sma_long and sma_short_prev <= sma_long_prev and current_position <= 0):
                df.loc[df.index[i], 'Position'] = 1
                current_position = 1
            
            # Sell signal: short SMA crosses below long SMA
            elif (sma_short < sma_long and sma_short_prev >= sma_long_prev and current_position >= 0):
                df.loc[df.index[i], 'Position'] = -1
                current_position = -1
        
        # Filter only actual signals
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
        df['RSI'] = self._calculate_rsi(df['Close'], rsi_period)
        
        # Generate signals with position tracking
        df['Position'] = 0
        current_position = 0
        
        for i in range(rsi_period, len(df)):
            rsi_val = df['RSI'].iloc[i]
            rsi_prev = df['RSI'].iloc[i-1] if i > 0 else rsi_val
            
            # Buy signal: RSI crosses below oversold level
            if (rsi_val < oversold and rsi_prev >= oversold and current_position <= 0):
                df.loc[df.index[i], 'Position'] = 1
                current_position = 1
            
            # Sell signal: RSI crosses above overbought level
            elif (rsi_val > overbought and rsi_prev <= overbought and current_position >= 0):
                df.loc[df.index[i], 'Position'] = -1
                current_position = -1
        
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
        df['EMA_Short'] = df['Close'].ewm(span=ema_short).mean()
        df['EMA_Long'] = df['Close'].ewm(span=ema_long).mean()
        df['RSI'] = self._calculate_rsi(df['Close'], rsi_period)
        
        # Generate signals with position tracking
        df['Position'] = 0
        df['Signal'] = 0
        current_position = 0
        
        for i in range(max(ema_long, rsi_period), len(df)):
            ema_short_val = df['EMA_Short'].iloc[i]
            ema_long_val = df['EMA_Long'].iloc[i]
            rsi_val = df['RSI'].iloc[i]
            
            # Buy signal: EMA trend is bullish and RSI is above threshold
            if (ema_short_val > ema_long_val and rsi_val > rsi_threshold and current_position <= 0):
                df.loc[df.index[i], 'Position'] = 1
                df.loc[df.index[i], 'Signal'] = 1
                current_position = 1
            
            # Sell signal: EMA trend is bearish or RSI drops below threshold
            elif ((ema_short_val < ema_long_val or rsi_val < rsi_threshold) and current_position >= 0):
                df.loc[df.index[i], 'Position'] = -1
                df.loc[df.index[i], 'Signal'] = -1
                current_position = -1
        
        # Filter only actual signals
        signals = df[df['Position'] != 0].copy()
        
        return signals[['Position']]
    
    def _custom_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Custom strategy based on user-selected indicators with proper position tracking.
        """
        df['Position'] = 0
        current_position = 0
        
        # Calculate all indicators first
        if self.parameters.get('use_sma', False):
            sma_period = self.parameters['sma_period']
            df['SMA'] = df['Close'].rolling(window=sma_period).mean()
        
        if self.parameters.get('use_ema', False):
            ema_period = self.parameters['ema_period']
            df['EMA'] = df['Close'].ewm(span=ema_period).mean()
        
        if self.parameters.get('use_rsi', False):
            rsi_period = self.parameters['rsi_period']
            df['RSI'] = self._calculate_rsi(df['Close'], rsi_period)
        
        # Determine the start index (after indicators are calculated)
        max_period = 1
        if self.parameters.get('use_sma', False):
            max_period = max(max_period, self.parameters['sma_period'])
        if self.parameters.get('use_ema', False):
            max_period = max(max_period, self.parameters['ema_period'])
        if self.parameters.get('use_rsi', False):
            max_period = max(max_period, self.parameters['rsi_period'])
        
        # Process signals with proper position tracking
        for i in range(max_period, len(df)):
            buy_conditions = []
            sell_conditions = []
            
            # Check SMA condition
            if self.parameters.get('use_sma', False):
                buy_conditions.append(df['Close'].iloc[i] > df['SMA'].iloc[i])
                sell_conditions.append(df['Close'].iloc[i] < df['SMA'].iloc[i])
            
            # Check EMA condition
            if self.parameters.get('use_ema', False):
                buy_conditions.append(df['Close'].iloc[i] > df['EMA'].iloc[i])
                sell_conditions.append(df['Close'].iloc[i] < df['EMA'].iloc[i])
            
            # Check RSI condition
            if self.parameters.get('use_rsi', False):
                rsi_buy = self.parameters['rsi_buy']
                rsi_sell = self.parameters['rsi_sell']
                buy_conditions.append(df['RSI'].iloc[i] < rsi_buy)
                sell_conditions.append(df['RSI'].iloc[i] > rsi_sell)
            
            # Evaluate combined conditions
            buy_signal = all(buy_conditions) if buy_conditions else False
            sell_signal = all(sell_conditions) if sell_conditions else False
            
            # Generate position changes based on current state
            if buy_signal and current_position <= 0:
                df.loc[df.index[i], 'Position'] = 1
                current_position = 1
            elif sell_signal and current_position >= 0:
                df.loc[df.index[i], 'Position'] = -1
                current_position = -1
        
        # Filter only actual signals
        signals = df[df['Position'] != 0].copy()
        
        return signals[['Position']]
    
    def _calculate_rsi(self, price_series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index) using pandas.
        """
        delta = price_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_indicator_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get calculated indicator data for visualization.
        """
        df = data.copy()
        
        if self.strategy_type == "Simple Moving Average Crossover":
            short_window = self.parameters['short_window']
            long_window = self.parameters['long_window']
            df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
            df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
            
        elif self.strategy_type == "RSI Mean Reversion":
            rsi_period = self.parameters['rsi_period']
            df['RSI'] = self._calculate_rsi(df['Close'], rsi_period)
            
        elif self.strategy_type == "EMA + RSI Combo":
            ema_short = self.parameters['ema_short']
            ema_long = self.parameters['ema_long']
            rsi_period = self.parameters['rsi_period']
            df['EMA_Short'] = df['Close'].ewm(span=ema_short).mean()
            df['EMA_Long'] = df['Close'].ewm(span=ema_long).mean()
            df['RSI'] = self._calculate_rsi(df['Close'], rsi_period)
        
        return df
