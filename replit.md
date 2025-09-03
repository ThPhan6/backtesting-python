# Trading Strategy Backtester

## Overview

A comprehensive web-based trading strategy backtester built with Streamlit that allows users to test various trading strategies against historical market data. The application supports both stocks and cryptocurrencies, providing detailed performance analytics including portfolio value tracking, risk metrics, and comparative analysis against benchmark performance. Users can configure multiple strategy types including moving average crossovers, RSI mean reversion, and custom strategies with adjustable parameters like position sizing, stop losses, and take profits.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application with sidebar configuration and main content area
- **Interactive Visualizations**: Plotly-based charts for price data, portfolio performance, and trading signals
- **Session State Management**: Maintains backtest results and strategy configurations across user interactions
- **Responsive Layout**: Wide layout with expandable sidebar for strategy configuration

### Backend Architecture
- **Modular Design**: Separated into distinct components for backtesting logic, strategy generation, and utilities
- **BacktestingEngine Class**: Core simulation engine that processes price data and trading signals to calculate portfolio performance
- **StrategyBuilder Class**: Strategy factory that generates trading signals based on configurable parameters
- **Object-Oriented Approach**: Clean separation of concerns with dedicated classes for specific functionality

### Data Processing
- **Market Data Integration**: Yahoo Finance API for historical price data retrieval
- **Technical Analysis**: pandas_ta library for technical indicators calculation
- **Performance Metrics**: Custom utilities for calculating returns, volatility, Sharpe ratio, maximum drawdown, and Calmar ratio
- **Data Validation**: Input sanitization and error handling for market data and strategy parameters

### Trading Strategy Framework
- **Multiple Strategy Types**: Support for SMA crossover, RSI mean reversion, EMA+RSI combo, and custom strategies
- **Configurable Parameters**: User-adjustable windows, thresholds, position sizing, and risk management
- **Signal Generation**: Standardized position signals (1 for buy, -1 for sell, 0 for hold)
- **Risk Management**: Built-in stop loss and take profit functionality

## External Dependencies

### Market Data
- **Yahoo Finance (yfinance)**: Primary data source for historical stock and cryptocurrency prices
- **Real-time Data**: OHLCV (Open, High, Low, Close, Volume) data retrieval with configurable date ranges

### Technical Analysis
- **pandas_ta**: Technical analysis library for indicator calculations
- **NumPy/Pandas**: Core data manipulation and numerical computing

### Visualization
- **Plotly**: Interactive charting library for financial data visualization
- **Streamlit Components**: Built-in UI components for user input and data display

### Development Stack
- **Streamlit**: Web application framework for rapid development
- **Python Standard Library**: Core functionality including datetime manipulation and mathematical operations