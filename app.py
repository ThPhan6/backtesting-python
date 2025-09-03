import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

from backtesting_engine import BacktestingEngine
from strategy_builder import StrategyBuilder
from utils import calculate_metrics, format_percentage, format_currency

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Trading Strategy Backtester")
st.markdown("Test your trading strategies against historical market data with comprehensive performance analytics.")

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'strategy_config' not in st.session_state:
    st.session_state.strategy_config = None

# Sidebar - Strategy Configuration
st.sidebar.header("🔧 Strategy Configuration")

# Asset Selection
asset_type = st.sidebar.selectbox(
    "Asset Type",
    ["Stock", "Cryptocurrency"],
    help="Select the type of asset to backtest"
)

if asset_type == "Stock":
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
    ).upper()
else:
    crypto_symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD"]
    symbol = st.sidebar.selectbox(
        "Cryptocurrency",
        crypto_symbols,
        help="Select cryptocurrency to backtest"
    )

# Time Period
st.sidebar.subheader("📅 Time Period")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Strategy Parameters
st.sidebar.subheader("📊 Technical Indicators")

strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    ["Simple Moving Average Crossover", "RSI Mean Reversion", "EMA + RSI Combo", "Custom"],
    help="Select a predefined strategy or create custom"
)

# Strategy specific parameters
strategy_params = {}

if strategy_type == "Simple Moving Average Crossover":
    strategy_params['short_window'] = st.sidebar.slider("Short SMA Period", 5, 50, 20)
    strategy_params['long_window'] = st.sidebar.slider("Long SMA Period", 20, 200, 50)
    
elif strategy_type == "RSI Mean Reversion":
    strategy_params['rsi_period'] = st.sidebar.slider("RSI Period", 5, 30, 14)
    strategy_params['rsi_oversold'] = st.sidebar.slider("RSI Oversold Level", 10, 40, 30)
    strategy_params['rsi_overbought'] = st.sidebar.slider("RSI Overbought Level", 60, 90, 70)
    
elif strategy_type == "EMA + RSI Combo":
    strategy_params['ema_short'] = st.sidebar.slider("Short EMA Period", 5, 30, 12)
    strategy_params['ema_long'] = st.sidebar.slider("Long EMA Period", 20, 100, 26)
    strategy_params['rsi_period'] = st.sidebar.slider("RSI Period", 5, 30, 14)
    strategy_params['rsi_threshold'] = st.sidebar.slider("RSI Threshold", 40, 60, 50)

elif strategy_type == "Custom":
    st.sidebar.markdown("**Custom Strategy Parameters**")
    strategy_params['use_sma'] = st.sidebar.checkbox("Use SMA")
    if strategy_params['use_sma']:
        strategy_params['sma_period'] = st.sidebar.slider("SMA Period", 5, 50, 20)
    
    strategy_params['use_ema'] = st.sidebar.checkbox("Use EMA")
    if strategy_params['use_ema']:
        strategy_params['ema_period'] = st.sidebar.slider("EMA Period", 5, 50, 20)
    
    strategy_params['use_rsi'] = st.sidebar.checkbox("Use RSI")
    if strategy_params['use_rsi']:
        strategy_params['rsi_period'] = st.sidebar.slider("RSI Period", 5, 30, 14)
        strategy_params['rsi_buy'] = st.sidebar.slider("RSI Buy Level", 20, 50, 30)
        strategy_params['rsi_sell'] = st.sidebar.slider("RSI Sell Level", 50, 80, 70)

# Risk Management
st.sidebar.subheader("⚠️ Risk Management")
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

position_size = st.sidebar.slider(
    "Position Size (%)",
    min_value=1,
    max_value=100,
    value=100,
    help="Percentage of capital to use per trade"
) / 100

stop_loss = st.sidebar.number_input(
    "Stop Loss (%)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.5,
    help="Set to 0 to disable stop loss"
) / 100

take_profit = st.sidebar.number_input(
    "Take Profit (%)",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=0.5,
    help="Set to 0 to disable take profit"
) / 100

# Run Backtest Button
if st.sidebar.button("🚀 Run Backtest", type="primary"):
    try:
        with st.spinner("Fetching data and running backtest..."):
            # Fetch data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Clean and validate data
            if data is None or data.empty:
                st.error(f"No data found for symbol {symbol} in the specified date range.")
            else:
                # Handle multi-level columns (sometimes yfinance returns ticker info in columns)
                if isinstance(data.columns, pd.MultiIndex):
                    # Flatten multi-level columns - take the first level (price type)
                    data.columns = [col[0] for col in data.columns.values]
                
                # Ensure we have the required OHLCV columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    # Clean the data - remove any rows with invalid data
                    data = data.dropna()
                    
                    # Ensure all price columns are numeric
                    for col in ['Open', 'High', 'Low', 'Close']:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                    # Remove any remaining invalid rows
                    data = data.dropna()
                    
                    if data.empty:
                        st.error("No valid data available after cleaning.")
                    else:
                        # Initialize strategy builder and backtesting engine
                        strategy_builder = StrategyBuilder(strategy_type, strategy_params)
                        engine = BacktestingEngine(
                            initial_capital=initial_capital,
                            position_size=position_size,
                            stop_loss=stop_loss if stop_loss > 0 else None,
                            take_profit=take_profit if take_profit > 0 else None
                        )
                        
                        # Generate signals
                        signals = strategy_builder.generate_signals(data)
                        
                        # Run backtest
                        results = engine.run_backtest(data, signals)
                        
                        # Store results in session state
                        st.session_state.backtest_results = results
                        st.session_state.strategy_config = {
                            'symbol': symbol,
                            'strategy_type': strategy_type,
                            'strategy_params': strategy_params,
                            'initial_capital': initial_capital,
                            'position_size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'start_date': start_date,
                            'end_date': end_date
                        }
                        
                        st.success("Backtest completed successfully!")
                
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")

# Main content area
if st.session_state.backtest_results is not None:
    results = st.session_state.backtest_results
    config = st.session_state.strategy_config
    
    # Performance Summary
    st.header("📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = results['performance_metrics']['total_return']
        st.metric(
            "Total Return",
            format_percentage(total_return),
            delta=format_percentage(total_return)
        )
    
    with col2:
        sharpe_ratio = results['performance_metrics']['sharpe_ratio']
        st.metric(
            "Sharpe Ratio",
            f"{sharpe_ratio:.2f}",
            delta=f"{sharpe_ratio:.2f}" if sharpe_ratio > 1 else None
        )
    
    with col3:
        max_drawdown = results['performance_metrics']['max_drawdown']
        st.metric(
            "Max Drawdown",
            format_percentage(max_drawdown),
            delta=format_percentage(max_drawdown)
        )
    
    with col4:
        win_rate = results['performance_metrics']['win_rate']
        st.metric(
            "Win Rate",
            format_percentage(win_rate),
            delta=format_percentage(win_rate - 0.5)
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        total_trades = results['performance_metrics']['total_trades']
        st.metric("Total Trades", total_trades)
    
    with col6:
        avg_trade_return = results['performance_metrics']['avg_trade_return']
        st.metric("Avg Trade Return", format_percentage(avg_trade_return))
    
    with col7:
        volatility = results['performance_metrics']['volatility']
        st.metric("Volatility", format_percentage(volatility))
    
    with col8:
        final_portfolio_value = results['portfolio_value'].iloc[-1]
        st.metric("Final Portfolio Value", format_currency(final_portfolio_value))
    
    # Portfolio Performance Chart
    st.header("📈 Portfolio Performance")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Portfolio Value vs Benchmark', 'Price with Buy/Sell Signals'),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4]
    )
    
    # Portfolio value chart
    portfolio_data = results['portfolio_value']
    benchmark_data = results['benchmark_value']
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data.values,
            name='Strategy',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=benchmark_data.index,
            y=benchmark_data.values,
            name='Buy & Hold',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=1
    )
    
    # Price with signals
    price_data = results['price_data']
    signals_data = results['signals']
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            name='Price',
            line=dict(color='black', width=1),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Buy signals
    buy_signals = signals_data[signals_data['Position'] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=price_data.loc[buy_signals.index, 'Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=8, color='green'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Sell signals
    sell_signals = signals_data[signals_data['Position'] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=price_data.loc[sell_signals.index, 'Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=8, color='red'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=700,
        title=f"{config['symbol']} - {config['strategy_type']} Strategy Backtest",
        xaxis2_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis2_title="Price ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade History
    st.header("📋 Trade History")
    
    trade_history = results['trade_history']
    if not trade_history.empty:
        st.dataframe(
            trade_history.style.format({
                'Entry Price': '${:.2f}',
                'Exit Price': '${:.2f}',
                'Return': '{:.2%}',
                'P&L': '${:.2f}'
            }),
            use_container_width=True
        )
        
        # Trade analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Return distribution
            fig_hist = px.histogram(
                trade_history,
                x='Return',
                title='Trade Return Distribution',
                nbins=20
            )
            fig_hist.update_layout(
                xaxis_title='Return (%)',
                yaxis_title='Frequency'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Cumulative P&L
            trade_history['Cumulative P&L'] = trade_history['P&L'].cumsum()
            fig_pnl = px.line(
                trade_history,
                x=trade_history.index,
                y='Cumulative P&L',
                title='Cumulative P&L by Trade'
            )
            fig_pnl.update_layout(
                xaxis_title='Trade Number',
                yaxis_title='Cumulative P&L ($)'
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.info("No trades were executed during the backtest period.")
    
    # Export Results
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export trade history
        if not trade_history.empty:
            csv_trades = trade_history.to_csv(index=False)
            st.download_button(
                label="📊 Download Trade History (CSV)",
                data=csv_trades,
                file_name=f"{config['symbol']}_trade_history.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export portfolio performance
        portfolio_df = pd.DataFrame({
            'Date': portfolio_data.index,
            'Portfolio Value': portfolio_data.values,
            'Benchmark Value': benchmark_data.values
        })
        csv_portfolio = portfolio_df.to_csv(index=False)
        st.download_button(
            label="📈 Download Portfolio Data (CSV)",
            data=csv_portfolio,
            file_name=f"{config['symbol']}_portfolio_performance.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export full report
        report_data = {
            'Strategy Configuration': config,
            'Performance Metrics': results['performance_metrics']
        }
        report_str = str(report_data)
        st.download_button(
            label="📄 Download Full Report (TXT)",
            data=report_str,
            file_name=f"{config['symbol']}_backtest_report.txt",
            mime="text/plain"
        )

else:
    # Welcome message
    st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Backtest' to get started!")
    
    # Instructions
    st.header("🎯 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Select Asset**
        - Choose between stocks and cryptocurrencies
        - Enter the symbol or select from popular cryptos
        
        **2. Set Time Period**
        - Choose start and end dates for backtesting
        - Longer periods provide more reliable results
        
        **3. Configure Strategy**
        - Select from predefined strategies or create custom
        - Adjust technical indicator parameters
        """)
    
    with col2:
        st.markdown("""
        **4. Risk Management**
        - Set initial capital amount
        - Configure position sizing
        - Optional stop-loss and take-profit levels
        
        **5. Run & Analyze**
        - Click 'Run Backtest' to execute
        - Review performance metrics and charts
        - Export results for further analysis
        """)
    
    st.header("📚 Available Strategies")
    
    strategy_info = {
        "Simple Moving Average Crossover": "Buy when short SMA crosses above long SMA, sell when it crosses below",
        "RSI Mean Reversion": "Buy when RSI is oversold, sell when overbought",
        "EMA + RSI Combo": "Combined EMA crossover with RSI confirmation",
        "Custom": "Build your own strategy with multiple indicators"
    }
    
    for strategy, description in strategy_info.items():
        st.markdown(f"**{strategy}**: {description}")
