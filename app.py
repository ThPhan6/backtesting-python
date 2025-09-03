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

# Initialize session state first
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'strategy_config' not in st.session_state:
    st.session_state.strategy_config = None
if 'backtest_history' not in st.session_state:
    st.session_state.backtest_history = {}
if 'is_running_backtest' not in st.session_state:
    st.session_state.is_running_backtest = False

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

# Check if symbol changed and there's an active backtest
symbol_changed = st.session_state.get('current_symbol', '') != symbol
if symbol_changed and st.session_state.backtest_results is not None:
    st.warning(f"⚠️ You have an active backtest for {st.session_state.get('current_symbol', 'previous symbol')}. Changing to {symbol} will clear current results.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear & Continue"):
            st.session_state.backtest_results = None
            st.session_state.strategy_config = None
            st.session_state.current_symbol = symbol
            st.rerun()
    with col2:
        if st.button("💾 Save to History"):
            # Save current backtest to history
            if st.session_state.strategy_config:
                history_key = f"{st.session_state.current_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.backtest_history[history_key] = {
                    'results': st.session_state.backtest_results,
                    'config': st.session_state.strategy_config,
                    'timestamp': datetime.now()
                }
            st.session_state.backtest_results = None
            st.session_state.strategy_config = None
            st.session_state.current_symbol = symbol
            st.rerun()

# Update current symbol
st.session_state.current_symbol = symbol

# Dynamic title that updates with the selected symbol
st.title(f"📈 Trading Strategy Backtester + {symbol}")
st.markdown("Test your trading strategies against historical market data with comprehensive performance analytics.")

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
    min_value=100,
    max_value=1000000,
    value=10000,
    step=100
)

position_size = st.sidebar.slider(
    "Position Size (%)",
    min_value=1,
    max_value=100,
    value=100,
    help="Percentage of capital to use per trade"
) / 100

# Risk/Reward Ratio Setting
use_risk_reward = st.sidebar.checkbox(
    "Use Risk/Reward Ratio",
    value=True,
    help="Use risk/reward ratio instead of individual SL/TP"
)

if use_risk_reward:
    risk_percent = st.sidebar.number_input(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Maximum loss per trade as % of capital"
    )
    
    reward_ratio = st.sidebar.number_input(
        "Reward Ratio (Risk:Reward)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Reward ratio (e.g., 2.0 means 1:2 risk/reward)"
    )
    
    stop_loss = risk_percent / 100
    take_profit = (risk_percent * reward_ratio) / 100
    
    st.sidebar.info(f"📊 Risk/Reward: {risk_percent:.1f}% / {risk_percent * reward_ratio:.1f}% (1:{reward_ratio:.1f})")
else:
    stop_loss = st.sidebar.number_input(
        "Stop Loss (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.5,
        help="Set to 0 to disable stop loss"
    ) / 100
    
    take_profit = st.sidebar.number_input(
        "Take Profit (%)",
        min_value=0.0,
        max_value=50.0,
        value=6.0,
        step=0.5,
        help="Set to 0 to disable take profit"
    ) / 100

# Backtest History Display
if st.session_state.backtest_history:
    st.sidebar.subheader("📚 Backtest History")
    history_options = []
    for key, item in st.session_state.backtest_history.items():
        config = item['config']
        timestamp = item['timestamp'].strftime('%m/%d %H:%M')
        history_options.append(f"{config['symbol']} - {config['strategy_type'][:10]} - {timestamp}")
    
    selected_history = st.sidebar.selectbox(
        "Load Previous Backtest:",
        [""] + history_options,
        help="Select a previous backtest to view"
    )
    
    if selected_history and selected_history != "":
        history_keys = list(st.session_state.backtest_history.keys())
        selected_key = history_keys[history_options.index(selected_history)]
        if st.sidebar.button("📊 Load Historical Backtest"):
            st.session_state.backtest_results = st.session_state.backtest_history[selected_key]['results']
            st.session_state.strategy_config = st.session_state.backtest_history[selected_key]['config']
            st.success(f"Loaded historical backtest for {st.session_state.backtest_history[selected_key]['config']['symbol']}")

# Run Backtest Button with enhanced states
button_text = "🚀 Run Backtest" if not st.session_state.is_running_backtest else "⏳ Running..."
button_disabled = st.session_state.is_running_backtest

if st.sidebar.button(button_text, type="primary", disabled=button_disabled):
    st.session_state.is_running_backtest = True
    try:
        # Create progress placeholders
        progress_container = st.empty()
        stage_container = st.empty()
        
        with progress_container.container():
            progress_bar = st.progress(0, text="Starting backtest...")
        
        # Stage 1: Fetch Data
        with stage_container.container():
            st.info("📥 Stage 1/4: Fetching market data...")
        progress_bar.progress(25, text="Fetching market data...")
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Clean and validate data
        if data is None or data.empty:
            st.error(f"No data found for symbol {symbol} in the specified date range.")
            st.session_state.is_running_backtest = False
        else:
            # Stage 2: Process Data
            with stage_container.container():
                st.info("🔧 Stage 2/4: Processing and cleaning data...")
            progress_bar.progress(50, text="Processing data...")
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns.values]
            
            # Ensure required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.session_state.is_running_backtest = False
            else:
                # Clean data
                data = data.dropna()
                for col in ['Open', 'High', 'Low', 'Close']:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                data = data.dropna()
                
                if data.empty:
                    st.error("No valid data available after cleaning.")
                    st.session_state.is_running_backtest = False
                else:
                    # Stage 3: Generate Signals
                    with stage_container.container():
                        st.info("📊 Stage 3/4: Generating trading signals...")
                    progress_bar.progress(75, text="Generating trading signals...")
                    
                    try:
                        strategy_builder = StrategyBuilder(strategy_type, strategy_params)
                        signals = strategy_builder.generate_signals(data)
                        
                        if len(signals) == 0:
                            st.warning("⚠️ No trading signals generated. Try adjusting strategy parameters.")
                            st.session_state.is_running_backtest = False
                        else:
                            # Stage 4: Run Backtest
                            with stage_container.container():
                                st.info(f"🚀 Stage 4/4: Running backtest with {len(signals)} signals...")
                            progress_bar.progress(100, text="Running backtest simulation...")
                            
                            engine = BacktestingEngine(
                                initial_capital=initial_capital,
                                position_size=position_size,
                                stop_loss=stop_loss if stop_loss > 0 else None,
                                take_profit=take_profit if take_profit > 0 else None
                            )
                            
                            results = engine.run_backtest(data, signals)
                            
                            # Success - Store results
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
                            
                            # Clear progress indicators and show success
                            progress_container.empty()
                            stage_container.empty()
                            st.success(f"✅ Backtest completed! Generated {len(signals)} signals, executed {len(results['trade_history'])} trades.")
                            st.session_state.is_running_backtest = False
                            
                    except Exception as strategy_error:
                        st.error(f"Strategy Error: {str(strategy_error)}")
                        st.session_state.is_running_backtest = False
                
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        st.session_state.is_running_backtest = False
        if 'progress_container' in locals():
            progress_container.empty()
        if 'stage_container' in locals():
            stage_container.empty()

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
