"""
Stock Price Tracker with Technical Indicators
=============================================
A beginner-friendly stock tracking dashboard built with Streamlit.

This app will:
- Track stock prices for stocks you're interested in
- Show price charts with moving averages
- Calculate technical indicators (SMA, RSI)
- Give simple buy/sell/hold recommendations

To run this app:
1. Install required packages: pip install streamlit yfinance pandas plotly
2. Run: streamlit run stock_tracker.py
3. Open your browser to the URL shown (usually http://localhost:8501)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# CONFIGURATION - Edit these to customize!
# ============================================

# Add or remove stock symbols here (use the ticker symbol, like "AAPL" for Apple)
DEFAULT_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "AVGO", "CRWV", "UBER", NVDA", "META", "KLAR"]

# How many days of historical data to fetch
LOOKBACK_DAYS = 365


# ============================================
# TECHNICAL INDICATOR FUNCTIONS
# ============================================

def calculate_sma(prices, window):
    """
    Calculate Simple Moving Average (SMA).

    What is SMA? It's the average price over a certain number of days.
    - A 20-day SMA is the average of the last 20 days' closing prices.
    - When current price is ABOVE the SMA, it might indicate an uptrend.
    - When current price is BELOW the SMA, it might indicate a downtrend.

    Args:
        prices: A pandas Series of stock prices
        window: Number of days to average (e.g., 20 for 20-day SMA)

    Returns:
        A pandas Series with the SMA values
    """
    return prices.rolling(window=window).mean()


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).

    What is RSI? It measures how fast and how much prices are changing.
    - RSI ranges from 0 to 100
    - RSI > 70 typically means "overbought" (might go down soon)
    - RSI < 30 typically means "oversold" (might go up soon)
    - RSI around 50 is neutral

    Args:
        prices: A pandas Series of stock prices
        period: Number of days to calculate RSI over (default is 14)

    Returns:
        A pandas Series with RSI values
    """
    # Calculate price changes from day to day
    delta = prices.diff()

    # Separate gains (positive changes) and losses (negative changes)
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0  # Keep only positive values
    losses[losses > 0] = 0  # Keep only negative values
    losses = abs(losses)  # Make losses positive for calculation

    # Calculate average gains and losses
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    # Calculate RSI
    # RS = Average Gain / Average Loss
    # RSI = 100 - (100 / (1 + RS))
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_recommendation(current_price, sma_20, sma_50, rsi):
    """
    Generate a buy/sell/hold recommendation based on technical indicators.

    This uses a simple strategy combining moving averages and RSI:

    BUY signals:
    - Price is above both SMAs (uptrend) AND RSI < 70 (not overbought)
    - OR RSI < 30 (oversold, potential bounce)

    SELL signals:
    - Price is below both SMAs (downtrend) AND RSI > 30 (not oversold)
    - OR RSI > 70 (overbought, potential pullback)

    HOLD: Everything else

    IMPORTANT: This is for educational purposes only! Real trading decisions
    should consider many more factors.
    """
    # Check for missing data
    if pd.isna(sma_20) or pd.isna(sma_50) or pd.isna(rsi):
        return "HOLD", "Not enough data", "gray"

    reasons = []

    # Strong buy signal: oversold
    if rsi < 30:
        return "BUY", f"RSI is {rsi:.1f} (oversold)", "green"

    # Strong sell signal: overbought
    if rsi > 70:
        return "SELL", f"RSI is {rsi:.1f} (overbought)", "red"

    # Check trend using moving averages
    above_sma_20 = current_price > sma_20
    above_sma_50 = current_price > sma_50

    if above_sma_20 and above_sma_50:
        return "BUY", "Price above both SMAs (uptrend)", "green"
    elif not above_sma_20 and not above_sma_50:
        return "SELL", "Price below both SMAs (downtrend)", "red"
    else:
        return "HOLD", "Mixed signals - wait for clearer trend", "orange"


def fetch_stock_data(symbol, days=LOOKBACK_DAYS):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        days: How many days of history to fetch

    Returns:
        A pandas DataFrame with the stock data, or None if there's an error
    """
    try:
        # Create a Ticker object for the stock
        stock = yf.Ticker(symbol)

        # Calculate the start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Download historical data
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            return None

        return data
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None


# ============================================
# STREAMLIT APP
# ============================================

def main():
    """Main function that runs the Streamlit app."""

    # Page configuration
    st.set_page_config(
        page_title="Stock Tracker",
        page_icon="📈",
        layout="wide"
    )

    # Title and description
    st.title("📈 Stock Price Tracker")
    st.markdown("""
    Track your stocks with technical indicators and get simple buy/sell recommendations.

    **Disclaimer:** This is for educational purposes only. Not financial advice!
    """)

    # Sidebar for stock selection
    st.sidebar.header("📊 Settings")

    # Let user add custom stocks
    custom_stocks = st.sidebar.text_input(
        "Add stocks (comma-separated)",
        placeholder="e.g., NVDA, META, NFLX"
    )

    # Combine default stocks with any custom ones
    stocks_to_track = DEFAULT_STOCKS.copy()
    if custom_stocks:
        # Split by comma and clean up whitespace
        new_stocks = [s.strip().upper() for s in custom_stocks.split(",")]
        stocks_to_track.extend(new_stocks)
        # Remove duplicates while preserving order
        stocks_to_track = list(dict.fromkeys(stocks_to_track))

    # Select which stocks to display
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to track",
        options=stocks_to_track,
        default=stocks_to_track[:5]  # Default to first 5
    )

    if not selected_stocks:
        st.warning("Please select at least one stock to track.")
        return

    # Main content area
    st.header("📋 Portfolio Overview")

    # Create columns for the summary cards
    cols = st.columns(len(selected_stocks))

    # Store data for detailed view
    all_stock_data = {}

    # Fetch and display summary for each stock
    for i, symbol in enumerate(selected_stocks):
        with cols[i]:
            with st.spinner(f"Loading {symbol}..."):
                data = fetch_stock_data(symbol)

                if data is not None and len(data) > 0:
                    # Calculate indicators
                    data['SMA_20'] = calculate_sma(data['Close'], 20)
                    data['SMA_50'] = calculate_sma(data['Close'], 50)
                    data['RSI'] = calculate_rsi(data['Close'])

                    # Get latest values
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    price_change = ((current_price - prev_price) / prev_price) * 100

                    sma_20 = data['SMA_20'].iloc[-1]
                    sma_50 = data['SMA_50'].iloc[-1]
                    rsi = data['RSI'].iloc[-1]

                    # Get recommendation
                    rec, reason, color = get_recommendation(current_price, sma_20, sma_50, rsi)

                    # Display summary card
                    st.subheader(symbol)
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:.2f}",
                        delta=f"{price_change:+.2f}%"
                    )

                    # Show recommendation with color
                    if rec == "BUY":
                        st.success(f"🟢 {rec}")
                    elif rec == "SELL":
                        st.error(f"🔴 {rec}")
                    else:
                        st.warning(f"🟡 {rec}")

                    st.caption(reason)

                    # Store for detailed view
                    all_stock_data[symbol] = data
                else:
                    st.error(f"Could not load {symbol}")

    # Detailed view for each stock
    st.header("📈 Detailed Charts")

    for symbol, data in all_stock_data.items():
        with st.expander(f"📊 {symbol} - Detailed Analysis", expanded=False):

            # Create price chart with moving averages
            fig = go.Figure()

            # Add candlestick or line chart
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))

            # Add 20-day SMA
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='20-day SMA',
                line=dict(color='orange', width=1, dash='dash')
            ))

            # Add 50-day SMA
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='50-day SMA',
                line=dict(color='red', width=1, dash='dash')
            ))

            fig.update_layout(
                title=f"{symbol} Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # RSI Chart
            fig_rsi = go.Figure()

            fig_rsi.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))

            # Add overbought/oversold lines
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                            annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                            annotation_text="Oversold (30)")

            fig_rsi.update_layout(
                title=f"{symbol} RSI (Relative Strength Index)",
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100])
            )

            st.plotly_chart(fig_rsi, use_container_width=True)

            # Show current indicator values
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("20-day SMA", f"${data['SMA_20'].iloc[-1]:.2f}")
            with col2:
                st.metric("50-day SMA", f"${data['SMA_50'].iloc[-1]:.2f}")
            with col3:
                st.metric("RSI (14-day)", f"{data['RSI'].iloc[-1]:.1f}")

    # Educational section
    st.header("📚 Learn About the Indicators")

    with st.expander("What is SMA (Simple Moving Average)?"):
        st.markdown("""
        **Simple Moving Average (SMA)** is the average price over a specific number of days.

        - **20-day SMA**: Short-term trend indicator
        - **50-day SMA**: Medium-term trend indicator

        **How to use it:**
        - When price is **above** the SMA, it suggests an **uptrend**
        - When price is **below** the SMA, it suggests a **downtrend**
        - When price **crosses above** the SMA, it might be a buy signal
        - When price **crosses below** the SMA, it might be a sell signal
        """)

    with st.expander("What is RSI (Relative Strength Index)?"):
        st.markdown("""
        **RSI** measures how fast and how much prices are changing. It ranges from 0 to 100.

        **How to interpret:**
        - **RSI > 70**: The stock might be "overbought" (possibly due for a pullback)
        - **RSI < 30**: The stock might be "oversold" (possibly due for a bounce)
        - **RSI around 50**: Neutral, no strong momentum either way

        **Remember:** RSI works best when combined with other indicators!
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    *Built with Python, Streamlit, and yfinance* |
    *Data from Yahoo Finance* |
    *This is for educational purposes only - not financial advice!*
    """)


# This is the entry point of the script
if __name__ == "__main__":
    main()
