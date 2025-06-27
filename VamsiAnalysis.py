import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
import toml  # Keep toml if config.toml is used for non-secret app config, otherwise remove
from pathlib import Path
import os
import json
from html import escape
from io import BytesIO
import re
import numpy as np
import requests
import time
from google.api_core import exceptions  # Corrected import for ResourceExhausted

# --- NEW: Firebase/Firestore Imports ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- Define Firestore Column Names ---
FIRESTORE_TRADE_COLUMNS = [
    "FirestoreDocId",  # To store the Firestore document ID for updates
    "Datetime", "Ticker", "Status", "Analysis Type",
    "Primary Buy Price", "Primary Stop Loss", "Primary Target Price",
    "Risk-Reward Ratio",  # Calculated: (Target - Buy) / (Buy - Stop Loss)
    "Actual Entry Price", "Entry Timestamp",
    "Actual Exit Price", "Exit Timestamp",
    "Quantity",
    "Capital_Invested_Dollar",  # RENAMED: from "Capital Invested ($)"
    "Profit_Loss_Dollar",  # RENAMED: from "Profit/Loss ($)"
    "Profit_Loss_Percent",  # RENAMED: from "Profit/Loss (%)"
    "OpenAI Prompt", "OpenAI Advice", "OpenAI Buy Price", "OpenAI Stop Loss", "OpenAI Target Price",
    "Gemini Prompt", "Gemini Advice", "Gemini Buy Price", "Gemini Stop Loss", "Gemini Target Price",
    "DeepSeek Prompt", "DeepSeek Advice", "DeepSeek Buy Price", "DeepSeek Stop Loss", "DeepSeek Target Price",
    "Grok Prompt", "Grok Advice", "Grok Buy Price", "Grok Stop Loss", "Grok Target Price",  # Added Grok
    "Trade Notes"  # New column for notes
]

FIRESTORE_ANALYSIS_COLUMNS = [
    "FirestoreDocId",  # To store the Firestore document ID
    "Datetime", "Ticker", "OpenAI Prompt", "Gemini Prompt", "DeepSeek Prompt", "Grok Prompt"  # Added Grok
]

# --- Load API Keys and initialize clients from Streamlit Secrets ---
client_openai = None

# ALL API keys and Telegram details are now loaded from Streamlit secrets
openai_api_key_from_secrets = st.secrets.get("openai_api_key", "")
gemini_api_key_from_config = st.secrets.get("gemini_api_key", "")  # Renamed this var, but still loads from secrets
deepseek_api_key_from_config = st.secrets.get("deepseek_api_key", "")  # Renamed this var, but still loads from secrets
grok_api_key_from_config = st.secrets.get("grok_api_key", "")  # NEW: Get Grok API key from secrets
telegram_bot_token = st.secrets.get("telegram_bot_token", "")
telegram_chat_id = st.secrets.get("telegram_chat_id", "")

if openai_api_key_from_secrets:
    client_openai = OpenAI(api_key=openai_api_key_from_secrets)
else:
    st.warning("OpenAI API key not found in Streamlit Secrets. OpenAI advice will not be available.")

# Check if at least one AI API key is present
if not openai_api_key_from_secrets and not gemini_api_key_from_config and not deepseek_api_key_from_config and not grok_api_key_from_config:
    st.warning("No AI API keys found in Streamlit Secrets. Please provide at least one to use the AI features.")

if not telegram_bot_token or not telegram_chat_id:
    st.warning("Telegram bot_token or chat_id not found in Streamlit Secrets. Telegram notifications will not be sent.")

# --- NEW: Functions for managing batch tickers in Firestore ---
BATCH_TICKERS_DOC_ID = "current_batch_list"  # A fixed document ID for the single document storing the list
BATCH_STATE_DOC_ID = "current_batch_run_state"  # A fixed document ID for the batch run state

# Initialize Firestore
db = None
firebase_credentials_json_string = st.secrets.get("firebase_credentials_json",
                                                  "")  # Assuming secrets.toml for production
if not firebase_credentials_json_string:
    st.warning(
        "Firebase credentials (firebase_credentials_json) not found in Streamlit Secrets. Trade logs and analysis will NOT be saved to Firestore.")
else:
    try:
        # Create a temporary file for credentials
        # This approach is generally safe in Streamlit Cloud as it's isolated
        temp_cred_file = Path("temp_firebase_credentials.json")
        temp_cred_file.write_text(firebase_credentials_json_string)

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(temp_cred_file))
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        # st.info("Firestore initialized successfully!") # Commented out to reduce clutter
    except Exception as e:
        st.error(
            f"Error initializing Firebase: {e}. Check 'firebase_credentials_json' secret format, content, or service account permissions.")
        db = None  # Ensure db is None if init fails
    finally:
        # Clean up the temporary credentials file in any case
        if temp_cred_file.exists():
            temp_cred_file.unlink()

# Global variable to track last successful Firestore operation time
last_firestore_success_time = time.time()
FIRESTORE_RATE_LIMIT_DELAY = 1  # Minimum delay between general Firestore reads/writes (in seconds)
FIRESTORE_ERROR_BACKOFF_FACTOR = 2  # Factor to increase delay on error
MAX_FIRESTORE_ERROR_DELAY = 60  # Max delay for backoff


def _throttle_firestore_call():
    """Ensures a minimum delay between Firestore calls."""
    global last_firestore_success_time
    time_since_last_call = time.time() - last_firestore_success_time
    if time_since_last_call < FIRESTORE_RATE_LIMIT_DELAY:
        time.sleep(FIRESTORE_RATE_LIMIT_DELAY - time_since_last_call)
    last_firestore_success_time = time.time()


def _handle_firestore_error(e, operation_name):
    """Logs Firestore errors and implements backoff."""
    st.error(f"Firestore Error during {operation_name}: {e}")
    # Implement exponential backoff for Firestore errors
    current_delay = st.session_state.get('firestore_error_delay', FIRESTORE_RATE_LIMIT_DELAY)
    current_delay = min(current_delay * FIRESTORE_ERROR_BACKOFF_FACTOR, MAX_FIRESTORE_ERROR_DELAY)
    st.session_state.firestore_error_delay = current_delay
    st.warning(f"Retrying {operation_name} in {current_delay:.1f} seconds due to error...")
    time.sleep(current_delay)  # Block execution for the backoff period
    # Optionally, you might want to stop the continuous loop if errors persist
    if current_delay >= MAX_FIRESTORE_ERROR_DELAY and st.session_state.get('batch_running', False):
        st.session_state.batch_running = False
        st.session_state.batch_start_time = None
        st.warning("Max Firestore error backoff reached. Stopping continuous batch analysis.")
        if db:
            save_batch_run_state_to_firestore(False, None, 0,
                                              st.session_state.configured_batch_refresh_interval,
                                              st.session_state.configured_batch_delay_between_stocks)
        st.rerun()  # Rerun to reflect stopped state


def save_batch_run_state_to_firestore(is_running, start_time, total_duration_seconds, refresh_interval,
                                      delay_between_stocks):
    """Saves the current batch run state to Firestore."""
    if not db:
        return False
    try:
        _throttle_firestore_call()
        doc_ref = db.collection("app_state").document(BATCH_STATE_DOC_ID)
        state_data = {
            "is_running": is_running,
            "start_time": start_time,
            "total_duration_seconds": total_duration_seconds,
            "refresh_interval": refresh_interval,
            "delay_between_stocks": delay_between_stocks,
            "last_updated": firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(state_data)
        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
        return True
    except exceptions.ResourceExhausted as e:  # Specific error for quota issues
        _handle_firestore_error(e, "saving batch run state")
        return False
    except Exception as e:
        _handle_firestore_error(e, "saving batch run state")
        return False


def load_batch_run_state_from_firestore():
    """Loads the batch run state from Firestore."""
    if not db:
        return {
            "is_running": False,
            "start_time": None,
            "total_duration_seconds": 0,
            "refresh_interval": 180,
            "delay_between_stocks": 2
        }
    try:
        _throttle_firestore_call()
        doc_ref = db.collection("app_state").document(BATCH_STATE_DOC_ID)
        doc = doc_ref.get()
        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
        if doc.exists:
            state = doc.to_dict()
            return {
                "is_running": state.get("is_running", False),
                "start_time": state.get("start_time"),
                "total_duration_seconds": state.get("total_duration_seconds", 0),
                "refresh_interval": state.get("refresh_interval", 180),
                "delay_between_stocks": state.get("delay_between_stocks", 2)
            }
        return {
            "is_running": False,
            "start_time": None,
            "total_duration_seconds": 0,
            "refresh_interval": 180,
            "delay_between_stocks": 2
        }
    except exceptions.ResourceExhausted as e:
        _handle_firestore_error(e, "loading batch run state")
        # Return current session state values if loading fails due to quota
        return {
            "is_running": st.session_state.get("batch_running", False),
            "start_time": st.session_state.get("batch_start_time", None),
            "total_duration_seconds": st.session_state.get("batch_total_duration_seconds", 0),
            "refresh_interval": st.session_state.get("configured_batch_refresh_interval", 180),
            "delay_between_stocks": st.session_state.get("configured_batch_delay_between_stocks", 2)
        }
    except Exception as e:
        _handle_firestore_error(e, "loading batch run state")
        # Return default state if there's an error loading
        return {
            "is_running": False,
            "start_time": None,
            "total_duration_seconds": 0,
            "refresh_interval": 180,
            "delay_between_stocks": 2
        }


# --- Initialize Streamlit Session State (Guaranteed Defaults at the very start) ---
# Load persisted state from Firestore only if DB is available, otherwise use defaults
persisted_state = {}
if db:
    persisted_state = load_batch_run_state_from_firestore()

st.session_state.setdefault('batch_running', persisted_state.get("is_running", False))
# Convert Timestamp object from Firestore to float for time.time() comparisons
if persisted_state.get("start_time") is not None and hasattr(persisted_state["start_time"], 'timestamp'):
    st.session_state.setdefault('batch_start_time', persisted_state["start_time"].timestamp())
else:
    st.session_state.setdefault('batch_start_time', persisted_state.get("start_time", None))

st.session_state.setdefault('batch_total_duration_seconds', persisted_state.get("total_duration_seconds", 0))
st.session_state.setdefault('configured_batch_refresh_interval', persisted_state.get("refresh_interval", 180))
st.session_state.setdefault('configured_batch_delay_between_stocks', persisted_state.get("delay_between_stocks", 2))
st.session_state.setdefault('manual_batch_run_triggered', False)
st.session_state.setdefault('batch_run_completed_once', False)
st.session_state.setdefault('main_app_mode', "Batch Analysis")  # Default mode for radio
st.session_state.setdefault('manual_tracker_run_triggered', False)
st.session_state.setdefault('batch_tickers_input_value', "")
# Default value for the new text input for LLM selection
st.session_state.setdefault('llm_selection_for_batch_text', "OpenAI, Gemini, DeepSeek, Grok")
st.session_state.setdefault('firestore_error_delay', FIRESTORE_RATE_LIMIT_DELAY)  # For backoff strategy

# --- READ OPTIMIZATION: Initialize as empty, load on demand ---
st.session_state.setdefault('trade_log_df', pd.DataFrame(columns=FIRESTORE_TRADE_COLUMNS))
st.session_state.setdefault('batch_tickers_from_firestore_list', [])

# NEW: Toggle for detailed analysis logging
st.session_state.setdefault('enable_detailed_analysis_logging', False)

# NEW: Default date range for trade tracker
st.session_state.setdefault('filter_end_date', datetime.now().date())
# Default to 30 days ago, will be adjusted by actual data later if available
st.session_state.setdefault('filter_start_date', datetime.now().date() - timedelta(days=30))

# --- Streamlit Setup ---
st.set_page_config(page_title="📊 Stock Technical Analyzer", layout="wide")
st.title("📈 Real-Time Stock Analyzer with Buy/Sell Recommendations")


# --- Helper function to safely extract scalar numeric values ---
def get_safe_scalar(series_element):
    """
    Safely extracts a scalar numeric value from a pandas Series element.
    Handles cases where an element might be a Series of length 1 or NaN.
    """
    if isinstance(series_element, pd.Series):
        if not series_element.empty:
            val = pd.to_numeric(series_element.iloc[0], errors='coerce')
            return val if pd.notna(val) else np.nan
        else:
            return np.nan
    else:
        val = pd.to_numeric(series_element, errors='coerce')
        return val if pd.notna(val) else np.nan


# --- Fetch Data ---
@st.cache_data(ttl=3600)
def load_data(ticker, period=None, interval=None, start=None, end=None):
    """
    Loads historical stock data using yfinance.
    """
    try:
        if period and interval:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        elif start and end and interval:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
        else:
            st.error("Invalid data loading parameters provided.")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            new_columns = []
            for col_tuple in data.columns:
                standard_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                found_standard = False
                for part in col_tuple:
                    if part in standard_names:
                        new_columns.append(part)
                        found_standard = True
                        break
                if not found_standard:
                    new_columns.append(
                        '_'.join(str(p) for p in col_tuple if pd.notna(p) and p != ''))
            data.columns = new_columns

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    if isinstance(data[col], pd.DataFrame) and not data[col].empty:
                        data[col] = data[col].iloc[:, 0]
                    elif isinstance(data[col], pd.DataFrame) and data[col].empty:
                        data[col] = pd.Series(index=data.index, dtype=float)

        if data.empty:
            # st.warning(f"No data fetched for {ticker} with period='{period}', interval='{interval}'.")
            return pd.DataFrame()

        required_yfinance_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_yfinance_cols = [col for col in required_yfinance_cols if col not in data.columns]
        if missing_yfinance_cols:
            st.warning(
                f"For ticker {ticker} (Period: '{period}', Interval: '{interval}'), missing essential data columns from Yahoo Finance: {', '.join(missing_yfinance_cols)}. Technical analysis might be incomplete.")

        return data
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker} (Period: '{period}', Interval: '{interval}'): {e}")
        return pd.DataFrame()


# --- Technical Analysis ---
def apply_indicators(df):
    """
    Adds various technical indicators to the DataFrame.
    """
    if df.empty:
        return df

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"Missing essential column '{col}'. Cannot apply all technical indicators.")
            return df
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df['Close'].isnull().all():
        st.warning(
            "The 'Close' price column contains no valid numeric data. Technical indicators cannot be calculated.")
        return df

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs)).fillna(0)

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = exp1 - exp2
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)

    if 'Volume' in df.columns and (df['Volume'] > 0).any():
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    else:
        df['VWAP'] = np.nan

    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)

    df['Chikou_Span'] = df['Close'].shift(-26)

    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Corrected +DM and -DM calculation logic
    plus_dm = df['High'].diff().where(df['High'].diff() > 0, 0)
    minus_dm = df['Low'].diff().where(df['Low'].diff() < 0, 0).abs()

    df['+DM'] = np.where(plus_dm > minus_dm, plus_dm, 0)
    df['-DM'] = np.where(minus_dm > plus_dm, minus_dm, 0)

    alpha_adx = 1 / 14

    df['+DMI14'] = df['+DM'].ewm(alpha=alpha_adx, adjust=False).mean()
    df['-DMI14'] = df['-DM'].ewm(alpha=alpha_adx, adjust=False).mean()
    df['ATR14'] = df['TR'].ewm(alpha=alpha_adx, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss  # Re-calculate rs here as it might be used globally
        df['+DI14'] = (df['+DMI14'] / df['ATR14']) * 100
        df['-DI14'] = (df['+DMI14'] / df['ATR14']) * 100  # Corrected to use -DMI14
    df['+DI14'] = df['+DI14'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['-DI14'] = df['-DI14'].replace([np.inf, -np.inf], np.nan).fillna(0)  # Corrected this line to use df['-DI14']

    with np.errstate(divide='ignore', invalid='ignore'):
        df['DX'] = (abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14'])) * 100
    df['DX'] = df['DX'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['ADX_14'] = df['DX'].ewm(alpha=alpha_adx, adjust=False).mean()

    df = df.drop(
        columns=['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TR', '+DM', '-DM', '+DMI14', '-DMI14', 'ATR14', 'DX'],
        errors='ignore')

    return df


# --- OpenAI GPT-3.5 Analysis ---
def gpt_trade_advice(df, ticker, client, analysis_type="Long-Term"):
    """
    Generates trading advice using OpenAI's GPT-3.5-turbo model.
    """
    prompt = ""
    advice = "OpenAI advice not available."
    buy_price, stop_loss, target_price = np.nan, np.nan, np.nan

    if client is None:
        return prompt, advice, buy_price, stop_loss, target_price

    try:
        if df.empty:
            return "", "No data to generate OpenAI advice.", np.nan, np.nan, np.nan

        last = df.iloc[-1]

        current_close = get_safe_scalar(last.get('Close', np.nan))
        rsi = get_safe_scalar(last.get('RSI_14', np.nan))
        sma_20 = get_safe_scalar(last.get('SMA_20', np.nan))
        ema_50 = get_safe_scalar(last.get('EMA_50', np.nan))
        ema_9 = get_safe_scalar(last.get('EMA_9', np.nan))
        macd = get_safe_scalar(last.get('MACD_12_26_9', np.nan))
        macds = get_safe_scalar(last.get('MACDs_12_26_9', np.nan))
        macd_h = get_safe_scalar(last.get('MACDh_12_26_9', np.nan))
        bb_upper = get_safe_scalar(last.get('BB_Upper', np.nan))
        bb_middle = get_safe_scalar(last.get('BB_Middle', np.nan))
        bb_lower = get_safe_scalar(last.get('BB_Lower', np.nan))
        vwap = get_safe_scalar(last.get('VWAP', np.nan))

        tenkan_sen = get_safe_scalar(last.get('Tenkan_sen', np.nan))
        kijun_sen = get_safe_scalar(last.get('Kijun_sen', np.nan))
        senkou_span_a = get_safe_scalar(last.get('Senkou_Span_A', np.nan))
        senkou_span_b = get_safe_scalar(last.get('Senkou_Span_B', np.nan))
        chikou_span = get_safe_scalar(last.get('Chikou_Span', np.nan))
        adx_14 = get_safe_scalar(last.get('ADX_14', np.nan))
        plus_di_14 = get_safe_scalar(last.get('+DI14', np.nan))
        minus_di_14 = get_safe_scalar(last.get('-DI14', np.nan))

        recent_history_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5)
        recent_history_str = recent_history_df.to_string(float_format="%.2f")

        price_change = 0.0
        if len(df) > 20 and pd.notna(df['Close'].iloc[-1]) and pd.notna(df['Close'].iloc[-20]) and df['Close'].iloc[
            -20] != 0:
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100)

        volatility = 0.0
        if len(df) >= 20 and 'Close' in df.columns:
            std_dev_series = df['Close'].rolling(window=20).std()
            if not std_dev_series.empty and pd.notna(std_dev_series.iloc[-1]):
                volatility = std_dev_series.iloc[-1]

        price_vs_sma20 = 0.0
        if pd.notna(current_close) and pd.notna(sma_20) and sma_20 != 0:
            price_vs_sma20 = ((current_close - sma_20) / sma_20 * 100)

        current_trend = 'N/A'
        if pd.notna(current_close) and pd.notna(ema_50):
            current_trend = 'Bullish' if current_close > ema_50 else 'Bearish'

        macd_crossover = 'N/A'
        if pd.notna(macd) and pd.notna(macds):
            macd_crossover = 'Yes' if macd > macds else 'No'

        system_message = "You are a highly experienced and cautious stock trading analyst. Provide clear buy/sell/hold recommendations with specific, data-driven price targets and a concise rationale. Consider all provided technical indicators, including Ichimoku Cloud and ADX, for a holistic view."
        prompt_intro = f"""Your goal is to provide **highly accurate, data-driven, and realistic trade recommendations** for {ticker}.
Carefully analyze the provided technical indicators and recent historical price data.
"""
        if analysis_type == "Intraday Analysis":
            system_message = "You are an expert intraday stock trading analyst. Focus on short-term price action, volume, and intraday indicators to provide precise, actionable buy/sell/hold recommendations for quick trades. Emphasize entry, stop, and target levels suitable for day trading. Utilize all available indicators, especially VWAP, EMA(9), and relevant Ichimoku signals for intraday momentum."
            prompt_intro += """**This is an INTRA-DAY analysis.** Focus on short-term price movements and volatility.
Consider indicators like VWAP and EMA(9) which are critical for intraday decisions.
"""
        else:
            prompt_intro += """**This is a LONG-TERM analysis.** Focus on broader trends, momentum, and major support/resistance levels.
"""

        price_guidance = """
If you identify a clear, high-conviction trading opportunity with strong momentum and volume, then provide the following numerical values, clearly labeled:
-   **Realistic Buy Price:** [numerical value]
-   **Strict Stop Loss:** [numerical value]
-   **Realistic Target Price:** [numerical value]
Otherwise, if a clear opportunity or strong signals for these specific prices are not present, simply provide your primary recommendation (e.g., "Hold", "Wait", "Observe") and rationale, omitting the specific price points.
"""

        prompt_data = f"""
1.  **Current Recommendation (Buy/Sell/Hold/Wait):** Clearly state your primary recommendation.
{price_guidance}
5.  **Concise Rationale:** Briefly explain the technical and price action reasons supporting your recommendation. Justify your recommendation based on the data.

Here's the current and recent data for {ticker}:

**Last Known Values:**
-   **Last Close Price:** {current_close:.2f}
-   **RSI (14-day):** {rsi:.2f}
-   **SMA (20-day):** {sma_20:.2f}
-   **EMA (50-day):** {ema_50:.2f}
"""
        if analysis_type == "Intraday Analysis":
            prompt_data += f"""-   **EMA (9-day):** {ema_9:.2f}
-   **VWAP:** {vwap:.2f}
"""

        prompt_data += f"""-   **MACD (12,26,9):** {macd:.2f}
-   **MACD Signal (12,26,9):** {macds:.2f}
-   **MACD Histogram:** {macd_h:.2f}
-   **Bollinger Bands (20-day):** Upper: {bb_upper:.2f}, Middle: {bb_middle:.2f}, Lower: {bb_lower:.2f}
-   **20-day Price Change (%):** {price_change:.2f}%
-   **Volatility (20d Std Dev):** {volatility:.2f}
-   **Price vs SMA_20 (%):** {price_vs_sma20:.2f}%
-   **Current Trend (based on Close vs EMA_50):** {current_trend}
-   **MACD Crossover (MACD > Signal):** {macd_crossover}

**New Indicators:**
-   **Ichimoku Tenkan-sen:** {tenkan_sen:.2f}
-   **Ichimoku Kijun-sen:** {kijun_sen:.2f}
-   **Ichimoku Senkou Span A:** {senkou_span_a:.2f}
-   **Ichimoku Senkou Span B:** {senkou_span_b:.2f}
-   **Ichimoku Chikou Span:** {chikou_span:.2f}
-   **ADX (14-day):** {adx_14:.2f}
-   **+DI (14-day):** {plus_di_14:.2f}
-   **-DI (14-day):** {minus_di_14:.2f}

**Recent 5-Day Historical Data (Open, High, Low, Close, Volume):**
{recent_history_str}
"""
        prompt = prompt_intro + prompt_data

        with open("latest_gpt_prompt_openai.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        advice = response.choices[0].message.content

        with open("latest_gpt_response_openai.txt", "w", encoding="utf-8") as f:
            f.write(advice)

        buy_price = extract_price_from_text(advice, r'(buy\s*price|entry\s*price|buy\s*at)')
        stop_loss = extract_price_from_text(advice, r'(stop\s*loss|sl|cut\s*loss\s*at)')
        target_price = extract_price_from_text(advice, r'(target\s*price|sell\s*price|take\s*profit)')

        return prompt, advice, buy_price, stop_loss, target_price
    except Exception as e:
        with open("gpt_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Ticker: {ticker}, OpenAI Error: {e}\n")
        return prompt, f"An error occurred while getting OpenAI advice: {e}. Check logs.", np.nan, np.nan, np.nan


# --- Google Gemini Analysis ---
def gemini_trade_advice(df, ticker, api_key, analysis_type="Long-Term"):
    """
    Generates trading advice using Google's Gemini-2.0-Flash model.
    """
    prompt = ""
    advice = "Gemini advice not available."
    buy_price, stop_loss, target_price = np.nan, np.nan, np.nan

    if not api_key:
        return prompt, advice, buy_price, stop_loss, target_price

    try:
        if df.empty:
            return "", "No data to generate Gemini advice.", np.nan, np.nan, np.nan

        last = df.iloc[-1]

        current_close = get_safe_scalar(last.get('Close', np.nan))
        rsi = get_safe_scalar(last.get('RSI_14', np.nan))
        sma_20 = get_safe_scalar(last.get('SMA_20', np.nan))
        ema_50 = get_safe_scalar(last.get('EMA_50', np.nan))
        ema_9 = get_safe_scalar(last.get('EMA_9', np.nan))
        macd = get_safe_scalar(last.get('MACD_12_26_9', np.nan))
        macds = get_safe_scalar(last.get('MACDs_12_26_9', np.nan))
        macd_h = get_safe_scalar(last.get('MACDh_12_26_9', np.nan))
        bb_upper = get_safe_scalar(last.get('BB_Upper', np.nan))
        bb_middle = get_safe_scalar(last.get('BB_Middle', np.nan))
        bb_lower = get_safe_scalar(last.get('BB_Lower', np.nan))
        vwap = get_safe_scalar(last.get('VWAP', np.nan))

        tenkan_sen = get_safe_scalar(last.get('Tenkan_sen', np.nan))
        kijun_sen = get_safe_scalar(last.get('Kijun_sen', np.nan))
        senkou_span_a = get_safe_scalar(last.get('Senkou_Span_A', np.nan))
        senkou_span_b = get_safe_scalar(last.get('Senkou_Span_B', np.nan))
        chikou_span = get_safe_scalar(last.get('Chikou_Span', np.nan))
        adx_14 = get_safe_scalar(last.get('ADX_14', np.nan))
        plus_di_14 = get_safe_scalar(last.get('+DI14', np.nan))
        minus_di_14 = get_safe_scalar(last.get('-DI14', np.nan))

        recent_history_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5)
        recent_history_str = recent_history_df.to_string(float_format="%.2f")

        price_change = 0.0
        if len(df) > 20 and pd.notna(df['Close'].iloc[-1]) and pd.notna(df['Close'].iloc[-20]) and df['Close'].iloc[
            -20] != 0:
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100)

        volatility = 0.0
        if len(df) >= 20 and 'Close' in df.columns:
            std_dev_series = df['Close'].rolling(window=20).std()
            if not std_dev_series.empty and pd.notna(std_dev_series.iloc[-1]):
                volatility = std_dev_series.iloc[-1]

        price_vs_sma20 = 0.0
        if pd.notna(current_close) and pd.notna(sma_20) and sma_20 != 0:
            price_vs_sma20 = ((current_close - sma_20) / sma_20 * 100)

        current_trend = 'N/A'
        if pd.notna(current_close) and pd.notna(ema_50):
            current_trend = 'Bullish' if current_close > ema_50 else 'Bearish'

        macd_crossover = 'N/A'
        if pd.notna(macd) and pd.notna(macds):
            macd_crossover = 'Yes' if macd > macds else 'No'

        system_message = "You are a highly experienced and cautious stock trading assistant. Provide clear buy/sell/hold recommendations with specific, data-driven price targets and a concise rationale. Consider all provided technical indicators, including Ichimoku Cloud and ADX, for a holistic view."
        prompt_intro = f"""Your goal is to provide **highly accurate, data-driven, and realistic trade recommendations** for {ticker}.
Carefully analyze the provided technical indicators and recent historical price data.
"""
        if analysis_type == "Intraday Analysis":
            system_message = "You are an expert intraday stock trading analyst. Focus on short-term price action, volume, and intraday indicators to provide precise, actionable buy/sell/hold recommendations for quick trades. Emphasize entry, stop, and target levels suitable for day trading. Utilize all available indicators, especially VWAP, EMA(9), and relevant Ichimoku signals for intraday momentum."
            prompt_intro += """**This is an INTRA-DAY analysis.** Focus on short-term price movements and volatility.
Consider indicators like VWAP and EMA(9) which are critical for intraday decisions.
"""
        else:
            prompt_intro += """**This is a LONG-TERM analysis.** Focus on broader trends, momentum, and major support/resistance levels."""

        price_guidance = """
If you identify a clear, high-conviction trading opportunity with strong momentum and volume, then provide the following numerical values, clearly labeled:
-   **Realistic Buy Price:** [numerical value]
-   **Strict Stop Loss:** [numerical value]
-   **Realistic Target Price:** [numerical value]
Otherwise, if a clear opportunity or strong signals for these specific prices are not present, simply provide your primary recommendation (e.g., "Hold", "Wait", "Observe") and rationale, omitting the specific price points.
"""

        prompt_data = f"""
1.  **Current Recommendation (Buy/Sell/Hold/Wait):** Clearly state your primary recommendation.
{price_guidance}
5.  **Concise Rationale:** Briefly explain the technical and price action reasons supporting your recommendation. Justify your recommendation based on the data.

Here's the current and recent data for {ticker}:

**Last Known Values:**
-   **Last Close Price:** {current_close:.2f}
-   **RSI (14-day):** {rsi:.2f}
-   **SMA (20-day):** {sma_20:.2f}
-   **EMA (50-day):** {ema_50:.2f}
"""
        if analysis_type == "Intraday Analysis":
            prompt_data += f"""-   **EMA (9-day):** {ema_9:.2f}
-   **VWAP:** {vwap:.2f}
"""

        prompt_data += f"""-   **MACD (12,26,9):** {macd:.2f}
-   **MACD Signal (12,26,9):** {macds:.2f}
-   **MACD Histogram:** {macd_h:.2f}
-   **Bollinger Bands (20-day):** Upper: {bb_upper:.2f}, Middle: {bb_middle:.2f}, Lower: {bb_lower:.2f}
-   **20-day Price Change (%):** {price_change:.2f}%
-   **Volatility (20d Std Dev):** {volatility:.2f}
-   **Price vs SMA_20 (%):** {price_vs_sma20:.2f}%
-   **Current Trend (based on Close vs EMA_50):** {current_trend}
-   **MACD Crossover (MACD > Signal):** {macd_crossover}

**New Indicators:**
-   **Ichimoku Tenkan-sen:** {tenkan_sen:.2f}
-   **Ichimoku Kijun-sen:** {kijun_sen:.2f}
-   **Ichimoku Senkou Span A:** {senkou_span_a:.2f}
-   **Ichimoku Senkou Span B:** {senkou_span_b:.2f}
-   **Ichimoku Chikou Span:** {chikou_span:.2f}
-   **ADX (14-day):** {adx_14:.2f}
-   **+DI (14-day):** {plus_di_14:.2f}
-   **-DI (14-day):** {minus_di_14:.2f}

**Recent 5-Day Historical Data (Open, High, Low, Close, Volume):**
{recent_history_str}
"""
        prompt = prompt_intro + prompt_data

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
        }

        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
                result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
                len(result['candidates'][0]['content'].get('parts')) > 0:
            advice = result['candidates'][0]['content']['parts'][0]['text']
        else:
            advice = "Gemini API did not return valid content or candidates."
            st.warning(f"Gemini API response structure unexpected for {ticker}: {result}")

        with open("latest_gpt_response_gemini.txt", "w", encoding="utf-8") as f:
            f.write(advice)

        buy_price = extract_price_from_text(advice, r'(buy\s*price|entry\s*price|buy\s*at)')
        stop_loss = extract_price_from_text(advice, r'(stop\s*loss|sl|cut\s*loss\s*at)')
        target_price = extract_price_from_text(advice, r'(target\s*price|sell\s*price|take\s*profit)')

        return prompt, advice, buy_price, stop_loss, target_price

    except requests.exceptions.RequestException as req_e:
        with open("gpt_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Ticker: {ticker}, Gemini API Request Error: {req_e}\n")
        return prompt, f"Gemini API request error: {req_e}. Check API key and network.", np.nan, np.nan, np.nan
    except Exception as e:
        with open("gpt_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Ticker: {ticker}, Gemini Unexpected Error: {e}.\n")
        return prompt, f"An unexpected error occurred getting Gemini advice: {e}.", np.nan, np.nan, np.nan


# --- DeepSeek R1 Analysis (NEW FUNCTION) ---
def deepseek_trade_advice(df, ticker, api_key, analysis_type="Long-Term"):
    """
    Generates trading advice using DeepSeek R1 model.
    """
    prompt = ""
    advice = "DeepSeek advice not available."
    buy_price, stop_loss, target_price = np.nan, np.nan, np.nan

    if not api_key:
        return prompt, advice, buy_price, stop_loss, target_price

    try:
        if df.empty:
            return "", "No data to generate DeepSeek advice.", np.nan, np.nan, np.nan

        last = df.iloc[-1]

        current_close = get_safe_scalar(last.get('Close', np.nan))
        rsi = get_safe_scalar(last.get('RSI_14', np.nan))
        sma_20 = get_safe_scalar(last.get('SMA_20', np.nan))
        ema_50 = get_safe_scalar(last.get('EMA_50', np.nan))
        ema_9 = get_safe_scalar(last.get('EMA_9', np.nan))
        macd = get_safe_scalar(last.get('MACD_12_26_9', np.nan))
        macds = get_safe_scalar(last.get('MACDs_12_26_9', np.nan))
        macd_h = get_safe_scalar(last.get('MACDh_12_26_9', np.nan))
        bb_upper = get_safe_scalar(last.get('BB_Upper', np.nan))
        bb_middle = get_safe_scalar(last.get('BB_Middle', np.nan))
        bb_lower = get_safe_scalar(last.get('BB_Lower', np.nan))
        vwap = get_safe_scalar(last.get('VWAP', np.nan))

        tenkan_sen = get_safe_scalar(last.get('Tenkan_sen', np.nan))
        kijun_sen = get_safe_scalar(last.get('Kijun_sen', np.nan))
        senkou_span_a = get_safe_scalar(last.get('Senkou_Span_A', np.nan))
        senkou_span_b = get_safe_scalar(last.get('Senkou_Span_B', np.nan))
        chikou_span = get_safe_scalar(last.get('Chikou_Span', np.nan))
        adx_14 = get_safe_scalar(last.get('ADX_14', np.nan))
        plus_di_14 = get_safe_scalar(last.get('+DI14', np.nan))
        minus_di_14 = get_safe_scalar(last.get('-DI14', np.nan))

        recent_history_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5)
        recent_history_str = recent_history_df.to_string(float_format="%.2f")

        price_change = 0.0
        if len(df) > 20 and pd.notna(df['Close'].iloc[-1]) and pd.notna(df['Close'].iloc[-20]) and df['Close'].iloc[
            -20] != 0:
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100)

        volatility = 0.0
        if len(df) >= 20 and 'Close' in df.columns:
            std_dev_series = df['Close'].rolling(window=20).std()
            if not std_dev_series.empty and pd.notna(std_dev_series.iloc[-1]):
                volatility = std_dev_series.iloc[-1]

        price_vs_sma20 = 0.0
        if pd.notna(current_close) and pd.notna(sma_20) and sma_20 != 0:
            price_vs_sma20 = ((current_close - sma_20) / sma_20 * 100)

        current_trend = 'N/A'
        if pd.notna(current_close) and pd.notna(ema_50):
            current_trend = 'Bullish' if current_close > ema_50 else 'Bearish'

        macd_crossover = 'N/A'
        if pd.notna(macd) and pd.notna(macds):
            macd_crossover = 'Yes' if macd > macds else 'No'

        system_message = "You are a highly experienced and cautious stock trading analyst. Provide clear buy/sell/hold recommendations with specific, data-driven price targets and a concise rationale. Consider all provided technical indicators, including Ichimoku Cloud and ADX, for a holistic view."
        prompt_intro = f"""Your goal is to provide **highly accurate, data-driven, and realistic trade recommendations** for {ticker}.
Carefully analyze the provided technical indicators and recent historical price data.
"""
        if analysis_type == "Intraday Analysis":
            system_message = "You are an expert intraday stock trading analyst. Focus on short-term price action, volume, and intraday indicators to provide precise, actionable buy/sell/hold recommendations for quick trades. Emphasize entry, stop, and target levels suitable for day trading. Utilize all available indicators, especially VWAP, EMA(9), and relevant Ichimoku signals for intraday momentum."
            prompt_intro += """**This is an INTRA-DAY analysis.** Focus on short-term price movements and volatility.
Consider indicators like VWAP and EMA(9) which are critical for intraday decisions.
"""
        else:
            prompt_intro += """**This is a LONG-TERM analysis.** Focus on broader trends, momentum, and major support/resistance levels."""

        price_guidance = """
If you identify a clear, high-conviction trading opportunity with strong momentum and volume, then provide the following numerical values, clearly labeled:
-   **Realistic Buy Price:** [numerical value]
-   **Strict Stop Loss:** [numerical value]
-   **Realistic Target Price:** [numerical value]
Otherwise, if a clear opportunity or strong signals for these specific prices are not present, simply provide your primary recommendation (e.g., "Hold", "Wait", "Observe") and rationale, omitting the specific price points.
"""

        prompt_data = f"""
1.  **Current Recommendation (Buy/Sell/Hold/Wait):** Clearly state your primary recommendation.
{price_guidance}
5.  **Concise Rationale:** Briefly explain the technical and price action reasons supporting your recommendation. Justify your recommendation based on the data.

Here's the current and recent data for {ticker}:

**Last Known Values:**
-   **Last Close Price:** {current_close:.2f}
-   **RSI (14-day):** {rsi:.2f}
-   **SMA (20-day):** {sma_20:.2f}
-   **EMA (50-day):** {ema_50:.2f}
"""
        if analysis_type == "Intraday Analysis":
            prompt_data += f"""-   **EMA (9-day):** {ema_9:.2f}
-   **VWAP:** {vwap:.2f}
"""

        prompt_data += f"""-   **MACD (12,26,9):** {macd:.2f}
-   **MACD Signal (12,26,9):** {macds:.2f}
-   **MACD Histogram:** {macd_h:.2f}
-   **Bollinger Bands (20-day):** Upper: {bb_upper:.2f}, Middle: {bb_middle:.2f}, Lower: {bb_lower:.2f}
-   **20-day Price Change (%):** {price_change:.2f}%
-   **Volatility (20d Std Dev):** {volatility:.2f}
-   **Price vs SMA_20 (%):** {price_vs_sma20:.2f}%
-   **Current Trend (based on Close vs EMA_50):** {current_trend}
-   **MACD Crossover (MACD > Signal):** {macd_crossover}

**New Indicators:**
-   **Ichimoku Tenkan-sen:** {tenkan_sen:.2f}
-   **Ichimoku Kijun-sen:** {kijun_sen:.2f}
-   **Ichimoku Senkou Span A:** {senkou_span_a:.2f}
-   **Ichimoku Senkou Span B:** {senkou_span_b:.2f}
-   **Ichimoku Chikou Span:** {chikou_span:.2f}
-   **ADX (14-day):** {adx_14:.2f}
-   **+DI (14-day):** {plus_di_14:.2f}
-   **-DI (14-day):** {minus_di_14:.2f}

**Recent 5-Day Historical Data (Open, High, Low, Close, Volume):**
{recent_history_str}
"""
        prompt = prompt_intro + prompt_data

        api_url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get('choices') and len(result['choices']) > 0 and \
                result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
            advice = result['choices'][0]['message']['content']
        else:
            advice = "DeepSeek API did not return valid content or choices."
            st.warning(f"DeepSeek API response structure unexpected for {ticker}: {result}")

        with open("latest_gpt_response_deepseek.txt", "w", encoding="utf-8") as f:
            f.write(advice)

        buy_price = extract_price_from_text(advice, r'(buy\s*price|entry\s*price|buy\s*at)')
        stop_loss = extract_price_from_text(advice, r'(stop\s*loss|sl|cut\s*loss\s*at)')
        target_price = extract_price_from_text(advice, r'(target\s*price|sell\s*price|take\s*profit)')

        return prompt, advice, buy_price, stop_loss, target_price

    except requests.exceptions.RequestException as req_e:
        with open("gpt_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Ticker: {ticker}, DeepSeek API Request Error: {req_e}\n")
        return prompt, f"DeepSeek API request error: {req_e}. Check API key and network.", np.nan, np.nan, np.nan
    except Exception as e:
        with open("gpt_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Ticker: {ticker}, DeepSeek Unexpected Error: {e}.\n")
        return prompt, f"An unexpected error occurred getting DeepSeek advice: {e}.", np.nan, np.nan, np.nan


# --- Grok AI Analysis (NEW FUNCTION) ---
def grok_trade_advice(df, ticker, api_key, analysis_type="Long-Term"):
    """
    Generates trading advice using Grok model.
    NOTE: Replace with actual Grok API endpoint and model name when available.
    This is a placeholder implementation.
    """
    prompt = ""
    advice = "Grok advice not available."
    buy_price, stop_loss, target_price = np.nan, np.nan, np.nan

    if not api_key:
        return prompt, advice, buy_price, stop_loss, target_price

    try:
        if df.empty:
            return "", "No data to generate Grok advice.", np.nan, np.nan, np.nan

        last = df.iloc[-1]

        current_close = get_safe_scalar(last.get('Close', np.nan))
        rsi = get_safe_scalar(last.get('RSI_14', np.nan))
        sma_20 = get_safe_scalar(last.get('SMA_20', np.nan))
        ema_50 = get_safe_scalar(last.get('EMA_50', np.nan))
        ema_9 = get_safe_scalar(last.get('EMA_9', np.nan))
        macd = get_safe_scalar(last.get('MACD_12_26_9', np.nan))
        macds = get_safe_scalar(last.get('MACDs_12_26_9', np.nan))
        macd_h = get_safe_scalar(last.get('MACDh_12_26_9', np.nan))
        bb_upper = get_safe_scalar(last.get('BB_Upper', np.nan))
        bb_middle = get_safe_scalar(last.get('BB_Middle', np.nan))
        bb_lower = get_safe_scalar(last.get('BB_Lower', np.nan))
        vwap = get_safe_scalar(last.get('VWAP', np.nan))

        tenkan_sen = get_safe_scalar(last.get('Tenkan_sen', np.nan))
        kijun_sen = get_safe_scalar(last.get('Kijun_sen', np.nan))
        senkou_span_a = get_safe_scalar(last.get('Senkou_Span_A', np.nan))
        senkou_span_b = get_safe_scalar(last.get('Senkou_Span_B', np.nan))
        chikou_span = get_safe_scalar(last.get('Chikou_Span', np.nan))
        adx_14 = get_safe_scalar(last.get('ADX_14', np.nan))
        plus_di_14 = get_safe_scalar(last.get('+DI14', np.nan))
        minus_di_14 = get_safe_scalar(last.get('-DI14', np.nan))

        recent_history_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5)
        recent_history_str = recent_history_df.to_string(float_format="%.2f")

        price_change = 0.0
        if len(df) > 20 and pd.notna(df['Close'].iloc[-1]) and pd.notna(df['Close'].iloc[-20]) and df['Close'].iloc[
            -20] != 0:
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100)

        volatility = 0.0
        if len(df) >= 20 and 'Close' in df.columns:
            std_dev_series = df['Close'].rolling(window=20).std()
            if not std_dev_series.empty and pd.notna(std_dev_series.iloc[-1]):
                volatility = std_dev_series.iloc[-1]

        price_vs_sma20 = 0.0
        if pd.notna(current_close) and pd.notna(sma_20) and sma_20 != 0:
            price_vs_sma20 = ((current_close - sma_20) / sma_20 * 100)

        current_trend = 'N/A'
        if pd.notna(current_close) and pd.notna(ema_50):
            current_trend = 'Bullish' if current_close > ema_50 else 'Bearish'

        macd_crossover = 'N/A'
        if pd.notna(macd) and pd.notna(macds):
            macd_crossover = 'Yes' if macd > macds else 'No'

        system_message = "You are a highly experienced and cautious stock trading analyst. Provide clear buy/sell/hold recommendations with specific, data-driven price targets and a concise rationale. Consider all provided technical indicators, including Ichimoku Cloud and ADX, for a holistic view."
        prompt_intro = f"""Your goal is to provide **highly accurate, data-driven, and realistic trade recommendations** for {ticker}.
Carefully analyze the provided technical indicators and recent historical price data.
"""
        if analysis_type == "Intraday Analysis":
            system_message = "You are an expert intraday stock trading analyst. Focus on short-term price action, volume, and intraday indicators to provide precise, actionable buy/sell/hold recommendations for quick trades. Emphasize entry, stop, and target levels suitable for day trading. Utilize all available indicators, especially VWAP, EMA(9), and relevant Ichimoku signals for intraday momentum."
            prompt_intro += """**This is an INTRA-DAY analysis.** Focus on short-term price movements and volatility.
Consider indicators like VWAP and EMA(9) which are critical for intraday decisions.
"""
        else:
            prompt_intro += """**This is a LONG-TERM analysis.** Focus on broader trends, momentum, and major support/resistance levels."""

        price_guidance = """
If you identify a clear, high-conviction trading opportunity with strong momentum and volume, then provide the following numerical values, clearly labeled:
-   **Realistic Buy Price:** [numerical value]
-   **Strict Stop Loss:** [numerical value]
-   **Realistic Target Price:** [numerical value]
Otherwise, if a clear opportunity or strong signals for these specific prices are not present, simply provide your primary recommendation (e.g., "Hold", "Wait", "Observe") and rationale, omitting the specific price points.
"""

        prompt_data = f"""
1.  **Current Recommendation (Buy/Sell/Hold/Wait):** Clearly state your primary recommendation.
{price_guidance}
5.  **Concise Rationale:** Briefly explain the technical and price action reasons supporting your recommendation. Justify your recommendation based on the data.

Here's the current and recent data for {ticker}:

**Last Known Values:**
-   **Last Close Price:** {current_close:.2f}
-   **RSI (14-day):** {rsi:.2f}
-   **SMA (20-day):** {sma_20:.2f}
-   **EMA (50-day):** {ema_50:.2f}
"""
        if analysis_type == "Intraday Analysis":
            prompt_data += f"""-   **EMA (9-day):** {ema_9:.2f}
-   **VWAP:** {vwap:.2f}
"""

        prompt_data += f"""-   **MACD (12,26,9):** {macd:.2f}
-   **MACD Signal (12,26,9):** {macds:.2f}
-   **MACD Histogram:** {macd_h:.2f}
-   **Bollinger Bands (20-day):** Upper: {bb_upper:.2f}, Middle: {bb_middle:.2f}, Lower: {bb_lower:.2f}
-   **20-day Price Change (%):** {price_change:.2f}%
-   **Volatility (20d Std Dev):** {volatility:.2f}
-   **Price vs SMA_20 (%):** {price_vs_sma20:.2f}%
-   **Current Trend (based on Close vs EMA_50):** {current_trend}
-   **MACD Crossover (MACD > Signal):** {macd_crossover}

**New Indicators:**
-   **Ichimoku Tenkan-sen:** {tenkan_sen:.2f}
-   **Ichimoku Kijun-sen:** {kijun_sen:.2f}
-   **Ichimoku Senkou Span A:** {senkou_span_a:.2f}
-   **Ichimoku Senkou Span B:** {senkou_span_b:.2f}
-   **Ichimoku Chikou Span:** {chikou_span:.2f}
-   **ADX (14-day):** {adx_14:.2f}
-   **+DI (14-day):** {plus_di_14:.2f}
-   **-DI (14-day):** {minus_di_14:.2f}

**Recent 5-Day Historical Data (Open, High, Low, Close, Volume):**
{recent_history_str}
"""
        prompt = prompt_intro + prompt_data

        # Placeholder for actual Grok API call.
        # This assumes a similar structure to DeepSeek or OpenAI.
        # Replace with the actual Grok API endpoint and headers/payload
        api_url = "https://api.grok.com/v1/chat/completions"  # Hypothetical endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "grok-1",  # Hypothetical model name
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get('choices') and len(result['choices']) > 0 and \
                result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
            advice = result['choices'][0]['message']['content']
        else:
            advice = "Grok API did not return valid content or choices."
            st.warning(f"Grok API response structure unexpected for {ticker}: {result}")

        with open("latest_gpt_response_grok.txt", "w", encoding="utf-8") as f:
            f.write(advice)

        buy_price = extract_price_from_text(advice, r'(buy\s*price|entry\s*price|buy\s*at)')
        stop_loss = extract_price_from_text(advice, r'(stop\s*loss|sl|cut\s*loss\s*at)')
        target_price = extract_price_from_text(advice, r'(target\s*price|sell\s*price|take\s*profit)')

        return prompt, advice, buy_price, stop_loss, target_price

    except requests.exceptions.RequestException as req_e:
        with open("gpt_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Ticker: {ticker}, Grok API Request Error: {req_e}\n")
        return prompt, f"Grok API request error: {req_e}. Check API key and network.", np.nan, np.nan, np.nan
    except Exception as e:
        with open("gpt_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Ticker: {ticker}, Grok Unexpected Error: {e}.\n")
        return prompt, f"An unexpected error occurred getting Grok advice: {e}.", np.nan, np.nan, np.nan


# --- Helper to extract numbers from advice for trade tracker ---
def extract_price_from_text(text, keyword_regex, default_val=np.nan):
    """
    Extracts a numeric price from a given text based on keywords.
    Uses regex to find numbers potentially near the keywords.
    Prioritizes numbers that look like actual prices (decimals, or larger integers).
    Returns np.nan if not found.
    """
    price_pattern = r'(?:\$|€|£)?\s*(\d+\.\d{1,}|\d{2,}(?!\.\d{0,1}))'

    combined_regex = rf'{keyword_regex}\s*[:]*\s*{price_pattern}'

    match = re.search(combined_regex, text, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            if val > 0.01:
                return val
        except ValueError:
            pass

    lines = [l for l in text.splitlines() if re.search(keyword_regex, l, re.IGNORECASE)]
    for line in lines:
        nums = [float(s) for s in re.findall(price_pattern, line) if float(s) > 0.01]
        if nums:
            return nums[0]

    return default_val


# --- Telegram Notification Function ---
def send_telegram_message(message):
    """Sends a message to a Telegram chat."""
    if not telegram_bot_token or not telegram_chat_id:
        st.warning("Telegram bot token or chat ID is not configured. Message not sent.")
        return

    telegram_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(telegram_url, json=payload)
        response.raise_for_status()
        if response.status_code == 200:
            st.success("Telegram notification sent!")
        else:
            st.error(f"Failed to send Telegram message. Status Code: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending Telegram message: {e}")


# --- Function to fetch all trade logs from Firestore ---
def fetch_trade_logs_from_firestore_db():
    """Fetches all trade logs from Firestore and returns them as a Pandas DataFrame."""
    if not db:
        st.warning("Firestore not initialized, cannot fetch trade logs.")
        return pd.DataFrame(columns=FIRESTORE_TRADE_COLUMNS)

    docs = []
    try:
        _throttle_firestore_call()
        for doc in db.collection("trade_logs").stream():
            doc_data = doc.to_dict()
            # Convert Firestore Timestamp to Python datetime, then to timezone-naive
            for col_name in ['Datetime', 'Entry Timestamp', 'Exit Timestamp']:  # Changed 'col' to 'col_name'
                if col_name in doc_data and hasattr(doc_data[col_name], 'astimezone'):
                    doc_data[col_name] = doc_data[col_name].astimezone(datetime.now().astimezone().tzinfo).replace(
                        tzinfo=None)
                elif col_name in doc_data and isinstance(doc_data[col_name], datetime) and doc_data[
                    col_name].tzinfo is not None:
                    doc_data[col_name] = doc_data[col_name].replace(tzinfo=None)
                elif col_name in doc_data and not isinstance(doc_data[col_name],
                                                             datetime):  # Handle cases where it's not a datetime object
                    doc_data[col_name] = pd.NaT  # Not a Time, equivalent to NaN for datetime

            doc_data['FirestoreDocId'] = doc.id
            docs.append(doc_data)
        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success

        df = pd.DataFrame(docs)
        if not df.empty:
            # Ensure numeric columns are actually numeric, coercing errors
            numeric_cols_to_convert = [
                'Primary Buy Price', 'Primary Stop Loss', 'Primary Target Price', 'Risk-Reward Ratio',
                'Actual Entry Price', 'Actual Exit Price', 'Quantity',
                "Capital_Invested_Dollar",  # RENAMED
                "Profit_Loss_Dollar",  # RENAMED
                "Profit_Loss_Percent",  # RENAMED
                'OpenAI Buy Price', 'OpenAI Stop Loss', 'OpenAI Target Price',
                'Gemini Buy Price', 'Gemini Stop Loss', 'Gemini Target Price',
                'DeepSeek Buy Price', 'DeepSeek Stop Loss', 'DeepSeek Target Price',
                "Grok Buy Price", "Grok Stop Loss", "Grok Target Price"  # Added Grok
            ]
            for col_name in numeric_cols_to_convert:  # Changed 'col' to 'col_name'
                if col_name in df.columns:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                else:
                    df[col_name] = np.nan  # Add missing numeric columns as NaN

            # Ensure all columns from FIRESTORE_TRADE_COLUMNS are present, filling missing with NaN
            missing_cols = [col for col in FIRESTORE_TRADE_COLUMNS if col not in df.columns]
            for col in missing_cols:
                df[col] = np.nan  # Use np.nan for all missing columns for consistency

            df = df[FIRESTORE_TRADE_COLUMNS]  # Reorder and select only defined columns
            df = df.sort_values(by='Datetime', ascending=False).reset_index(drop=True)
        return df
    except exceptions.ResourceExhausted as e:
        _handle_firestore_error(e, "fetching trade logs")
        return pd.DataFrame(columns=FIRESTORE_TRADE_COLUMNS)  # Return empty DF on error
    except Exception as e:
        _handle_firestore_error(e, "fetching trade logs")
        return pd.DataFrame(columns=FIRESTORE_TRADE_COLUMNS)  # Return empty DF on error


# --- Function to fetch all analysis logs from Firestore ---
def fetch_all_analysis_from_firestore_for_download():  # Renamed to emphasize on-demand
    """Fetches all analysis logs from Firestore and returns them as a Pandas DataFrame."""
    if not db:
        st.warning("Firestore not initialized, cannot fetch all analysis logs.")
        return pd.DataFrame(columns=FIRESTORE_ANALYSIS_COLUMNS)

    docs = []
    try:
        _throttle_firestore_call()
        for doc in db.collection("all_analysis_logs").stream():
            doc_data = doc.to_dict()
            # FIX: Corrected reference from 'col' to 'Datetime'
            if 'Datetime' in doc_data and hasattr(doc_data['Datetime'], 'astimezone'):
                doc_data['Datetime'] = doc_data['Datetime'].astimezone(datetime.now().astimezone().tzinfo).replace(
                    tzinfo=None)
            elif 'Datetime' in doc_data and isinstance(doc_data['Datetime'], datetime) and doc_data[
                'Datetime'].tzinfo is not None:
                doc_data['Datetime'] = doc_data['Datetime'].replace(tzinfo=None)
            elif 'Datetime' in doc_data and not isinstance(doc_data['Datetime'],
                                                           datetime):  # FIX: Corrected reference from 'col' to 'Datetime'
                doc_data['Datetime'] = pd.NaT  # Not a Time, equivalent to NaN for datetime
            else:  # If 'Datetime' is not in doc_data at all
                doc_data['Datetime'] = pd.NaT

            doc_data['FirestoreDocId'] = doc.id
            docs.append(doc_data)
        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success

        df = pd.DataFrame(docs)
        if not df.empty:
            missing_cols = [col for col in FIRESTORE_ANALYSIS_COLUMNS if col not in df.columns]
            for col in missing_cols:
                df[col] = np.nan
            df = df[FIRESTORE_ANALYSIS_COLUMNS]
            df = df.sort_values(by='Datetime', ascending=False).reset_index(drop=True)
        return df
    except exceptions.ResourceExhausted as e:
        _handle_firestore_error(e, "fetching all analysis logs")
        return pd.DataFrame(columns=FIRESTORE_ANALYSIS_COLUMNS)
    except Exception as e:
        _handle_firestore_error(e, "fetching all analysis logs")
        return pd.DataFrame(columns=FIRESTORE_ANALYSIS_COLUMNS)


# --- Function to log all analysis to Firestore ---
def log_all_analysis(ticker, openai_prompt, gemini_prompt, deepseek_prompt, grok_prompt):  # Added grok_prompt
    """Logs every analysis performed to Firestore."""
    if not db:
        st.warning("Firestore not initialized, cannot log all analysis.")
        return

    # Only log if the user has enabled detailed analysis logging
    if not st.session_state.enable_detailed_analysis_logging:
        return

    analysis_data = {
        "Datetime": firestore.SERVER_TIMESTAMP,
        "Ticker": ticker,
        "OpenAI Prompt": openai_prompt,
        "Gemini Prompt": gemini_prompt,
        "DeepSeek Prompt": deepseek_prompt,
        "Grok Prompt": grok_prompt  # Added Grok
    }
    try:
        _throttle_firestore_call()
        db.collection("all_analysis_logs").add(analysis_data)
        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
    except exceptions.ResourceExhausted as e:
        _handle_firestore_error(e, "logging all analysis")
    except Exception as e:
        _handle_firestore_error(e, "logging all analysis")


# --- NEW: Functions for managing batch tickers in Firestore ---
def fetch_batch_tickers_from_firestore_db():
    """Fetches the list of batch tickers from Firestore."""
    if not db:
        return []
    try:
        _throttle_firestore_call()
        doc_ref = db.collection("batch_tickers").document(BATCH_TICKERS_DOC_ID)
        doc = doc_ref.get()
        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
        if doc.exists:
            tickers_data = doc.to_dict()
            tickers = tickers_data.get('tickers', [])
            return [t for t in tickers if t and isinstance(t, str)]  # Filter out empty or non-string entries
        return []
    except exceptions.ResourceExhausted as e:
        _handle_firestore_error(e, "fetching batch tickers")
        return []
    except Exception as e:
        _handle_firestore_error(e, "fetching batch tickers")
        return []


def save_batch_tickers_to_firestore(tickers_list):
    """Saves (overwrites) the list of batch tickers to Firestore."""
    if not db:
        st.warning("Firestore not initialized, cannot save batch tickers.")
        return False
    try:
        _throttle_firestore_call()
        doc_ref = db.collection("batch_tickers").document(BATCH_TICKERS_DOC_ID)
        # Overwrite the document with the new list
        doc_ref.set({"tickers": tickers_list})
        st.success("✅ Batch tickers updated in Firestore!")
        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
        return True
    except exceptions.ResourceExhausted as e:
        _handle_firestore_error(e, "saving batch tickers")
        return False
    except Exception as e:
        _handle_firestore_error(e, "saving batch tickers")
        return False


st.sidebar.markdown("---")
st.sidebar.subheader("App Modes")

main_app_mode_selection = st.sidebar.radio(
    "Select Main Application Mode",
    ("Batch Analysis", "Trade Tracker"),
    index=["Batch Analysis", "Trade Tracker"].index(st.session_state.main_app_mode),
    key="main_app_mode_radio"
)

# Check if app mode changed, then update session state and fetch initial data if necessary
if main_app_mode_selection != st.session_state.main_app_mode:
    st.session_state.main_app_mode = main_app_mode_selection
    # --- READ OPTIMIZATION: Load only if db is available AND data is not already loaded ---
    if st.session_state.main_app_mode == "Trade Tracker":
        if db and st.session_state.trade_log_df.empty:  # Only fetch if empty and db is ready
            with st.spinner("Loading Trade Log from Firestore..."):
                st.session_state.trade_log_df = fetch_trade_logs_from_firestore_db()
                # Initialize date filters based on loaded data
                if not st.session_state.trade_log_df.empty and 'Datetime' in st.session_state.trade_log_df.columns:
                    earliest_date = st.session_state.trade_log_df['Datetime'].min().date()
                    # Set filter_start_date to earliest date or 30 days ago, whichever is later
                    st.session_state.filter_start_date = max(earliest_date, datetime.now().date() - timedelta(days=30))
                else:
                    st.session_state.filter_start_date = datetime.now().date() - timedelta(days=30)
                st.session_state.filter_end_date = datetime.now().date()

        elif not db:
            st.session_state.trade_log_df = pd.DataFrame(columns=FIRESTORE_TRADE_COLUMNS)
            st.warning("Firestore not initialized, trade log cannot be loaded.")
    elif st.session_state.main_app_mode == "Batch Analysis":
        if db and not st.session_state.batch_tickers_from_firestore_list:  # Only fetch if empty and db is ready
            with st.spinner("Loading Batch Tickers from Firestore..."):
                st.session_state.batch_tickers_from_firestore_list = fetch_batch_tickers_from_firestore_db()
        elif not db:
            st.session_state.batch_tickers_from_firestore_list = []
            st.warning("Firestore not initialized, batch tickers cannot be loaded.")
    st.rerun()  # Rerun to apply mode change and data load

# Now use st.session_state.main_app_mode for logic
main_app_mode = st.session_state.main_app_mode

# Define timeframes/intervals for Batch Analysis
BATCH_TIMEFRAMES = []
batch_analysis_mode_sub = None

if main_app_mode == "Batch Analysis":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Batch Analysis Configuration")
    batch_analysis_mode_sub = st.sidebar.radio(
        "Batch Analysis Type",
        ("Intraday Analysis", "Long-Term Analysis")
    )
    if batch_analysis_mode_sub == "Intraday Analysis":
        BATCH_TIMEFRAMES = [
            ("1d", "5m"),
            ("5d", "15m"),
            ("2d", "30m"),
            ("1mo", "1h")
        ]
    elif batch_analysis_mode_sub == "Long-Term Analysis":
        BATCH_TIMEFRAMES = [("1y", "1d"), ("5y", "1wk"), ("max", "1mo")]

    # NEW: LLM Selection for Batch Analysis (text input)
    st.session_state.llm_selection_for_batch_text = st.sidebar.text_input(
        "Select LLMs for Batch Analysis (comma-separated, e.g., OpenAI, Gemini)",
        value=st.session_state.llm_selection_for_batch_text,
        key="llm_batch_text_input"
    )
    # Parse the string into a list, filter for valid options
    selected_llms_raw = [llm.strip() for llm in st.session_state.llm_selection_for_batch_text.split(',') if llm.strip()]

    # Filter based on configured API keys
    available_llm_options_api_check = []
    if client_openai is not None:
        available_llm_options_api_check.append("OpenAI")
    if gemini_api_key_from_config:
        available_llm_options_api_check.append("Gemini")
    if deepseek_api_key_from_config:
        available_llm_options_api_check.append("DeepSeek")
    if grok_api_key_from_config:
        available_llm_options_api_check.append("Grok")

    st.session_state.llm_selection_for_batch = [
        llm for llm in selected_llms_raw if llm in available_llm_options_api_check
    ]

    if not st.session_state.llm_selection_for_batch and st.session_state.get('main_app_mode_radio') == "Batch Analysis":
        st.sidebar.warning("Please select at least one LLM for Batch Analysis. Enter names like 'OpenAI, Gemini'.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Batch Stock List (Firestore Managed)")

    # Use session state variable for batch tickers
    batch_tickers_text = "\n".join(st.session_state.batch_tickers_from_firestore_list)

    batch_tickers_input = st.sidebar.text_area(
        "Enter stock tickers (one per line, e.g., AAPL, INFY.BO)",
        value=batch_tickers_text,
        height=200,
        key="batch_tickers_input"
    )

    if st.sidebar.button("Save Batch Tickers to Firestore", key="save_batch_tickers_button"):
        new_tickers = [t.strip().upper() for t in batch_tickers_input.split('\n') if t.strip()]
        if db:
            with st.spinner("Saving tickers..."):
                if save_batch_tickers_to_firestore(new_tickers):
                    st.session_state.batch_tickers_from_firestore_list = new_tickers  # Update session state after successful save
                    st.rerun()
                else:
                    st.error("Failed to save batch tickers to Firestore.")
        else:
            st.warning("Firestore not initialized, cannot save batch tickers.")

    enable_continuous_batch = st.sidebar.checkbox("Enable Continuous Batch Analysis",
                                                  value=st.session_state.batch_running,
                                                  key="enable_continuous_batch_checkbox")
    # Optimize write: Only save batch state if this checkbox is explicitly toggled
    if enable_continuous_batch != st.session_state.batch_running:
        st.session_state.batch_running = enable_continuous_batch
        if st.session_state.batch_running:
            st.session_state.batch_start_time = time.time()
        else:
            st.session_state.batch_start_time = None
        if db:
            save_batch_run_state_to_firestore(st.session_state.batch_running, st.session_state.batch_start_time,
                                              st.session_state.batch_total_duration_seconds,
                                              st.session_state.configured_batch_refresh_interval,
                                              st.session_state.configured_batch_delay_between_stocks)

    refresh_input_str_batch = st.sidebar.text_input("Batch Cycle Refresh (seconds)",
                                                    value=str(st.session_state.configured_batch_refresh_interval),
                                                    key="batch_refresh_input")
    try:
        val = int(refresh_input_str_batch)
        if val < 60:
            st.sidebar.warning("Batch refresh must be at least 60 seconds. Setting to 60.")
            st.session_state.configured_batch_refresh_interval = 60
        else:
            st.session_state.configured_batch_refresh_interval = val
        if st.session_state.batch_total_duration_seconds != 0 and \
                st.session_state.configured_batch_refresh_interval > st.session_state.batch_total_duration_seconds:
            st.sidebar.warning("Refresh interval cannot be greater than total duration. Setting to total duration.")
            st.session_state.configured_batch_refresh_interval = st.session_state.batch_total_duration_seconds
    except ValueError:
        st.sidebar.warning("Invalid batch refresh value. Setting to 180.")
        st.session_state.configured_batch_refresh_interval = 180

    delay_input_str = st.sidebar.text_input("Delay between stocks (seconds)",
                                            value=str(st.session_state.configured_batch_delay_between_stocks),
                                            key="batch_delay_input")
    try:
        val = int(delay_input_str)
        if val < 0:
            st.sidebar.warning("Delay cannot be negative. Setting to 0.")
            st.session_state.configured_batch_delay_between_stocks = 0
        else:
            st.session_state.configured_batch_delay_between_stocks = val
    except ValueError:
        st.sidebar.warning("Invalid delay value. Setting to 0.")
        st.session_state.configured_batch_delay_between_stocks = 0

    batch_duration_options = {
        "Indefinite": 0,
        "5 Minutes": 300, "10 Minutes": 600, "30 Minutes": 1800,
        "1 Hour": 3600, "2 Hours": 7200, "4 Hours": 14400
    }
    selected_batch_duration_label = st.sidebar.selectbox("Batch Run Duration",
                                                         options=list(batch_duration_options.keys()),
                                                         index=list(batch_duration_options.values()).index(
                                                             st.session_state.batch_total_duration_seconds),
                                                         key="batch_duration_select")
    # Only update session state; persistence happens on start/stop/checkbox toggle
    st.session_state.batch_total_duration_seconds = batch_duration_options[selected_batch_duration_label]

    # NEW: Toggle for detailed analysis logging
    st.session_state.enable_detailed_analysis_logging = st.sidebar.checkbox(
        "Enable Detailed Analysis Logging (to Firestore: all_analysis_logs)",
        value=st.session_state.enable_detailed_analysis_logging,
        key="enable_detailed_analysis_logging_checkbox",
        help="If checked, every AI prompt and response will be logged to Firestore. Uncheck to save reads/writes."
    )


else:  # If main_app_mode is not Batch Analysis, ensure batch running state is reset and persisted
    if st.session_state.batch_running:
        st.session_state.batch_running = False
        st.session_state.batch_start_time = None
        st.session_state.batch_total_duration_seconds = 0
        if db:
            save_batch_run_state_to_firestore(False, None, 0,
                                              st.session_state.configured_batch_refresh_interval,
                                              st.session_state.configured_batch_delay_between_stocks)
        st.warning("Continuous Batch Analysis is only available when 'Batch Analysis' mode is selected. Stopping.")

st.sidebar.markdown("---")

is_analysis_running = st.session_state.batch_running  # Only batch analysis can be continuous now
run_button_label = "⏹️ Stop Analysis" if is_analysis_running else "▶️ Start Analysis"

if st.sidebar.button(run_button_label, key="main_run_button"):
    if is_analysis_running:  # User clicked Stop
        st.session_state.batch_running = False
        st.session_state.batch_start_time = None
        st.session_state.batch_total_duration_seconds = 0
        if db:
            save_batch_run_state_to_firestore(False, None, 0,
                                              st.session_state.configured_batch_refresh_interval,
                                              st.session_state.configured_batch_delay_between_stocks)
        st.info("Analysis stopped by user.")
        st.rerun()  # Rerun to refresh UI
    else:  # User clicked Start
        if main_app_mode == "Batch Analysis":
            if not st.session_state.llm_selection_for_batch:
                st.error("Please select at least one LLM for Batch Analysis before starting.")
                # Do not trigger rerun if no LLMs selected
            else:
                st.session_state.batch_running = enable_continuous_batch  # Ensure consistency
                st.session_state.manual_batch_run_triggered = not enable_continuous_batch  # If not continuous, it's manual

                if st.session_state.batch_running:
                    st.session_state.batch_start_time = time.time()
                else:
                    st.session_state.batch_start_time = None  # No start time for one-time run

                if db:  # Always save the current configuration on Start/Stop
                    save_batch_run_state_to_firestore(st.session_state.batch_running, st.session_state.batch_start_time,
                                                      st.session_state.batch_total_duration_seconds,
                                                      st.session_state.configured_batch_refresh_interval,
                                                      st.session_state.configured_batch_delay_between_stocks)

                # --- READ OPTIMIZATION: Load tickers only when starting batch analysis ---
                if db:
                    with st.spinner("Loading Batch Tickers from Firestore..."):
                        st.session_state.batch_tickers_from_firestore_list = fetch_batch_tickers_from_firestore_db()
                else:
                    st.session_state.batch_tickers_from_firestore_list = []
                    st.warning("Firestore not initialized, batch tickers cannot be loaded for analysis.")

                if st.session_state.batch_running:
                    st.info(f"Starting Continuous Batch Analysis for "
                            f"{(st.session_state.batch_total_duration_seconds / 60):.0f} minutes "
                            f" (refresh every {st.session_state.configured_batch_refresh_interval} seconds).")
                else:
                    st.info("Starting a one-time Batch Analysis run.")
                st.rerun()  # Rerun to start the batch loop
        elif main_app_mode == "Trade Tracker":
            st.session_state.manual_tracker_run_triggered = True
            st.info("Refreshing Trade Tracker and checking for hits...")
            # On manual refresh, ensure we fetch the latest data from DB
            if db:
                st.session_state.trade_log_df = fetch_trade_logs_from_firestore_db()
                # Re-initialize date filters based on newly fetched data
                if not st.session_state.trade_log_df.empty and 'Datetime' in st.session_state.trade_log_df.columns:
                    earliest_date = st.session_state.trade_log_df['Datetime'].min().date()
                    st.session_state.filter_start_date = max(earliest_date, datetime.now().date() - timedelta(days=30))
                else:
                    st.session_state.filter_start_date = datetime.now().date() - timedelta(days=30)
                st.session_state.filter_end_date = datetime.now().date()
            st.rerun()  # Rerun immediately to trigger the tracker refresh logic

# Dummy Data Button for Telegram Test (Now adds to Firestore)
if st.sidebar.button("➕ Add Dummy Buy Signal (Telegram Test)", key="dummy_button"):
    if db:
        dummy_ticker = "DUMMY"
        dummy_buy_price = 100.50
        dummy_stop_loss = 98.00
        dummy_target_price = 105.75
        dummy_analysis_type = "Test Intraday"

        dummy_trade_data = {
            "Datetime": firestore.SERVER_TIMESTAMP,
            "Ticker": dummy_ticker,
            "Status": "Pending",
            "Analysis Type": dummy_analysis_type,
            "Primary Buy Price": dummy_buy_price,
            "Primary Stop Loss": dummy_stop_loss,
            "Primary Target Price": dummy_target_price,
            "Risk-Reward Ratio": np.nan,
            "Actual Entry Price": np.nan, "Entry Timestamp": np.nan,
            "Actual Exit Price": np.nan, "Exit Timestamp": np.nan,
            "Quantity": np.nan,
            "Capital_Invested_Dollar": np.nan,  # RENAMED
            "Profit_Loss_Dollar": np.nan,  # RENAMED
            "Profit_Loss_Percent": np.nan,  # RENAMED
            "OpenAI Prompt": "Dummy OpenAI Prompt", "OpenAI Advice": "Dummy OpenAI Advice: Buy signal detected.",
            "OpenAI Buy Price": dummy_buy_price, "OpenAI Stop Loss": dummy_stop_loss,
            "OpenAI Target Price": dummy_target_price,
            "Gemini Prompt": "Dummy Gemini Prompt", "Gemini Advice": "Dummy Gemini Advice: Buy signal identified.",
            "Gemini Buy Price": dummy_buy_price, "Gemini Stop Loss": dummy_stop_loss,
            "Gemini Target Price": dummy_target_price,
            "DeepSeek Prompt": "Dummy DeepSeek Prompt",
            "DeepSeek Advice": "Dummy DeepSeek Advice: Buy signal identified.",
            "DeepSeek Buy Price": dummy_buy_price, "DeepSeek Stop Loss": dummy_stop_loss,
            "DeepSeek Target Price": dummy_target_price,
            "Grok Prompt": "Dummy Grok Prompt",  # Added Grok
            "Grok Advice": "Dummy Grok Advice: Buy signal identified.",  # Added Grok
            "Grok Buy Price": dummy_buy_price, "Grok Stop Loss": dummy_stop_loss,
            "Grok Target Price": dummy_target_price,  # Added Grok
            "Trade Notes": "This is a dummy trade for testing Telegram notifications."
        }
        try:
            _throttle_firestore_call()
            db.collection("trade_logs").add(dummy_trade_data)
            st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
            st.success(f"✅ Dummy BUY signal for {dummy_ticker} added to Firestore.")
            # Update the session state DataFrame after adding
            st.session_state.trade_log_df = fetch_trade_logs_from_firestore_db()  # Refresh data after adding
            telegram_msg = (
                f"📈 *DUMMY BUY Alert for {dummy_ticker}!* 📈\n"
                f"📊 *Analysis Type:* {dummy_analysis_type}\n"
                f"💰 *Buy Price:* {dummy_buy_price:.2f}\n"
                f"⛔️ *Stop Loss:* {dummy_stop_loss:.2f}\n"
                f"🎯 *Target Price:* {dummy_target_price:.2f}\n"
                f"\n_This is a test notification._"
            )
            send_telegram_message(telegram_msg)
            st.rerun()
        except exceptions.ResourceExhausted as e:
            _handle_firestore_error(e, "adding dummy data")
        except Exception as e:
            _handle_firestore_error(e, "adding dummy data")
    else:
        st.warning("Firestore not initialized, cannot add dummy data.")

# New: Download All Analysis Log button (from Firestore)
if db:
    # This function is now only called when the button is pressed
    def get_analysis_excel_for_download():
        df_analysis = fetch_all_analysis_from_firestore_for_download()  # Call the on-demand fetcher
        if not df_analysis.empty:
            df_analysis_download = df_analysis.drop(columns=['FirestoreDocId'], errors='ignore')
            # Ensure 'Datetime' column is timezone-naive
            if 'Datetime' in df_analysis_download.columns and df_analysis_download['Datetime'].dt.tz is not None:
                df_analysis_download['Datetime'] = df_analysis_download['Datetime'].dt.tz_localize(None)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_analysis_download.to_excel(writer, index=False, sheet_name='AllAnalysisLog')
            output.seek(0)
            return output.getvalue()
        return None


    # Moved update_firestore_for_trade function definition here, before its first usage
    def update_firestore_for_trade(doc_id, new_status, actual_entry_price, entry_timestamp, actual_exit_price,
                                   exit_timestamp,
                                   quantity, capital_invested, profit_loss_dollar, profit_loss_percent,
                                   risk_reward_ratio, trade_notes):
        if not db:
            st.error("Firestore not initialized, cannot update trade log.")
            return

        # Ensure trade_notes is a string and handle NaN/None explicitly if it comes from pandas Series
        trade_notes_cleaned = str(trade_notes) if pd.notna(trade_notes) else ""

        update_payload = {
            "Status": new_status,
            "Actual Entry Price": actual_entry_price if pd.notna(actual_entry_price) else firestore.DELETE_FIELD,
            "Entry Timestamp": entry_timestamp if pd.notna(entry_timestamp) else firestore.DELETE_FIELD,
            "Actual Exit Price": actual_exit_price if pd.notna(actual_exit_price) else firestore.DELETE_FIELD,
            "Exit Timestamp": exit_timestamp if pd.notna(exit_timestamp) else firestore.DELETE_FIELD,
            "Quantity": quantity if pd.notna(quantity) else firestore.DELETE_FIELD,
            "Capital_Invested_Dollar": capital_invested if pd.notna(capital_invested) else firestore.DELETE_FIELD,
            # RENAMED
            "Profit_Loss_Dollar": profit_loss_dollar if pd.notna(profit_loss_dollar) else firestore.DELETE_FIELD,
            # RENAMED
            "Profit_Loss_Percent": profit_loss_percent if pd.notna(profit_loss_percent) else firestore.DELETE_FIELD,
            # RENAMED
            "Risk-Reward Ratio": risk_reward_ratio if pd.notna(risk_reward_ratio) else firestore.DELETE_FIELD,
            "Trade Notes": trade_notes_cleaned  # Always save as string, do not use DELETE_FIELD for optional text
        }
        try:
            _throttle_firestore_call()
            doc_ref = db.collection("trade_logs").document(doc_id)
            doc_ref.update(update_payload)
            st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
        except exceptions.ResourceExhausted as e:
            _handle_firestore_error(e, f"updating trade log document {doc_id}")
        except Exception as e:
            _handle_firestore_error(e, f"updating trade log document {doc_id}")


    # Analysis Log Download Button
    if st.sidebar.button("📥 Generate & Download All Analysis Log (Excel)", key="download_all_analysis_log_button_main"):
        with st.spinner("Generating Analysis Log... This may take a moment."):  # Add spinner
            analysis_excel_data = get_analysis_excel_for_download()
            if analysis_excel_data:
                st.download_button(
                    label="Click to Download Analysis Log",
                    data=analysis_excel_data,
                    file_name="all_analysis_log.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_analysis_log_final",
                    help="Click to download the generated Excel file."
                )
                st.success("Analysis log ready for download!")
            else:
                st.info("No analysis logs in Firestore to download. Generate some first in Batch Analysis mode.")


    # Trade Log Download Button
    def get_trade_log_excel_for_download_on_demand():
        df_trades = fetch_trade_logs_from_firestore_db()  # Call the regular trade log fetcher
        if not df_trades.empty:
            df_trades_download = df_trades.drop(columns=['FirestoreDocId'], errors='ignore')
            for col_name in ['Datetime', 'Entry Timestamp', 'Exit Timestamp']:  # Changed 'col' to 'col_name'
                if col_name in df_trades_download.columns and df_trades_download[col_name].dt.tz is not None:
                    df_trades_download[col_name] = df_trades_download[col_name].dt.tz_localize(None)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_trades_download.to_excel(writer, index=False, sheet_name='TradeLog')
            output.seek(0)
            return output.getvalue()
        return None


    if st.sidebar.button("📥 Generate & Download Trade Log (Excel)", key="download_trade_log_button_main"):
        with st.spinner("Generating Trade Log... This may take a moment."):  # Add spinner
            trade_log_excel_data = get_trade_log_excel_for_download_on_demand()
            if trade_log_excel_data:
                st.download_button(
                    label="Click to Download Trade Log",
                    data=trade_log_excel_data,
                    file_name="trade_log.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_trade_log_final",
                    help="Click to download the generated Excel file."
                )
                st.success("Trade log ready for download!")
            else:
                st.info("No trade logs in Firestore to download.")


else:  # This block handles the general fallback if db is not initialized
    st.sidebar.info("Firestore not initialized. Download functions are disabled.")

# --- Main Content Rendering based on main_app_mode ---

if main_app_mode == "Batch Analysis":
    # If the app just loaded and batch_running is True from Firestore, start the batch loop
    if st.session_state.batch_running or (
            st.session_state.manual_batch_run_triggered and not st.session_state.batch_run_completed_once):
        # BUG FIX: Reset manual trigger AFTER checking its state for the current run cycle
        if st.session_state.manual_batch_run_triggered:
            st.session_state.manual_batch_run_triggered = False
            st.session_state.batch_run_completed_once = False  # Ensure it runs at least once if manually triggered

        # Display progress and status outside the main analysis loop for continuous updates
        progress_bar_placeholder = st.empty()
        status_text_placeholder = st.empty()

        # Loop should continue as long as batch_running is True (continuous) or it's a one-time manual run that hasn't completed
        # The while loop itself should not cause multiple Streamlit reruns. Instead, sleep and then rerun explicitly if continuous.
        # For a one-time run, it should just complete and then not rerun.

        run_this_cycle = True
        while run_this_cycle:  # This loop represents one "cycle" of batch processing
            if st.session_state.batch_running:  # Only for continuous mode
                if st.session_state.batch_start_time is not None and st.session_state.batch_total_duration_seconds != 0:
                    elapsed_time = time.time() - st.session_state.batch_start_time
                    remaining_batch_seconds = max(0,
                                                  st.session_state.batch_total_duration_seconds - elapsed_time)
                    mins, secs = divmod(int(remaining_batch_seconds), 60)
                    hours, mins = divmod(mins, 60)
                    st.sidebar.text(f"Batch Remaining: {hours:02d}:{mins:02d}:{secs:02d}")
                    if remaining_batch_seconds <= 0:
                        st.session_state.batch_running = False
                        st.session_state.batch_start_time = None
                        st.session_state.batch_total_duration_seconds = 0
                        if db:
                            save_batch_run_state_to_firestore(False, None, 0,
                                                              st.session_state.configured_batch_refresh_interval,
                                                              st.session_state.configured_batch_delay_between_stocks)
                        st.warning("Continuous Batch Analysis duration completed.")
                        run_this_cycle = False  # Stop the current cycle
                        st.rerun()  # Rerun to reflect stopped state and clean up UI
                        break  # Exit the while loop
                elif st.session_state.batch_total_duration_seconds == 0:
                    st.sidebar.text("Batch Running: Indefinite")

            st.subheader("📑 Batch Analysis for Entered Tickers")

            # Use the session state variable for tickers
            tickers = st.session_state.batch_tickers_from_firestore_list

            if not tickers:
                st.warning("No tickers found in your batch list to analyze. Please add some using the sidebar input.")
                run_this_cycle = False  # Stop this cycle if no tickers
                if st.session_state.batch_running:  # If continuous, wait and rerun
                    time.sleep(st.session_state.configured_batch_refresh_interval)
                    st.rerun()
                break  # Exit the while loop

            if not st.session_state.llm_selection_for_batch:
                st.error("No LLMs selected for Batch Analysis. Please select at least one.")
                run_this_cycle = False  # Stop this cycle if no LLMs
                if st.session_state.batch_running:
                    time.sleep(st.session_state.configured_batch_refresh_interval)
                    st.rerun()
                break  # Exit the while loop

            st.info(
                f"Analyzing {len(tickers)} tickers from your Firestore-managed list in {batch_analysis_mode_sub}. Current run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            progress_bar = progress_bar_placeholder.progress(0)
            status_text = status_text_placeholder.empty()

            for i, ticker_item in enumerate(tickers):
                status_text.text(f"Processing {ticker_item} ({i + 1}/{len(tickers)})")

                combined_df = pd.DataFrame()
                for tf, iv in BATCH_TIMEFRAMES:
                    df_tf = load_data(ticker_item, period=tf, interval=iv)
                    if df_tf.empty:
                        continue
                    df_tf = apply_indicators(df_tf)
                    df_tf["Timeframe_Interval"] = f"{tf}-{iv}"
                    combined_df = pd.concat([combined_df, df_tf], ignore_index=False)

                if combined_df.empty:
                    st.warning(f"Could not fetch data for {ticker_item} across any timeframes. Skipping.")
                    time.sleep(st.session_state.configured_batch_delay_between_stocks)
                    continue

                # Initialize all LLM advice and prompts to default NaN/empty
                openai_prompt, openai_advice, openai_buy_price, openai_stop_loss, openai_target_price = [np.nan] * 5
                gemini_prompt, gemini_advice, gemini_buy_price, gemini_stop_loss, gemini_target_price = [np.nan] * 5
                deepseek_prompt, deepseek_advice, deepseek_buy_price, deepseek_stop_loss, deepseek_target_price = [np.nan] * 5
                grok_prompt, grok_advice, grok_buy_price, grok_stop_loss, grok_target_price = [np.nan] * 5

                if "OpenAI" in st.session_state.llm_selection_for_batch:
                    openai_prompt, openai_advice, openai_buy_price, openai_stop_loss, openai_target_price = \
                        gpt_trade_advice(combined_df, ticker_item, client_openai, analysis_type=batch_analysis_mode_sub)
                if "Gemini" in st.session_state.llm_selection_for_batch:
                    gemini_prompt, gemini_advice, gemini_buy_price, gemini_stop_loss, gemini_target_price = \
                        gemini_trade_advice(combined_df, ticker_item, gemini_api_key_from_config,
                                            analysis_type=batch_analysis_mode_sub)
                if "DeepSeek" in st.session_state.llm_selection_for_batch:
                    deepseek_prompt, deepseek_advice, deepseek_buy_price, deepseek_stop_loss, deepseek_target_price = \
                        deepseek_trade_advice(combined_df, ticker_item, deepseek_api_key_from_config,
                                              analysis_type=batch_analysis_mode_sub)
                if "Grok" in st.session_state.llm_selection_for_batch:
                    grok_prompt, grok_advice, grok_buy_price, grok_stop_loss, grok_target_price = \
                        grok_trade_advice(combined_df, ticker_item, grok_api_key_from_config,
                                          analysis_type=batch_analysis_mode_sub)

                log_all_analysis(ticker_item, openai_prompt, gemini_prompt, deepseek_prompt,
                                 grok_prompt)  # Now conditional

                primary_buy_price, primary_stop_loss, primary_target_price = np.nan, np.nan, np.nan
                recommendation_status = "No Signal"

                llm_buys_found = False

                if pd.notna(openai_buy_price):
                    primary_buy_price = openai_buy_price
                    primary_stop_loss = openai_stop_loss
                    primary_target_price = openai_target_price
                    recommendation_status = "Pending"
                    llm_buys_found = True
                elif pd.notna(gemini_buy_price):
                    primary_buy_price = gemini_buy_price
                    primary_stop_loss = gemini_stop_loss
                    primary_target_price = gemini_target_price
                    recommendation_status = "Pending"
                    llm_buys_found = True
                elif pd.notna(deepseek_buy_price):
                    primary_buy_price = deepseek_buy_price
                    primary_stop_loss = deepseek_stop_loss
                    primary_target_price = deepseek_target_price
                    recommendation_status = "Pending"
                    llm_buys_found = True
                elif pd.notna(grok_buy_price):
                    primary_buy_price = grok_buy_price
                    primary_stop_loss = grok_stop_loss
                    primary_target_price = grok_target_price
                    recommendation_status = "Pending"
                    llm_buys_found = True

                risk_reward_ratio = np.nan
                if pd.notna(primary_buy_price) and pd.notna(primary_stop_loss) and pd.notna(primary_target_price):
                    risk = primary_buy_price - primary_stop_loss
                    reward = primary_target_price - primary_buy_price
                    if risk > 0:
                        risk_reward_ratio = reward / risk

                if llm_buys_found and db:
                    trade_data_to_save = {
                        "Datetime": firestore.SERVER_TIMESTAMP,
                        "Ticker": ticker_item,
                        "Status": recommendation_status,
                        "Analysis Type": batch_analysis_mode_sub,
                        "Primary Buy Price": primary_buy_price,
                        "Primary Stop Loss": primary_stop_loss,
                        "Primary Target Price": primary_target_price,
                        "Risk-Reward Ratio": risk_reward_ratio,
                        "Actual Entry Price": np.nan, "Entry Timestamp": np.nan,
                        "Actual Exit Price": np.nan, "Exit Timestamp": np.nan,
                        "Quantity": np.nan,
                        "Capital_Invested_Dollar": np.nan,  # RENAMED
                        "Profit_Loss_Dollar": np.nan,  # RENAMED
                        "Profit_Loss_Percent": np.nan,  # RENAMED
                        "OpenAI Prompt": openai_prompt, "OpenAI Advice": openai_advice,
                        "OpenAI Buy Price": openai_buy_price, "OpenAI Stop Loss": openai_stop_loss,
                        "OpenAI Target Price": openai_target_price,
                        "Gemini Prompt": gemini_prompt, "Gemini Advice": gemini_advice,
                        "Gemini Buy Price": gemini_buy_price, "Gemini Stop Loss": gemini_stop_loss,
                        "Gemini Target Price": gemini_target_price,
                        "DeepSeek Prompt": deepseek_prompt, "DeepSeek Advice": deepseek_advice,
                        "DeepSeek Buy Price": deepseek_buy_price, "DeepSeek Stop Loss": deepseek_stop_loss,
                        "DeepSeek Target Price": deepseek_target_price,
                        "Grok Prompt": grok_prompt, "Grok Advice": grok_advice,
                        "Grok Buy Price": grok_buy_price, "Grok Stop Loss": grok_stop_loss,
                        "Grok Target Price": grok_target_price,
                        "Trade Notes": ""
                    }
                    try:
                        _throttle_firestore_call()
                        db.collection("trade_logs").add(trade_data_to_save)
                        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
                        st.success(f"✅ Processed {ticker_item} and logged a BUY signal to Firestore.")
                    except exceptions.ResourceExhausted as e:
                        _handle_firestore_error(e, f"logging batch recommendation for {ticker_item}")
                    except Exception as e:
                        _handle_firestore_error(e, f"logging batch recommendation for {ticker_item}")

                    telegram_msg = (
                        f"📈 *New BUY Alert for {ticker_item}!* 📈\n"
                        f"📊 *Analysis Type:* {batch_analysis_mode_sub}\n"
                        f"💰 *Buy Price:* {primary_buy_price:.2f}\n"
                        f"⛔️ *Stop Loss:* {primary_stop_loss:.2f}\n"
                        f"  *Target Price:* {primary_target_price:.2f}\n"
                        f"\n_Check app for detailed rationale._"
                    )
                    send_telegram_message(telegram_msg)
                elif not db:
                    st.warning("Firestore not initialized, recommendations not logged for persistence.")
                else:
                    st.info(f"ℹ️ {ticker_item}: No strong buy signal found from any LLM. Not logged to tracker.")

                progress_bar.progress((i + 1) / len(tickers))
                time.sleep(st.session_state.configured_batch_delay_between_stocks)
            status_text.text("Batch analysis complete for this cycle!")
            st.success("All selected tickers have been processed and updated in Firestore.")

            st.session_state.batch_run_completed_once = True
            if st.session_state.batch_running:
                time.sleep(st.session_state.configured_batch_refresh_interval)
                # After sleeping, if it's a continuous run, we need to explicitly rerun
                st.rerun()  # This will re-execute the script from top
            else:  # If it was a manual one-time run, break the loop
                run_this_cycle = False
                progress_bar_placeholder.empty()
                status_text_placeholder.empty()

elif main_app_mode == "Trade Tracker":
    if st.session_state.batch_running:
        st.session_state.batch_running = False
        st.session_state.batch_start_time = None
        st.session_state.batch_total_duration_seconds = 0
        if db:
            save_batch_run_state_to_firestore(False, None, 0,
                                              st.session_state.configured_batch_refresh_interval,
                                              st.session_state.configured_batch_delay_between_stocks)
        st.info("Continuous Batch Analysis stopped to view Trade Tracker.")
        st.rerun()

    # Add a button to explicitly refresh the trade log from Firestore
    if st.button("🔄 Refresh Trade Log Data", key="refresh_trade_log_button"):
        if db:
            st.session_state.trade_log_df = fetch_trade_logs_from_firestore_db()
            # Re-initialize date filters based on newly fetched data
            if not st.session_state.trade_log_df.empty and 'Datetime' in st.session_state.trade_log_df.columns:
                earliest_date = st.session_state.trade_log_df['Datetime'].min().date()
                st.session_state.filter_start_date = max(earliest_date, datetime.now().date() - timedelta(days=30))
            else:
                st.session_state.filter_start_date = datetime.now().date() - timedelta(days=30)
            st.session_state.filter_end_date = datetime.now().date()
            st.success("Trade log data refreshed from Firestore.")
        else:
            st.warning("Firestore not initialized. Cannot refresh trade log.")
        st.rerun()

    # Use the session state DataFrame for the trade log
    df_log = st.session_state.trade_log_df

    if df_log.empty:
        st.info("Trade log is empty. Click '🔄 Refresh Trade Log Data' to load it from Firestore.")
    else:
        # Ensure all numeric columns are numeric (especially new ones)
        numeric_cols = [
            "Primary Buy Price", "Primary Stop Loss", "Primary Target Price", "Risk-Reward Ratio",
            "Actual Entry Price", "Actual Exit Price", "Quantity",
            "Capital_Invested_Dollar",  # RENAMED
            "Profit_Loss_Dollar",  # RENAMED
            "Profit_Loss_Percent",  # RENAMED
            "OpenAI Buy Price", "OpenAI Stop Loss", "OpenAI Target Price",
            "Gemini Buy Price", "Gemini Stop Loss", "Gemini Target Price",
            "DeepSeek Buy Price", "DeepSeek Stop Loss", "DeepSeek Target Price",
            "Grok Buy Price", "Grok Stop Loss", "Grok Target Price"
        ]
        for col in numeric_cols:
            if col in df_log.columns:
                df_log[col] = pd.to_numeric(df_log[col], errors='coerce')
            else:
                df_log[col] = np.nan

        datetime_cols = ["Datetime", "Entry Timestamp", "Exit Timestamp"]
        for col in datetime_cols:
            if col in df_log.columns:
                df_log[col] = pd.to_datetime(df_log[col], errors='coerce')
            else:
                df_log[col] = pd.NaT

        st.write("### Current Trade Log")

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])  # Added one more column for date filters
        with col1:
            unique_tickers = ['All'] + sorted(df_log['Ticker'].dropna().unique().tolist())
            filter_ticker = st.selectbox("Filter by Ticker", unique_tickers, key="filter_ticker")
        with col2:
            unique_statuses = ['All'] + sorted(df_log['Status'].dropna().unique().tolist())
            filter_status = st.selectbox("Filter by Status", unique_statuses, key="filter_status")
        with col3:
            unique_analysis_types = ['All'] + sorted(df_log['Analysis Type'].dropna().unique().tolist())
            filter_analysis_type = st.selectbox("Filter by Analysis Type", unique_analysis_types,
                                                key="filter_analysis_type")
        with col4:  # New date filter
            filter_start_date = st.date_input("Start Date", value=st.session_state.filter_start_date,
                                              key="filter_start_date")
        with col5:  # New date filter
            filter_end_date = st.date_input("End Date", value=st.session_state.filter_end_date, key="filter_end_date")

        assumed_trade_value = st.number_input(
            "Assumed Investment per Trade ($) (Used if Quantity is empty)",
            min_value=1.0,
            value=100.0,
            step=10.0,
            format="%.2f",
            key="assumed_trade_value_tracker"
        )

        filtered_df_log = df_log.copy()
        if filter_ticker != 'All':
            filtered_df_log = filtered_df_log[filtered_df_log['Ticker'] == filter_ticker]
        if filter_status != 'All':
            filtered_df_log = filtered_df_log[filtered_df_log['Status'] == filter_status]
        if filter_analysis_type != 'All':
            filtered_df_log = filtered_df_log[filtered_df_log['Analysis Type'] == filter_analysis_type]

        # Apply date range filter
        if not filtered_df_log.empty and 'Datetime' in filtered_df_log.columns:
            filtered_df_log = filtered_df_log[
                (filtered_df_log['Datetime'].dt.date >= filter_start_date) &
                (filtered_df_log['Datetime'].dt.date <= filter_end_date)
                ]

        # Recalculate P/L, Capital Invested, Risk-Reward for DISPLAY in filtered_df_log
        for idx, row in filtered_df_log.iterrows():
            # Recalculate Risk-Reward for display (if not yet calculated or needs refresh)
            if (pd.isna(row['Risk-Reward Ratio']) or st.session_state.manual_tracker_run_triggered) and \
                    pd.notna(row['Primary Buy Price']) and pd.notna(row['Primary Stop Loss']) and pd.notna(
                row['Primary Target Price']):
                risk = row['Primary Buy Price'] - row['Primary Stop Loss']
                reward = row['Primary Target Price'] - row['Primary Buy Price']
                if risk > 0:
                    filtered_df_log.at[idx, 'Risk-Reward Ratio'] = reward / risk
                else:
                    filtered_df_log.at[idx, 'Risk-Reward Ratio'] = np.nan

            # Calculate Capital Invested if Quantity and Actual Entry Price are present
            if pd.notna(row['Quantity']) and pd.notna(row['Actual Entry Price']):
                filtered_df_log.at[idx, 'Capital_Invested_Dollar'] = row['Actual Entry Price'] * row[
                    'Quantity']  # RENAMED
            else:
                filtered_df_log.at[idx, 'Capital_Invested_Dollar'] = np.nan  # RENAMED

            # Calculate Profit/Loss ($) and (%) based on Actual Entry/Exit
            if row['Status'] in ['Target Hit', 'SL Hit'] and pd.notna(row['Actual Entry Price']) and pd.notna(
                    row['Actual Exit Price']):
                if pd.notna(row['Quantity']):
                    profit_loss_dollar = (row['Actual Exit Price'] - row['Actual Entry Price']) * row['Quantity']
                    filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                    if row['Actual Entry Price'] != 0:
                        filtered_df_log.at[idx, 'Profit_Loss_Percent'] = (profit_loss_dollar / (
                                    row['Actual Entry Price'] * row['Quantity'])) * 100  # RENAMED
                    else:
                        filtered_df_log.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                else:
                    if row['Primary Buy Price'] != 0 and pd.notna(row['Primary Buy Price']) and pd.notna(
                            row['Actual Exit Price']):
                        profit_loss_dollar = ((row['Actual Exit Price'] - row['Primary Buy Price']) / row[
                            'Primary Buy Price']) * assumed_trade_value
                        filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                        filtered_df_log.at[idx, 'Profit_Loss_Percent'] = ((row['Actual Exit Price'] - row[
                            'Primary Buy Price']) / row['Primary Buy Price']) * 100  # RENAMED
                    else:
                        filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                        filtered_df_log.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
            elif row['Status'] == 'Entry Hit' and pd.notna(row['Actual Entry Price']):
                current_price_data = load_data(row['Ticker'], period="1d", interval="1m")
                if not current_price_data.empty and 'Close' in current_price_data.columns and pd.notna(
                        current_price_data.iloc[-1]['Close']):
                    current_close = get_safe_scalar(current_price_data.iloc[-1]['Close'])
                    if pd.notna(current_close):
                        if pd.notna(row['Quantity']):
                            profit_loss_dollar = (current_close - row['Actual Entry Price']) * row['Quantity']
                            filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                            if row['Actual Entry Price'] != 0:
                                filtered_df_log.at[idx, 'Profit_Loss_Percent'] = (profit_loss_dollar / (
                                            row['Actual Entry Price'] * row['Quantity'])) * 100  # RENAMED
                            else:
                                filtered_df_log.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                        else:
                            if row['Actual Entry Price'] != 0:
                                profit_loss_dollar = ((current_close - row['Actual Entry Price']) / row[
                                    'Actual Entry Price']) * assumed_trade_value
                                filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                                filtered_df_log.at[idx, 'Profit_Loss_Percent'] = ((current_close - row[
                                    'Actual Entry Price']) / row['Actual Entry Price']) * 100  # RENAMED
                            else:
                                filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                                filtered_df_log.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                else:
                    filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                    filtered_df_log.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
            else:
                filtered_df_log.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                filtered_df_log.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                filtered_df_log.at[idx, 'Capital_Invested_Dollar'] = np.nan  # RENAMED

        display_cols_for_editor = [
            "Datetime", "Ticker", "Status", "Analysis Type",
            "Primary Buy Price", "Primary Stop Loss", "Primary Target Price", "Risk-Reward Ratio",
            "Actual Entry Price", "Entry Timestamp", "Actual Exit Price", "Exit Timestamp",
            "Quantity", "Capital_Invested_Dollar", "Profit_Loss_Dollar", "Profit_Loss_Percent",  # RENAMED
            "Trade Notes"
        ]

        data_editor_column_config = {
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Pending", "Entry Hit", "Target Hit", "SL Hit", "Closed", "Canceled"],
                required=True,
            ),
            "FirestoreDocId": None,
            "Datetime": st.column_config.DatetimeColumn("Datetime", format="YYYY-MM-DD HH:mm:ss", disabled=True),
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Analysis Type": st.column_config.TextColumn("Analysis Type", disabled=True),
            "Primary Buy Price": st.column_config.NumberColumn("Rec. Buy Price", format="%.2f", disabled=True),
            "Primary Stop Loss": st.column_config.NumberColumn("Rec. SL", format="%.2f", disabled=True),
            "Primary Target Price": st.column_config.NumberColumn("Rec. TP", format="%.2f", disabled=True),
            "Risk-Reward Ratio": st.column_config.NumberColumn("R:R Ratio", format="%.2f", disabled=True),
            "Actual Entry Price": st.column_config.NumberColumn("Actual Entry Price", format="%.2f", disabled=True),
            "Entry Timestamp": st.column_config.DatetimeColumn("Entry TS", format="YYYY-MM-DD HH:mm:ss", disabled=True),
            "Actual Exit Price": st.column_config.NumberColumn("Actual Exit Price", format="%.2f", disabled=True),
            "Exit Timestamp": st.column_config.DatetimeColumn("Exit TS", format="YYYY-MM-DD HH:mm:ss", disabled=True),
            "Quantity": st.column_config.NumberColumn("Quantity", format="%d"),
            "Capital_Invested_Dollar": st.column_config.NumberColumn("Capital Inv. ($)", format="%.2f", disabled=True),
            # RENAMED, display name remains
            "Profit_Loss_Dollar": st.column_config.NumberColumn("P/L ($)", format="%.2f", disabled=True),
            # RENAMED, display name remains
            "Profit_Loss_Percent": st.column_config.NumberColumn("P/L (%)", format="%.2f", disabled=True),
            # RENAMED, display name remains
            "Trade Notes": st.column_config.TextColumn("Notes", width="large")
        }

        for col_name in FIRESTORE_TRADE_COLUMNS:
            if "Prompt" in col_name or "Advice" in col_name or (
                    "Buy Price" in col_name and not col_name.startswith("Primary")) or (
                    "Stop Loss" in col_name and not col_name.startswith("Primary")) or (
                    "Target Price" in col_name and not col_name.startswith("Primary")):
                if col_name not in data_editor_column_config:
                    data_editor_column_config[col_name] = None

        edited_data = st.data_editor(
            filtered_df_log[display_cols_for_editor],
            column_config=data_editor_column_config,
            num_rows="dynamic",
            use_container_width=True,
            key="trade_log_data_editor"
        )

        if st.button("🔄 Check for Target/SL/Entry Hits and Update Metrics"):
            updated_rows_count = 0
            progress_bar_tracker = st.progress(0)
            status_text_tracker = st.empty()

            current_df_log_for_update = fetch_trade_logs_from_firestore_db()  # Fetch latest from DB for comparison

            trades_to_check = current_df_log_for_update[
                (current_df_log_for_update['Status'] == 'Pending') |
                (current_df_log_for_update['Status'] == 'Entry Hit')
                ].copy()

            if trades_to_check.empty:
                st.info("No 'Pending' or 'Entry Hit' trades to update.")
            else:
                for i, (idx, row) in enumerate(trades_to_check.iterrows()):
                    status_text_tracker.text(f"Checking {row['Ticker']} ({i + 1}/{len(trades_to_check)})")

                    original_status = row['Status']
                    symbol = row['Ticker']

                    # Use the original recommendation Datetime or Entry Timestamp as start for fetching historical data
                    fetch_start_date_hist = row['Entry Timestamp'] if pd.notna(row['Entry Timestamp']) else row[
                        'Datetime']
                    fetch_start_date_hist = pd.to_datetime(fetch_start_date_hist).normalize()

                    # Fetch data up to today + a buffer for complete market data
                    fetch_end_date_hist = datetime.now() + timedelta(days=5)

                    hist = load_data(symbol, start=fetch_start_date_hist.strftime('%Y-%m-%d'),
                                     end=fetch_end_date_hist.strftime('%Y-%m-%d'), interval="1d")

                    if hist.empty:
                        status_text_tracker.write(f"No recent historical data for {symbol} to check status. Skipping.")
                        progress_bar_tracker.progress((i + 1) / len(trades_to_check))
                        continue

                    hist_high = pd.to_numeric(hist.get('High'), errors='coerce').dropna()
                    hist_low = pd.to_numeric(hist.get('Low'), errors='coerce').dropna()
                    hist_close = pd.to_numeric(hist.get('Close'), errors='coerce').dropna()

                    if hist_high.empty or hist_low.empty or hist_close.empty:
                        progress_bar_tracker.progress((i + 1) / len(trades_to_check))
                        continue

                    min_hist_low = hist_low.min()
                    max_hist_high = hist_high.max()

                    latest_close_price = hist_close.iloc[-1] if not hist_close.empty else np.nan

                    primary_buy_price = row.get('Primary Buy Price', np.nan)
                    primary_stop_loss = row.get('Primary Stop Loss', np.nan)
                    primary_target_price = row.get('Primary Target Price', np.nan)

                    new_status = original_status
                    actual_entry_price = row.get('Actual Entry Price', np.nan)
                    entry_timestamp = row.get('Entry Timestamp', pd.NaT)
                    actual_exit_price = row.get('Actual Exit Price', np.nan)
                    exit_timestamp = row.get('Exit Timestamp', pd.NaT)

                    quantity = float(row['Quantity']) if pd.notna(row['Quantity']) else np.nan

                    status_changed_in_this_loop = False
                    fields_changed_in_this_loop = False  # NEW: Flag for any field change

                    # 1. Check for Entry Hit (if Pending)
                    if new_status == "Pending" and pd.notna(primary_buy_price) and primary_buy_price > 0:
                        entry_hit_dates = hist_close[hist_close <= primary_buy_price].index
                        if not entry_hit_dates.empty:
                            first_entry_date = entry_hit_dates[0]
                            new_status = 'Entry Hit'
                            actual_entry_price = hist_close.loc[first_entry_date]
                            entry_timestamp = first_entry_date
                            st.toast(
                                f"🎉 {symbol}: Entry Price {primary_buy_price:.2f} was hit on {entry_timestamp.strftime('%Y-%m-%d')}!")
                            status_changed_in_this_loop = True
                            fields_changed_in_this_loop = True  # Mark for write

                    # 2. Check for Target Hit or SL Hit (if Entry Hit)
                    if new_status == "Entry Hit" and pd.notna(actual_entry_price):
                        relevant_hist_for_exit = hist[
                            hist.index.normalize() >= pd.to_datetime(entry_timestamp).normalize()] if pd.notna(
                            entry_timestamp) else hist

                        relevant_hist_high = pd.to_numeric(relevant_hist_for_exit.get('High'), errors='coerce').dropna()
                        relevant_hist_low = pd.to_numeric(relevant_hist_for_exit.get('Low'), errors='coerce').dropna()
                        relevant_hist_close = pd.to_numeric(relevant_hist_for_exit.get('Close'),
                                                            errors='coerce').dropna()

                        if pd.notna(
                                primary_target_price) and primary_target_price > 0 and max_hist_high >= primary_target_price:
                            target_hit_dates = relevant_hist_close[relevant_hist_close >= primary_target_price].index
                            if not target_hit_dates.empty:
                                first_target_date = target_hit_dates[0]
                                new_status = 'Target Hit'
                                actual_exit_price = relevant_hist_close.loc[first_target_date]
                                exit_timestamp = first_target_date
                                st.toast(
                                    f"✅ {symbol}: Target Price {primary_target_price:.2f} was hit on {exit_timestamp.strftime('%Y-%m-%d')}!")
                                status_changed_in_this_loop = True
                                fields_changed_in_this_loop = True  # Mark for write

                        elif pd.notna(
                                primary_stop_loss) and primary_stop_loss > 0 and min_hist_low <= primary_stop_loss:
                            sl_hit_dates = relevant_hist_close[relevant_hist_close <= primary_stop_loss].index
                            if not sl_hit_dates.empty:
                                first_sl_date = sl_hit_dates[0]
                                new_status = 'SL Hit'
                                actual_exit_price = relevant_hist_close.loc[first_sl_date]
                                exit_timestamp = first_sl_date
                                st.toast(
                                    f"💥 {symbol}: Stop Loss {primary_stop_loss:.2f} was hit on {exit_timestamp.strftime('%Y-%m-%d')}!")
                                status_changed_in_this_loop = True
                                fields_changed_in_this_loop = True  # Mark for write

                    calculated_risk_reward = np.nan
                    if pd.notna(primary_buy_price) and pd.notna(primary_stop_loss) and pd.notna(primary_target_price):
                        risk = primary_buy_price - primary_stop_loss
                        reward = primary_target_price - primary_buy_price
                        if risk > 0:
                            calculated_risk_reward = reward / risk

                    # Only update if the value is different OR was NaN and now is a value
                    if pd.isna(row['Risk-Reward Ratio']) and pd.notna(calculated_risk_reward):
                        fields_changed_in_this_loop = True
                    elif pd.notna(row['Risk-Reward Ratio']) and pd.notna(calculated_risk_reward) and \
                            not np.isclose(row['Risk-Reward Ratio'], calculated_risk_reward):
                        fields_changed_in_this_loop = True

                    calculated_capital_invested = np.nan
                    if pd.notna(actual_entry_price) and pd.notna(quantity):
                        calculated_capital_invested = actual_entry_price * quantity

                    if pd.isna(row['Capital_Invested_Dollar']) and pd.notna(calculated_capital_invested):  # RENAMED
                        fields_changed_in_this_loop = True
                    elif pd.notna(row['Capital_Invested_Dollar']) and pd.notna(calculated_capital_invested) and \
                            not np.isclose(row['Capital_Invested_Dollar'], calculated_capital_invested):  # RENAMED
                        fields_changed_in_this_loop = True

                    calculated_profit_loss_dollar = np.nan
                    calculated_profit_loss_percent = np.nan

                    if new_status in ['Target Hit', 'SL Hit'] and pd.notna(actual_entry_price) and pd.notna(
                            actual_exit_price):
                        if pd.notna(quantity):
                            calculated_profit_loss_dollar = (actual_exit_price - actual_entry_price) * quantity
                            if actual_entry_price != 0:
                                calculated_profit_loss_percent = (calculated_profit_loss_dollar / (
                                            actual_entry_price * quantity)) * 100
                        else:
                            if actual_entry_price != 0:  # Changed from Primary Buy Price to Actual Entry Price
                                calculated_profit_loss_dollar = ((
                                                                             actual_exit_price - actual_entry_price) / actual_entry_price) * assumed_trade_value
                                calculated_profit_loss_percent = ((
                                                                              actual_exit_price - actual_entry_price) / actual_entry_price) * 100
                            else:
                                calculated_profit_loss_dollar = np.nan
                                calculated_profit_loss_percent = np.nan  # Ensure both are NaN if calc failed
                    elif new_status == 'Entry Hit' and pd.notna(actual_entry_price):
                        current_price_data = load_data(row['Ticker'], period="1d", interval="1m")
                        if not current_price_data.empty and 'Close' in current_price_data.columns and pd.notna(
                                current_price_data.iloc[-1]['Close']):
                            current_close = get_safe_scalar(current_price_data.iloc[-1]['Close'])
                            if pd.notna(current_close):
                                if pd.notna(quantity):
                                    calculated_profit_loss_dollar = (current_close - actual_entry_price) * quantity
                                    if actual_entry_price != 0:
                                        calculated_profit_loss_percent = (calculated_profit_loss_dollar / (
                                                    actual_entry_price * quantity)) * 100
                                else:
                                    if actual_entry_price != 0:
                                        calculated_profit_loss_dollar = ((
                                                                                     current_close - actual_entry_price) / actual_entry_price) * assumed_trade_value
                                        calculated_profit_loss_percent = ((
                                                                                      current_close - actual_entry_price) / actual_entry_price) * 100
                                    else:
                                        calculated_profit_loss_dollar = np.nan
                                        calculated_profit_loss_percent = np.nan  # Ensure both are NaN if calc failed
                        else:
                            calculated_profit_loss_dollar = np.nan
                            calculated_profit_loss_percent = np.nan
                    else:
                        calculated_profit_loss_dollar = np.nan
                        calculated_profit_loss_percent = np.nan

                    # Check if P/L changed
                    if pd.isna(row['Profit_Loss_Dollar']) and pd.notna(calculated_profit_loss_dollar):  # RENAMED
                        fields_changed_in_this_loop = True
                    elif pd.notna(row['Profit_Loss_Dollar']) and pd.notna(calculated_profit_loss_dollar) and \
                            not np.isclose(row['Profit_Loss_Dollar'], calculated_profit_loss_dollar):  # RENAMED
                        fields_changed_in_this_loop = True

                    # Ensure trade_notes is a string
                    current_trade_notes = str(row.get('Trade Notes', '') if pd.notna(row.get('Trade Notes')) else '')

                    # Only call update_firestore_for_trade if anything actually changed
                    if status_changed_in_this_loop or fields_changed_in_this_loop:
                        update_firestore_for_trade(
                            doc_id=row['FirestoreDocId'],
                            new_status=new_status,
                            actual_entry_price=actual_entry_price,
                            entry_timestamp=entry_timestamp,
                            actual_exit_price=actual_exit_price,
                            # FIX: Corrected this from exit_timestamp to actual_exit_price
                            exit_timestamp=exit_timestamp,
                            quantity=quantity,
                            capital_invested=calculated_capital_invested,
                            profit_loss_dollar=calculated_profit_loss_dollar,
                            profit_loss_percent=calculated_profit_loss_percent,
                            risk_reward_ratio=calculated_risk_reward,
                            trade_notes=current_trade_notes  # Pass the casted string here
                        )
                        updated_rows_count += 1

                    progress_bar_tracker.progress((i + 1) / len(trades_to_check))

                status_text_tracker.empty()
                if updated_rows_count > 0:
                    st.success(f"Trade status check complete. {updated_rows_count} trades updated.")
                else:
                    st.info("Trade status check complete. No updates needed for current trades.")
                st.session_state.trade_log_df = fetch_trade_logs_from_firestore_db()  # Refresh data after update
                st.rerun()

        if not edited_data.equals(filtered_df_log[display_cols_for_editor]):
            st.info("Detecting manual changes... Updating Firestore.")
            for index, edited_row in edited_data.iterrows():
                # Locate the original row in the unfiltered df_log using FirestoreDocId
                original_row_full = df_log[df_log['FirestoreDocId'] == edited_row['FirestoreDocId']].iloc[0]

                status_changed = edited_row['Status'] != original_row_full['Status']
                quantity_changed = pd.notna(edited_row['Quantity']) and (
                            edited_row['Quantity'] != original_row_full['Quantity'])
                # FIX: Ensure edited_row.get('Trade Notes') is handled for potential None/NaN correctly
                new_notes = str(edited_row.get('Trade Notes', '') if pd.notna(edited_row.get('Trade Notes')) else '')
                notes_changed = new_notes != (str(original_row_full.get('Trade Notes', '') if pd.notna(
                    original_row_full.get('Trade Notes')) else ''))

                if status_changed or quantity_changed or notes_changed:  # Only proceed if any of these explicitly changed
                    doc_id_to_update = original_row_full['FirestoreDocId']

                    # Recalculate based on potentially changed quantity
                    new_status = edited_row['Status']
                    new_quantity = edited_row['Quantity']

                    actual_entry_price = pd.to_numeric(original_row_full['Actual Entry Price'], errors='coerce')
                    actual_exit_price = pd.to_numeric(original_row_full['Actual Exit Price'], errors='coerce')

                    calculated_capital_invested = np.nan
                    if pd.notna(actual_entry_price) and pd.notna(new_quantity):
                        calculated_capital_invested = actual_entry_price * new_quantity

                    calculated_profit_loss_dollar = np.nan
                    calculated_profit_loss_percent = np.nan

                    if new_status in ['Target Hit', 'SL Hit'] and pd.notna(actual_entry_price) and pd.notna(
                            actual_exit_price):
                        if pd.notna(new_quantity):
                            calculated_profit_loss_dollar = (actual_exit_price - actual_entry_price) * new_quantity
                            if actual_entry_price != 0:
                                calculated_profit_loss_percent = (calculated_profit_loss_dollar / (
                                            actual_entry_price * new_quantity)) * 100
                        else:
                            if actual_entry_price != 0:  # Changed from Primary Buy Price to Actual Entry Price
                                calculated_profit_loss_dollar = ((
                                                                             actual_exit_price - actual_entry_price) / actual_entry_price) * assumed_trade_value
                                calculated_profit_loss_percent = ((
                                                                              actual_exit_price - actual_entry_price) / actual_entry_price) * 100
                            else:
                                calculated_profit_loss_dollar = np.nan
                                calculated_profit_loss_percent = np.nan  # Ensure both are NaN if calc failed
                    elif new_status == 'Entry Hit' and pd.notna(actual_entry_price):
                        current_price_data = load_data(original_row_full['Ticker'], period="1d", interval="1m")
                        if not current_price_data.empty and 'Close' in current_price_data.columns and pd.notna(
                                current_price_data.iloc[-1]['Close']):
                            current_close = get_safe_scalar(current_price_data.iloc[-1]['Close'])
                            if pd.notna(current_close):
                                if pd.notna(new_quantity):
                                    calculated_profit_loss_dollar = (current_close - actual_entry_price) * new_quantity
                                    if actual_entry_price != 0:
                                        calculated_profit_loss_percent = (calculated_profit_loss_dollar / (
                                                    actual_entry_price * new_quantity)) * 100
                                else:
                                    if actual_entry_price != 0:
                                        calculated_profit_loss_dollar = ((
                                                                                     current_close - actual_entry_price) / actual_entry_price) * assumed_trade_value
                                        calculated_profit_loss_percent = ((
                                                                                      current_close - actual_entry_price) / actual_entry_price) * 100
                                    else:
                                        calculated_profit_loss_dollar = np.nan
                                        calculated_profit_loss_percent = np.nan  # Ensure both are NaN if calc failed
                        else:
                            calculated_profit_loss_dollar = np.nan
                            calculated_profit_loss_percent = np.nan
                    else:
                        calculated_profit_loss_dollar = np.nan
                        calculated_profit_loss_percent = np.nan

                    update_payload = {
                        "Status": new_status,
                        "Quantity": new_quantity if pd.notna(new_quantity) else firestore.DELETE_FIELD,
                        "Capital_Invested_Dollar": calculated_capital_invested if pd.notna(
                            calculated_capital_invested) else firestore.DELETE_FIELD,  # RENAMED
                        "Profit_Loss_Dollar": calculated_profit_loss_dollar if pd.notna(
                            calculated_profit_loss_dollar) else firestore.DELETE_FIELD,  # RENAMED
                        "Profit_Loss_Percent": calculated_profit_loss_percent if pd.notna(
                            calculated_profit_loss_percent) else firestore.DELETE_FIELD,  # RENAMED
                        "Trade Notes": new_notes  # Always save as string, not DELETE_FIELD
                    }

                    if new_status == 'Entry Hit' and original_row_full['Status'] == 'Pending' and pd.isna(
                            original_row_full['Actual Entry Price']):
                        current_price_data = load_data(original_row_full['Ticker'], period="1d", interval="1m")
                        if not current_price_data.empty and 'Close' in current_price_data.columns and pd.notna(
                                current_price_data.iloc[-1]['Close']):
                            update_payload["Actual Entry Price"] = get_safe_scalar(current_price_data.iloc[-1]['Close'])
                            update_payload["Entry Timestamp"] = firestore.SERVER_TIMESTAMP
                            st.toast(f"Entry set for {original_row_full['Ticker']} manually.")
                        else:
                            st.warning(
                                f"Could not get current price for {original_row_full['Ticker']} to set Actual Entry Price.")

                    if (new_status == 'Target Hit' or new_status == 'SL Hit') and original_row_full[
                        'Status'] == 'Entry Hit' and pd.isna(original_row_full['Actual Exit Price']):
                        current_price_data = load_data(original_row_full['Ticker'], period="1d", interval="1m")
                        if not current_price_data.empty and 'Close' in current_price_data.columns and pd.notna(
                                current_price_data.iloc[-1]['Close']):
                            update_payload["Actual Exit Price"] = get_safe_scalar(current_price_data.iloc[-1]['Close'])
                            update_payload["Exit Timestamp"] = firestore.SERVER_TIMESTAMP
                            st.toast(f"Exit set for {original_row_full['Ticker']} manually.")
                        else:
                            st.warning(
                                f"Could not get current price for {original_row_full['Ticker']} to set Actual Exit Price.")

                    try:
                        _throttle_firestore_call()
                        doc_ref = db.collection("trade_logs").document(doc_id_to_update)
                        doc_ref.update(update_payload)
                        st.session_state.firestore_error_delay = FIRESTORE_RATE_LIMIT_DELAY  # Reset delay on success
                        st.success(
                            f"Updated trade {original_row_full['Ticker']} (ID: {doc_id_to_update}) in Firestore.")
                    except exceptions.ResourceExhausted as e:
                        _handle_firestore_error(e, f"updating document {doc_id_to_update} in Firestore")
                    except Exception as e:
                        _handle_firestore_error(e, f"updating document {doc_id_to_update} in Firestore")
            st.session_state.trade_log_df = fetch_trade_logs_from_firestore_db()  # Refresh session state after manual edits
            st.rerun()

        st.markdown("---")
        st.write("### Trade Statistics & Visualizations")

        llm_selection_options = ["All LLMs", "OpenAI", "Gemini", "DeepSeek", "Grok"]
        selected_llm_for_metrics = st.selectbox(
            "Select LLM for Metrics Display",
            options=llm_selection_options,
            key="llm_metrics_display_selection_2"
        )

        df_metrics_filtered = df_log.copy()

        if selected_llm_for_metrics == "OpenAI":
            df_metrics_filtered = df_metrics_filtered[pd.notna(df_metrics_filtered["OpenAI Buy Price"])]
        elif selected_llm_for_metrics == "Gemini":
            df_metrics_filtered = df_metrics_filtered[pd.notna(df_metrics_filtered["Gemini Buy Price"])]
        elif selected_llm_for_metrics == "DeepSeek":
            df_metrics_filtered = df_metrics_filtered[pd.notna(df_metrics_filtered["DeepSeek Buy Price"])]
        elif selected_llm_for_metrics == "Grok":
            df_metrics_filtered = df_metrics_filtered[pd.notna(df_metrics_filtered["Grok Buy Price"])]

        # Apply date range filter to metrics display as well
        if not df_metrics_filtered.empty and 'Datetime' in df_metrics_filtered.columns:
            df_metrics_filtered = df_metrics_filtered[
                (df_metrics_filtered['Datetime'].dt.date >= filter_start_date) &
                (df_metrics_filtered['Datetime'].dt.date <= filter_end_date)
                ]

        for idx, row in df_metrics_filtered.iterrows():
            if (pd.isna(row['Risk-Reward Ratio']) or st.session_state.manual_tracker_run_triggered) and \
                    pd.notna(row['Primary Buy Price']) and pd.notna(row['Primary Stop Loss']) and pd.notna(
                row['Primary Target Price']):
                risk = row['Primary Buy Price'] - row['Primary Stop Loss']
                reward = row['Primary Target Price'] - row['Primary Buy Price']
                if risk > 0:
                    df_metrics_filtered.at[idx, 'Risk-Reward Ratio'] = reward / risk
                else:
                    df_metrics_filtered.at[idx, 'Risk-Reward Ratio'] = np.nan

            if pd.notna(row['Quantity']) and pd.notna(row['Actual Entry Price']):
                df_metrics_filtered.at[idx, 'Capital_Invested_Dollar'] = row['Actual Entry Price'] * row[
                    'Quantity']  # RENAMED
            else:
                df_metrics_filtered.at[idx, 'Capital_Invested_Dollar'] = np.nan  # RENAMED

            if row['Status'] in ['Target Hit', 'SL Hit'] and pd.notna(row['Actual Entry Price']) and pd.notna(
                    row['Actual Exit Price']):
                if pd.notna(row['Quantity']):
                    profit_loss_dollar = (row['Actual Exit Price'] - row['Actual Entry Price']) * row['Quantity']
                    df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                    if row['Actual Entry Price'] != 0:
                        df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = (profit_loss_dollar / (
                                    row['Actual Entry Price'] * row['Quantity'])) * 100  # RENAMED
                    else:
                        df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                else:
                    if row['Primary Buy Price'] != 0 and pd.notna(row['Primary Buy Price']) and pd.notna(
                            row['Actual Exit Price']):
                        profit_loss_dollar = ((row['Actual Exit Price'] - row['Primary Buy Price']) / row[
                            'Primary Buy Price']) * assumed_trade_value
                        df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                        df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = ((row['Actual Exit Price'] - row[
                            'Primary Buy Price']) / row['Primary Buy Price']) * 100  # RENAMED
                    else:
                        df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                        df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
            elif row['Status'] == 'Entry Hit' and pd.notna(row['Actual Entry Price']):
                current_price_data = load_data(row['Ticker'], period="1d", interval="1m")
                if not current_price_data.empty and 'Close' in current_price_data.columns and pd.notna(
                        current_price_data.iloc[-1]['Close']):
                    current_close = get_safe_scalar(current_price_data.iloc[-1]['Close'])
                    if pd.notna(current_close):
                        if pd.notna(row['Quantity']):
                            profit_loss_dollar = (current_close - row['Actual Entry Price']) * row['Quantity']
                            df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                            if row['Actual Entry Price'] != 0:
                                df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = (profit_loss_dollar / (
                                            row['Actual Entry Price'] * row['Quantity'])) * 100  # RENAMED
                            else:
                                df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                        else:
                            if row['Actual Entry Price'] != 0:
                                profit_loss_dollar = ((current_close - row['Actual Entry Price']) / row[
                                    'Actual Entry Price']) * assumed_trade_value
                                df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                                df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = ((current_close - row[
                                    'Actual Entry Price']) / row['Actual Entry Price']) * 100  # RENAMED
                            else:
                                df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                                df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                else:
                    df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                    df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
            else:
                df_metrics_filtered.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                df_metrics_filtered.at[idx, 'Profit_Loss_Percent'] = np.nan  # RENAMED
                df_metrics_filtered.at[idx, 'Capital_Invested_Dollar'] = np.nan  # RENAMED

        if not df_metrics_filtered.empty:
            total_profit = df_metrics_filtered[df_metrics_filtered['Profit_Loss_Dollar'] > 0][
                'Profit_Loss_Dollar'].sum()  # RENAMED
            total_loss = df_metrics_filtered[df_metrics_filtered['Profit_Loss_Dollar'] < 0][
                'Profit_Loss_Dollar'].sum()  # RENAMED
            net_profit_loss = total_profit + total_loss

            st.subheader(f"Financial Summary for {selected_llm_for_metrics}")
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Total Profit", f"${total_profit:.2f}")
            with metric_cols[1]:
                st.metric("Total Loss", f"${total_loss:.2f}")
            with metric_cols[2]:
                color = "green" if net_profit_loss >= 0 else "red"
                st.markdown(
                    f"**Net P/L:** <span style='color:{color}; font-size: 1.5em;'>${net_profit_loss:.2f}</span>",
                    unsafe_allow_html=True)

            st.markdown("---")

            status_counts = df_metrics_filtered['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            st.subheader(f"Trade Status Distribution for {selected_llm_for_metrics} Trades")
            st.bar_chart(status_counts.set_index('Status'))

            analysis_type_counts = df_metrics_filtered['Analysis Type'].value_counts().reset_index()
            analysis_type_counts.columns = ['Analysis Type', 'Count']
            st.subheader(f"Analysis Type Distribution for {selected_llm_for_metrics} Trades")
            st.bar_chart(analysis_type_counts.set_index('Analysis Type'))

            if df_metrics_filtered['Profit_Loss_Dollar'].notna().sum() > 1:  # RENAMED
                st.subheader(f"Profit/Loss Distribution for {selected_llm_for_metrics} Trades")
                min_val = df_metrics_filtered['Profit_Loss_Dollar'].min()  # RENAMED
                max_val = df_metrics_filtered['Profit_Loss_Dollar'].max()  # RENAMED
                num_bins = 10

                if min_val == max_val:
                    bins = [min_val - abs(min_val * 0.1) - 1, min_val, min_val + abs(min_val * 0.1) + 1]
                else:
                    bins = np.linspace(min_val, max_val, num_bins)

                pnl_bins = pd.cut(df_metrics_filtered['Profit_Loss_Dollar'].dropna(), bins=bins,  # RENAMED
                                  include_lowest=True)
                pnl_counts = pnl_bins.value_counts().sort_index().reset_index()
                pnl_counts.columns = ['P/L Range', 'Count']
                pnl_counts['P/L Range'] = pnl_counts['P/L Range'].astype(str)
                st.bar_chart(pnl_counts.set_index('P/L Range'))
            else:
                st.info(
                    f"Not enough data with calculated Profit/Loss for {selected_llm_for_metrics} to generate a distribution chart.")
        else:
            st.info(
                f"No data available for {selected_llm_for_metrics} after filtering to generate charts and financial summaries.")

        st.markdown("---")
        st.write("### Comparison Between All LLMs")
        llm_comparison_data = []

        for llm_name in ["OpenAI", "Gemini", "DeepSeek", "Grok"]:
            llm_buy_col = f"{llm_name} Buy Price"

            if llm_buy_col in df_log.columns:
                df_llm_specific_trades = df_log[pd.notna(df_log[llm_buy_col])].copy()
            else:
                df_llm_specific_trades = pd.DataFrame(columns=df_log.columns)

            # Apply date range filter to comparison as well
            if not df_llm_specific_trades.empty and 'Datetime' in df_llm_specific_trades.columns:
                df_llm_specific_trades = df_llm_specific_trades[
                    (df_llm_specific_trades['Datetime'].dt.date >= filter_start_date) &
                    (df_llm_specific_trades['Datetime'].dt.date <= filter_end_date)
                    ]

            if not df_llm_specific_trades.empty:
                for idx, row in df_llm_specific_trades.iterrows():
                    if row['Status'] in ['Target Hit', 'SL Hit'] and pd.notna(row['Actual Entry Price']) and pd.notna(
                            row['Actual Exit Price']):
                        if pd.notna(row['Quantity']):
                            profit_loss_dollar = (row['Actual Exit Price'] - row['Actual Entry Price']) * row[
                                'Quantity']
                            df_llm_specific_trades.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                        else:
                            if row['Primary Buy Price'] != 0 and pd.notna(row['Primary Buy Price']) and pd.notna(
                                    row['Actual Exit Price']):
                                profit_loss_dollar = ((row['Actual Exit Price'] - row['Primary Buy Price']) / row[
                                    'Primary Buy Price']) * assumed_trade_value
                                df_llm_specific_trades.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                            else:
                                df_llm_specific_trades.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                    elif row['Status'] == 'Entry Hit' and pd.notna(row['Actual Entry Price']):
                        current_price_data = load_data(row['Ticker'], period="1d", interval="1m")
                        if not current_price_data.empty and 'Close' in current_price_data.columns and pd.notna(
                                current_price_data.iloc[-1]['Close']):
                            current_close = get_safe_scalar(current_price_data.iloc[-1]['Close'])
                            if pd.notna(current_close):
                                if pd.notna(row['Quantity']):
                                    profit_loss_dollar = (current_close - row['Actual Entry Price']) * row['Quantity']
                                    df_llm_specific_trades.at[idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                                else:
                                    if row['Actual Entry Price'] != 0:
                                        profit_loss_dollar = ((current_close - row['Actual Entry Price']) / row[
                                            'Actual Entry Price']) * assumed_trade_value
                                        df_llm_specific_trades.at[
                                            idx, 'Profit_Loss_Dollar'] = profit_loss_dollar  # RENAMED
                            else:
                                df_llm_specific_trades.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                        else:
                            df_llm_specific_trades.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED
                    else:
                        df_llm_specific_trades.at[idx, 'Profit_Loss_Dollar'] = np.nan  # RENAMED

                hit_count_llm = (df_llm_specific_trades['Status'] == 'Target Hit').sum()
                sl_count_llm = (df_llm_specific_trades['Status'] == 'SL Hit').sum()
                total_trades_llm = len(df_llm_specific_trades)

                resolved_trades_llm = df_llm_specific_trades[
                    (df_llm_specific_trades['Status'] == 'Target Hit') |
                    (df_llm_specific_trades['Status'] == 'SL Hit')
                    ].shape[0]

                hit_rate_llm = (hit_count_llm / resolved_trades_llm * 100) if resolved_trades_llm > 0 else 0

                total_profit_loss_llm = df_llm_specific_trades['Profit_Loss_Dollar'].sum()  # RENAMED

                llm_comparison_data.append({
                    "LLM": llm_name,
                    "Trades Recommended": total_trades_llm,
                    "Target Hits": hit_count_llm,
                    "SL Hits": sl_count_llm,
                    "Hit Rate (%)": f"{hit_rate_llm:.2f}%",
                    "Total P/L ($)": f"{total_profit_loss_llm:.2f}"
                })
            else:
                llm_comparison_data.append({
                    "LLM": llm_name,
                    "Trades Recommended": 0,
                    "Target Hits": 0,
                    "SL Hits": 0,
                    "Hit Rate (%)": "0.00%",
                    "Total P/L ($)": "0.00"
                })

        if llm_comparison_data:
            df_comparison = pd.DataFrame(llm_comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
        else:
            st.info("No LLM trade recommendations found for comparison.")

# --- About Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This app provides real-time stock technical analysis and AI-powered trade recommendations using Streamlit, yfinance, OpenAI, Google Gemini, DeepSeek R1, and Grok, with persistent data storage in Firestore!")
