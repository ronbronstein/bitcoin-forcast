"""
Bitcoin Forecast Dashboard

Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from src.data import DataLoader
from src.forecast import UnifiedForecastEngine
from src.backtest import BacktestEngine, BacktestConfig
from src.models import HistoricalMeanModel, ConditionalMeanModel

# Page config
st.set_page_config(
    page_title="Bitcoin Forecast Engine",
    page_icon="₿",
    layout="wide"
)

# Cache data loading
@st.cache_data
def load_data():
    loader = DataLoader(Path('data'))
    return loader.load_full_dataset(exclude_usdt=True)

# Cache forecast
@st.cache_data
def run_forecast(n_sims, n_months, seed):
    df = load_data()
    engine = UnifiedForecastEngine(
        data=df,
        n_simulations=n_sims,
        n_months=n_months,
        random_seed=seed
    )
    return engine.run()

# Cache backtest
@st.cache_data
def run_backtest(model_type, start_date, end_date):
    df = load_data()
    model_class = ConditionalMeanModel if model_type == "Conditional" else HistoricalMeanModel
    config = BacktestConfig(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        min_training_months=36
    )
    engine = BacktestEngine(model_class, config)
    return engine.run(df, verbose=False)

# Title
st.title("₿ Bitcoin Forecast Engine")
st.markdown("*Combining regime classification, scenario matching, and Monte Carlo simulation*")

# Sidebar
st.sidebar.header("Settings")

tab = st.sidebar.radio("View", ["Forecast", "Backtest", "Scenarios"])

n_sims = st.sidebar.slider("Simulations", 500, 5000, 2000, 500)
n_months = st.sidebar.slider("Forecast Months", 3, 24, 12)
seed = st.sidebar.number_input("Random Seed", value=42, min_value=1)

if tab == "Forecast":
    # Run forecast
    with st.spinner("Running forecast..."):
        result = run_forecast(n_sims, n_months, seed)

    # Current Conditions
    st.header("Current Conditions")
    cond = result.conditions

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${cond.price:,.0f}")
    col2.metric("RSI", f"{cond.rsi:.1f}", cond.rsi_state)
    col3.metric("DXY", cond.dxy_trend)
    col4.metric("Regime", cond.regime.value.replace("_", " ").title())

    if cond.halving_phase:
        st.info(f"📅 {cond.halving_phase:.1f} months since last halving (Post-Halving Year {int(cond.halving_phase // 12) + 1})")

    # Matching Scenarios
    st.header("Matching Scenarios")

    scenario_df = pd.DataFrame([
        {
            "Scenario": sf.name,
            "Win Rate": f"{sf.win_rate:.0f}%",
            "Avg Return": f"{sf.avg_return:+.1f}%",
            "Samples": sf.count,
            "Quality": sf.quality
        }
        for sf in result.scenario_forecasts
    ])
    st.dataframe(scenario_df, use_container_width=True)

    # Weighted forecast
    wf = result.weighted_forecast
    st.subheader("Weighted Scenario Forecast")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{wf['expected_return']:+.1f}%")
    col2.metric("Win Rate", f"{wf['win_rate']:.0f}%")
    col3.metric("Confidence", wf['confidence'])

    # Monte Carlo Chart
    st.header("Monte Carlo Simulation")

    # Get summary data
    summary = result.simulation.get_summary()

    fig = go.Figure()

    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=summary['Date'],
        y=summary['Price_P90'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=summary['Date'],
        y=summary['Price_P10'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(width=0),
        name='P10-P90 Range'
    ))

    # Add median line
    fig.add_trace(go.Scatter(
        x=summary['Date'],
        y=summary['Price_P50'],
        mode='lines+markers',
        line=dict(color='blue', width=2),
        name='Median (P50)'
    ))

    # Add current price line
    fig.add_hline(y=cond.price, line_dash="dash", line_color="gray",
                  annotation_text=f"Current: ${cond.price:,.0f}")

    fig.update_layout(
        title="Price Forecast Distribution",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Final Distribution
    st.header("Final Forecast")

    final = result.simulation.get_final_distribution()
    final_ret = result.simulation.get_return_percentiles(result.simulation.n_months)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        price_df = pd.DataFrame({
            "Percentile": ["P10", "P25", "P50", "P75", "P90", "Mean"],
            "Price": [
                f"${final['p10']:,.0f}",
                f"${final['p25']:,.0f}",
                f"${final['p50']:,.0f}",
                f"${final['p75']:,.0f}",
                f"${final['p90']:,.0f}",
                f"${final['mean']:,.0f}",
            ],
            "Return": [
                f"{final_ret['p10']:+.1f}%",
                f"{final_ret['p25']:+.1f}%",
                f"{final_ret['p50']:+.1f}%",
                f"{final_ret['p75']:+.1f}%",
                f"{final_ret['p90']:+.1f}%",
                f"{final_ret['mean']:+.1f}%",
            ]
        })
        st.dataframe(price_df, use_container_width=True)

    with col2:
        st.subheader("Probability Analysis")
        probs = result.get_return_probabilities()
        targets = result.get_price_targets()

        st.metric("Positive Return", f"{probs['positive']:.1f}%")
        st.metric(">100% Gain", f"{probs['above_100pct']:.1f}%")
        st.metric(">$200K", f"{targets.get(200000, 0):.1f}%")

    # Combined Signal
    st.header("Combined Signal")

    mc_bullish = final_ret["p50"] > 20
    scenario_bullish = wf["win_rate"] > 55 and wf["expected_return"] > 5

    if mc_bullish and scenario_bullish:
        signal = "BULLISH"
        color = "green"
    elif mc_bullish or scenario_bullish:
        signal = "NEUTRAL-BULLISH"
        color = "orange"
    elif wf["win_rate"] < 45 or final_ret["p50"] < -10:
        signal = "BEARISH"
        color = "red"
    else:
        signal = "NEUTRAL"
        color = "gray"

    st.markdown(f"### :{color}[{signal}]")

elif tab == "Backtest":
    st.header("Model Backtest")

    col1, col2, col3 = st.columns(3)
    model_type = col1.selectbox("Model", ["Baseline", "Conditional"])
    start_date = col2.date_input("Start", pd.Timestamp("2020-01-01"))
    end_date = col3.date_input("End", pd.Timestamp("2024-12-01"))

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            results = run_backtest(model_type, start_date, end_date)

        metrics = results['metrics']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Directional Accuracy", f"{metrics['directional_accuracy']*100:.1f}%")
        col2.metric("MAPE", f"{metrics['mape']:.1f}%")
        col3.metric("P10-P90 Capture", f"{metrics['p10_p90_capture']*100:.1f}%")
        col4.metric("Predictions", metrics['n_predictions'])

        # Results chart
        results_df = results['results']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['date'],
            y=results_df['actual_return'],
            mode='lines+markers',
            name='Actual Return'
        ))
        fig.add_trace(go.Scatter(
            x=results_df['date'],
            y=results_df['predicted_return'],
            mode='lines+markers',
            name='Predicted Return'
        ))

        fig.update_layout(
            title="Backtest: Predicted vs Actual Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Direction accuracy by year
        results_df['year'] = pd.to_datetime(results_df['date']).dt.year
        results_df['correct'] = results_df['predicted_direction'] == results_df['actual_direction']

        yearly = results_df.groupby('year').agg({
            'correct': 'mean',
            'actual_return': 'count'
        }).reset_index()
        yearly.columns = ['Year', 'Accuracy', 'Count']
        yearly['Accuracy'] = (yearly['Accuracy'] * 100).round(1).astype(str) + '%'

        st.subheader("Accuracy by Year")
        st.dataframe(yearly, use_container_width=True)

else:  # Scenarios
    st.header("Scenario Analysis")

    from src.scenarios import ScenarioMatcher

    df = load_data()
    matcher = ScenarioMatcher(df)
    matrix = matcher.run_matrix_analysis()

    # Current scenarios
    current = matcher.get_current_scenarios(df.index.max())
    st.subheader(f"Current Matching Scenarios ({len(current)})")
    st.write(", ".join([s.value.replace("_", " ").title() for s in current]))

    # Heatmap
    st.subheader("Win Rate Heatmap")

    # Pivot for heatmap
    pivot = matrix.pivot(index='Scenario', columns='Month', values='Win_Rate')

    # Reorder months
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = pivot[month_order]

    fig = px.imshow(
        pivot,
        color_continuous_scale='RdYlGn',
        labels=dict(color="Win Rate %"),
        aspect="auto"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Top scenarios
    st.subheader("Top Scenarios (by Win Rate)")
    top = matrix[matrix['Count'] >= 5].nlargest(10, 'Win_Rate')[
        ['Month', 'Scenario', 'Win_Rate', 'Avg_Return', 'Count']
    ]
    st.dataframe(top, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Built with Point-in-Time data integrity. 209 tests passing.*")
