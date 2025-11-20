"""
Backtest Visualization Generator
Creates interactive HTML dashboard showing backtest performance
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.utils
import pandas as pd
import numpy as np
import json

class BacktestViz:
    def __init__(self, results_df):
        """
        Args:
            results_df: DataFrame from BacktestEngine.run_backtest()
        """
        # Validate input
        if results_df.empty:
            raise ValueError("Results DataFrame is empty")

        required_cols = ['test_date', 'target_date', 'actual_price_end', 'forecast_p50',
                         'forecast_p10', 'forecast_p90', 'direction_correct']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean data: remove rows with NaN in critical columns
        results_clean = results_df.dropna(subset=['actual_price_end', 'forecast_p50',
                                                   'forecast_p10', 'forecast_p90']).copy()

        # Filter out corrupted rows with impossible date gaps (> 45 days or < 20 days)
        results_clean['days_diff'] = (pd.to_datetime(results_clean['target_date']) -
                                       pd.to_datetime(results_clean['test_date'])).dt.days
        results_clean = results_clean[(results_clean['days_diff'] >= 20) &
                                       (results_clean['days_diff'] <= 45)].copy()
        results_clean = results_clean.drop('days_diff', axis=1)

        if results_clean.empty:
            raise ValueError("No valid data remaining after filtering")

        self.results = results_clean
        
    def generate_report(self):
        """Generate full HTML report with all visualizations"""
        print("🎨 Generating Backtest Visualization Report...")
        
        html_parts = []
        
        # Header
        html_parts.append(self._generate_header())
        
        # Summary Stats
        html_parts.append(self._generate_summary_stats())
        
        # Chart 1: Timeline - Forecast vs Actual
        html_parts.append(self._generate_timeline_chart())
        
        # Chart 2: Error Distribution
        html_parts.append(self._generate_error_distribution())
        
        # Chart 3: Accuracy by Condition
        html_parts.append(self._generate_accuracy_by_condition())
        
        # Chart 4: Sample Size Impact
        html_parts.append(self._generate_sample_size_analysis())
        
        # Chart 5: Monthly Win/Loss Grid
        html_parts.append(self._generate_monthly_grid())
        
        # Recommendations
        html_parts.append(self._generate_recommendations())
        
        # Footer
        html_parts.append("</div></body></html>")
        
        return "".join(html_parts)
    
    def _generate_header(self):
        """HTML header and styling"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bitcoin Forecast Backtest Report</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                :root { 
                    --bg: #0d1117; 
                    --card: #161b22; 
                    --border: #30363d; 
                    --text: #e6edf3; 
                    --text-dim: #8b949e; 
                    --green: #3fb950; 
                    --red: #da3633; 
                    --accent: #1f6feb; 
                }
                body { 
                    background: var(--bg); 
                    color: var(--text); 
                    font-family: 'Inter', sans-serif; 
                    margin: 0; 
                    padding: 30px; 
                }
                .container { max-width: 1400px; margin: 0 auto; }
                h1 { font-size: 28px; font-weight: 800; margin: 0 0 10px 0; }
                h2 { font-size: 20px; font-weight: 700; margin: 30px 0 15px 0; color: var(--accent); }
                .subtitle { color: var(--text-dim); font-size: 14px; margin-bottom: 30px; }
                .card { 
                    background: var(--card); 
                    border: 1px solid var(--border); 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin-bottom: 20px; 
                }
                .stat-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .stat-box {
                    background: rgba(31, 111, 235, 0.1);
                    border: 1px solid var(--accent);
                    border-radius: 6px;
                    padding: 15px;
                    text-align: center;
                }
                .stat-label {
                    font-size: 12px;
                    color: var(--text-dim);
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 5px;
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: 700;
                    font-family: 'JetBrains Mono', monospace;
                }
                .good { color: var(--green); }
                .bad { color: var(--red); }
                .neutral { color: var(--text); }
                .chart-container { height: 400px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 Backtest Report: Bitcoin Forecast Model</h1>
                <div class="subtitle">Walk-Forward Validation • 2020-2024 • Month-by-Month Analysis</div>
        """
    
    def _generate_summary_stats(self):
        """Generate summary statistics cards"""
        total = len(self.results)
        dir_acc = self.results['direction_correct'].mean() * 100
        p10_p90 = self.results['within_p10_p90'].mean() * 100
        mape = self.results['abs_price_error_pct'].mean()
        
        # Determine colors
        dir_color = 'good' if dir_acc > 55 else 'bad' if dir_acc < 45 else 'neutral'
        band_color = 'good' if 70 <= p10_p90 <= 90 else 'bad'
        mape_color = 'good' if mape < 15 else 'bad' if mape > 25 else 'neutral'
        
        return f"""
        <div class="card">
            <h2>📊 Overall Performance</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-label">Tests Run</div>
                    <div class="stat-value neutral">{total}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Directional Accuracy</div>
                    <div class="stat-value {dir_color}">{dir_acc:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">P10-P90 Capture Rate</div>
                    <div class="stat-value {band_color}">{p10_p90:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Avg Price Error (MAPE)</div>
                    <div class="stat-value {mape_color}">{mape:.1f}%</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_timeline_chart(self):
        """Chart showing forecast bands vs actual prices over time"""
        fig = go.Figure()

        # Convert to lists to avoid binary encoding issues
        dates = self.results['target_date'].tolist()
        actual_prices = self.results['actual_price_end'].tolist()
        p10 = self.results['forecast_p10'].tolist()
        p50 = self.results['forecast_p50'].tolist()
        p90 = self.results['forecast_p90'].tolist()

        # P10-P90 Band (shaded area) - add FIRST so it's in the background
        fig.add_trace(go.Scatter(
            x=dates,
            y=p90,
            mode='lines',
            name='P90',
            line=dict(width=1, color='rgba(31, 111, 235, 0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=dates,
            y=p10,
            mode='lines',
            name='P10-P90 Range',
            fill='tonexty',
            fillcolor='rgba(31, 111, 235, 0.2)',
            line=dict(width=1, color='rgba(31, 111, 235, 0.3)'),
            hoverinfo='skip'
        ))

        # Forecast P50 (Median)
        fig.add_trace(go.Scatter(
            x=dates,
            y=p50,
            mode='lines+markers',
            name='Forecast (Median)',
            line=dict(color='#1f6feb', width=2, dash='dash'),
            marker=dict(size=4)
        ))

        # Actual prices - Changed to GREEN for better visibility
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual_prices,
            mode='lines+markers',
            name='Actual Price',
            line=dict(color='#3fb950', width=3),
            marker=dict(size=6, color='#3fb950')
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Forecast Bands vs Actual Prices",
            xaxis=dict(title="Month", gridcolor="#30363d"),
            yaxis=dict(title="Bitcoin Price (USD)", gridcolor="#30363d"),
            hovermode='x unified',
            height=500,
            hoverlabel=dict(
                bgcolor="#1f2937",
                font_size=14,
                font_family="JetBrains Mono",
                font_color="#e6edf3"
            ),
            legend=dict(
                bgcolor="rgba(22, 27, 34, 0.8)",
                bordercolor="#30363d",
                borderwidth=1
            )
        )
        
        plot_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
        return f"""
        <div class="card">
            <h2>📈 Timeline: Forecast vs Reality</h2>
            <div id="timeline_chart" class="chart-container" style="height: 500px;"></div>
            <script>
                var timelineData = {plot_json};
                Plotly.newPlot('timeline_chart', timelineData.data, timelineData.layout, {{responsive: true}});
            </script>
        </div>
        """
    
    def _generate_error_distribution(self):
        """Histogram of forecast errors"""
        fig = go.Figure()

        # Convert to list
        errors = self.results['price_error_pct'].tolist()

        # Create histogram with better visibility
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=20,
            name='Forecast Error',
            marker=dict(
                color='#1f6feb',
                line=dict(color='#30363d', width=1)
            ),
            opacity=0.8
        ))

        # Add vertical line at zero
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="#3fb950",
                      annotation_text="Perfect Forecast", annotation_position="top")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Distribution of Forecast Errors",
            xaxis=dict(title="Forecast Error (%)", gridcolor="#30363d"),
            yaxis=dict(title="Number of Tests", gridcolor="#30363d"),
            height=400,
            showlegend=False,
            hoverlabel=dict(
                bgcolor="#1f2937",
                font_size=14,
                font_color="#e6edf3"
            )
        )
        
        plot_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
        return f"""
        <div class="card">
            <h2>📊 Error Distribution</h2>
            <div id="error_chart" class="chart-container"></div>
            <script>
                var errorData = {plot_json};
                Plotly.newPlot('error_chart', errorData.data, errorData.layout, {{responsive: true}});
            </script>
        </div>
        """
    
    def _generate_accuracy_by_condition(self):
        """Accuracy broken down by market conditions"""
        # Group by RSI state - use copy to avoid mutation
        results_temp = self.results.copy()
        results_temp['rsi_state'] = pd.cut(
            results_temp['condition_rsi_btc'],
            bins=[0, 45, 65, 100],
            labels=['Oversold<br>RSI < 45', 'Neutral<br>RSI 45-65', 'Overbought<br>RSI > 65']
        )

        accuracy_by_rsi = results_temp.groupby('rsi_state', observed=True)['direction_correct'].agg(['mean', 'count'])
        accuracy_by_rsi['mean'] *= 100

        # Assign colors based on performance
        colors = []
        for acc in accuracy_by_rsi['mean']:
            if acc >= 55:
                colors.append('#3fb950')  # Green for good
            elif acc >= 45:
                colors.append('#f0883e')  # Orange for neutral
            else:
                colors.append('#da3633')  # Red for bad

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=accuracy_by_rsi.index.astype(str),
            y=accuracy_by_rsi['mean'],
            text=[f"<b>{v:.1f}%</b><br>{int(accuracy_by_rsi.loc[i, 'count'])} tests"
                  for i, v in accuracy_by_rsi['mean'].items()],
            textposition='outside',
            textfont=dict(size=14, color='#e6edf3'),
            marker=dict(
                color=colors,
                line=dict(color='#30363d', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>'
        ))

        # Add reference line at 50%
        fig.add_hline(y=50, line_width=2, line_dash="dash", line_color="#8b949e",
                      annotation_text="Random (50%)", annotation_position="right")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="How Well Does the Model Predict in Different Market Conditions?<br><sub>Higher is better. >50% means better than random coin flip.</sub>",
            xaxis=dict(title="Bitcoin Market Condition (Based on RSI)", gridcolor="#30363d"),
            yaxis=dict(title="Directional Accuracy (%)", gridcolor="#30363d", range=[0, max(accuracy_by_rsi['mean'].max() + 15, 65)]),
            height=450,
            showlegend=False,
            hoverlabel=dict(
                bgcolor="#1f2937",
                font_size=14,
                font_color="#e6edf3"
            )
        )
        
        plot_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
        return f"""
        <div class="card">
            <h2>🎯 Performance by Market Condition</h2>
            <div id="condition_chart" class="chart-container"></div>
            <script>
                var conditionData = {plot_json};
                Plotly.newPlot('condition_chart', conditionData.data, conditionData.layout, {{responsive: true}});
            </script>
        </div>
        """
    
    def _generate_sample_size_analysis(self):
        """How does sample size affect accuracy?"""
        # Bin by sample size - use copy to avoid mutation
        results_temp = self.results.copy()
        results_temp['sample_bin'] = pd.cut(
            results_temp['sample_size'],
            bins=[0, 5, 10, 15, 1000],
            labels=['< 5 samples', '5-9 samples', '10-14 samples', '15+ samples']
        )

        # Group and filter out empty bins
        accuracy_by_size = results_temp.groupby('sample_bin', observed=True).agg({
            'direction_correct': ['mean', 'count'],
            'abs_price_error_pct': 'mean'
        })

        # Remove rows with NaN
        accuracy_by_size = accuracy_by_size.dropna()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Direction accuracy
        fig.add_trace(
            go.Bar(
                x=accuracy_by_size.index.astype(str),
                y=accuracy_by_size[('direction_correct', 'mean')] * 100,
                name='Directional Accuracy',
                marker=dict(color='#3fb950'),
                text=[f"{v:.1f}%" for v in accuracy_by_size[('direction_correct', 'mean')] * 100],
                textposition='outside'
            ),
            secondary_y=False
        )
        
        # MAPE
        fig.add_trace(
            go.Scatter(
                x=accuracy_by_size.index.astype(str),
                y=accuracy_by_size[('abs_price_error_pct', 'mean')],
                name='Avg Error (MAPE)',
                mode='lines+markers',
                marker=dict(color='#da3633', size=10),
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Forecast Quality vs Sample Size",
            xaxis=dict(title="Training Sample Size", gridcolor="#30363d"),
            height=400
        )
        
        fig.update_yaxes(title_text="Accuracy (%)", secondary_y=False, gridcolor="#30363d")
        fig.update_yaxes(title_text="MAPE (%)", secondary_y=True)
        
        plot_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
        return f"""
        <div class="card">
            <h2>📏 Sample Size Impact</h2>
            <div id="sample_chart" class="chart-container"></div>
            <script>
                var sampleData = {plot_json};
                Plotly.newPlot('sample_chart', sampleData.data, sampleData.layout, {{responsive: true}});
            </script>
        </div>
        """
    
    def _generate_monthly_grid(self):
        """Heatmap showing wins/losses by month and year"""
        # Create pivot table - use copy to avoid mutation
        results_temp = self.results.copy()
        results_temp['year'] = results_temp['target_date'].dt.year
        results_temp['month'] = results_temp['target_date'].dt.month

        # Convert boolean to int for better display
        results_temp['direction_int'] = results_temp['direction_correct'].astype(int)

        # Use 'first' to handle single values per cell
        pivot = results_temp.pivot_table(
            values='direction_int',
            index='month',
            columns='year',
            aggfunc='first'
        )

        # Create text labels
        text_labels = []
        for i, row in enumerate(pivot.values):
            text_row = []
            for val in row:
                if pd.isna(val):
                    text_row.append('')
                elif val == 1:
                    text_row.append('✓')
                else:
                    text_row.append('✗')
            text_labels.append(text_row)

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=[month_names[i-1] for i in pivot.index],
            colorscale=[[0, '#da3633'], [0.5, '#2d333b'], [1, '#3fb950']],
            text=text_labels,
            texttemplate='%{text}',
            textfont={"size": 20, "color": "#e6edf3"},
            hovertemplate='<b>%{y} %{x}</b><br>Result: %{customdata}<extra></extra>',
            customdata=[['Correct ✓' if v == 1 else 'Wrong ✗' if v == 0 else 'No test'
                        for v in row] for row in pivot.values],
            showscale=False
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Monthly Win/Loss Grid<br><sub>✓ = Predicted direction correctly, ✗ = Predicted wrong direction</sub>",
            xaxis=dict(title="Year", side='bottom', dtick=1),
            yaxis=dict(title="Month"),
            height=450,
            hoverlabel=dict(
                bgcolor="#1f2937",
                font_size=14,
                font_color="#e6edf3"
            )
        )
        
        plot_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
        return f"""
        <div class="card">
            <h2>📅 Monthly Performance Grid</h2>
            <div id="monthly_chart" class="chart-container"></div>
            <script>
                var monthlyData = {plot_json};
                Plotly.newPlot('monthly_chart', monthlyData.data, monthlyData.layout, {{responsive: true}});
            </script>
        </div>
        """
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on results"""
        recommendations = []

        # Check directional accuracy
        dir_acc = self.results['direction_correct'].mean() * 100
        if dir_acc < 50:
            recommendations.append("🚨 <strong>Critical:</strong> Directional accuracy is below 50% (coin flip). The model may be worse than random. Consider simplifying to baseline-only forecasts.")
        elif dir_acc < 55:
            recommendations.append("⚠️ <strong>Warning:</strong> Directional accuracy is barely above random. Conditional logic may not be adding value.")
        else:
            recommendations.append("✅ <strong>Good:</strong> Directional accuracy exceeds random chance, suggesting the model has some predictive power.")

        # Check band calibration
        p10_p90 = self.results['within_p10_p90'].mean() * 100
        if p10_p90 < 70:
            recommendations.append("⚠️ <strong>Bands Too Narrow:</strong> P10-P90 bands capture less than 70% of outcomes. Model is overconfident. Widen uncertainty bands.")
        elif p10_p90 > 90:
            recommendations.append("⚠️ <strong>Bands Too Wide:</strong> P10-P90 bands capture over 90% of outcomes. Model is underconfident. Tighten bands or use more specific scenarios.")
        else:
            recommendations.append("✅ <strong>Good:</strong> Uncertainty bands are well-calibrated (70-90% capture rate).")

        # Check sample size impact
        small_sample = self.results[self.results['sample_size'] < 5]
        if len(small_sample) > 0:
            small_acc = small_sample['direction_correct'].mean() * 100
            large_sample = self.results[self.results['sample_size'] >= 10]
            if len(large_sample) > 0:
                large_acc = large_sample['direction_correct'].mean() * 100

                if small_acc < large_acc - 10:
                    recommendations.append(f"⚠️ <strong>Sample Size Matters:</strong> Forecasts with <5 samples have {small_acc:.1f}% accuracy vs {large_acc:.1f}% with 10+ samples. Flag low-confidence forecasts to users.")

        # Check if any condition clearly outperforms - create rsi_state on the fly
        if 'condition_rsi_btc' in self.results.columns:
            results_temp = self.results.copy()
            results_temp['rsi_state'] = pd.cut(
                results_temp['condition_rsi_btc'],
                bins=[0, 45, 65, 100],
                labels=['Low RSI (<45)', 'Mid RSI (45-65)', 'High RSI (>65)']
            )
            rsi_groups = results_temp.groupby('rsi_state', observed=True)['direction_correct'].mean() * 100
            if len(rsi_groups) > 1 and rsi_groups.max() - rsi_groups.min() > 20:
                best_state = rsi_groups.idxmax()
                recommendations.append(f"💡 <strong>Insight:</strong> Model performs significantly better in '{best_state}' conditions. Consider different strategies for different regimes.")
        
        # Build HTML
        html = '<div class="card"><h2>💡 Recommendations</h2><ul style="line-height: 2;">'
        for rec in recommendations:
            html += f'<li>{rec}</li>'
        html += '</ul></div>'
        
        return html


if __name__ == "__main__":
    # Load backtest results
    results = pd.read_csv('backtest_results.csv')
    results['test_date'] = pd.to_datetime(results['test_date'])
    results['target_date'] = pd.to_datetime(results['target_date'])
    
    # Generate report
    viz = BacktestViz(results)
    html = viz.generate_report()
    
    # Save
    with open('backtest_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("✅ Report saved to: backtest_report.html")