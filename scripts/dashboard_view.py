import plotly.graph_objects as go
import plotly.io as pio
import plotly.utils
import pandas as pd
import numpy as np
import json

class DashboardView:
    def render_dashboard(self, matrix_df, forecast_data, current_conditions):
        """
        Generates the HTML Dashboard with:
        1. Current Status Panel
        2. The HUGE Matrix Table (Interactive)
        3. Forecast Chart (Bear/Base/Bull Lines)
        """
        print("🎨 Rendering Dashboard...")
        
        dates, price_paths, match_logic, sample_size = forecast_data
        
        # Calculate Forecast Stats
        final_prices = price_paths[:, -1]
        median_path = np.median(price_paths, axis=0)
        p10_path = np.percentile(price_paths, 10, axis=0)
        p90_path = np.percentile(price_paths, 90, axis=0)
        
        print(f"DEBUG VIEW: Start Price={median_path[0]}, End Price={median_path[-1]}")
        print(f"DEBUG VIEW: P10 Path Sample: {p10_path[:3]}")
        
        # Calculate Next Month Win Rate (Immediate short term signal)
        # price_paths[:, 1] is the price at Month 1
        next_month_wins = np.mean(price_paths[:, 1] > price_paths[:, 0]) * 100
        
        # Keep existing 12-month calculation
        year_end_wins = np.mean(final_prices > price_paths[0,0]) * 100
        
        # Get Final Targets for Display
        target_bull = p90_path[-1]
        target_base = median_path[-1]
        target_bear = p10_path[-1]
        
        # Calculate regime comparison for current month
        current_month = current_conditions['Date'].month
        month_name = current_conditions['Date'].strftime('%B')
        
        # Get baseline vs recent regime stats
        baseline_row = matrix_df[
            (matrix_df['Scenario'] == '1. Baseline (All History)') & 
            (matrix_df['Month'] == month_name)
        ]
        recent_row = matrix_df[
            (matrix_df['Scenario'] == '1b. Recent Regime (2020+)') & 
            (matrix_df['Month'] == month_name)
        ]
        
        if not baseline_row.empty and not recent_row.empty:
            baseline_ret = baseline_row['Avg_Return'].values[0]
            recent_ret = recent_row['Avg_Return'].values[0]
            baseline_wr = baseline_row['Win_Rate'].values[0]
            recent_wr = recent_row['Win_Rate'].values[0]
            
            # Create comparison chart
            fig_regime = go.Figure()
            
            fig_regime.add_trace(go.Bar(
                name='All History',
                x=['Avg Return', 'Win Rate'],
                y=[baseline_ret, baseline_wr],
                marker_color='#8b949e',
                text=[f'{baseline_ret:+.1f}%', f'{baseline_wr:.0f}%'],
                textposition='outside'
            ))
            
            fig_regime.add_trace(go.Bar(
                name='Recent (2020+)',
                x=['Avg Return', 'Win Rate'],
                y=[recent_ret, recent_wr],
                marker_color='#1f6feb',
                text=[f'{recent_ret:+.1f}%', f'{recent_wr:.0f}%'],
                textposition='outside'
            ))
            
            fig_regime.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=40, b=40),
                barmode='group',
                title=f'{month_name} Performance: Historical vs Recent Regime',
                yaxis=dict(title='Percentage'),
                showlegend=True
            )
            
            plot_json_regime = json.dumps(fig_regime.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        else:
            # Empty chart if data not available
            plot_json_regime = json.dumps({'data': [], 'layout': {}})
        
        # --- 1. HTML TEMPLATE START ---
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BTC Matrix Forecast (V4.1)</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3; --text-dim: #8b949e; --green: #3fb950; --red: #da3633; --accent: #1f6feb; }
                body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 30px; }
                .container { max-width: 1400px; margin: 0 auto; }
                
                /* Header */
                h1 { font-size: 24px; font-weight: 800; margin: 0 0 5px 0; }
                .subtitle { color: var(--text-dim); font-size: 14px; margin-bottom: 30px; }
                
                /* Grid */
                .grid { display: grid; grid-template-columns: 300px 1fr; gap: 20px; margin-bottom: 30px; }
                
                /* Cards */
                .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; overflow: hidden; }
                .card h3 { margin: 0 0 15px 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); }
                
                /* Status Panel */
                .status-row { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 13px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
                .status-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
                
                /* The Matrix Table */
                .matrix-container { overflow-x: auto; }
                table { width: 100%; border-collapse: collapse; font-size: 12px; }
                th, td { padding: 10px; text-align: center; border-bottom: 1px solid var(--border); }
                th { text-align: left; position: sticky; left: 0; background: var(--card); z-index: 2; border-right: 1px solid var(--border); }
                td:first-child { position: sticky; left: 0; background: var(--card); z-index: 1; text-align: left; font-weight: 600; border-right: 1px solid var(--border); }
                
                .cell-val { font-weight: 700; display: block; }
                .cell-sub { font-size: 10px; opacity: 0.7; display: block; margin-top: 2px; }
                
                /* Heatmap Colors */
                .pos-high { color: #3fb950; } 
                .pos-mid { color: #7ee787; }
                .neg-mid { color: #ffa198; }
                .neg-high { color: #da3633; }
                
                .low-confidence { opacity: 0.5; font-style: italic; }
                
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Bitcoin Scenario Matrix (V4.1)</h1>
                <div class="subtitle">Conditional Probability Engine • 10-Year Historical Analysis</div>
                
                <div class="grid">
                    <!-- LEFT: Current Status -->
                    <div class="card">
                        <h3>Current Conditions</h3>
        """
        
        # Calculate Halving Phase
        from scenario_engine import get_halving_phase
        current_date = current_conditions.get('Date')
        
        if current_date:
            months_since_halving = get_halving_phase(current_date)
            if months_since_halving is not None:
                years = int(months_since_halving / 12)
                months = int(months_since_halving % 12)
                
                # Determine phase name
                if months_since_halving < 12:
                    phase_name = "🚀 Year 1 Post-Halving"
                    phase_color = "#3fb950"  # Green
                elif months_since_halving < 24:
                    phase_name = "📈 Year 2 Post-Halving"
                    phase_color = "#58a6ff"  # Blue
                elif months_since_halving >= 36:
                    phase_name = "🔄 Pre-Halving Year"
                    phase_color = "#e3b341"  # Yellow
                else:
                    phase_name = "📊 Mid-Cycle"
                    phase_color = "#8b949e"  # Gray
                
                html += f"""
                    <div class="status-row" style="border-bottom: 2px solid var(--border); padding-bottom: 12px; margin-bottom: 12px;">
                        <span style="font-weight: 700;">Halving Phase</span>
                        <span class="status-val" style="color: {phase_color}">{phase_name}</span>
                    </div>
                    <div class="status-row">
                        <span>Time Since Halving</span>
                        <span class="status-val">{years}y {months}m</span>
                    </div>
                """
        
        # Add Status Rows
        for k, v in current_conditions.items():
            val_str = f"{v:.2f}" if isinstance(v, (float, int)) and k != 'Date' else str(v)
            html += f"""
                <div class="status-row">
                    <span>{k}</span>
                    <span class="status-val">{val_str}</span>
                </div>
            """
            
        html += f"""
                        <div style="margin-top: 20px; padding-top: 10px; border-top: 1px dashed var(--border);">
                            <h3>Forecast Targets (12m)</h3>
                            <div class="status-row">
                                <span>Logic</span>
                                <span class="status-val" style="color: var(--accent)">{match_logic}</span>
                            </div>
                            <div class="status-row">
                                <span>Sample Size (N)</span>
                                <span class="status-val" style="color: {'#3fb950' if sample_size >= 10 else '#da3633' if sample_size < 5 else '#e3b341'}">{sample_size} years</span>
                            </div>
                            <div class="status-row">
                                <span>🚀 Bull Case (P90)</span>
                                <span class="status-val" style="color: #3fb950">${target_bull:,.0f}</span>
                            </div>
                            <div class="status-row">
                                <span>🎯 Base Case (Median)</span>
                                <span class="status-val" style="color: #e6edf3">${target_base:,.0f}</span>
                            </div>
                            <div class="status-row">
                                <span>🐻 Bear Case (P10)</span>
                                <span class="status-val" style="color: #da3633">${target_bear:,.0f}</span>
                            </div>
                            <div class="status-row">
                                <span>1-Mo Win Prob</span>
                                <span class="status-val" style="color: {'var(--green)' if next_month_wins > 50 else 'var(--red)'}">{next_month_wins:.1f}%</span>
                            </div>
                            <div class="status-row">
                                <span>12-Mo Win Prob</span>
                                <span class="status-val" style="color: {'var(--green)' if year_end_wins > 50 else 'var(--red)'}">{year_end_wins:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- RIGHT: Forecast Chart -->
                    <div class="card">
                        <h3>Projected Scenarios (Bear / Base / Bull)</h3>
                        <div id="chart_div" style="height: 300px;"></div>
                    </div>
                </div>
                
                <!-- REGIME COMPARISON SECTION -->
                <div class="card" style="margin-top: 20px;">
                    <h3>Regime Comparison: All History vs Recent Era (2020+)</h3>
                    <div id="regime_chart" style="height: 250px;"></div>
                </div>
                
                <!-- BOTTOM: THE MATRIX -->
                <div class="card">
                    <h3>Historical Scenario Matrix (Win Rate % / Avg Return %)</h3>
                    <div class="matrix-container">
                        <table>
                            <thead>
                                <tr>
                                    <th style="min-width: 200px;">Scenario</th>
        """
        
        # Table Headers (Months)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for m in months:
            html += f"<th>{m}</th>"
        html += "</tr></thead><tbody>"
        
        # Table Body
        scenarios = matrix_df['Scenario'].unique()
        
        for sc in scenarios:
            html += f"<tr><td>{sc}</td>"
            
            for m in months:
                # Get data for this cell
                row = matrix_df[(matrix_df['Scenario'] == sc) & (matrix_df['Month'] == m)]
                
                if not row.empty:
                    ret = row['Avg_Return'].values[0]
                    win = row['Win_Rate'].values[0]
                    count = row['Count'].values[0]
                    
                    # Get confidence interval if available
                    ci_lower = row['CI_Lower_90'].values[0] if 'CI_Lower_90' in row.columns else None
                    ci_upper = row['CI_Upper_90'].values[0] if 'CI_Upper_90' in row.columns else None
                    
                    # Determine cell opacity based on sample size
                    opacity = min(1.0, count / 10)  # Fade cells with <10 samples
                    
                    # Color Logic
                    color_class = ""
                    if win >= 65: color_class = "pos-high"
                    elif win > 50: color_class = "pos-mid"
                    elif win <= 35: color_class = "neg-high"
                    elif win <= 50: color_class = "neg-mid"
                    
                    # Warning emoji for low samples
                    warning = ""
                    if count < 5:
                        warning = " ⚠️"
                        color_class += " low-confidence"
                    elif count < 10:
                        warning = " ⚠"
                    
                    # Build cell with confidence interval if available
                    ci_text = ""
                    if ci_lower is not None and not np.isnan(ci_lower):
                        ci_text = f"<br><span style='font-size: 9px; opacity: 0.6;'>CI: {ci_lower:+.1f}% to {ci_upper:+.1f}%</span>"
                    
                    html += f"""
                        <td style="opacity: {opacity}">
                            <span class="cell-val {color_class}">{ret:+.1f}%{warning}</span>
                            <span class="cell-sub">{win:.0f}% WR ({count}y)</span>
                            {ci_text}
                        </td>
                    """
                else:
                    html += "<td><span class='cell-sub'>No Data</span></td>"
            
            html += "</tr>"
            
        html += """
                        </tbody>
                    </table>
                </div>
                <div style="margin-top: 10px; font-size: 11px; color: var(--text-dim);">
                    *Data based on 10 years of monthly history. 'Prev RSI' refers to the condition at the START of the month.
                </div>
            </div>
        </div>
        """
        
        # --- CHART GENERATION (JS) ---
        # Convert dates to strings for JSON safety
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        
        # We embed the Plotly JSON directly
        fig = go.Figure()
        
        # 1. Bear Case (P10) - RED
        fig.add_trace(go.Scatter(
            x=date_strs, y=p10_path.tolist(), 
            mode='lines+markers', 
            line=dict(width=2, color='#da3633', dash='dot'),
            marker=dict(size=4),
            name=f'Bear Case (${target_bear:,.0f})',
            hovertemplate='<b>Bear Case</b><br>Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
        ))

        # 2. Base Case (Median) - GREY/WHITE
        fig.add_trace(go.Scatter(
            x=date_strs, y=median_path.tolist(), 
            mode='lines+markers', 
            line=dict(width=3, color='#e6edf3'),
            marker=dict(size=4),
            name=f'Base Case (${target_base:,.0f})',
            hovertemplate='<b>Base Case</b><br>Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
        ))
        
        # 3. Bull Case (P90) - GREEN
        fig.add_trace(go.Scatter(
            x=date_strs, y=p90_path.tolist(), 
            mode='lines+markers', 
            line=dict(width=2, color='#3fb950', dash='dash'),
            marker=dict(size=4),
            name=f'Bull Case (${target_bull:,.0f})',
            hovertemplate='<b>Bull Case</b><br>Date: %{x}<br>Price: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=30, b=40), # Increased right margin for hover labels if needed
            yaxis=dict(gridcolor="#30363d", title="Price (USD)", automargin=True),
            xaxis=dict(gridcolor="#30363d"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#161b22", font_color="#e6edf3") # Fix Hover Colors
        )
        
        plot_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
        html += f"""
        <script>
            var plotData = {plot_json};
            Plotly.newPlot('chart_div', plotData.data, plotData.layout, {{responsive: true}});
            
            var regimeData = {plot_json_regime};
            Plotly.newPlot('regime_chart', regimeData.data, regimeData.layout, {{responsive: true}});
        </script>
        </body>
        </html>
        """
        
        return html
