import plotly.graph_objects as go
import pandas as pd

def generate_interactive_chart(df, company_name):
    company_df = df[df['company'] == company_name].sort_values('date')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=company_df['date'], y=company_df['close'],
        mode='lines', name=company_name,
        line=dict(color='#00D4FF', width=2)
    ))
    fig.update_layout(
        template='plotly_dark',
        title=f"{company_name} — Interactive 10-Year Price Chart",
        xaxis_title="Date", yaxis_title="Price ($)",
        height=450
    )
    return fig