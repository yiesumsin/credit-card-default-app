"""Look and feel of the app: colour palette, custom CSS, and the Plotly charts.

Keeping this in its own file keeps app.py short. Everything visual (colours, card styles, chart styles) is
defined here, so to re-theme the app you only need to change the PALETTE block below.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------------------------------------
# 1. Colour palette: purple, blue and white, plus red / green that are used ONLY for risk messages.
# ---------------------------------------------------------------------------------------------------------
PURPLE = "#6D3FD9"        # main brand colour
DEEP_PURPLE = "#4B2FB5"
BLUE = "#3B82F6"
LIGHT_PURPLE = "#F4F1FF"  # soft background for cards and the sidebar
LIGHT_BLUE = "#EAF2FF"
BORDER = "#E2DBFA"
INK = "#1E1B4B"           # dark text
MUTED = "#6B7280"
RED, RED_BG = "#DC2626", "#FEF2F2"
GREEN, GREEN_BG = "#16A34A", "#F0FDF4"
GRID = "#EEE9FF"

FONT = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------------------------------------
# 2. Custom CSS (injected once at the top of app.py). It styles the banner, sidebar, buttons, tabs and cards.
#    .streamlit/config.toml sets the base theme colours; this CSS adds the finishing touches.
# ---------------------------------------------------------------------------------------------------------
def inject_css():
    css = f"""
    <style>
      html, body, [class*="css"] {{ font-family: {FONT}; color: {INK}; }}
      .block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1250px; }}

      /* Banner at the top of the page */
      .hero {{ background: linear-gradient(120deg, {DEEP_PURPLE} 0%, {PURPLE} 45%, {BLUE} 100%);
               border-radius: 18px; padding: 26px 32px; color: white; margin-bottom: 18px;
               box-shadow: 0 10px 28px rgba(75, 47, 181, 0.25); }}
      .hero h1 {{ color: white; font-size: 2.0rem; margin: 0 0 4px 0; padding: 0; font-weight: 700; }}
      .hero p {{ color: #E9E4FF; margin: 0; font-size: 1.02rem; }}
      .hero .chip {{ display: inline-block; background: rgba(255,255,255,0.18); border-radius: 999px;
                     padding: 3px 12px; margin: 10px 8px 0 0; font-size: 0.82rem; }}

      /* Sidebar */
      [data-testid="stSidebar"] {{ background: linear-gradient(180deg, {LIGHT_PURPLE} 0%, {LIGHT_BLUE} 100%);
                                   border-right: 1px solid {BORDER}; }}
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {DEEP_PURPLE}; }}
      [data-testid="stSidebar"] [data-testid="stExpander"] {{ background: white; border: 1px solid {BORDER};
                                                              border-radius: 12px; }}

      /* Buttons */
      .stButton > button, .stDownloadButton > button {{
          background: linear-gradient(120deg, {PURPLE}, {BLUE}); color: white; border: 0; border-radius: 10px;
          padding: 0.5rem 1.1rem; font-weight: 600; transition: transform .08s ease, box-shadow .15s ease; }}
      .stButton > button:hover, .stDownloadButton > button:hover {{
          color: white; transform: translateY(-1px); box-shadow: 0 6px 16px rgba(109, 63, 217, 0.35); }}
      .stButton > button:focus, .stDownloadButton > button:focus {{ color: white; }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
      .stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600; }}
      .stTabs [aria-selected="true"] {{ color: {PURPLE}; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background-color: {PURPLE}; }}

      /* Metric cards */
      .card {{ background: white; border: 1px solid {BORDER}; border-top: 4px solid {PURPLE}; border-radius: 14px;
               padding: 14px 18px; box-shadow: 0 4px 14px rgba(75, 47, 181, 0.07); height: 100%; }}
      .card.blue {{ border-top-color: {BLUE}; }}
      .card .label {{ color: {MUTED}; font-size: 0.80rem; text-transform: uppercase; letter-spacing: .06em; }}
      .card .value {{ color: {INK}; font-size: 1.85rem; font-weight: 700; line-height: 1.25; }}
      .card .sub {{ color: {MUTED}; font-size: 0.82rem; }}

      /* Risk result boxes: red for high risk, green for low risk */
      .risk-card {{ border-radius: 16px; padding: 20px 24px; border: 2px solid; }}
      .risk-card .risk-label {{ font-size: 1.6rem; font-weight: 800; }}
      .risk-card .risk-text {{ font-size: 1.0rem; margin-top: 4px; }}
      .risk-high {{ background: {RED_BG}; border-color: {RED}; color: #7F1D1D; }}
      .risk-low {{ background: {GREEN_BG}; border-color: {GREEN}; color: #14532D; }}

      /* Small helper text */
      .note {{ background: {LIGHT_PURPLE}; border-left: 4px solid {PURPLE}; border-radius: 8px;
               padding: 10px 14px; color: {INK}; font-size: 0.92rem; }}
      h2, h3, h4 {{ color: {DEEP_PURPLE}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list):
    """The gradient banner at the top of the page."""
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", blue: bool = False) -> str:
    """HTML for one metric card. Use with st.markdown(card(...), unsafe_allow_html=True)."""
    cls = "card blue" if blue else "card"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def risk_card(is_high: bool, probability: float, threshold: float) -> str:
    """The big red (high risk) or green (low risk) result box."""
    if is_high:
        return (f'<div class="risk-card risk-high"><div class="risk-label">⚠️ Defaulter: high risk</div>'
                f'<div class="risk-text">Predicted probability of default is <b>{probability:.1%}</b>, '
                f'at or above the {threshold:.0%} cutoff. Review this client before extending credit.</div></div>')
    return (f'<div class="risk-card risk-low"><div class="risk-label">✅ Non-defaulter: low risk</div>'
            f'<div class="risk-text">Predicted probability of default is <b>{probability:.1%}</b>, '
            f'below the {threshold:.0%} cutoff.</div></div>')


# ---------------------------------------------------------------------------------------------------------
# 3. Plotly charts, all styled with the same palette
# ---------------------------------------------------------------------------------------------------------
def _style(fig, height=360, title=None):
    """Apply the shared chart style: white background, light purple grid, matching fonts."""
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, font=dict(size=16, color=DEEP_PURPLE)) if title else None,
        font=dict(family=FONT, color=INK), paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=60, t=55 if title else 20, b=40), legend=dict(orientation="h", y=-0.2),
        colorway=[PURPLE, BLUE, "#A78BFA", "#93C5FD"])
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def gauge(probability: float, threshold: float):
    """Half-circle gauge of the default probability, with a marker at the decision cutoff."""
    high = probability >= threshold
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number=dict(suffix="%", font=dict(size=44, color=INK)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=MUTED),
            bar=dict(color=RED if high else GREEN, thickness=0.28),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, threshold * 100], color=LIGHT_BLUE), dict(range=[threshold * 100, 100], color="#E6DDFB")],
            threshold=dict(line=dict(color=DEEP_PURPLE, width=4), thickness=0.9, value=threshold * 100))))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=25, b=5), paper_bgcolor="white", font=dict(family=FONT))
    return fig


def what_if_chart(labels, probabilities, threshold, current_index):
    """Bar chart: how the predicted probability would change for different latest repayment statuses."""
    colors = [DEEP_PURPLE if i == current_index else (BLUE if p < threshold else PURPLE) for i, p in enumerate(probabilities)]
    fig = go.Figure(go.Bar(x=labels, y=[p * 100 for p in probabilities], marker_color=colors,
                           text=[f"{p:.0%}" for p in probabilities], textposition="inside", insidetextanchor="end",
                           textfont=dict(color="white", size=13)))
    fig.add_hline(y=threshold * 100, line_dash="dash", line_color=RED)
    fig.add_annotation(x=1, xref="paper", y=threshold * 100, text=f"Cutoff<br>{threshold:.0%}", showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=RED, size=12))
    fig.update_yaxes(title="Probability of default (%)", range=[0, max(60, max(probabilities) * 100 + 8)])
    fig.update_layout(margin=dict(r=60))
    return _style(fig, 330, "What if the latest repayment status were different?")


def confusion_matrix_fig(tp, fp, tn, fn, title="Confusion matrix"):
    """2 x 2 confusion matrix. Correct predictions are solid (purple = caught defaulters, blue = correct non-defaulters),
    mistakes are pale. Each cell shows the count and its share of all clients."""
    total = max(tp + fp + tn + fn, 1)
    cells = [  # (row, column, name, count, fill colour, text colour)
        (0, 0, "True negative", tn, BLUE, "white"), (0, 1, "False positive", fp, "#DBEAFE", INK),
        (1, 0, "False negative", fn, "#EDE9FE", INK), (1, 1, "True positive", tp, PURPLE, "white")]
    fig = go.Figure(go.Heatmap(z=[[3, 1], [0, 2]], x=["Predicted non-default", "Predicted default"],
                               y=["Actual non-default", "Actual default"], showscale=False, xgap=5, ygap=5, hoverinfo="skip",
                               colorscale=[[0, "#EDE9FE"], [0.3333, "#DBEAFE"], [0.6667, PURPLE], [1, BLUE]], zmin=0, zmax=3))
    for row, col, name, count, _, text_color in cells:
        fig.add_annotation(x=col, y=row, showarrow=False, font=dict(color=text_color, size=15),
                           text=f"<b>{name}</b><br>{count:,}<br>{count / total:.1%}")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 380, title)


def roc_fig(curves, title="ROC curve"):
    """curves = list of (name, fpr, tpr, auc). The dashed diagonal is a coin flip (AUC 0.5)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess (AUC 0.50)",
                             line=dict(color="#9CA3AF", dash="dash")))
    palette = [PURPLE, BLUE, "#A78BFA"]
    for i, (name, fpr, tpr, auc) in enumerate(curves):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})", fill="tozeroy" if i == 0 else None,
                                 fillcolor="rgba(109, 63, 217, 0.10)", line=dict(color=palette[i % 3], width=3)))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    return _style(fig, 380, title)


def probability_histogram(probabilities, threshold):
    """Histogram of predicted probabilities for a batch of clients, with the cutoff marked."""
    p = np.asarray(probabilities) * 100
    fig = go.Figure(go.Histogram(x=p, xbins=dict(start=0, end=100, size=2.5), marker_color=PURPLE, opacity=0.9))
    fig.add_vline(x=threshold * 100, line_dash="dash", line_color=RED, annotation_text=f"Cutoff {threshold:.0%}")
    fig.update_xaxes(title="Predicted probability of default (%)")
    fig.update_yaxes(title="Number of clients")
    return _style(fig, 340, "Distribution of predicted risk")


def risk_donut(n_high, n_low):
    """Donut chart of how many clients fall on each side of the cutoff."""
    fig = go.Figure(go.Pie(labels=["High risk (defaulter)", "Low risk (non-defaulter)"], values=[n_high, n_low], hole=0.62,
                           marker=dict(colors=[RED, BLUE]), textinfo="percent", sort=False))
    return _style(fig, 340, "Predicted outcome")
