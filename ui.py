"""Look and feel of the app: colour palette, custom CSS, and the Plotly charts.

Keeping this in its own file keeps app.py short. Everything visual (colours, card styles, chart styles) is
defined here, so to re-theme the app you only need to change the PALETTE block below.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------------------------------------
# 1. Colour palette: forest green, slate and white, plus red / green that are used ONLY for risk messages.
# ---------------------------------------------------------------------------------------------------------
PURPLE = "#315D54"        # main brand colour
DEEP_PURPLE = "#243B35"
BLUE = "#617C90"
LIGHT_PURPLE = "#F5F6F4"  # soft background for cards and the sidebar
LIGHT_BLUE = "#EDF1F3"
BORDER = "#DDE2DD"
INK = "#242C29"           # dark text
MUTED = "#67716B"
RED, RED_BG = "#A34436", "#FAF3F0"
GREEN, GREEN_BG = "#315D54", "#EFF5F1"
GRID = "#E8ECE8"

FONT = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------------------------------------
# 2. Custom CSS (injected once at the top of app.py). It styles the banner, sidebar, buttons, tabs and cards.
#    .streamlit/config.toml sets the base theme colours; this CSS adds the finishing touches.
# ---------------------------------------------------------------------------------------------------------
def inject_css():
    css = f"""
    <style>
      .block-container {{ padding-top: 3rem; padding-bottom: 3rem; max-width: 1160px; }}
      .hero {{ padding: 0 0 22px; border-bottom: 1px solid {BORDER}; margin-bottom: 18px; }}
      .hero h1 {{ color: {INK}; font-size: 2rem; margin: 0; padding: 0; font-weight: 600; letter-spacing: -.03em; }}
      .hero p {{ color: {MUTED}; margin: 8px 0 0; font-size: 1rem; }}
      .hero .chip {{ display: inline-block; color: {MUTED}; margin: 12px 20px 0 0; font-size: .8rem; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 20px; }}
      .stTabs [data-baseweb="tab"] {{ padding: 10px 0; font-weight: 500; }}
      [data-testid="stForm"] {{ border: 1px solid {BORDER}; border-radius: 6px; padding: 24px; }}
      .card {{ background: white; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 16px 18px; height: 100%; }}
      .card .label {{ color: {MUTED}; font-size: .85rem; }}
      .card .value {{ color: {INK}; font-size: 1.85rem; font-weight: 600; line-height: 1.4; }}
      .card .sub {{ color: {MUTED}; font-size: .82rem; }}
      .risk-card {{ border-radius: 4px; padding: 18px 22px; border-left: 4px solid; margin-bottom: 20px; }}
      .risk-card .risk-label {{ font-size: 1.3rem; font-weight: 600; }}
      .risk-card .risk-text {{ font-size: .95rem; margin-top: 6px; }}
      .risk-high {{ background: {RED_BG}; border-color: {RED}; color: {INK}; }}
      .risk-low {{ background: {GREEN_BG}; border-color: {GREEN}; color: {INK}; }}
      .note {{ background: {LIGHT_PURPLE}; border-left: 3px solid {BORDER};
               padding: 12px 16px; color: {INK}; font-size: .92rem; }}
      h2, h3, h4 {{ color: {INK}; }}
      @media (max-width: 640px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        [data-testid="stForm"] {{ padding: 12px; }}
        .hero h1 {{ font-size: 1.65rem; }}
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list):
    """The understated page title and supporting context."""
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", blue: bool = False) -> str:
    """HTML for one metric card. Use with st.markdown(card(...), unsafe_allow_html=True)."""
    cls = "card blue" if blue else "card"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def risk_card(is_high: bool, probability: float, threshold: float) -> str:
    """The big red (high risk) or green (low risk) result box."""
    if is_high:
        return (f'<div class="risk-card risk-high"><div class="risk-label">Above the decision cutoff</div>'
                f'<div class="risk-text">Predicted probability of default is <b>{probability:.1%}</b>, '
                f'at or above the {threshold:.0%} cutoff. This is a model estimate, not a confirmed outcome.</div></div>')
    return (f'<div class="risk-card risk-low"><div class="risk-label">Below the decision cutoff</div>'
            f'<div class="risk-text">Predicted probability of default is <b>{probability:.1%}</b>, '
            f'below the {threshold:.0%} cutoff.</div></div>')


# ---------------------------------------------------------------------------------------------------------
# 3. Plotly charts, all styled with the same palette
# ---------------------------------------------------------------------------------------------------------
def _style(fig, height=360, title=None):
    """Apply the shared chart style: white background, light neutral grid, matching fonts."""
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, font=dict(size=16, color=DEEP_PURPLE)) if title else None,
        font=dict(family=FONT, color=INK), paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=60, t=55 if title else 20, b=40), legend=dict(orientation="h", y=-0.2),
        colorway=[PURPLE, BLUE, "#91A59A", "#B2C2CA"])
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
        (0, 0, "True negative", tn, BLUE, "white"), (0, 1, "False positive", fp, "#E7EDF0", INK),
        (1, 0, "False negative", fn, "#EEF1EC", INK), (1, 1, "True positive", tp, PURPLE, "white")]
    fig = go.Figure(go.Heatmap(z=[[3, 1], [0, 2]], x=["Predicted non-default", "Predicted default"],
                               y=["Actual non-default", "Actual default"], showscale=False, xgap=5, ygap=5, hoverinfo="skip",
                               colorscale=[[0, "#EEF1EC"], [0.3333, "#E7EDF0"], [0.6667, PURPLE], [1, BLUE]], zmin=0, zmax=3))
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
    palette = [PURPLE, BLUE, "#91A59A"]
    for i, (name, fpr, tpr, auc) in enumerate(curves):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})", fill="tozeroy" if i == 0 else None,
                                 fillcolor="rgba(49, 93, 84, 0.08)", line=dict(color=palette[i % 3], width=3)))
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

