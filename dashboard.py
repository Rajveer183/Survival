import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from main import run_pipeline
from src.visualization import plot_km_curves, plot_hazard_curves, plot_forest_cox, plot_distribution, plot_metric_comparison, COLORS, CHART_LINE_WIDTH
from src.diagnostics import check_cox_assumptions, compute_residuals
from src.nonparametric_models import fit_kaplan_meier, fit_nelson_aalen, compare_groups
from src.utils import clinical_risk_stratification
from scipy import stats as scipy_stats
import os
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Survival Analysis Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Hide Streamlit Menu and Deploy button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton, [data-testid="stDeployButton"] {display:none !important;}
    header[data-testid="stHeader"] {display:none !important;}
    .stApp > header {display:none !important;}
    
    /* Full page background with light blue shades */
    /* Ensure all text is inherently bigger and clearer */
    html {
        font-size: 110%;
    }
    
    /* Force st.dataframe text to be larger and bold */
    [data-testid="stTable"], [data-testid="stDataFrame"] {
        font-size: 1.15rem !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #dbeafe 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling with Shade */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e0f2fe 100%) !important;
        border-right: 1px solid #bae6fd;
    }
    
    /* Header Container - Clean and Top-aligned */
    .header-container {
        padding: 1rem 0;
        margin-top: 0px;
        margin-bottom: 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 15px rgba(3, 105, 161, 0.05);
    }
    
    .main-title {
        background: linear-gradient(90deg, #0369a1, #1d4ed8, #7e22ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.2rem;
        margin: 0;
        letter-spacing: -0.03em;
        padding-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #0c4a6e;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
    }

    /* Premium Metric Flashcards */
    .metric-card {
        padding: 30px;
        border-radius: 28px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        border-top: 8px solid #3b82f6;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left: 1px solid rgba(255,255,255,0.2);
        border-right: 1px solid rgba(255,255,255,0.2);
        cursor: default;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    /* Info Boxes for Results */
    .info-box {
        background-color: #f0f9ff;
        border-left: 6px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        color: #1e40af;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.02);
        font-size: 1.1rem;
    }

    .stSidebar {
        background: linear-gradient(180deg, #f8fafc 0%, #e0f2fe 100%) !important;
        border-right: 1px solid #bae6fd;
    }

    /* Additional UI Colors */
    .highlight-blue { color: #0369a1; font-weight: bold; }
    .highlight-green { color: #166534; font-weight: bold; }
    .highlight-red { color: #991b1b; font-weight: bold; }
    
    /* Remove padding at top and bottom of main container */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    /* Ensure no extra whitespace in sidebar */
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        padding-top: 0rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- UTILS ---
def get_sev_color(p):
    """Global utility for severity-based colors (Green, Yellow, Red)"""
    if p > 0.7: return "#dcfce7", "#166534", "#22c55e" # High (Green)
    if p > 0.4: return "#fef9c3", "#854d0e", "#eab308" # Moderate (Yellow)
    return "#fee2e2", "#991b1b", "#ef4444" # Low (Red)

# --- DISPLAY NAME MAPPINGS ---
DISPLAY_NAME_MAP = {
    "Adj_Chem": "Chemotherapy",
    "Adj_Radio": "Radiotherapy"
}

def get_display_name(col):
    return DISPLAY_NAME_MAP.get(col, col)

# --- DATA LOADING ---
DATA_PATH = "dataset.csv"

@st.cache_data(ttl=3600)
def get_analysis_results_v6(data_path, mtime):
    # Load and run pipeline directly in memory
    results = run_pipeline(data_path)
    # Preserve raw ID_REF for display (aligned with processed rows)
    raw = pd.read_csv(data_path)
    if 'ID_REF' in raw.columns:
        results['df_display'] = raw.loc[results['df'].index]
    else:
        results['df_display'] = results['df']
    return results

RESULTS_MTIME = os.path.getmtime(DATA_PATH)
results = get_analysis_results_v6(DATA_PATH, RESULTS_MTIME)
df = results['df']
df_display = results['df_display']
meta = results['meta']
stats = results['summary_stats']

# --- HEADER & NAVIGATION ---
nav_options = [
    "🏠 Dashboard Overview",
    "📉 Non-Parametric Analysis",
    "📈 Parametric Survival Models",
    "📑 Model Comparison",
    "📊 Multivariate Analysis",
    "💡 Results Interpretation",
    "🧮 Survival Prediction Tool"
]

# Permanent Full-Width Centered Title
st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 class="main-title" style="font-size: 3.5rem; line-height: 1.1;">Multivariate Survival Modeling and <br> Visualization of Cancer Patient Data</h1>
    </div>
""", unsafe_allow_html=True)

# Define Section Label Mapping for the Teal Headers
section_labels = {
    "🏠 Dashboard Overview": "Dashboard Overview",
    "📉 Non-Parametric Analysis": "Non-Parametric Survival Analysis",
    "📈 Parametric Survival Models": "Parametric Survival Modeling",
    "📑 Model Comparison": "Model Comparison & Selection",
    "📊 Multivariate Analysis": "Multivariate Survival Analysis",
    "💡 Results Interpretation": "Clinical & Statistical Interpretation",
    "🧮 Survival Prediction Tool": "Patient Survival Prediction Tool"
}

# Unified Section Header Row (Centered Title + Navigation in Top Right of section)
# [1.5, 5, 1.5] ratio ensures the center title has dominance while staying centered
header_spacer, header_label_col, nav_dropdown_col = st.columns([1.5, 7, 1.5])

# Capturing selection initially
with nav_dropdown_col:
    st.markdown('<div style="padding: 1.2rem 0 0 0;">', unsafe_allow_html=True)
    selection = st.selectbox("Navigation", options=nav_options, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with header_label_col:
    # Remove the leading emoji/icon from the display title for a cleaner centered look
    # E.g., "🏠 Dashboard Overview" -> "Dashboard Overview"
    parts = selection.split(' ', 1)
    display_title = parts[1] if len(parts) > 1 else selection
    
    st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem 0;">
            <h2 style="
                background: linear-gradient(90deg, #0d9488, #14b8a6, #5eead4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                font-size: 2.4rem;
                margin: 0;
                letter-spacing: -0.02em;
            ">{display_title}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="margin-bottom: 1.5rem; border-bottom: 2px solid #f1f5f9;"></div>', unsafe_allow_html=True)

if selection == "🏠 Dashboard Overview":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card" style="background: #e0f2fe; border-top-color: #0ea5e9;"><div style="color: #0369a1; font-weight: 700; font-size: 1.4rem; margin-bottom: 10px;">Cohort Size</div><div style="color: #0369a1; font-size: 3.2rem; font-weight: 900;">{stats["total_obs"]}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="background: #fee2e2; border-top-color: #ef4444;"><div style="color: #991b1b; font-weight: 700; font-size: 1.4rem; margin-bottom: 10px;">Mortality Events</div><div style="color: #991b1b; font-size: 3.2rem; font-weight: 900;">{stats["events"]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="background: #ffedd5; border-top-color: #f97316;"><div style="color: #9a3412; font-weight: 700; font-size: 1.4rem; margin-bottom: 10px;">Censored Rate</div><div style="color: #9a3412; font-size: 3.2rem; font-weight: 900;">{stats["censored"]}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card" style="background: #dcfce7; border-top-color: #166534;"><div style="color: #166534; font-weight: 700; font-size: 1.4rem; margin-bottom: 10px;">Median Surv.</div><div style="color: #166534; font-size: 3.2rem; font-weight: 900;">{stats["median_surv"]} Mo.</div></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Clinical Cohort Preview")
    # Set ID_REF as the index, align content and style the header row (first row of table)
    styled_df = df_display.head(10).set_index('ID_REF').style.set_properties(**{
        'text-align': 'center',
        'font-size': '1.2rem' # Bold and large content
    })\
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('font-weight', 'bold'), 
                ('text-align', 'center'), 
                ('background-color', '#1e293b'), #  background for headers
                ('color', 'white'), # White text for contrast
                ('font-size', '1.25rem') # Even larger headers
            ]},
            {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '1.2rem')]}
        ])
    st.dataframe(styled_df, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    # Define a shared height and dark color palette for chart elements (titles/labels)
    CHART_HEIGHT = 500
    TEXT_COLOR = "#000000"
    AXIS_COLOR = "#000000"
    TITLE_FONT_SIZE = 28
    AXIS_FONT_SIZE = 20
    TICK_FONT_SIZE = 16
    
    with col_a:
        dist_fig = plot_distribution(df[meta['duration_col']])
        dist_fig.update_traces(
            hovertemplate="<span style='font-size:36px; font-weight:bold;'>Duration: %{x} Mo.</span><br><span style='font-size:36px;'>Count: %{y}</span><extra></extra>"
        )
        dist_fig.update_layout(hoverlabel=dict(font_size=36, font_family="Arial", bgcolor="white"))
        st.plotly_chart(dist_fig, use_container_width=True)
    with col_b:
        count_df = pd.DataFrame({'Status': ['Mortality', 'Censored'], 'Count': [stats['events'], stats['censored']]})
        fig = px.pie(
            count_df, values='Count', names='Status', 
            hole=0.6, 
            color_discrete_sequence=['#ef4444', '#3b82f6']
        )
        fig.update_traces(
            textinfo='percent',
            textfont_size=18,
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            hoverlabel=dict(font_size=36, font_family="Arial", bgcolor="white")
        )
        fig.update_layout(
            title={'text': "<b>Vital Status Distribution</b>", 'x': 0.5, 'xanchor': 'center', 'font': {'size': TITLE_FONT_SIZE, 'color': TEXT_COLOR}},
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=TICK_FONT_SIZE, color=TEXT_COLOR)),
            height=CHART_HEIGHT,
            margin=dict(t=100, b=80)
        )
        fig.update_xaxes(showline=True, linecolor='#000000')
        fig.update_yaxes(showline=True, linecolor='#000000')
        st.plotly_chart(fig, use_container_width=True)

elif selection == "📉 Non-Parametric Analysis":
    # Selection of Stratification Factor displayed horizontally
    strata = st.radio(
        "Select Stratification Factor:", 
        meta['categorical_cols'],
        format_func=get_display_name,
        horizontal=True
    )
    
    display_strata = get_display_name(strata)
    kmf_dict = fit_kaplan_meier(df, meta['duration_col'], meta['event_col'], group_col=strata)
    naf_dict = fit_nelson_aalen(df, meta['duration_col'], meta['event_col'], group_col=strata)
    
    p_val = None
    lr_res = compare_groups(df, meta['duration_col'], meta['event_col'], strata)
    if lr_res:
        p_val = lr_res.p_value
        
    col_left, col_right = st.columns(2)
    
    with col_left:
        km_title = f"Kaplan-Meier Survival Probability (By {display_strata})"
        km_fig = plot_km_curves(kmf_dict, title=km_title, p_value=p_val)
        km_fig.update_traces(
            hovertemplate="<span style='font-size:36px;'>Surv: %{y:.1%}</span><extra></extra>"
        )
        km_fig.update_xaxes(tickprefix="Duration: ", ticksuffix=" Month", showtickprefix="first", showticksuffix="first")
        km_fig.update_layout(hoverlabel=dict(font_size=36, font_family="Arial", bgcolor="white"))
        st.plotly_chart(km_fig, use_container_width=True)

    with col_right:
        haz_title = f"Nelson-Aalen Cumulative Hazard (By {display_strata})"
        haz_fig = plot_hazard_curves(naf_dict, title=haz_title)
        haz_fig.update_traces(
            hovertemplate="<span style='font-size:36px;'>Haz: %{y:.2f}</span><extra></extra>"
        )
        haz_fig.update_xaxes(tickprefix="Duration: ", ticksuffix=" Month", showtickprefix="first", showticksuffix="first")
        haz_fig.update_layout(hoverlabel=dict(font_size=36, font_family="Arial", bgcolor="white"))
        st.plotly_chart(haz_fig, use_container_width=True)
        
    st.markdown("#### Clinical Prognostics (KM)")
    for label, kmf in kmf_dict.items():
        st.markdown(f"**Cohort: {label}**")
        cols = st.columns(4)
        m_val = kmf.median_survival_time_
        disp_med = f"{m_val:.1f} Mo." if m_val != np.inf and not np.isnan(m_val) else "Not Reached"
        
        p1 = kmf.survival_function_at_times(12).iloc[0]
        p2 = kmf.survival_function_at_times(24).iloc[0]
        p5 = kmf.survival_function_at_times(60).iloc[0] if 60 <= kmf.timeline.max() else None
        
        p1_bg, p1_text, p1_border = get_sev_color(p1)
        p2_bg, p2_text, p2_border = get_sev_color(p2)
        p5_bg, p5_text, p5_border = get_sev_color(p5 if p5 is not None else 0)
        
        with cols[0]: st.markdown(f'<div class="info-box" style="border-left: 5px solid #3b82f6; padding: 10px;"><small>Median</small><br/><strong>{disp_med}</strong></div>', unsafe_allow_html=True)
        with cols[1]: st.markdown(f'<div class="info-box" style="background-color: {p1_bg}; border-left: 5px solid {p1_border}; color: {p1_text}; padding: 10px;"><small>1-Yr</small><br/><strong>{p1*100:.0f}%</strong></div>', unsafe_allow_html=True)
        with cols[2]: st.markdown(f'<div class="info-box" style="background-color: {p2_bg}; border-left: 5px solid {p2_border}; color: {p2_text}; padding: 10px;"><small>2-Yr</small><br/><strong>{p2*100:.0f}%</strong></div>', unsafe_allow_html=True)
        with cols[3]: 
            disp_p5 = f"{p5*100:.0f}%" if p5 is not None else "N/A"
            st.markdown(f'<div class="info-box" style="background-color: {p5_bg if p5 else "#f0f9ff"}; border-left: 5px solid {p5_border if p5 else "#3b82f6"}; color: {p5_text if p5 else "#1e40af"}; padding: 10px;"><small>5-Yr</small><br/><strong>{disp_p5}</strong></div>', unsafe_allow_html=True)

elif selection == "📈 Parametric Survival Models":
    p_results = results['parametric_results']
    sel_mod = st.radio("Select Model Fit:", list(p_results.keys()), horizontal=True)
    info = p_results[sel_mod]
    fitter = info['fitter']
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        real_km = results['kmf_overall']['Overall'].survival_function_
        fig.add_trace(go.Scatter(x=real_km.index.tolist(), y=real_km['Overall'].values, name='KM Estimate', line=dict(color='black', dash='dash'), hovertemplate="<span style='font-size:36px;'>KM Surv: %{y:.1%}</span><extra></extra>"))
        timeline = np.linspace(0, df[meta['duration_col']].max(), 100)
        fig.add_trace(go.Scatter(x=timeline.tolist(), y=fitter.survival_function_at_times(timeline).values, 
                                 name=f'{sel_mod} Fit', line=dict(color=COLORS[0], width=CHART_LINE_WIDTH), hovertemplate="<span style='font-size:36px;'>Fit: %{y:.1%}</span><extra></extra>"))
        fig.update_layout(
            template="plotly_white", 
            title={
                'text': "<b>Parametric vs. Empirical Survival Rate</b>",
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
            }, 
            xaxis={'title': 'Follow-up (Months)', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000', 'tickprefix': 'Duration: ', 'ticksuffix': ' Month', 'showtickprefix': 'first', 'showticksuffix': 'first', 'hoverformat': '.0f'},
            yaxis={'title': 'Survival Probability', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000'},
            margin=dict(t=80, b=80, l=80),
            hovermode="x unified",
            hoverlabel=dict(font_size=36, font_family="Arial", bgcolor="white")
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline.tolist(), y=fitter.hazard_at_times(timeline).values, 
                                 name=f'{sel_mod} Hazard', line=dict(color=COLORS[1], width=CHART_LINE_WIDTH), hovertemplate="<span style='font-size:36px;'>Haz: %{y:.4f}</span><extra></extra>"))
        fig.update_layout(
            template="plotly_white", 
            title={
                'text': "<b>Instantaneous Hazard Rate Analysis</b>",
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
            },
            xaxis={'title': 'Duration (Months)', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000', 'tickprefix': 'Duration: ', 'ticksuffix': ' Month', 'showtickprefix': 'first', 'showticksuffix': 'first', 'hoverformat': '.0f'},
            yaxis={'title': 'Hazard Density', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000'},
            margin=dict(t=80, b=80, l=80),
            hovermode="x unified",
            hoverlabel=dict(font_size=36, font_family="Arial", bgcolor="white")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### ⏳ Predicted Survival Milestones")
    m_cols = st.columns(3)
    p1 = fitter.survival_function_at_times(12).iloc[0]
    p2 = fitter.survival_function_at_times(24).iloc[0]
    p5 = fitter.survival_function_at_times(60).iloc[0]
    p1_bg, p1_text, p1_border = get_sev_color(p1)
    p2_bg, p2_text, p2_border = get_sev_color(p2)
    p5_bg, p5_text, p5_border = get_sev_color(p5)
    
    with m_cols[0]: st.markdown(f'<div class="info-box" style="background-color: {p1_bg}; border-left: 5px solid {p1_border}; color: {p1_text};"><strong>12 Mo: {p1*100:.1f}%</strong></div>', unsafe_allow_html=True)
    with m_cols[1]: st.markdown(f'<div class="info-box" style="background-color: {p2_bg}; border-left: 5px solid {p2_border}; color: {p2_text};"><strong>24 Mo: {p2*100:.1f}%</strong></div>', unsafe_allow_html=True)
    with m_cols[2]: st.markdown(f'<div class="info-box" style="background-color: {p5_bg}; border-left: 5px solid {p5_border}; color: {p5_text};"><strong>60 Mo: {p5*100:.1f}%</strong></div>', unsafe_allow_html=True)
    st.write("**Model Parameters:**")
    # Flashcard style for parameters
    p_params = info['params']
    p_cols = st.columns(len(p_params))
    for i, (p_name, p_val) in enumerate(p_params.items()):
        with p_cols[i % len(p_cols)]:
            st.markdown(f"""
            <div class="info-box" style="text-align: center; border-radius: 12px; border: 1px solid #e0f2fe;">
                <small style="color: #64748b; font-weight: 600;">{p_name}</small><br/>
                <span style="font-size: 1.2rem; font-weight: 800; color: #0369a1;">{p_val:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

elif selection == "📊 Multivariate Analysis":
    aft_models = results['aft_models']
    aft_name = st.radio("Select AFT Variant:", list(aft_models.keys()), horizontal=True)
    aft_model = aft_models[aft_name]
    st.write(f"**{aft_name} Model Summary**")
    summ_aft = aft_model.summary.copy()
    
    # Filter programmatic terms
    drop_terms = ['intercept', 'rho_', 'alpha_', 'beta_']
    mask = [not (any(term in str(idx).lower() for term in drop_terms) or str(idx).strip().lower() == 'age' or 'age (in years)' in str(idx).lower()) for idx in summ_aft.index]
    summ_aft = summ_aft[mask]
    
    # Mapping labels
    if isinstance(summ_aft.index, pd.MultiIndex):
        # Keep only the primary location parameter to prevent duplicates in Gamma/Piecewise
        mask_primary = [l1 in ('lambda_', 'mu_', 'lambda_0_') for l1, l2 in summ_aft.index]
        summ_aft = summ_aft[mask_primary]
        
        # Strip the structural prefix and map just the covariate name
        summ_aft.index = [get_display_name(l2) for l1, l2 in summ_aft.index]
    else:
        summ_aft.index = [get_display_name(x) for x in summ_aft.index]
        
    # Groupby to flatten any rogue duplicates cleanly
    summ_aft = summ_aft[~summ_aft.index.duplicated(keep='first')]
        
    summ_aft = summ_aft.reset_index()
    # Flatten MultiIndex columns if they exist
    if isinstance(summ_aft.columns, pd.MultiIndex):
        summ_aft.columns = ['_'.join(map(str, col)).strip() for col in summ_aft.columns.values]
    
    # Prepare data for plotting
    plot_df = summ_aft.copy()
    param_col = plot_df.columns[0]
    plot_df['Parameter'] = plot_df[param_col].astype(str)
    fig = px.bar(plot_df, x='Parameter', y='coef', title=f"<b>AFT Model: {aft_name} Clinical Effects</b>", color='coef', color_continuous_scale='Portland')
    fig.update_traces(
        hovertemplate="<span style='font-size:32px; font-weight:bold;'>%{x}</span><br><span style='font-size:28px;'>Coefficient: %{y:.2f}</span><extra></extra>"
    )
    fig.update_layout(
        template="plotly_white", 
        margin=dict(t=80, b=80, l=80),
        xaxis={'title': 'Covariate', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000'},
        yaxis={'title': 'Effect Intensity', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000'},
        hoverlabel=dict(font_size=32, font_family="Arial", bgcolor="white")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(summ_aft, use_container_width=True)

elif selection == "💡 Results Interpretation":
    # 1. Nature of Hazard Function
    shape = results['hazard_shape']
    st.markdown(f"#### 1. Nature of Hazard Function: :blue[{shape}]")
    
    if "Increasing" in shape:
        st.info("**Analytical Insight (IFR):** The hazard of mortality increases over time. For cancer patients, this suggests a 'cumulative biological burden' pattern, where the risk of relapse or complications grows the longer the patient survives post-diagnosis. This warrants **intensified late-stage surveillance**.")
    elif "Decreasing" in shape:
        st.info("**Analytical Insight (DFR):** The hazard decreases over time. This reflects a 'selection effect' where the most vulnerable patients experience events early, and survivors demonstrate higher long-term resilience. Clinical focus should be on **immediate post-operative or early-cycle intensive care**.")
    elif "Constant" in shape:
        st.info("**Analytical Insight (Exponential):** The risk is stable over time. This suggests mortality is driven by random or external factors rather than aging or disease progression. Surveillance can be **consistent and scheduled**.")
    else:
        st.info("**Analytical Insight:** The hazard follows a complex or bathtub-shaped pattern, typically seen when high early risk (surgical/treatment toxicity) is followed by a stable period and eventual late-stage recurrence.")

    # 2. Clinical Evaluation of Covariate Impacts (Categorized)
    st.markdown("#### 2. Clinical Evaluation of Covariate Impacts")
    summ = results['cph_model'].summary
    
    categories = {
        "Age Distribution": "age",
        "Dukes Stage Severity": "stage",
        "Gender Differences": "gender",
        "Tumor Location": "location",
        "Chemotherapy Outcomes": "chem",
        "Radiotherapy Outcomes": "radio"
    }

    selected_cat = st.radio("Select Diagnostic Feature to Analyze:", list(categories.keys()), horizontal=True)
    search_str = categories[selected_cat]
    
    if search_str == 'age':
        # Protect against 'stAGE' pulling stage data into age queries
        matches = [idx for idx in summ.index if 'age' in str(idx).lower() and 'stage' not in str(idx).lower()]
    else:
        matches = [idx for idx in summ.index if search_str.lower() in str(idx).lower()]
    
    if not matches:
        st.markdown(f"<div style='padding: 10px; color: #94a3b8; font-style: italic; font-size: 0.9rem;'>{selected_cat} is not statistically registered in the current active clinical models.</div>", unsafe_allow_html=True)
    else:
        n = len(matches)
        
        # Center cards proportionally on the screen by buffering empty columns on the sides
        if n == 1:
            cols = st.columns([1.5, 1, 1.5])  # 37.5%, 25%, 37.5%
        elif n == 2:
            cols = st.columns([1, 1, 1, 1])   # 25% each
        elif n == 3:
            cols = st.columns([0.5, 1, 1, 1, 0.5]) # 12.5%, 25%, 25%, 25%, 12.5%
        else:
            cols = st.columns(4)              # Default 4-grid

        for i, idx in enumerate(matches):
            if n == 1: col_target = cols[1]
            elif n in [2, 3]: col_target = cols[i + 1]
            else: col_target = cols[i % 4]
            row = summ.loc[idx]
            display_idx = get_display_name(idx)
            hr = row['exp(coef)']
            p_val = row['p']
            direction = "Hazard Elevation" if hr > 1 else "Protective Effect"
            magnitude = abs(hr - 1) * 100
            
            if p_val < 0.01: marker = "🔴 Critical (p < 0.01)"
            elif p_val < 0.05: marker = "🟠 Significant (p < 0.05)"
            else: marker = "🔵 Observation (p >= 0.05)"
            
            with col_target:
                st.markdown(f"""
                <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border-top: 4px solid #3b82f6; border-left: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                    <strong style="font-size: 1.1rem; color: #1e293b;">{display_idx}</strong><br/>
                    <span style="font-size: 0.85rem; font-weight: 600;">{marker}</span><br/><br/>
                    <span style="font-size: 0.95rem; color: #334155;">
                    <b>Direction:</b> {direction}<br/>
                    <b>Impact:</b> {magnitude:.1f}% {'increase' if hr > 1 else 'reduction'} in risk.<br/>
                    <b>Hazard Ratio:</b> {hr:.4f}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # 3. Practical Implications with better styling
    st.markdown("---")
    st.markdown("#### 🚀 Clinical Strategy & Implications")
    imp_col1, imp_col2 = st.columns(2)
    with imp_col1:
        st.markdown(f"""
        <div style="background-color: #f0fdf4; padding: 20px; border-radius: 12px; border-left: 6px solid #22c55e;">
            <h5 style="color: #166534;">Tiered Risk Management</h5>
            <p style="font-size: 0.95rem; color: #14532d;">
            Patients identified as <b>High Risk</b> (Severe Category in Prediction Tool) 
            should be considered for prioritized adjuvant systemic therapy and quarterly screening.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with imp_col2:
        # Calculate dynamic window around median survival
        med_val = stats["median_surv"]
        try:
            med_float = float(med_val)
            if np.isfinite(med_float):
                window_start = max(0, med_float - 6)
                window_end = med_float + 6
                window_text = f"the {window_start:.0f}-{window_end:.1f} month window"
            else:
                window_text = "the late-stage recovery phase"
        except:
            window_text = "the post-observation phase"
            
        st.markdown(f"""
        <div style="background-color: #eff6ff; padding: 20px; border-radius: 12px; border-left: 6px solid #3b82f6;">
            <h5 style="color: #1e40af;">Observation Optimization</h5>
            <p style="font-size: 0.95rem; color: #1e3a8a;">
            Critical survival dynamics are observed around <b>{window_text}</b>. 
            Intensive radiological monitoring is critical during this period.
            </p>
        </div>
        """, unsafe_allow_html=True)

elif selection == "📑 Model Comparison":
    comp = []
    for m, i in results['parametric_results'].items():
        comp.append({'Model': m, 'AIC': i['aic'], 'BIC': i['bic']})
    st.dataframe(pd.DataFrame(comp).sort_values('AIC'), use_container_width=True)

elif selection == "🧮 Survival Prediction Tool":
    inputs = {}
    cols = st.columns(2)
    for i, cov in enumerate(meta['covariates']):
        with cols[i % 2]:
            display_label = get_display_name(cov)
            if cov in meta['categorical_cols']:
                inputs[cov] = st.selectbox(f"{display_label}", df[cov].unique())
            else:
                # Force integers for Age or other whole-number typical fields
                col_min = int(df[cov].min())
                col_max = int(df[cov].max())
                col_val = int(df[cov].mean())
                
                if "age" in cov.lower():
                    inputs[cov] = st.slider(f"{display_label}", col_min, col_max, col_val, step=1)
                else:
                    inputs[cov] = st.slider(f"{display_label}", float(df[cov].min()), float(df[cov].max()), float(df[cov].mean()))
    
    # Dynamic Prediction (auto-updates on change)
    st.markdown("---")
    input_df = pd.DataFrame([inputs])
    
    # Ensure categorical types are consistent with training data for correct dummy encoding
    for col in meta['categorical_cols']:
        if col in input_df.columns:
            input_df[col] = pd.Categorical(input_df[col], categories=results['df'][col].cat.categories)
    
    input_enc = pd.get_dummies(input_df, columns=meta['categorical_cols'], drop_first=True)
    
    train_cols = results['df_encoded'].drop(columns=[meta['duration_col'], meta['event_col']]).columns
    for c in train_cols:
        if c not in input_enc.columns: input_enc[c] = 0
    input_enc = input_enc[train_cols]
    
    cph = results['cph_model']
    pred_surv = cph.predict_survival_function(input_enc)
    
    # Area Under Curve for Life Expectancy (RMST)
    times = pred_surv.index.values
    surv_vals = pred_surv.iloc[:, 0].values
    # Use np.trapezoid (NumPy 2.0+) or fallback for compatibility
    try:
        expected_months = np.trapezoid(surv_vals, times)
    except AttributeError:
        expected_months = np.trapz(surv_vals, times)
    
    # Survival probabilities at milestones
    milestones = [12, 24, 60]
    prob_milestones = {}
    for m in milestones:
        if m <= times.max():
            prob = cph.predict_survival_function(input_enc, times=[m]).iloc[0, 0]
            prob_milestones[m] = f"{prob*100:.1f}%"
        else:
            prob_milestones[m] = "Beyond Data"

    st.markdown("### 📊 Patient-Specific Survival Prognosis")
    
    # Check median to conditionally display
    try: med = cph.predict_median(input_enc).iloc[0]
    except: med = np.inf
    
    if med != np.inf and not np.isnan(med):
        col_m1, col_m2, col_m3 = st.columns(3)
    else:
        col_m1, col_m2 = st.columns(2)
        col_m3 = None
        
    with col_m1:
        st.metric("Predicted Survival Time", f"{expected_months:.1f} Months", help="Technically RMST: The average expected months of survival based on the cohort data.")
    with col_m2:
        st.metric("Relative Hazard Score", f"{cph.predict_partial_hazard(input_enc).iloc[0]:.2f}")
    if col_m3:
        st.metric("Median Survival", f"{med:.1f} Mo.")

    st.markdown("#### Chance of Survival by Timeframe")
    p1_bg, p1_text, p1_border = get_sev_color(float(prob_milestones[12].strip('%'))/100 if "Beyond" not in prob_milestones[12] else 0)
    p2_bg, p2_text, p2_border = get_sev_color(float(prob_milestones[24].strip('%'))/100 if "Beyond" not in prob_milestones[24] else 0)
    p5_bg, p5_text, p5_border = get_sev_color(float(prob_milestones[60].strip('%'))/100 if "Beyond" not in prob_milestones[60] else 0)
    
    m_cols = st.columns(3)
    milestone_data = [
        (12, prob_milestones[12], p1_bg, p1_text, p1_border),
        (24, prob_milestones[24], p2_bg, p2_text, p2_border),
        (60, prob_milestones[60], p5_bg, p5_text, p5_border)
    ]
    
    for i, (m, p, bg, txt, brd) in enumerate(milestone_data):
        m_cols[i].markdown(f"""
        <div class="info-box" style="text-align: center; background-color: {bg}; border-left: 5px solid {brd}; color: {txt};">
            <small>{m} Months Survival Chance</small><br/>
            <strong style="font-size: 1.4rem;">{p}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    fig = go.Figure(go.Scatter(
        x=pred_surv.index.tolist(), 
        y=pred_surv.iloc[:,0].values, 
        name="Clinical Pathway", 
        line=dict(color='#0ea5e9', width=CHART_LINE_WIDTH),
        fill='tozeroy',
        fillcolor='rgba(14, 165, 233, 0.05)',
        hovertemplate="<span style='font-size:36px;'>Expected: %{y:.1%}</span><extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Personalized Survival Projection</b>",
            'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
        },
        xaxis={'title': 'Time Post-Diagnosis (Months)', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000', 'tickprefix': 'Duration: ', 'ticksuffix': ' Month', 'showtickprefix': 'first', 'showticksuffix': 'first', 'hoverformat': '.0f'},
        yaxis={'title': 'Survival Probability', 'title_font': {'size': 18, 'color': '#000000'}, 'tickfont': {'size': 14, 'color': '#000000'}, 'showline': True, 'linecolor': '#000000'},
        template="plotly_white",
        hovermode="x unified",
        hoverlabel=dict(font_size=36, font_family="Arial", bgcolor="white"),
        height=500,
        margin=dict(t=100, b=80, l=80)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Stratification Logic
    risk_score = cph.predict_partial_hazard(input_enc).iloc[0]
    mean_risk = cph.predict_partial_hazard(results['df_encoded'].drop(columns=[meta['duration_col'], meta['event_col']])).mean()
    strat = clinical_risk_stratification(risk_score, mean_risk)
    st.subheader(f"Risk Category: :blue[{strat}]")

# Final Cleanup
st.markdown('</div>', unsafe_allow_html=True)