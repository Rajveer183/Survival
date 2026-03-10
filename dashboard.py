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
    header {visibility: hidden;}
    
    /* Full page background with light blue shades */
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
        margin-top: -30px;
        margin-bottom: 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 15px rgba(3, 105, 161, 0.05);
    }
    
    .main-title {
        color: #0369a1;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
    }
    
    .sub-title {
        color: #0c4a6e;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.1);
        text-align: center;
        border-top: 6px solid #3b82f6;
    }

    /* Info Boxes for Results */
    .info-box {
        background-color: #f0f9ff;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #1e40af;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.02);
    }

    .stSidebar {
        background: linear-gradient(180deg, #f8fafc 0%, #e0f2fe 100%) !important;
        border-right: 1px solid #bae6fd;
    }

    /* Additional UI Colors */
    .highlight-blue { color: #0369a1; font-weight: bold; }
    .highlight-green { color: #166534; font-weight: bold; }
    .highlight-red { color: #991b1b; font-weight: bold; }
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
DATA_PATH = "Colorectal Cancer Patient Data.csv"

@st.cache_data
def get_analysis_results():
    # Load and run pipeline directly in memory
    return run_pipeline(DATA_PATH)

results = get_analysis_results()
df = results['df']
meta = results['meta']
stats = results['summary_stats']

# --- HEADER ---
st.markdown(f"""
    <div class="header-container">
        <h1 class="main-title">Multivariate Survival Modeling and Visualization of Cancer Patient Data</h1>
        <p class="sub-title">A complete survival suite featuring Non-parametric, Parametric, and Multivariate statistical modeling with diagnostic evaluation.</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3233/3233483.png", width=80)
st.sidebar.title("Survival Suite")
selection = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard Overview",
        "📉 Non-Parametric Analysis",
        "📈 Parametric Survival Models",
        "📊 Multivariate Analysis",
        "🔬 Assumptions & Diagnostics",
        "💡 Results Interpretation",
        "📑 Model Comparison",
        "🧮 Survival Prediction Tool"
    ]
)

# --- SECTIONS ---

if selection == "🏠 Dashboard Overview":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><small>Cohort Size</small><h3>{stats["total_obs"]}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><small>Mortality Events</small><h3>{stats["events"]}</h3></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><small>Censored Rate</small><h3>{stats["censoring_rate"]}%</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><small>Median Surv.</small><h3>{stats["median_surv"]} Mo.</h3></div>', unsafe_allow_html=True)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_distribution(df[meta['duration_col']]), use_container_width=True)
    with col_b:
        count_df = pd.DataFrame({'Status': ['Event (Mortality)', 'Censored (Survivor/Lost)'], 'Count': [stats['events'], stats['censored']]})
        fig = px.pie(
            count_df, values='Count', names='Status', 
            title="Cohort Vital Status Distribution", 
            hole=0.6, 
            color_discrete_sequence=['#ef4444', '#3b82f6']
        )
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

elif selection == "📉 Non-Parametric Analysis":
    st.subheader("Kaplan-Meier & Nelson-Aalen Estimation")
    strata = st.selectbox(
        "Stratification", 
        ["None"] + meta['categorical_cols'],
        format_func=get_display_name
    )
    
    display_strata = get_display_name(strata)
    if strata == "None":
        kmf_dict, naf_dict = results['kmf_overall'], results['naf_overall']
    else:
        kmf_dict = fit_kaplan_meier(df, meta['duration_col'], meta['event_col'], group_col=strata)
        naf_dict = fit_nelson_aalen(df, meta['duration_col'], meta['event_col'], group_col=strata)
    
    tab1, tab2 = st.tabs(["Kaplan-Meier Survival Analysis", "Nelson-Aalen Cumulative Hazard"])
    with tab1:
        p_val = None
        if strata != "None":
            lr_res = compare_groups(df, meta['duration_col'], meta['event_col'], strata)
            if lr_res:
                p_val = lr_res.p_value
        
        km_title = f"Kaplan-Meier Survival Probability Estimates (Stratified by {display_strata})" if strata != "None" else "Overall Kaplan-Meier Survival Probability Estimate"
        st.plotly_chart(plot_km_curves(kmf_dict, title=km_title, p_value=p_val), use_container_width=True)
        
        st.markdown("#### Clinical Prognostics: Median & Survival Probabilities")
        for label, kmf in kmf_dict.items():
            cols = st.columns(4)
            m_val = kmf.median_survival_time_
            disp_med = f"{m_val:.1f} Mo." if m_val != np.inf and not np.isnan(m_val) else "Not Reached"
            
            # Probability at milestones
            p1 = kmf.survival_function_at_times(12).iloc[0]
            p2 = kmf.survival_function_at_times(24).iloc[0]
            p5 = kmf.survival_function_at_times(60).iloc[0] if 60 <= kmf.timeline.max() else None
            
            p1_bg, p1_text, p1_border = get_sev_color(p1)
            p2_bg, p2_text, p2_border = get_sev_color(p2)
            p5_bg, p5_text, p5_border = get_sev_color(p5 if p5 is not None else 0)
            
            with cols[0]: st.markdown(f'<div class="info-box" style="border-left: 5px solid #3b82f6;"><small>{label} Median</small><br/><strong>{disp_med}</strong></div>', unsafe_allow_html=True)
            with cols[1]: st.markdown(f'<div class="info-box" style="background-color: {p1_bg}; border-left: 5px solid {p1_border}; color: {p1_text};"><small>1-Yr Surv.</small><br/><strong>{p1*100:.1f}%</strong></div>', unsafe_allow_html=True)
            with cols[2]: st.markdown(f'<div class="info-box" style="background-color: {p2_bg}; border-left: 5px solid {p2_border}; color: {p2_text};"><small>2-Yr Surv.</small><br/><strong>{p2*100:.1f}%</strong></div>', unsafe_allow_html=True)
            with cols[3]: 
                disp_p5 = f"{p5*100:.1f}%" if p5 is not None else "N/A"
                if p5 is not None:
                    st.markdown(f'<div class="info-box" style="background-color: {p5_bg}; border-left: 5px solid {p5_border}; color: {p5_text};"><small>5-Yr Surv.</small><br/><strong>{disp_p5}</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-box"><small>5-Yr Surv.</small><br/><strong>{disp_p5}</strong></div>', unsafe_allow_html=True)
    with tab2:
        haz_title = f"Nelson-Aalen Cumulative Hazard Function (Stratified by {display_strata})" if strata != "None" else "Overall Nelson-Aalen Cumulative Hazard Function"
        st.plotly_chart(plot_hazard_curves(naf_dict, title=haz_title), use_container_width=True)

elif selection == "📈 Parametric Survival Models":
    p_results, best_model = results['parametric_results'], results['best_parametric']
    st.info(f"Recommended Model: **{best_model}**")
    sel_mod = st.selectbox("Model Fit", list(p_results.keys()))
    info = p_results[sel_mod]
    fitter = info['fitter']
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        real_km = results['kmf_overall']['Overall'].survival_function_
        fig.add_trace(go.Scatter(x=real_km.index.tolist(), y=real_km['Overall'].values, name='KM Estimate', line=dict(color='black', dash='dash')))
        timeline = np.linspace(0, df[meta['duration_col']].max(), 100)
        fig.add_trace(go.Scatter(x=timeline.tolist(), y=fitter.survival_function_at_times(timeline).values, 
                                 name=f'{sel_mod} Fit', line=dict(color=COLORS[0], width=CHART_LINE_WIDTH)))
        fig.update_layout(template="plotly_white", title="<b>Parametric Survival Function Against Kaplan-Meier Estimate</b>", margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline.tolist(), y=fitter.hazard_at_times(timeline).values, 
                                 name=f'{sel_mod} Hazard', line=dict(color=COLORS[1], width=CHART_LINE_WIDTH)))
        fig.update_layout(template="plotly_white", title="<b>Estimated Hazards Rate (Parametric Modeling)</b>", margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Model-Based Predicted Survival Chances")
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
    tab1, tab2 = st.tabs(["Cox PH", "AFT Models"])
    with tab1:
        cph = results['cph_model']
        summary_to_show = cph.summary.copy()
        summary_to_show.index = [get_display_name(x) for x in summary_to_show.index]
        st.plotly_chart(plot_forest_cox(cph, display_map=DISPLAY_NAME_MAP), use_container_width=True)
        # Robust summary display
        if isinstance(summary_to_show.index, pd.MultiIndex): 
            summary_to_show.index = ['_'.join(map(str, x)) for x in summary_to_show.index]
        st.dataframe(summary_to_show, use_container_width=True)
    with tab2:
        aft_models = results['aft_models']
        aft_name = st.selectbox("AFT Variant", list(aft_models.keys()))
        aft_model = aft_models[aft_name]
        st.write(f"**{aft_name} Model Summary**")
        summ_aft = aft_model.summary.copy()
        
        # Mapping labels
        if isinstance(summ_aft.index, pd.MultiIndex):
            summ_aft.index = [f"{get_display_name(l1)}_{l2}" for l1, l2 in summ_aft.index]
        else:
            summ_aft.index = [get_display_name(x) for x in summ_aft.index]
            
        summ_aft = summ_aft.reset_index()
        # Flatten MultiIndex columns if they exist
        if isinstance(summ_aft.columns, pd.MultiIndex):
            summ_aft.columns = ['_'.join(map(str, col)).strip() for col in summ_aft.columns.values]
        st.dataframe(summ_aft, use_container_width=True)
        
        # Prepare data for plotting
        plot_df = summ_aft.copy()
        param_col = plot_df.columns[0]
        plot_df['Parameter'] = plot_df[param_col].astype(str)
        fig = px.bar(plot_df, x='Parameter', y='coef', title=f"Accelerated Failure Time (AFT): {aft_name} Effect Sizes", color='coef', color_continuous_scale='Portland')
        fig.update_layout(template="plotly_white", margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

elif selection == "🔬 Assumptions & Diagnostics":
    cph, df_enc = results['cph_model'], results['df_encoded']
    col1, col2 = st.columns(2)
    with col1:
        try:
            cph.check_assumptions(df_enc, p_value_threshold=0.05)
            st.success("✅ Proportional Hazards assumption met.")
        except Exception as e:
            st.warning("⚠️ PH Violations detected. See details below.")
            st.expander("Diagnostic Output").write(str(e))
    with col2:
        res = compute_residuals(cph, df_enc)
        # Ensure clean_res is a 1D array with enough data points
        mr = res['martingale'].dropna()
        if len(mr) > 5:  # Need sufficient points for a meaningful Q-Q plot and regression
            clean_res = np.asarray(mr.values.flatten(), dtype=np.float64)
            try:
                # Scipy probplot can be sensitive to shapes in newer versions
                (osm, osr), (slope, intercept, r) = scipy_stats.probplot(clean_res, dist="norm")
                osm = np.asarray(osm).flatten()
                osr = np.asarray(osr).flatten()
                
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=osm.tolist(), y=osr.tolist(), mode='markers', name='Residuals', marker=dict(color='#3b82f6')))
                fig_qq.add_trace(go.Scatter(x=osm.tolist(), y=(slope*osm + intercept).tolist(), mode='lines', name='Normal Fit', line=dict(color='#ef4444', width=2)))
                fig_qq.update_layout(title="Normal Q-Q Plot of Martingale Residuals", xaxis_title="Theoretical Quantiles", yaxis_title="Ordered Values")
                st.plotly_chart(fig_qq, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating Q-Q plot: {e}")
        else:
            st.warning("Insufficient residuals to perform Q-Q diagnostics.")
            
    col3, col4 = st.columns(2)
    with col3:
        # INDESTRUCTIBLE ALIGNMENT LOGIC
        # Fix: compute_residuals may return a DataFrame or Series. Force to Series if multi-column.
        m_ser = res['martingale']
        d_ser = res['deviance']
        
        # Extract first column if DataFrame (to fix the length 186 vs 62 issue)
        if hasattr(m_ser, 'iloc'): m_ser = m_ser.iloc[:, 0] if m_ser.ndim > 1 else m_ser
        if hasattr(d_ser, 'iloc'): d_ser = d_ser.iloc[:, 0] if d_ser.ndim > 1 else d_ser
        
        # Use pandas automatic index alignment (join) matching the fitted dataframe
        # This solves the "Length of original vs index" mismatch
        res_df = pd.DataFrame({'M': m_ser, 'D': d_ser})
        
        # Join with the time column from the source data
        diag_df = res_df.join(df_enc[[meta['duration_col']]]).dropna()
        diag_df.rename(columns={meta['duration_col']: 'Time', 'M': 'Martingale', 'D': 'Deviance'}, inplace=True)
        
        # COLORFUL DIAGNOSTIC PLOT
        fig = px.scatter(diag_df, x='Time', y='Martingale', 
                         labels={'Time': 'Duration (Months)', 'Martingale': 'Martingale Residual'},
                         title="Martingale Trend Analysis", color_discrete_sequence=[COLORS[0]])
        fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white')))
        fig.add_hline(y=0, line_dash="dash", line_color="#1e293b", line_width=2)
        fig.update_layout(template="plotly_white", margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        # COLORFUL DIAGNOSTIC PLOT
        fig = px.scatter(diag_df, x='Time', y='Deviance', 
                         labels={'Time': 'Duration (Months)', 'Deviance': 'Deviance Residual'},
                         title="Deviance Trend Analysis", color_discrete_sequence=[COLORS[1]])
        fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white')))
        fig.add_hline(y=0, line_dash="dash", line_color="#1e293b", line_width=2)
        fig.update_layout(template="plotly_white", margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

elif selection == "💡 Results Interpretation":
    st.markdown("### 🔬 Clinical & Statistical Interpretation")
    
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

    # 2. Effect of Covariates
    st.markdown("#### 2. Effect of Covariates on Survival Outcomes")
    summ = results['cph_model'].summary
    sig = summ[summ['p'] < 0.05]
    
    if not sig.empty:
        st.markdown("Analytical Evaluation of Significant Variables:")
        for idx, row in sig.iterrows():
            display_idx = get_display_name(idx)
            hr = row['exp(coef)']
            direction = "Hazard Elevation" if hr > 1 else "Protective Effect"
            magnitude = abs(hr - 1) * 100
            
            detail = f"""
            - **{display_idx}**: Demonstrates a **{direction}**.
              - *Quantification*: Every unit increase in {display_idx} results in a **{magnitude:.1f}%** {'increase' if hr > 1 else 'reduction'} in the instantaneous risk of mortality.
              - *Statistical Weight*: The p-value ({row['p']:.4f}) indicates that this covariate is a critical differentiator in patient survival trajectories. Clinical treatment plans should be aggressively adjusted for patients with high values of this predictor.
            """
            st.markdown(detail)
    else:
        st.warning("No statistically significant factors found at p < 0.05.")

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
        st.markdown(f"""
        <div style="background-color: #eff6ff; padding: 20px; border-radius: 12px; border-left: 6px solid #3b82f6;">
            <h5 style="color: #1e40af;">Observation Optimization</h5>
            <p style="font-size: 0.95rem; color: #1e3a8a;">
            The <b>24-36 month window</b> shows the sharpest decline in survival. 
            Intensive radiological monitoring is critical during this period.
            </p>
        </div>
        """, unsafe_allow_html=True)

elif selection == "📑 Model Comparison":
    comp = []
    for m, i in results['parametric_results'].items():
        comp.append({'Model': m, 'AIC': i['aic'], 'BIC': i['bic']})
    cph = results['cph_model']
    comp.append({'Model': 'Cox PH', 'AIC': cph.AIC_partial_, 'BIC': np.nan})
    st.dataframe(pd.DataFrame(comp).sort_values('AIC'), use_container_width=True)

elif selection == "🧮 Survival Prediction Tool":
    st.subheader("Patient Clinical Assessment")
    inputs = {}
    cols = st.columns(2)
    for i, cov in enumerate(meta['covariates']):
        with cols[i % 2]:
            display_label = get_display_name(cov)
            if cov in meta['categorical_cols']:
                inputs[cov] = st.selectbox(f"{display_label}", df[cov].unique())
            else:
                inputs[cov] = st.slider(f"{display_label}", float(df[cov].min()), float(df[cov].max()), float(df[cov].mean()))
    
    if st.button("Generate Comprehensive Clinical Prediction"):
        input_enc = pd.get_dummies(pd.DataFrame([inputs]))
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
            fillcolor='rgba(14, 165, 233, 0.05)'
        ))
        
        fig.update_layout(
            title="<b>Projected Patient Survival Trajectory</b>",
            xaxis_title="Time Following Diagnosis (Months)", 
            yaxis_title="Probability of Survival", 
            template="plotly_white",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stratification Logic
        risk_score = cph.predict_partial_hazard(input_enc).iloc[0]
        mean_risk = cph.predict_partial_hazard(results['df_encoded'].drop(columns=[meta['duration_col'], meta['event_col']])).mean()
        strat = clinical_risk_stratification(risk_score, mean_risk)
        st.subheader(f"Risk Category: :blue[{strat}]")

# Final Cleanup
st.markdown('</div>', unsafe_allow_html=True)
