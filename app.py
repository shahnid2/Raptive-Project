import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Time on Page vs Revenue Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("📈 Time on Page vs Revenue Analysis")
st.write("""
This interactive dashboard explores the relationship between time spent on page (TOP) and revenue,
while controlling for browser type, platform, and site.
""")

# Load data
@st.cache_data
def load_data():
    # Method 1: Direct download from GitHub (recommended)
    url = "https://raw.githubusercontent.com/shahnid2/Raptive-Project/main/testdata.csv"
    df = pd.read_csv(url)
    
    # Method 2: If you want to load locally (alternative)
    # df = pd.read_csv("testdata.csv")
    
    return df

df = load_data()

# Data cleaning
@st.cache_data
def clean_data(df):
    df_no_duplicates = df.drop_duplicates()
    return df_no_duplicates

df_no_duplicates = clean_data(df)

# Sidebar controls
st.sidebar.header("🔍 Data Filters")
selected_browser = st.sidebar.multiselect(
    "Select browser(s)",
    options=df_no_duplicates['browser'].unique(),
    default=df_no_duplicates['browser'].unique()
)

selected_platform = st.sidebar.multiselect(
    "Select platform(s)",
    options=df_no_duplicates['platform'].unique(),
    default=df_no_duplicates['platform'].unique()
)

selected_site = st.sidebar.multiselect(
    "Select site(s)",
    options=df_no_duplicates['site'].unique(),
    default=df_no_duplicates['site'].unique()
)

# Visualization settings
st.sidebar.header("🎨 Visualization Settings")
show_regression = st.sidebar.checkbox("Show regression line", value=True)
alpha = st.sidebar.slider("Point transparency", 0.1, 1.0, 0.6)
point_size = st.sidebar.slider("Point size", 10, 100, 30)

# Data download
st.sidebar.markdown("---")
st.sidebar.header("💾 Export Data")
st.sidebar.download_button(
    label="Download filtered data as CSV",
    data=df_no_duplicates.to_csv(index=False).encode('utf-8'),
    file_name='filtered_data.csv',
    mime='text/csv'
)

# Filter data based on selections
filtered_df = df_no_duplicates[
    (df_no_duplicates['browser'].isin(selected_browser)) &
    (df_no_duplicates['platform'].isin(selected_platform)) &
    (df_no_duplicates['site'].isin(selected_site))
]

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overall Relationship", 
    "🌐 By Browser", 
    "📱 By Platform", 
    "🏢 By Site",
    "🧪 Advanced Analysis"
])

with tab1:
    st.header("Overall Relationship Between TOP and Revenue")
    
    # Scatter plot with regression line
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    if show_regression:
        sns.regplot(x='top', y='revenue', data=filtered_df, 
                   scatter_kws={'alpha':alpha, 's':point_size}, 
                   line_kws={'color':'red'}, ax=ax1)
    else:
        sns.scatterplot(x='top', y='revenue', data=filtered_df, 
                       alpha=alpha, s=point_size, ax=ax1)
    ax1.set_title('Relationship Between Time on Page and Revenue')
    ax1.set_xlabel('Time on Page (TOP)')
    ax1.set_ylabel('Revenue')
    st.pyplot(fig1)
    
    # Show correlation
    correlation = filtered_df['top'].corr(filtered_df['revenue'])
    st.metric("Correlation coefficient", f"{correlation:.4f}")
    
    # Simple regression results
    st.subheader("Simple Regression Analysis")
    X = sm.add_constant(filtered_df['top'])
    y = filtered_df['revenue']
    model = sm.OLS(y, X).fit()
    
    # Display results in expander
    with st.expander("View Regression Results"):
        st.text(model.summary().as_text())

with tab2:
    st.header("Relationship by Browser")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='top', y='revenue', 
                   hue='browser', alpha=alpha, s=point_size, ax=ax2)
    if show_regression:
        sns.regplot(x='top', y='revenue', data=filtered_df, 
                   scatter=False, ci=None, ax=ax2)
    ax2.set_title('Revenue vs Time on Page by Browser')
    ax2.set_xlabel('Time on Page (top)')
    ax2.set_ylabel('Revenue')
    ax2.legend(title='Browser')
    st.pyplot(fig2)
    
    # Calculate correlations by browser
    browser_corr = filtered_df.groupby('browser')[['top', 'revenue']].corr().iloc[0::2,-1].reset_index()
    browser_corr = browser_corr.rename(columns={'revenue': 'correlation'}).drop('level_1', axis=1)
    st.write("Correlations by browser:")
    st.dataframe(browser_corr.style.format({'correlation': '{:.4f}'}))

with tab3:
    st.header("Relationship by Platform")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='top', y='revenue', 
                   hue='platform', alpha=alpha, s=point_size, ax=ax3)
    if show_regression:
        sns.regplot(x='top', y='revenue', data=filtered_df, 
                   scatter=False, ci=None, ax=ax3)
    ax3.set_title('Revenue vs Time on Page by Platform')
    ax3.set_xlabel('Time on Page (top)')
    ax3.set_ylabel('Revenue')
    ax3.legend(title='Platform')
    st.pyplot(fig3)
    
    # Calculate correlations by platform
    platform_corr = filtered_df.groupby('platform')[['top', 'revenue']].corr().iloc[0::2,-1].reset_index()
    platform_corr = platform_corr.rename(columns={'revenue': 'correlation'}).drop('level_1', axis=1)
    st.write("Correlations by platform:")
    st.dataframe(platform_corr.style.format({'correlation': '{:.4f}'}))

with tab4:
    st.header("Relationship by Site")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='top', y='revenue', 
                   hue='site', alpha=alpha, s=point_size, 
                   palette='viridis', ax=ax4)
    if show_regression:
        sns.regplot(x='top', y='revenue', data=filtered_df, 
                   scatter=False, ci=None, ax=ax4)
    ax4.set_title('Revenue vs Time on Page by Site')
    ax4.set_xlabel('Time on Page (top)')
    ax4.set_ylabel('Revenue')
    ax4.legend(title='Site')
    st.pyplot(fig4)
    
    # Calculate correlations by site
    site_corr = filtered_df.groupby('site')[['top', 'revenue']].corr().iloc[0::2,-1].reset_index()
    site_corr = site_corr.rename(columns={'revenue': 'correlation'}).drop('level_1', axis=1)
    st.write("Correlations by site:")
    st.dataframe(site_corr.style.format({'correlation': '{:.4f}'}))

with tab5:
    st.header("Advanced Regression Analysis")
    
    st.write("Build your custom regression model by selecting predictors below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predictor selection
        predictors = st.multiselect(
            "Select predictors for regression model",
            options=['top', 'browser', 'platform', 'site'],
            default=['top', 'browser', 'platform']
        )
        
        # Model options
        remove_outliers = st.checkbox("Remove outliers (top/bottom 1%)", value=False)
    
    with col2:
        # Model diagnostics
        show_diagnostics = st.checkbox("Show model diagnostics", value=True)
        show_predictions = st.checkbox("Show predictions plot", value=True)
    
    if st.button("Run Regression Analysis", type="primary"):
        if not predictors:
            st.warning("Please select at least one predictor")
        else:
            # Prepare data
            temp_df = filtered_df.copy()
            
            if remove_outliers:
                for col in ['top', 'revenue']:
                    if col in predictors or col == 'revenue':
                        q_low = temp_df[col].quantile(0.01)
                        q_hi = temp_df[col].quantile(0.99)
                        temp_df = temp_df[(temp_df[col] < q_hi) & (temp_df[col] > q_low)]
            
            X = temp_df[predictors]
            y = temp_df['revenue']
            
            # Convert categoricals to dummy variables
            if 'browser' in predictors:
                X = pd.get_dummies(X, columns=['browser'], drop_first=True)
            if 'platform' in predictors:
                X = pd.get_dummies(X, columns=['platform'], drop_first=True)
            if 'site' in predictors:
                X = pd.get_dummies(X, columns=['site'], drop_first=True)
            
            X = sm.add_constant(X)
            
            # Run regression
            model = sm.OLS(y, X).fit()
            
            # Display results
            st.subheader("Regression Results")
            
            with st.expander("View Detailed Results"):
                st.text(model.summary().as_text())
            
            # Model diagnostics
            if show_diagnostics:
                st.subheader("Model Diagnostics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residuals plot
                    fig_resid, ax_resid = plt.subplots(figsize=(8, 6))
                    sns.residplot(x=model.predict(), y=model.resid, lowess=True, 
                                 line_kws={'color': 'red'}, ax=ax_resid)
                    ax_resid.set_title("Residuals vs Fitted")
                    ax_resid.set_xlabel("Fitted values")
                    ax_resid.set_ylabel("Residuals")
                    st.pyplot(fig_resid)
                
                with col2:
                    # QQ plot
                    fig_qq, ax_qq = plt.subplots(figsize=(8, 6))
                    sm.qqplot(model.resid, line='s', ax=ax_qq)
                    ax_qq.set_title("Normal Q-Q Plot")
                    st.pyplot(fig_qq)
            
            # Plot actual vs predicted
            if show_predictions:
                st.subheader("Actual vs Predicted Values")
                fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
                sns.regplot(x=model.predict(), y=y, ax=ax_pred)
                ax_pred.set_xlabel("Predicted Revenue")
                ax_pred.set_ylabel("Actual Revenue")
                ax_pred.set_title("Actual vs Predicted Revenue")
                st.pyplot(fig_pred)

# Key findings section
st.markdown("---")
st.header("🔑 Key Findings")

# Calculate metrics for findings
correlation = filtered_df['top'].corr(filtered_df['revenue'])
browser_corr = filtered_df.groupby('browser')[['top', 'revenue']].corr().iloc[0::2,-1].reset_index()
browser_corr = browser_corr.rename(columns={'revenue': 'correlation'}).drop('level_1', axis=1)

# Simple regression for coefficient
X_simple = sm.add_constant(filtered_df['top'])
y_simple = filtered_df['revenue']
model_simple = sm.OLS(y_simple, X_simple).fit()

st.write(f"""
1. **Overall Relationship**: The correlation between time on page and revenue is {correlation:.4f}, indicating a {'positive' if correlation > 0 else 'negative'} relationship.

2. **By Browser**: The relationship varies between browsers:
   - Chrome: {browser_corr.loc[browser_corr['browser'] == 'chrome', 'correlation'].values[0]:.4f}
   - Safari: {browser_corr.loc[browser_corr['browser'] == 'safari', 'correlation'].values[0]:.4f}

3. **Regression Analysis**: The simple regression shows that each additional unit of time on page is associated with a change of {model_simple.params['top']:.6f} in revenue (p-value: {model_simple.pvalues['top']:.4f}).
""")

# Add interpretation from your analysis
st.markdown("""
## 📝 Interpretation

The analysis reveals several important insights:

1. **Correlation vs. Causation**: While the simple correlation shows a [positive/negative] relationship, the regression analysis that controls for other variables reveals [different insights].

2. **Browser Differences**: The relationship appears stronger for [browser] users compared to [other browser].

3. **Platform Impact**: Mobile users show [different pattern] compared to desktop users.

4. **Site Variations**: The strength of the relationship varies across sites, with Site [X] showing the strongest correlation.

Use the Advanced Analysis tab to explore custom regression models that control for different combinations of factors.
""")

# Add footer
st.markdown("---")
st.markdown("""
**Note**: This dashboard is interactive - use the filters in the sidebar to explore different segments of the data.
""")
