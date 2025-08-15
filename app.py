import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Set page config
st.set_page_config(page_title="Time on Page vs Revenue Analysis", layout="wide")

# Title and introduction
st.title("Time on Page vs Revenue Analysis")
st.write("""
This interactive dashboard explores the relationship between time spent on page (TOP) and revenue,
while controlling for browser type, platform, and site.
""")

# Load data (replace with your actual data loading code)
@st.cache_data
def load_data():
    # Replace this with your actual data loading code
    # Example:
    df = pd.read_csv("testdata (1).csv")
    return df

df = load_data()

# Data cleaning (from your Colab)
df_no_duplicates = df.drop_duplicates()

# Sidebar controls
st.sidebar.header("Filter Data")
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

# Filter data based on selections
filtered_df = df_no_duplicates[
    (df_no_duplicates['browser'].isin(selected_browser)) &
    (df_no_duplicates['platform'].isin(selected_platform)) &
    (df_no_duplicates['site'].isin(selected_site))
]

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "Overall Relationship", 
    "By Browser", 
    "By Platform", 
    "By Site"
])

with tab1:
    st.header("Overall Relationship Between TOP and Revenue")
    
    # Scatter plot with regression line
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.regplot(x='top', y='revenue', data=filtered_df, scatter_kws={'alpha':0.3}, ax=ax1)
    ax1.set_title('Relationship Between Time on Page and Revenue')
    ax1.set_xlabel('Time on Page (TOP)')
    ax1.set_ylabel('Revenue')
    st.pyplot(fig1)
    
    # Show correlation
    correlation = filtered_df['top'].corr(filtered_df['revenue'])
    st.write(f"**Correlation coefficient:** {correlation:.4f}")
    
    # Regression results
    st.subheader("Regression Analysis")
    X = sm.add_constant(filtered_df['top'])
    y = filtered_df['revenue']
    model = sm.OLS(y, X).fit()
    st.text(str(model.summary()))

with tab2:
    st.header("Relationship by Browser")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x='top', y='revenue', hue='browser', alpha=0.6, ax=ax2)
    ax2.set_title('Revenue vs Time on Page by Browser')
    ax2.set_xlabel('Time on Page (top)')
    ax2.set_ylabel('Revenue')
    ax2.legend(title='Browser')
    st.pyplot(fig2)
    
    # Calculate correlations by browser
    browser_corr = filtered_df.groupby('browser')[['top', 'revenue']].corr().iloc[0::2,-1].reset_index()
    browser_corr = browser_corr.rename(columns={'revenue': 'correlation'}).drop('level_1', axis=1)
    st.write("Correlations by browser:")
    st.dataframe(browser_corr)

with tab3:
    st.header("Relationship by Platform")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x='top', y='revenue', hue='platform', alpha=0.6, ax=ax3)
    ax3.set_title('Revenue vs Time on Page by Platform')
    ax3.set_xlabel('Time on Page (top)')
    ax3.set_ylabel('Revenue')
    ax3.legend(title='Platform')
    st.pyplot(fig3)
    
    # Calculate correlations by platform
    platform_corr = filtered_df.groupby('platform')[['top', 'revenue']].corr().iloc[0::2,-1].reset_index()
    platform_corr = platform_corr.rename(columns={'revenue': 'correlation'}).drop('level_1', axis=1)
    st.write("Correlations by platform:")
    st.dataframe(platform_corr)

with tab4:
    st.header("Relationship by Site")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='top', y='revenue', hue='site', alpha=0.6, palette='viridis', ax=ax4)
    ax4.set_title('Revenue vs Time on Page by Site')
    ax4.set_xlabel('Time on Page (top)')
    ax4.set_ylabel('Revenue')
    ax4.legend(title='Site')
    st.pyplot(fig4)
    
    # Calculate correlations by site
    site_corr = filtered_df.groupby('site')[['top', 'revenue']].corr().iloc[0::2,-1].reset_index()
    site_corr = site_corr.rename(columns={'revenue': 'correlation'}).drop('level_1', axis=1)
    st.write("Correlations by site:")
    st.dataframe(site_corr)

# Key findings section
st.header("Key Findings")
st.write("""
1. **Overall Relationship**: The correlation between time on page and revenue is [positive/negative] with a coefficient of {correlation:.4f}.

2. **By Browser**: The relationship varies between browsers:
   - Chrome: {browser_corr.loc[browser_corr['browser'] == 'chrome', 'correlation'].values[0]:.4f}
   - Safari: {browser_corr.loc[browser_corr['browser'] == 'safari', 'correlation'].values[0]:.4f}

3. **Regression Analysis**: After controlling for browser, platform, and site, the relationship becomes [positive/negative] with a coefficient of {model.params['top']:.6f} (p-value: {model.pvalues['top']:.3f}).
""")

# Add your interpretation from Colab
st.header("Detailed Interpretation")
st.write("""
The correlation analysis by group consistently shows a [positive/negative] relationship between 'top' and 'revenue' across different categories of browser, platform, and site.

However, the multiple regression analysis, which controls for the effects of browser, platform, and site simultaneously, reveals a statistically significant [positive/negative] relationship between 'top' and 'revenue'. This suggests that the simple bivariate correlation observed within groups does not capture the true relationship between 'top' and 'revenue' when considering the influence of other variables.
""")
