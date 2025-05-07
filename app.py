import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Streamlit setup
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("Global COVID-19 Dashboard")

# Sidebar settings
st.sidebar.title(" Dashboard Settings")
refresh = st.sidebar.checkbox("🔁 Refresh data", value=False)
top_n = st.sidebar.slider("Top N countries to display", 5, 30, 10)
selected_metric = st.sidebar.selectbox("Choropleth Metric", ["cases", "deaths", "mortality_rate", "recovery_rate"])
selected_country = st.sidebar.selectbox("Country profile", [])

st.sidebar.markdown("### Compare Countries")
compare_countries = st.sidebar.multiselect("Choose up to 2 countries", [])

# -------------------- Data Fetching --------------------
@st.cache_data(ttl=300 if refresh else None)
def fetch_country_data():
    url = "https://disease.sh/v3/covid-19/countries"
    r = requests.get(url)
    data = pd.json_normalize(r.json())
    df = data[['country', 'cases', 'todayCases', 'deaths', 'todayDeaths', 'recovered', 'population']].dropna()
    df['mortality_rate'] = df['deaths'] / df['cases']
    df['recovery_rate'] = df['recovered'] / df['cases']
    df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()
    return df.sort_values(by='cases', ascending=False)

@st.cache_data
def fetch_global_history():
    url = "https://disease.sh/v3/covid-19/historical/all?lastdays=30"
    r = requests.get(url)
    if r.status_code != 200:
        return pd.DataFrame()
    d = r.json()
    df = pd.DataFrame({
        'date': list(d['cases'].keys()),
        'cases': list(d['cases'].values()),
        'deaths': list(d['deaths'].values())
    })
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def fetch_country_history(country):
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays=30"
    r = requests.get(url)
    try:
        data = r.json()['timeline']
        df = pd.DataFrame({
            'date': list(data['cases'].keys()),
            'cases': list(data['cases'].values()),
            'deaths': list(data['deaths'].values())
        })
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

# -------------------- Load Data --------------------
df = fetch_country_data()
df_hist = fetch_global_history()

# Update dropdowns after loading
selected_country = st.sidebar.selectbox("Country profile", df['country'].unique())
compare_countries = st.sidebar.multiselect("Choose up to 2 countries", df['country'].unique(), default=['USA', 'India'])

# -------------------- KPIs --------------------
st.subheader(" Global Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric(" Total Cases", f"{df['cases'].sum():,}")
col2.metric(" Total Deaths", f"{df['deaths'].sum():,}")
col3.metric(" Mortality Rate", f"{df['mortality_rate'].mean()*100:.2f}%")
col4.metric(" Recovery Rate", f"{df['recovery_rate'].mean()*100:.2f}%")

# -------------------- Bar Chart --------------------
st.subheader(" Top Affected Countries")
top_df = df.nlargest(top_n, 'cases')
fig_bar = px.bar(top_df, x='cases', y='country', orientation='h', color='cases',
                 labels={'cases': 'Total Cases'}, title="Top Countries by Total Cases")
st.plotly_chart(fig_bar, use_container_width=True)

# -------------------- Choropleth Map --------------------
st.subheader(f" World Map by {selected_metric.replace('_', ' ').title()}")
fig_map = px.choropleth(df,
    locations="country",
    locationmode="country names",
    color=selected_metric,
    hover_name="country",
    color_continuous_scale="Reds" if selected_metric != 'recovery_rate' else 'Greens',
    title=f"Global {selected_metric.replace('_', ' ').title()}"
)
st.plotly_chart(fig_map, use_container_width=True)

# -------------------- Clustering --------------------
st.subheader(" Risk Clustering (KMeans)")
features = df[['cases', 'deaths', 'population', 'mortality_rate', 'recovery_rate']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

fig_cluster = px.scatter(df, x='cases', y='deaths', color='cluster',
                         hover_name='country', size='population',
                         title='COVID-19 Risk Clusters',
                         labels={'cases': 'Cases', 'deaths': 'Deaths'})
st.plotly_chart(fig_cluster, use_container_width=True)

# -------------------- Cluster Summaries --------------------
st.subheader("Cluster Summaries")
for c in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == c]
    with st.expander(f"Cluster {c} ({len(subset)} countries)"):
        st.markdown(f"""
        - Avg Cases: **{int(subset['cases'].mean()):,}**  
        - Avg Deaths: **{int(subset['deaths'].mean()):,}**  
        - Mortality Rate: **{subset['mortality_rate'].mean()*100:.2f}%**  
        - Recovery Rate: **{subset['recovery_rate'].mean()*100:.2f}%**  
        - Examples: {', '.join(subset['country'].head(3))}...
        """)

# -------------------- Country Profile --------------------
st.subheader(f" Country Profile: {selected_country}")
c_data = df[df['country'] == selected_country].squeeze()
st.markdown(f"""
- **Population:** {c_data['population']:,}  
- **Total Cases:** {c_data['cases']:,}  
- **Today's Cases:** {c_data['todayCases']:,}  
- **Total Deaths:** {c_data['deaths']:,}  
- **Today's Deaths:** {c_data['todayDeaths']:,}  
- **Recovered:** {c_data['recovered']:,}  
- **Mortality Rate:** {c_data['mortality_rate']*100:.2f}%  
- **Recovery Rate:** {c_data['recovery_rate']*100:.2f}%
""")

# -------------------- Per-Country Trend --------------------
st.subheader(f" {selected_country} - 30 Day Trend")
history_df = fetch_country_history(selected_country)

if not history_df.empty:
    fig_country_trend = px.line(history_df, x='date', y=['cases', 'deaths'], markers=True,
                                title=f"{selected_country} - COVID-19 Trend (Last 30 Days)",
                                labels={'value': 'Count', 'variable': 'Metric'})
    st.plotly_chart(fig_country_trend, use_container_width=True)
else:
    st.warning("No historical data available.")

# -------------------- Global History --------------------
if not df_hist.empty:
    st.subheader(" Global Trend (Last 30 Days)")
    fig_hist = px.line(df_hist, x='date', y=['cases', 'deaths'], markers=True,
                       labels={'value': 'Count', 'date': 'Date', 'variable': 'Metric'},
                       title="Global COVID-19 Trend")
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------- Country Comparison --------------------
if len(compare_countries) == 2:
    st.subheader(f" {compare_countries[0]} vs {compare_countries[1]}")
    comp_df = df[df['country'].isin(compare_countries)]
    fig_compare = px.bar(comp_df.melt(id_vars='country',
                         value_vars=['cases', 'deaths', 'mortality_rate', 'recovery_rate']),
                         x='variable', y='value', color='country', barmode='group',
                         title="Country Comparison")
    st.plotly_chart(fig_compare, use_container_width=True)

# -------------------- Table & Download --------------------
with st.expander(" Full Data Table"):
    st.dataframe(df)

st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="covid_data.csv", mime="text/csv")

st.success("✅ Dashboard loaded successfully.")
