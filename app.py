import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Title
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("🌍 COVID-19 Global Dashboard")
st.markdown("Interactive analysis of COVID-19 data with clustering insights.")

# Fetch data
@st.cache_data
def fetch_data():
    url = "https://disease.sh/v3/covid-19/countries"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.json_normalize(data)
        columns = ['country', 'cases', 'todayCases', 'deaths', 'todayDeaths', 'recovered', 'population']
        df = df[columns].dropna()
        df['mortality_rate'] = df['deaths'] / df['cases']
        df['recovery_rate'] = df['recovered'] / df['cases']
        df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()
        return df.sort_values(by='cases', ascending=False)
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return pd.DataFrame()

df = fetch_data()

# Sidebar Controls
st.sidebar.header("🔧 Controls")
top_n = st.sidebar.slider("Top N Countries", 5, 30, 10)
chart_type = st.sidebar.selectbox("Chart Type", ["Total Cases", "Total Deaths", "Mortality Rate"])

# Show Data Table
if st.sidebar.checkbox("Show Raw Data"):
    st.dataframe(df)

# Bar Charts
st.subheader("📊 Top Countries")
if chart_type == "Total Cases":
    top = df.nlargest(top_n, 'cases')
    fig = px.bar(top, x='cases', y='country', orientation='h', color='cases',
                 title="Top Countries by COVID-19 Cases")
elif chart_type == "Total Deaths":
    top = df.nlargest(top_n, 'deaths')
    fig = px.bar(top, x='deaths', y='country', orientation='h', color='deaths',
                 title="Top Countries by COVID-19 Deaths")
else:
    top = df.nlargest(top_n, 'mortality_rate')
    fig = px.bar(top, x='mortality_rate', y='country', orientation='h', color='mortality_rate',
                 title="Top Countries by COVID-19 Mortality Rate")

st.plotly_chart(fig, use_container_width=True)

# Clustering
features = df[['cases', 'deaths', 'population', 'mortality_rate', 'recovery_rate']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
df['risk_cluster'] = kmeans.fit_predict(scaled)

# Scatter Plot
st.subheader("🧬 Risk Clustering (Cases vs Deaths)")
scatter = px.scatter(df, x='cases', y='deaths', color='risk_cluster',
                     hover_name='country', title="COVID-19 Risk Clusters",
                     labels={'cases': 'Total Cases', 'deaths': 'Total Deaths'})
st.plotly_chart(scatter, use_container_width=True)

# Summary
st.subheader("📋 Cluster Summary")
for cluster_id in sorted(df['risk_cluster'].unique()):
    cluster_data = df[df['risk_cluster'] == cluster_id]
    avg_cases = int(cluster_data['cases'].mean())
    avg_deaths = int(cluster_data['deaths'].mean())
    avg_mortality = round(cluster_data['mortality_rate'].mean() * 100, 2)
    avg_recovery = round(cluster_data['recovery_rate'].mean() * 100, 2)
    count = len(cluster_data)
    example_countries = ', '.join(cluster_data['country'].head(3))

    with st.expander(f"🟢 Cluster {cluster_id} ({count} countries)"):
        st.markdown(f"""
        - **Avg Cases:** {avg_cases:,}
        - **Avg Deaths:** {avg_deaths:,}
        - **Mortality Rate:** {avg_mortality}%
        - **Recovery Rate:** {avg_recovery}%
        - **Example Countries:** {example_countries}...
        """)

st.success("✅ Analysis Complete")
