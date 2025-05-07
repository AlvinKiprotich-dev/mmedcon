# STEP 1: Import libraries
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
sys.stdout.reconfigure(encoding='utf-8')

# STEP 2: Fetch and clean COVID-19 data from API
url = "https://disease.sh/v3/covid-19/countries"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
else:
    print("Error fetching data:", response.status_code)
    exit()

df = pd.json_normalize(data)

# Keep relevant columns
columns = ['country', 'cases', 'todayCases', 'deaths', 'todayDeaths', 'recovered', 'population']
df = df[columns]
df = df.dropna()

# Calculate mortality and recovery rates
df['mortality_rate'] = df['deaths'] / df['cases']
df['recovery_rate'] = df['recovered'] / df['cases']
df = df.replace([float('inf'), -float('inf')], pd.NA).dropna()

# Sort for reporting
df = df.sort_values(by='cases', ascending=False)

# STEP 3: Save visualizations
sns.set(style="whitegrid")

# --- Top 10 countries by total cases ---
top_cases = df.nlargest(10, 'cases')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_cases, x='cases', y='country', palette='Blues_r', hue='country', legend=False)
plt.title('Top 10 Countries by COVID-19 Cases')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig("top_cases.png")
plt.close()

# --- Top 10 countries by total deaths ---
top_deaths = df.nlargest(10, 'deaths')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_deaths, x='deaths', y='country', palette='Reds_r', hue='country', legend=False)
plt.title('Top 10 Countries by COVID-19 Deaths')
plt.xlabel('Total Deaths')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig("top_deaths.png")
plt.close()

# --- Top 10 countries by mortality rate ---
top_mortality = df.nlargest(10, 'mortality_rate')
plt.figure(figsize=(12, 6))
sns.barplot(data=top_mortality, x='mortality_rate', y='country', palette='Greys', hue='country', legend=False)
plt.title('Top 10 Countries by COVID-19 Mortality Rate')
plt.xlabel('Mortality Rate')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig("top_mortality.png")
plt.close()

# STEP 4: Clustering using enriched features
features = df[['cases', 'deaths', 'population', 'mortality_rate', 'recovery_rate']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
df['risk_cluster'] = kmeans.fit_predict(scaled_features)

# STEP 5: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x='cases', y='deaths',
    hue='risk_cluster', palette='Set2'
)
plt.title('COVID-19 Risk Clusters')
plt.xlabel('Total Cases')
plt.ylabel('Total Deaths')
plt.tight_layout()
plt.savefig("risk_clusters.png")
plt.close()

# STEP 6: Print Summary
print("\n" + "="*40)
print("📊 COVID-19 Risk Cluster Summary")
print("="*40)

for cluster_id in sorted(df['risk_cluster'].unique()):
    cluster_data = df[df['risk_cluster'] == cluster_id]
    avg_cases = int(cluster_data['cases'].mean())
    avg_deaths = int(cluster_data['deaths'].mean())
    avg_mortality = round(cluster_data['mortality_rate'].mean() * 100, 2)
    avg_recovery = round(cluster_data['recovery_rate'].mean() * 100, 2)
    count = len(cluster_data)
    example_countries = ', '.join(cluster_data['country'].head(3))

    print(f"""
🟢 Cluster {cluster_id}:
- {count} countries
- Avg Cases: {avg_cases:,}
- Avg Deaths: {avg_deaths:,}
- Mortality Rate: {avg_mortality}%
- Recovery Rate: {avg_recovery}%
- Example countries: {example_countries}...
""")

print("✅ Analysis Complete. Images saved.")
