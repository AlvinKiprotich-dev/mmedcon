# STEP 2: Fetch and clean COVID-19 data from API

import os  # For folder creation
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Create a folder to store plots ---
os.makedirs('plots', exist_ok=True)

# --- Fetch COVID-19 data per country ---
url = "https://disease.sh/v3/covid-19/countries"
response = requests.get(url)

# Check response
if response.status_code == 200:
    data = response.json()
else:
    print("Error fetching data:", response.status_code)
    exit()

# Convert to DataFrame
df = pd.json_normalize(data)

# Keep relevant columns
columns = ['country', 'cases', 'todayCases', 'deaths', 'todayDeaths', 'recovered', 'population']
df = df[columns]

# Clean missing values
df = df.dropna()

# Sort by total cases
df = df.sort_values(by='cases', ascending=False)

# View first 5 rows
print(df.head())

# --- Top 10 countries by total cases ---
top_cases = df.nlargest(10, 'cases')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_cases, x='cases', y='country', palette='Blues_r')
plt.title('Top 10 Countries by COVID-19 Cases')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('plots/top_cases.png')  # Save figure
plt.close()  # Close figure

# --- Top 10 countries by total deaths ---
top_deaths = df.nlargest(10, 'deaths')

plt.figure(figsize=(12, 6))
sns.barplot(data=top_deaths, x='deaths', y='country', palette='Reds_r')
plt.title('Top 10 Countries by COVID-19 Deaths')
plt.xlabel('Total Deaths')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('plots/top_deaths.png')
plt.close()

# --- Clustering using KMeans ---
features = df[['cases', 'deaths', 'population']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
df['risk_cluster'] = kmeans.fit_predict(scaled_features)

# Sample cluster result
print(df[['country', 'cases', 'deaths', 'population', 'risk_cluster']].head())

# --- Scatter plot for clusters ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x='cases', y='deaths',
    hue='risk_cluster', palette='Set2'
)
plt.title('COVID-19 Risk Clusters')
plt.xlabel('Total Cases')
plt.ylabel('Total Deaths')
plt.tight_layout()
plt.savefig('plots/risk_clusters.png')
plt.close()

# ----------------------------
# Step 5: AI-like summary report
# ----------------------------
print("\n" + "="*40)
print("📊 COVID-19 Risk Cluster Summary")
print("="*40)

cluster_descriptions = []

for cluster_id in sorted(df['risk_cluster'].unique()):
    cluster_data = df[df['risk_cluster'] == cluster_id]
    avg_cases = int(cluster_data['cases'].mean())
    avg_deaths = int(cluster_data['deaths'].mean())
    country_count = len(cluster_data)
    sample_countries = ', '.join(cluster_data['country'].head(3))

    description = (
        f"\n🟢 Cluster {cluster_id}:\n"
        f"- {country_count} countries\n"
        f"- Avg cases: {avg_cases:,}\n"
        f"- Avg deaths: {avg_deaths:,}\n"
        f"- Example countries: {sample_countries}..."
    )
    print(description)
    cluster_descriptions.append(description)

print("\n✅ Summary complete.")
