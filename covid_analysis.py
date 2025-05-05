# STEP 2: Fetch and clean COVID-19 data from API

import pandas as pd
import requests

# Fetch COVID-19 data per country
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
