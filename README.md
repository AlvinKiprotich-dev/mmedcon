# COVID-19 Global Dashboard Site
- Access web app from: [`site`](https://alvinkiprotich-dev-mmedcon-app-hdcbit.streamlit.app/)

An interactive, data-driven dashboard for visualizing and analyzing global COVID-19 statistics using real-time API data. This project demonstrates my ability to work with public health data, clean and transform it, apply machine learning, and present insights through modern visual tools.

> **Built entirely by me using Python, Streamlit, Plotly, Pandas, and Scikit-Learn.**

---

##  Features

-  **Live API integration** for up-to-date global and per-country COVID-19 stats  
-  **Bar charts** of top N countries by total cases  
-  **Choropleth map** to visualize spread by country and metric  
-  **KMeans clustering** to categorize countries by risk (using unsupervised machine learning)  
-  **Country profiles** with key health indicators  
-  **Global and country-level historical trends** (30-day)  
-  **Country comparison** across multiple health indicators  
-  **Downloadable data** (CSV)  
-  **Auto-refresh toggle** for dynamic updates  

---

##  Sample Visualizations

| Global Metrics                | Risk Clustering (KMeans)       |
|------------------------------|--------------------------------|
| ![Metrics](assets/metrics.png) | ![Clusters](assets/clusters.png) |

---

##  Tech Stack

- **Python**  
- **Streamlit** for UI & interactivity  
- **Plotly** for interactive charts & maps  
- **Pandas** for data manipulation  
- **Requests** to call the `disease.sh` API  
- **Scikit-Learn** for KMeans clustering  

---

##  Data Source

- COVID-19 data from: [`disease.sh`](https://disease.sh/)
  - Endpoint used: `/v3/covid-19/countries`
  - Historical data: `/v3/covid-19/historical/all` and `/historical/{country}`

---

##  Setup Instructions

1. Clone this repo:
   ```bash
   git clone https://github.com/AlvinKiprotich-dev/covid19-dashboard.git
   cd covid19-dashboard
