# Applicant-for-SimPPL

# Social Media Analytics Dashboard - Mastodon

## Overview
The **Social Media Analytics Dashboard** is a tool designed to collect, analyze, and visualize engagement metrics for unreliable media URLs shared on Mastodon. This project fulfills part of the assignment to analyze social media data and tell a story through graphs.

---

## Features
1. **Data Collection:**
   - Scrapes engagement metrics (likes, shares, replies) for the top 50 URLs based on Mastodon posts.
   - Includes timestamped engagement data for time-based analysis.

2. **Data Visualization:**
   - Bar chart: Total engagement by URL.
   - Line chart: Engagement trends over time.
   - Pie chart: Engagement breakdown by URL.
   - Scatter plot: Likes vs. Replies.
   - Heatmap: Hourly engagement trends.
   - Semantic clustering visualization for posts.

3. **Interactivity:**
   - Filter data by URLs, date range, and engagement scores.
   - Explore trends and patterns dynamically.

4. **Semantic Clustering:**
   - Groups similar posts using SentenceTransformer embeddings and KMeans clustering.

5. **Data Export:**
   - Saves collected engagement data into CSV for further analysis.

---

## Setup Instructions
### Prerequisites
1. Python 3.8+
2. Install required libraries:
   ```bash
   pip install pandas plotly dash dash-bootstrap-components mastodon.py sklearn sentence-transformers
   ```
3. Access to a Mastodon instance and API token.

### Files in Repository
1. **`matsodon_dashboard.py`**: Main script for data collection, analysis, and dashboard creation.
2. **`data.csv`**: Input file containing website details (optional; replace with your own).
3. **`mastodon_engagement_data_with_timestamps.csv`**: Output file storing engagement metrics.

### Steps to Run
1. Clone the repository or download the script.
2. Replace the `access_token` and `api_base_url` in the script with your Mastodon API credentials:
   ```python
   mastodon = Mastodon(
       access_token='your_access_token',
       api_base_url='https://mastodon.social'
   )
   ```
3. Run the script:
   ```bash
   python matsodon_dashboard.py
   ```

---

## Dashboard Components
### Filters
- **URL Filter:** Select specific URLs to analyze.
- **Date Range Picker:** Choose a date range for engagement trends.
- **Engagement Range Slider:** Filter data by engagement score.

### Visualizations
- **Bar Chart:** Total engagement per URL.
  ![Bar Chart](https://github.com/ketan1403/Applicant-for-SimPPL/blob/c4c555e4873ff7a537f6c200eca48a8edfc21959/Total%20Engagement%20by%20URL.png)

- **Line Chart:** Engagement trends over time.
  ![Line Chart](https://github.com/ketan1403/Applicant-for-SimPPL/blob/180b2d766000c61ba77d612bd36b620dfb7be672/Engagement%20Trends%20over%20Time.png)

- **Pie Chart:** Engagement distribution across URLs.
  ![Pie Chart](https://github.com/ketan1403/Applicant-for-SimPPL/blob/80156f2c20e52e2c9a583782488590769f47eb15/Engagement%20Breakdown%20by%20URLs.png)

- **Scatter Plot:** Relationship between likes and replies.
  ![Scatter Plot](https://github.com/ketan1403/Applicant-for-SimPPL/blob/0190590a8ed16cf56940894805e5d1c70dcdde3a/Likes%20vs%20Replies.png)

- **Heatmap:** Hourly engagement trends.
  ![Heatmap](https://github.com/ketan1403/Applicant-for-SimPPL/blob/b4edbbee16e33d86c54ab62d6e1e8e0d829c6404/Hourly%20Engagement%20Trends.png)

- **Cluster Visualization:** Semantic clusters of posts.
  ![Cluster Visualization Example](https://via.placeholder.com/600x300?text=Cluster+Visualization+Screenshot)

### Data Table
- Displays URL-specific engagement metrics, including likes, shares, replies, and engagement scores.
  ![Data Table Example](https://via.placeholder.com/600x300?text=Data+Table+Screenshot)

---

## Code Details
### Key Functions
1. **Data Cleaning:**
   - Converts columns like `Year online`, `MisinfoMe`, and `Site Rank` to numeric values.

2. **Visualization Functions:**
   - `plot_fact_rating_dist`: Creates bar charts for MBFC Fact Ratings.
   - `plot_language_dist`: Generates pie charts for language distribution.
   - `plot_age_vs_misinfo`: Displays scatter plots for site age vs. MisinfoMe scores.

3. **Engagement Data Collection:**
   - Collects likes, shares, and replies for the top 50 URLs from Mastodon.

4. **Semantic Clustering:**
   - Uses SentenceTransformer and KMeans for clustering post content.

5. **Dashboard Components:**
   - Built using Dash and Dash Bootstrap Components.

---

## Sample Output
### Engagement Data (Sample)
| URL                | Likes | Shares | Replies | Engagement Score | Date       |
|--------------------|-------|--------|---------|------------------|------------|
| example.com/news1  | 50    | 10     | 5       | 65               | 2024-01-20 |
| another.com/story2 | 30    | 15     | 3       | 48               | 2024-01-21 |
