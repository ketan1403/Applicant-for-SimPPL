# Install required libraries
!pip install Mastodon.py pandas plotly dash dash-bootstrap-components

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Read the CSV file
df = pd.read_csv('/content/data.csv')

# Clean and prepare data
def clean_data(df):
    # Convert Year online to numeric, handling any errors
    df['Year online'] = pd.to_numeric(df['Year online'], errors='coerce')

    # Convert MisinfoMe to numeric, handling any errors
    df['MisinfoMe'] = pd.to_numeric(df['MisinfoMe'], errors='coerce')

    # Convert Site Rank to numeric, handling any errors
    df['Site Rank'] = pd.to_numeric(df['Site Rank'], errors='coerce')

    return df

df = clean_data(df)

# 1. MBFC Fact Rating Distribution
def plot_fact_rating_dist():
    fact_dist = df['MBFC Fact'].value_counts().reset_index()
    fact_dist.columns = ['Rating', 'Count']

    fig = px.bar(fact_dist,
                 x='Rating',
                 y='Count',
                 title='Distribution of MBFC Fact Ratings',
                 labels={'Rating': 'MBFC Fact Rating', 'Count': 'Number of Sites'},
                 color='Count')

    fig.show()

# 2. Language Distribution
def plot_language_dist():
    lang_dist = df['Lang'].value_counts().reset_index()
    lang_dist.columns = ['Language', 'Count']

    fig = px.pie(lang_dist,
                 values='Count',
                 names='Language',
                 title='Distribution of Languages',
                 hole=0.4)  # Makes it a donut chart

    fig.show()

# 3. Site Age vs MisinfoMe Score
def plot_age_vs_misinfo():
    fig = px.scatter(df,
                    x='Year online',
                    y='MisinfoMe',
                    title='Site Age vs MisinfoMe Score',
                    labels={'Year online': 'Year Website Went Online',
                           'MisinfoMe': 'MisinfoMe Score'},
                    hover_data=['Domain'])

    fig.show()

# 4. MBFC Bias Distribution
def plot_bias_dist():
    bias_dist = df['MBFC Bias'].value_counts().reset_index()
    bias_dist.columns = ['Bias', 'Count']

    fig = px.bar(bias_dist,
                 x='Bias',
                 y='Count',
                 title='Distribution of MBFC Bias Ratings',
                 labels={'Bias': 'MBFC Bias Rating', 'Count': 'Number of Sites'},
                 color='Count')

    fig.show()

# 5. Site Rank Analysis
def plot_site_rank_analysis():

    df['Log Site Rank'] = np.log10(df['Site Rank'])

    fig = px.box(df,
                 y='Log Site Rank',
                 x='MBFC Fact',
                 title='Site Rank Distribution by MBFC Fact Rating',
                 labels={'Log Site Rank': 'Log10(Site Rank)',
                        'MBFC Fact': 'MBFC Fact Rating'})

    fig.show()

# 6. Top Sites Table
def display_top_sites(n=100):

    top_sites = df.sort_values('Site Rank').head(n)[
        ['Domain', 'Site Rank', 'Year online', 'MBFC Fact', 'MBFC Bias', 'MisinfoMe']
    ]

    return top_sites

# Create summary statistics
def display_summary_stats():
    summary = pd.DataFrame({
        'Total Sites': len(df),
        'Average MisinfoMe Score': df['MisinfoMe'].mean(),
        'Median Site Rank': df['Site Rank'].median(),
        'Most Common Language': df['Lang'].mode().iloc[0],
        'Most Common MBFC Fact Rating': df['MBFC Fact'].mode().iloc[0],
        'Average Site Age': 2024 - df['Year online'].mean()
    }, index=[0])

    return summary

print("\nGenerating visualizations and analyses...\n")

print("Summary Statistics:")
display_summary_stats()

print("\nTop 100 Sites by Rank:")
print(display_top_sites())

plot_fact_rating_dist()
plot_language_dist()
plot_age_vs_misinfo()
plot_bias_dist()
plot_site_rank_analysis()

def get_top_100_urls(dataframe):
    top_urls = dataframe.sort_values('Site Rank').head(100)['Domain'].tolist()
    return top_urls

top_100_urls = get_top_100_urls(df)
print("Top 100 URLs extracted:", top_100_urls)

# Initialize Mastodon API
from mastodon import Mastodon

mastodon = Mastodon(
    access_token='mSSacgIUUP7GhUdsk93S3BaZf8nEaff9qVyP5Z3v6l8',
    api_base_url='https://mastodon.social'
)

# Collect engagement metrics for the top 100 URLs
def fetch_engagement_data(mastodon, target_urls, max_posts=100):
    engagement_data = []

    for url in target_urls:
        print(f"Fetching posts for URL: {url}")
        results = mastodon.search(url, result_type='statuses')

        for status in results['statuses'][:max_posts]:
            engagement_data.append({
                'url': url,
                'content': status['content'],
                'likes': status['favourites_count'],
                'shares': status['reblogs_count'],
                'replies': status.get('replies_count', 0),
                'post_url': status['url']
            })

    return pd.DataFrame(engagement_data)

engagement_df = fetch_engagement_data(mastodon, top_100_urls)

engagement_df.to_csv('mastodon_engagement_data.csv', index=False)
print("Engagement data collected and saved to 'mastodon_engagement_data.csv'")

def fetch_engagement_data(mastodon, target_urls, max_posts=100):
    engagement_data = []

    for url in target_urls:
        print(f"Fetching posts for URL: {url}")
        results = mastodon.search(url, result_type='statuses')

        for status in results['statuses'][:max_posts]:
            engagement_data.append({
                'url': url,
                'content': status['content'],
                'likes': status['favourites_count'],
                'shares': status['reblogs_count'],
                'replies': status.get('replies_count', 0),
                'post_url': status['url'],
                'timestamp': pd.to_datetime(status['created_at']),
            })

    return pd.DataFrame(engagement_data)

import pandas as pd

df['Site Rank'] = pd.to_numeric(df['Site Rank'], errors='coerce')
df = df.dropna(subset=['Site Rank'])

top_50_urls = df.sort_values('Site Rank').head(100)['Domain'].tolist()

print(f"Top 100 URLs: {top_100_urls}")

def fetch_engagement_data(mastodon, target_urls, max_posts=100):
    engagement_data = []

    for url in target_urls:
        print(f"Fetching posts for URL: {url}")
        results = mastodon.search(url, result_type='statuses')

        for status in results['statuses'][:max_posts]:
            engagement_data.append({
                'url': url,
                'content': status['content'],
                'likes': status['favourites_count'],
                'shares': status['reblogs_count'],
                'replies': status.get('replies_count', 0),
                'post_url': status['url'],
                'timestamp': pd.to_datetime(status['created_at']),
            })

    return pd.DataFrame(engagement_data)

from mastodon import Mastodon

mastodon = Mastodon(
    access_token='mSSacgIUUP7GhUdsk93S3BaZf8nEaff9qVyP5Z3v6l8',
    api_base_url='https://mastodon.social'
)

engagement_df = fetch_engagement_data(mastodon, top_100_urls)

engagement_df.to_csv('mastodon_engagement_data_with_timestamps.csv', index=False)
print("Engagement data saved to 'mastodon_engagement_data_with_timestamps.csv'")

engagement_df = pd.read_csv('mastodon_engagement_data_with_timestamps.csv')

print(engagement_df.head())

print(engagement_df.isnull().sum())

engagement_df.dropna(inplace=True)  # Remove rows with missing values

# Convert the 'timestamp' column to a datetime object
engagement_df['timestamp'] = pd.to_datetime(engagement_df['timestamp'], format='ISO8601', errors='coerce')

# Extract the hour from the timestamp (optional for hourly analysis)
engagement_df['hour'] = engagement_df['timestamp'].dt.hour

# Extract the date from the timestamp
engagement_df['date'] = engagement_df['timestamp'].dt.date

# Verify the new columns
print(engagement_df[['url', 'timestamp', 'hour', 'date']].head())

from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

engagement_df = pd.read_csv('mastodon_engagement_data_with_timestamps.csv')

engagement_df['timestamp'] = pd.to_datetime(engagement_df['timestamp'], format='ISO8601', errors='coerce')
engagement_df['date'] = pd.to_datetime(engagement_df['timestamp']).dt.date
engagement_df['hour'] = pd.to_datetime(engagement_df['timestamp']).dt.hour
engagement_df['engagement_score'] = (
    engagement_df['likes'] + engagement_df['shares'] + engagement_df['replies']
)

# Step 2: Semantic clustering
def semantic_clustering(dataframe):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = embedder.encode(dataframe['content'].fillna(""))

    kmeans = KMeans(n_clusters=5, random_state=42)
    dataframe['cluster'] = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    dataframe['x'] = reduced_embeddings[:, 0]
    dataframe['y'] = reduced_embeddings[:, 1]

    return dataframe

engagement_df = semantic_clustering(engagement_df)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Mastodon Social Media Dashboard", className="text-center my-4 text-primary"),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='url-dropdown',
                options=[{'label': url, 'value': url} for url in engagement_df['url'].unique()],
                multi=True,
                placeholder="Select URLs",
                className="mb-3"
            ),
        ], width=4),
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker',
                start_date=engagement_df['date'].min(),
                end_date=engagement_df['date'].max(),
                display_format='YYYY-MM-DD',
                className="mb-3"
            ),
        ], width=4),
        dbc.Col([
            dcc.RangeSlider(
                id='engagement-slider',
                min=engagement_df['engagement_score'].min(),
                max=engagement_df['engagement_score'].max(),
                value=[engagement_df['engagement_score'].min(), engagement_df['engagement_score'].max()],
                marks={
                    int(engagement_df['engagement_score'].min()): 'Low',
                    int(engagement_df['engagement_score'].max()): 'High',
                },
                tooltip={"placement": "bottom", "always_visible": True},
                className="mb-3"
            ),
        ], width=4),
    ]),

    dbc.Row([
        dbc.Col([dcc.Graph(id='bar-chart')], width=6),
        dbc.Col([dcc.Graph(id='line-chart')], width=6),
    ]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='pie-chart')], width=6),
        dbc.Col([dcc.Graph(id='scatter-plot')], width=6),
    ]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='heatmap-chart')], width=12),
    ]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='cluster-visualization')], width=12),
    ]),

    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='engagement-table',
                columns=[
                    {'name': 'URL', 'id': 'url'},
                    {'name': 'Likes', 'id': 'likes'},
                    {'name': 'Shares', 'id': 'shares'},
                    {'name': 'Replies', 'id': 'replies'},
                    {'name': 'Engagement Score', 'id': 'engagement_score'},
                    {'name': 'Date', 'id': 'date'},
                    {'name': 'Cluster', 'id': 'cluster'}
                ],
                style_table={'overflowX': 'auto'},
                filter_action="native",
                sort_action="native",
                page_size=10
            )
        ])
    ])
], fluid=True)

@app.callback(
    [
        Output('bar-chart', 'figure'),
        Output('line-chart', 'figure'),
        Output('pie-chart', 'figure'),
        Output('scatter-plot', 'figure'),
        Output('heatmap-chart', 'figure'),
        Output('cluster-visualization', 'figure'),
        Output('engagement-table', 'data')
    ],
    [
        Input('url-dropdown', 'value'),
        Input('date-picker', 'start_date'),
        Input('date-picker', 'end_date'),
        Input('engagement-slider', 'value')
    ]
)
def update_dashboard(selected_urls, start_date, end_date, engagement_range):

    filtered_df = engagement_df.copy()
    if selected_urls:
        filtered_df = filtered_df[filtered_df['url'].isin(selected_urls)]
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date).date()) &
            (filtered_df['date'] <= pd.to_datetime(end_date).date())
        ]
    filtered_df = filtered_df[
        (filtered_df['engagement_score'] >= engagement_range[0]) &
        (filtered_df['engagement_score'] <= engagement_range[1])
    ]

    # Bar chart: Total engagement by URL
    bar_chart = px.bar(
        filtered_df.groupby('url')['engagement_score'].sum().reset_index(),
        x='url',
        y='engagement_score',
        title='Total Engagement by URL',
        color='engagement_score'
    )

    # Line chart: Engagement trends over time
    line_chart = px.line(
        filtered_df.groupby('date')['engagement_score'].sum().reset_index(),
        x='date',
        y='engagement_score',
        title='Engagement Trends Over Time'
    )

    # Pie chart: Engagement breakdown
    pie_chart = px.pie(
        filtered_df,
        names='url',
        values='engagement_score',
        title='Engagement Breakdown by URL'
    )

    # Scatter plot: Likes vs Replies
    scatter_plot = px.scatter(
        filtered_df,
        x='likes',
        y='replies',
        size='engagement_score',
        color='url',
        hover_data=['url'],
        title='Likes vs Replies Scatter Plot'
    )

    # Heatmap: Hourly engagement trends
    heatmap_data = filtered_df.groupby('hour')['engagement_score'].sum().reset_index()
    heatmap_chart = px.density_heatmap(
        heatmap_data,
        x='hour',
        y='engagement_score',
        title='Hourly Engagement Trends',
        labels={'hour': 'Hour of Day', 'engagement_score': 'Engagement Score'}
    )

    # Cluster visualization
    cluster_visualization = px.scatter(
        filtered_df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['content'],
        title='Semantic Clusters of Posts'
    )

    table_data = filtered_df.to_dict('records')

    return bar_chart, line_chart, pie_chart, scatter_plot, heatmap_chart, cluster_visualization, table_data

# Run the App.
if __name__ == '__main__':
    app.run_server(debug=True)



