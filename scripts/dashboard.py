import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
import os
from load_data import load_data, connect_to_db
from experience_analytics import analyze_customer_experience
from aggregiate_metrics import aggregate_user_metrics
from satistfaction_anlysis import SatisfactionAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def get_processed_data():
    """Cache the data processing steps"""
    engagement_data = aggregate_user_metrics()
    experience_results = analyze_customer_experience()
    experience_data = experience_results['processed_data']
    
    # Get common users
    common_users = set(engagement_data['msisdn']).intersection(set(experience_data['msisdn']))
    engagement_data = engagement_data[engagement_data['msisdn'].isin(common_users)]
    experience_data = experience_data[experience_data['msisdn'].isin(common_users)]
    
    return engagement_data, experience_data

@st.cache_resource
def get_model_results():
    """Cache or load the model results"""
    model_path = 'models/satisfaction_model.joblib'
    results_path = 'models/satisfaction_results.joblib'
    
    if os.path.exists(results_path):
        # Load cached results
        results = load(results_path)
    else:
        # Get data
        engagement_data, experience_data = get_processed_data()
        
        # Initialize analyzer and calculate scores
        analyzer = SatisfactionAnalyzer()
        engagement_scores = analyzer.calculate_engagement_score(engagement_data, None)
        experience_scores = analyzer.calculate_experience_score(experience_data, None)
        satisfaction_scores = analyzer.calculate_satisfaction_score(engagement_scores, experience_scores)
        
        # Prepare features
        X = pd.concat([
            engagement_data[['session_count', 'total_duration', 'total_dl', 'total_ul']],
            experience_data[['tcp_retrans_rate', 'avg_rtt', 'avg_throughput']]
        ], axis=1)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, satisfaction_scores)
        
        # Get predictions
        X_train, X_test, y_train, y_test = train_test_split(X, satisfaction_scores, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store all results
        results = {
            'model': model,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred,
            'engagement_scores': engagement_scores,
            'experience_scores': experience_scores,
            'satisfaction_scores': satisfaction_scores,
            'clusters': analyzer.cluster_users(engagement_scores, experience_scores)
        }
        
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save results
        dump(results, results_path)
        dump(model, model_path)
    
    return results


# Page configuration
st.set_page_config(
    page_title="Telecom Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["User Overview", "User Engagement", "Experience Analysis", "Satisfaction Analysis"]
)


# User Overview Page
if page == "User Overview":
    st.title("User Overview Analysis")
    
    # Load data
    conn = connect_to_db()
    query = "SELECT * FROM xdr_data;"
    df = load_data(query, conn)
    
    # First row - 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Handsets Used by Customers")
        top_10_handsets = df['Handset Type'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, palette='viridis')
        plt.title('Top 10 Handsets Used by Customers')
        plt.xlabel('Number of Users')
        plt.ylabel('Handset Type')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top 3 Handset Manufacturers")
        top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_3_manufacturers.values, y=top_3_manufacturers.index, palette='viridis')
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Number of Users')
        plt.ylabel('Handset Manufacturer')
        st.pyplot(fig)

    # Second row - Manufacturers' top handsets
    st.subheader("Top 5 Handsets for each of the top 3 manufacturers")
    top_5_handsets_per_manufacturer = {}
    
    for manufacturer in top_3_manufacturers.index:
        manufacturer_df = df[df['Handset Manufacturer'] == manufacturer]
        top_5_handsets = manufacturer_df['Handset Type'].value_counts().head(5)
        top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_5_handsets.values, y=top_5_handsets.index, palette='cubehelix')
        plt.title(f'Top 5 Handsets for {manufacturer}')
        plt.xlabel('Number of Users')
        plt.ylabel('Handset Type')
        st.pyplot(fig)

    # Third row - 2 columns
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Distribution of Total Session Duration")
        user_stats = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',
            'Dur. (ms)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()
        
        fig, ax = plt.subplots()
        sns.histplot(data=user_stats['Dur. (ms)'], bins=100, kde=True)
        plt.title('Distribution of Total Session Duration')
        plt.xlabel('Total Session Duration (ms)')
        plt.ylabel('Frequency')
        st.pyplot(fig)
    
    with col4:
        st.subheader("Session Duration vs Total Data Volume")
        user_stats['total_data_volume'] = user_stats['Total DL (Bytes)'] + user_stats['Total UL (Bytes)']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=user_stats, x='Dur. (ms)', y='total_data_volume', alpha=0.5)
        plt.title('Scatter Plot of Total Session Duration vs Total Data Volume')
        plt.xlabel('Total Session Duration (ms)')
        plt.ylabel('Total Data Volume (Bytes)')
        st.pyplot(fig)

    # Fourth row - Log transformed plots
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Log-Transformed Distribution of Total Session Duration")
        user_stats['log_total_session_duration'] = np.log1p(user_stats['Dur. (ms)'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(data=user_stats['log_total_session_duration'], bins=50, kde=True, color='darkblue')
        plt.title('Log-Transformed Distribution of Total Session Duration', fontsize=20, fontweight='bold')
        plt.xlabel('Log of Total Session Duration (ms)', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        st.pyplot(fig)
    
    with col6:
        st.subheader("Log-Transformed Session Duration vs Data Volume")
        user_stats['log_total_data_volume'] = np.log1p(user_stats['total_data_volume'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=user_stats, x='log_total_session_duration', y='log_total_data_volume', 
                       s=80, color='darkred', alpha=0.7)
        plt.title('Log-Transformed Scatter Plot of Session Duration vs Data Volume', 
                 fontsize=20, fontweight='bold')
        plt.xlabel('Log of Total Session Duration (ms)', fontsize=16)
        plt.ylabel('Log of Total Data Volume (Bytes)', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

# User Engagement Page
elif page == "User Engagement":
    st.title("User Engagement Analysis")
    
    # Load data
    conn = connect_to_db()
    df = load_data("SELECT * FROM xdr_data;", conn)
    
    # Aggregate metrics
    user_metrics = df.groupby('MSISDN/Number').agg({
        'Social Media DL (Bytes)': 'sum',
        'Social Media UL (Bytes)': 'sum',
        'Google DL (Bytes)': 'sum',
        'Google UL (Bytes)': 'sum',
        'Email DL (Bytes)': 'sum',
        'Email UL (Bytes)': 'sum',
        'Youtube DL (Bytes)': 'sum',
        'Youtube UL (Bytes)': 'sum',
        'Netflix DL (Bytes)': 'sum',
        'Netflix UL (Bytes)': 'sum',
        'Gaming DL (Bytes)': 'sum',
        'Gaming UL (Bytes)': 'sum',
        'Other DL (Bytes)': 'sum',
        'Other UL (Bytes)': 'sum'
    }).reset_index()

    # Calculate total traffic per application
    user_metrics['Social Media Total'] = user_metrics['Social Media DL (Bytes)'] + user_metrics['Social Media UL (Bytes)']
    user_metrics['Google Total'] = user_metrics['Google DL (Bytes)'] + user_metrics['Google UL (Bytes)']
    user_metrics['Email Total'] = user_metrics['Email DL (Bytes)'] + user_metrics['Email UL (Bytes)']
    user_metrics['Youtube Total'] = user_metrics['Youtube DL (Bytes)'] + user_metrics['Youtube UL (Bytes)']
    user_metrics['Netflix Total'] = user_metrics['Netflix DL (Bytes)'] + user_metrics['Netflix UL (Bytes)']
    user_metrics['Gaming Total'] = user_metrics['Gaming DL (Bytes)'] + user_metrics['Gaming UL (Bytes)']
    user_metrics['Other Total'] = user_metrics['Other DL (Bytes)'] + user_metrics['Other UL (Bytes)']

    # First row - Full width
    st.subheader("Correlation Matrix of Application Usage")
    app_columns = ['Social Media Total', 'Google Total', 'Email Total', 
                  'Youtube Total', 'Netflix Total', 'Gaming Total', 'Other Total']
    
    corr_matrix = user_metrics[app_columns].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Application Usage', fontsize=16, pad=20)
    st.pyplot(fig)

    # Second row - 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Engagement Clusters")
        # Prepare data for clustering
        cluster_features = ['Social Media Total', 'Youtube Total', 'Gaming Total']
        X = user_metrics[cluster_features]
        X = np.log1p(X)  # Log transformation
        
        # Normalize the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        user_metrics['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create scatter plot
        fig = px.scatter(
            user_metrics,
            x=np.log1p(user_metrics['Social Media Total']),
            y=np.log1p(user_metrics['Youtube Total']),
            color='Cluster',
            title='User Engagement Clusters (Log Scale)',
            labels={
                'x': 'Log(Social Media Traffic + 1)',
                'y': 'Log(Youtube Traffic + 1)'
            }
        )
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Top 3 Applications by Total Traffic")
        total_traffic = pd.DataFrame({
            'Application': app_columns[:-1],  # Excluding 'Other Total'
            'Total Traffic': [user_metrics[col].sum() for col in app_columns[:-1]]
        })
        total_traffic = total_traffic.sort_values('Total Traffic', ascending=False).head(3)
        
        fig = px.bar(
            total_traffic,
            x='Application',
            y='Total Traffic',
            title='Top 3 Applications by Total Traffic',
            color='Application'
        )
        fig.update_layout(
            xaxis_title="Application",
            yaxis_title="Total Traffic (Bytes)",
            showlegend=False
        )
        st.plotly_chart(fig)

    # Add explanatory text
    st.markdown("""
    ### Key Insights:
    - The correlation matrix shows relationships between different application usages
    - User clusters reveal distinct patterns in application usage
    - The top 3 applications represent the most used services by traffic volume
    """)

# ... (previous imports and page config remain the same)

elif page == "Experience Analysis":
    st.title("Experience Analysis")
    
    # Load data
    results = analyze_customer_experience()
    df = results['processed_data']
    distribution_stats = results['distribution_stats']
    handset_metrics = results['handset_metrics']
    
    # First row - Distribution Analysis
    st.subheader("1. Distribution Analysis")
    
    metrics = ['tcp_retrans_rate', 'avg_rtt', 'avg_throughput']
    titles = ['TCP Retransmission', 'Average RTT', 'Throughput']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, metric, title in zip(axes, metrics, titles):
        sns.histplot(data=df, x=metric, ax=ax, kde=True)
        ax.set_title(f'Distribution of {title}')
        ax.set_xlabel(f'{title} Value')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)

    # Second row - Cluster Characteristics
    st.subheader("2. Clustering Analysis")
    
    clustering_results = results['clustering_results']
    cluster_data = clustering_results['clustered_data']
    cluster_stats = clustering_results['cluster_stats']
    cluster_centers = clustering_results['cluster_centers']
    
    # Plot cluster characteristics using cluster_stats
    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_centers_normalized = (cluster_centers - cluster_centers.min()) / (cluster_centers.max() - cluster_centers.min())
    cluster_centers_normalized.plot(kind='bar', ax=plt.gca())
    plt.title('Cluster Characteristics')
    plt.xlabel('Cluster')
    plt.ylabel('Average Value')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

    # Display cluster interpretations
    st.write("\nCluster Interpretations:")
    cluster_interpretations = {
        0: "Average Users: Moderate values across all metrics",
        1: "Poor Experience Users: High RTT, low TCP retransmission, low throughput",
        2: "High Performance Users: Low RTT, low TCP retransmission, high throughput"
    }

    for cluster, interpretation in cluster_interpretations.items():
        st.write(f"Cluster {cluster}: {interpretation}")


    # Third row - Correlation Analysis
    st.subheader("3. Correlation Analysis")
    
    correlation_matrix = df[metrics].corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Experience Metrics')
    plt.tight_layout()
    st.pyplot(fig)

    # Fourth row - Performance Score
    df_normalized = (df[metrics] - df[metrics].min()) / (df[metrics].max() - df[metrics].min())
    df['performance_score'] = (
        (1 - df_normalized['tcp_retrans_rate']) + 
        (1 - df_normalized['avg_rtt']) + 
        df_normalized['avg_throughput']
    ) / 3
    
    st.subheader("4. Performance Score Distribution")
    
    fig = plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='performance_score', kde=True)
    plt.title('Distribution of User Performance Scores')
    plt.xlabel('Performance Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)

    # Fifth row - Top Performers
    st.subheader("5. Top Performers Analysis")
    
    # Top performers by handset type
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x='performance_score', y='handset_type', 
                data=df.nlargest(10, 'performance_score'))
    plt.title('Top 10 Performing users by handset type')
    plt.xlabel('Performance Score')
    plt.ylabel('Handset Type')
    plt.tight_layout()
    st.pyplot(fig)

    # Top performing handset types
    handset_performance = df.groupby('handset_type')['performance_score'].agg([
        'mean',
        'count'
    ]).reset_index()

    top_handsets = handset_performance[handset_performance['count'] >= 10].nlargest(10, 'mean')

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x='mean', y='handset_type', data=top_handsets)
    plt.title('Top 10 Performing Handset Types (minimum 10 users)')
    plt.xlabel('Average Performance Score')
    plt.ylabel('Handset Type')
    plt.tight_layout()
    st.pyplot(fig)

    # Add insights
    st.markdown("""
    ### Key Insights:
    - Distribution analysis shows the spread of key performance metrics
    - Cluster characteristics reveal distinct user experience patterns
    - Performance scores indicate the overall quality of service
    - Handset type analysis helps identify optimal device configurations
    """)


elif page == "Satisfaction Analysis":
    st.title("Satisfaction Analysis")
    
    # Get cached results
    results = get_model_results()
    
    # Create results dataframe
    engagement_data, experience_data = get_processed_data()
    results_df = pd.DataFrame({
        'msisdn': experience_data['msisdn'],
        'engagement_score': results['engagement_scores'],
        'experience_score': results['experience_scores'],
        'satisfaction_score': results['satisfaction_scores'],
        'cluster': results['clusters']
    })
    
    # First row - Satisfaction Score Distribution
    st.subheader("Distribution of Customer Satisfaction Scores")
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='satisfaction_score', bins=30)
    plt.title('Distribution of Customer Satisfaction Scores')
    plt.xlabel('Satisfaction Score')
    plt.ylabel('Count')
    st.pyplot(fig)
    
    # Second row - Feature Importance
    st.subheader("Feature Importance in Predicting Satisfaction Score")
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=results['feature_importance'])
    plt.title('Feature Importance in Predicting Satisfaction Score')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(fig)
    
    # Third row - Predicted vs Actual
    st.subheader("Predicted vs Actual Satisfaction Scores")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(results['y_test'], results['y_pred'], alpha=0.5)
    plt.plot([results['y_test'].min(), results['y_test'].max()], 
             [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
    plt.xlabel('Actual Satisfaction Score')
    plt.ylabel('Predicted Satisfaction Score')
    plt.title('Predicted vs Actual Satisfaction Scores')
    st.pyplot(fig)
    
    # Fourth row - Customer Clusters
    st.subheader("Customer Clusters based on Engagement and Experience")
    fig = plt.figure(figsize=(10, 6))
    for cluster in [0, 1]:
        cluster_data = results_df[results_df['cluster'] == cluster]
        plt.scatter(
            cluster_data['engagement_score'],
            cluster_data['experience_score'],
            label=f'Cluster {cluster}',
            alpha=0.5
        )
    plt.xlabel('Engagement Score')
    plt.ylabel('Experience Score')
    plt.title('Customer Clusters based on Engagement and Experience')
    plt.legend()
    st.pyplot(fig)
    
    # Fifth row - Average Scores by Cluster
    st.subheader("Average Scores by Cluster")
    cluster_means = results_df.groupby('cluster')[
        ['satisfaction_score', 'experience_score', 'engagement_score']
    ].mean()
    
    fig = plt.figure(figsize=(10, 6))
    cluster_means.plot(kind='bar', ax=plt.gca())
    plt.title('Average Scores by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Score')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)


    # Add insights
    st.markdown("""
    ### Key Insights:
    - The satisfaction score distribution shows the overall customer satisfaction levels
    - Feature importance highlights which factors most influence satisfaction
    - The clustering analysis reveals distinct customer segments
    - Predicted vs actual scores demonstrate model performance
    - Average scores by cluster help understand segment characteristics
    """)