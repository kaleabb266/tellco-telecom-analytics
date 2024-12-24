import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from aggregiate_metrics import aggregate_user_metrics

def normalize_metrics(df):
    """Normalize user engagement metrics using MinMaxScaler."""
    # Update column names to match the actual DataFrame columns
    columns_to_normalize = ['session_count', 'total_duration', 'total_dl']
    
    # Create a copy of the DataFrame with only the columns we want to normalize
    df_norm = df[columns_to_normalize].copy()
    
    # Apply normalization
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df_norm)
    
    return df

def run_kmeans(df, n_clusters=3):
    """Run K-means clustering on normalized metrics."""
    # Update column names to match the normalized columns
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['engagement_cluster'] = kmeans.fit_predict(
        df[['session_count', 'total_duration', 'total_dl']]
    )
    return df

if __name__ == "__main__":
    df = aggregate_user_metrics()
    df = normalize_metrics(df)
    df = run_kmeans(df)
    print(df['engagement_cluster'].value_counts())