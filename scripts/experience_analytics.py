import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from load_data import load_data

class DataPreprocessor:
    @staticmethod
    def handle_missing_values(df, columns):
        """Fill missing values with mean or mode."""
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    @staticmethod
    def handle_outliers(df, columns, method='iqr'):
        """Handle outliers using specified method."""
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        return df
    
class ExperienceMetrics:

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        
    def get_customer_metrics(self):
    
        """Get aggregated customer experience metrics."""
        query = """
        SELECT 
            "MSISDN/Number" as msisdn,
            SUM("TCP DL Retrans. Vol (Bytes)" + "TCP UL Retrans. Vol (Bytes)") as tcp_retrans_vol,
            AVG("Avg RTT DL (ms)") as rtt_dl,
            AVG("Avg RTT UL (ms)") as rtt_ul,
            MAX("Handset Manufacturer") as handset_manufacturer,
            MAX("Handset Type") as handset_type,
            AVG("Avg Bearer TP DL (kbps)") as tp_dl,
            AVG("Avg Bearer TP UL (kbps)") as tp_ul,
            SUM("Total DL (Bytes)") as total_dl_bytes,
            SUM("Total UL (Bytes)") as total_ul_bytes,
            SUM("Dur. (ms)") as duration
        FROM xdr_data
        GROUP BY "MSISDN/Number"
        """
        df = load_data(query)
        
        if df is not None:
            # Calculate derived metrics
            df['avg_rtt'] = (df['rtt_dl'] + df['rtt_ul']) / 2
            
            # Calculate average throughput in Mbps
            df['throughput_dl'] = df['tp_dl'] / 1000  # Convert kbps to Mbps
            df['throughput_ul'] = df['tp_ul'] / 1000  # Convert kbps to Mbps
            df['avg_throughput'] = (df['throughput_dl'] + df['throughput_ul']) / 2
            
            # Calculate TCP retransmission rate (bytes retransmitted / total bytes)
            df['tcp_retrans_rate'] = df['tcp_retrans_vol'] / (df['total_dl_bytes'] + df['total_ul_bytes'])
            
            # Select final columns for analysis
            df_final = df[[
                'msisdn', 
                'handset_manufacturer',
                'handset_type',
                'tcp_retrans_rate',
                'avg_rtt',
                'avg_throughput'
            ]]
            return df_final
        
        else:
            raise Exception("Failed to load data from database")
    
    def process_metrics(self, df):
        """Process and clean the metrics data."""
        numeric_cols = ['tcp_retrans_rate', 'avg_rtt', 'avg_throughput']
        categorical_cols = ['handset_manufacturer', 'handset_type']

		# Handle missing values
        df = self.preprocessor.handle_missing_values(df, numeric_cols + categorical_cols)
	
		# Handle outliers in numeric columns
        df = self.preprocessor.handle_outliers(df, numeric_cols)
    
        return df
    
class MetricsAnalyzer:
    @staticmethod
    def get_distribution_stats(df, columns, n=10):
        """Get distribution statistics for specified columns."""
        results = {}
        for col in columns:
            results[col] = {
                'top_n': df[col].nlargest(n).tolist(),
                'bottom_n': df[col].nsmallest(n).tolist(),
                'most_frequent': df[col].value_counts().head(n).index.tolist()
            }
        return results

    @staticmethod
    def get_handset_metrics(df):
        """Get metrics grouped by handset type."""
        metrics = ['avg_throughput', 'tcp_retrans_rate', 'avg_rtt']
        results = {}
        
        for metric in metrics:
            results[metric] = df.groupby(['handset_manufacturer', 'handset_type'])[metric].agg([
                'mean', 'median', 'std', 'count'
            ]).sort_values('mean', ascending=False)
            
        return results
    
class ExperienceClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def prepare_features(self, df):
        """Prepare features for clustering."""
        features = ['tcp_retrans_rate', 'avg_rtt', 'avg_throughput']
        X = df[features]
        return self.scaler.fit_transform(X), features
    def perform_clustering(self, df):
        """Perform clustering analysis."""
        X_scaled, features = self.prepare_features(df)
        
        # Fit clustering
        df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Get cluster statistics
        cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=features
        )
        
        cluster_stats = df.groupby('cluster')[features].agg([
            'mean', 'median', 'std', 'count'
        ])
        
        return {
            'cluster_stats': cluster_stats,
            'cluster_sizes': df['cluster'].value_counts(),
            'cluster_centers': cluster_centers,
            'clustered_data': df
        }





def analyze_customer_experience():
    """Main function to perform complete experience analysis."""
    # Initialize classes
    metrics = ExperienceMetrics()
    analyzer = MetricsAnalyzer()
    clusterer = ExperienceClusterer()

    # Get and process data
    df = metrics.get_customer_metrics()
    df = metrics.process_metrics(df)

    # Perform analyses
    distribution_stats = analyzer.get_distribution_stats(
        df, 
        ['tcp_retrans_rate', 'avg_rtt', 'avg_throughput']
    )
    handset_metrics = analyzer.get_handset_metrics(df)
    clustering_results = clusterer.perform_clustering(df)

    return {
        'processed_data': df,
        'distribution_stats': distribution_stats,
        'handset_metrics': handset_metrics,
        'clustering_results': clustering_results
    }

