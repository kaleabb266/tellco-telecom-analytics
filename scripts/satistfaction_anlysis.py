import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestRegressor

class SatisfactionAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.engagement_kmeans = KMeans(n_clusters=3, random_state=42)
        self.experience_kmeans = KMeans(n_clusters=3, random_state=42)
        
    def calculate_engagement_score(self, user_metrics,kmeans_model=None):
        """Calculate engagement score based on distance from least engaged cluster"""
        # Remove rows with NaN MSISDN
        user_metrics = user_metrics.dropna(subset=['msisdn'])
        # Scale the features
        features = ['session_count', 'total_duration', 'total_dl', 'total_ul']
        X = self.scaler.fit_transform(user_metrics[features])
        
         # Fit kmeans if not already fitted
        if kmeans_model is None:
           self.engagement_kmeans.fit(X)
           kmeans_model = self.engagement_kmeans
        
        # Calculate distances to all cluster centers
        distances = euclidean_distances(X, kmeans_model.cluster_centers_)
        
        
        # Get distance to least engaged cluster (assuming cluster 0)
        engagement_distances = distances[:, 0]
        
        # Normalize scores to 0-1 range
        scores = (engagement_distances - engagement_distances.min()) / (engagement_distances.max() - engagement_distances.min())
        return scores

    def calculate_experience_score(self, experience_data ,kmeans_model=None):
            """Calculate experience score based on distance from worst experience cluster"""
            # Remove rows with NaN MSISDN
            experience_data = experience_data.dropna(subset=['msisdn'])
            
            # Scale the features
            features = ['tcp_retrans_rate', 'avg_rtt', 'avg_throughput']
            X = self.scaler.fit_transform(experience_data[features])
            
            # Fit kmeans if not already fitted
            if kmeans_model is None:
                self.experience_kmeans.fit(X)
                kmeans_model = self.experience_kmeans
            
           # Calculate distances to all cluster centers
            distances = euclidean_distances(X, kmeans_model.cluster_centers_)
            
            # Get distance to worst experience cluster (assuming cluster 0)
            experience_distances = distances[:, 0]
            
            # Normalize scores to 0-1 range
            scores = (experience_distances - experience_distances.min()) / (experience_distances.max() - experience_distances.min())
            return scores
        
    
    def calculate_satisfaction_score(self, engagement_scores, experience_scores):
        """Calculate overall satisfaction score as average of engagement and experience"""
        return (engagement_scores + experience_scores) / 2
    
    def build_prediction_model(self, X, y):
        """Build and train a Random Forest model to predict satisfaction scores"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    def cluster_users(self, engagement_scores, experience_scores):
        """Perform k-means clustering on engagement and experience scores"""
        X = np.column_stack((engagement_scores, experience_scores))
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X)
        return clusters
    
    def cluster_users(self, engagement_scores, experience_scores):
        """Perform k-means clustering on engagement and experience scores"""
        X = np.column_stack((engagement_scores, experience_scores))
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X)
        return clusters
    
    
    def aggregate_cluster_scores(self, clusters, satisfaction_scores, experience_scores):
        """Calculate average satisfaction and experience scores per cluster"""
        df = pd.DataFrame({
            'cluster': clusters,
            'satisfaction_score': satisfaction_scores,
            'experience_score': experience_scores
        })
        return df.groupby('cluster').mean()

        

      


