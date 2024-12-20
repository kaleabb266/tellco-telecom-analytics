import pandas as pd

def handle_missing_data(df):
    """Handle missing data by filling with the column mean."""
    return df.fillna(df.mean())

def remove_outliers(df, column_name):
    """Remove outliers by filtering the top 1% and bottom 1% of the column."""
    q1 = df[column_name].quantile(0.01)
    q99 = df[column_name].quantile(0.99)
    return df[(df[column_name] >= q1) & (df[column_name] <= q99)]
