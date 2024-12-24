import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()

# Get database credentials from .env file
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')


def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("Connected to the database")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def load_data(query, conn=None):
    # """
    # Load data from the PostgreSQL database into a Pandas DataFrame.
    
    # Args:
    # - query (str): The SQL query to execute.
    # - conn (connection, optional): Existing database connection.
    
    # Returns:
    # - pd.DataFrame: Data from the query as a DataFrame.
    # """
    try:
        if conn is None:
            conn = connect_to_db()
        if conn is not None:
            df=pd.read_sql(query, conn)
            print("Data loaded successfully")
            return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None 
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed")

def export_to_db(df, table_name, if_exists='replace'):
    """
    Export DataFrame to PostgreSQL database.
    
    Args:
    - df (pd.DataFrame): DataFrame to export
    - table_name (str): Name of the table to create/update
    - if_exists (str): How to behave if table exists ('fail', 'replace', or 'append')
    
    Returns:
    - bool: True if successful, False otherwise
    """
    try:
        # Create SQLAlchemy engine
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        
        # Export to PostgreSQL
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        print(f"Data successfully exported to table: {table_name}")
        
        # Verify the export
        verification_query = f"SELECT * FROM {table_name} LIMIT 5;"
        verification_result = pd.read_sql(verification_query, engine)
        print("\nVerification of exported data:")
        print(verification_result)
        
        return True
        
    except Exception as e:
        print(f"Error exporting to PostgreSQL: {str(e)}")
        return False