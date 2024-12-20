from load_data import load_data

def aggregate_user_metrics():
    # """Aggregate session metrics (session frequency, duration, total traffic) per user."""
    query = """
    SELECT 
        "MSISDN/Number" as msisdn, 
        COUNT("Bearer Id") AS session_frequency, 
        SUM("Dur. (ms)") AS total_session_duration, 
        SUM("Total UL (Bytes)") + SUM("Total DL (Bytes)") AS total_session_traffic 
    FROM xdr_data 
    GROUP BY "MSISDN/Number";
    """
    df = load_data(query)
    return df