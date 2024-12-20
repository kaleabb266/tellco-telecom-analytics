from load_data import load_data

def aggregate_application_traffic():
    # """Aggregate total user traffic for specific applications (YouTube, Gaming, etc.)."""
    query = """
    SELECT 
        "MSISDN/Number" as msisdn, 
        SUM("Youtube DL (Bytes)") + SUM("Youtube UL (Bytes)") AS youtube_traffic, 
        SUM("Gaming DL (Bytes)") + SUM("Gaming UL (Bytes)") AS gaming_traffic, 
        SUM("Social Media DL (Bytes)") + SUM("Social Media UL (Bytes)") AS social_media_traffic 
    FROM xdr_data 
    GROUP BY "MSISDN/Number";
    """
    df = load_data(query)
    return df
