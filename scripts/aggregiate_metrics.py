from load_data import load_data


def aggregate_user_metrics():
    query = """
    SELECT 
        "MSISDN/Number" as msisdn,
        COUNT(*) as session_count,
        SUM("Dur. (ms)") as total_duration,
        SUM("Total DL (Bytes)") as total_dl,
        SUM("Total UL (Bytes)") as total_ul,
        SUM("Social Media DL (Bytes)" + "Social Media UL (Bytes)") as social_media_total,
        SUM("Google DL (Bytes)" + "Google UL (Bytes)") as google_total,
        SUM("Email DL (Bytes)" + "Email UL (Bytes)") as email_total,
        SUM("Youtube DL (Bytes)" + "Youtube UL (Bytes)") as youtube_total,
        SUM("Netflix DL (Bytes)" + "Netflix UL (Bytes)") as netflix_total,
        SUM("Gaming DL (Bytes)" + "Gaming UL (Bytes)") as gaming_total,
        SUM("Other DL (Bytes)" + "Other UL (Bytes)") as other_total
    FROM xdr_data
    GROUP BY "MSISDN/Number"
    """
    return load_data(query)