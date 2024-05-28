def advancedStats(df):
    """Advanced stats should leverage pandas to calculate
    some relevant statistics on the data."""
    # Calculate skewness and kurtosis for each numeric column
    numeric_columns = [col for col in df.columns if df[col].dtype.kind in 'ifc']
    count = 0
    for col in numeric_columns:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        print(f"\nColumn {count+1} statistics:")
        print(f"Skewness: {skewness}    Kurtosis: {kurtosis}")
        count += 1
    # Print the decriptive statistics for the dataset
    print(df.describe())
