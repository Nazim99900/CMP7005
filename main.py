import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os



# ===== TASK 1: Data Handling =====
def load_and_merge_data():
    """
    Task 1: Data Handling
    - Select and download one site from urban, suburban, rural, and Industrial/hotspots
    - Import the datasets into the development environment
    - Merge the datasets into one to start the analysis
    """
    print("===== TASK 1: Data Handling =====")
    
    # List all available CSV data files
    file_paths = glob.glob('PRSA_Data_*.csv')
    available_sites = [path.split('_')[2] for path in file_paths]
    print(f"Available sites: {available_sites}")
    
    # Categorize sites based on their characteristics
    # Urban: Dongsi - Urban commercial/residential area
    # Suburban: Shunyi - Suburban area
    # Rural: Aotizhongxin - Olympic Sports Center, less dense
    # Industrial/hotspot: Nongzhanguan - Agricultural exhibition center, industrial area
    
    selected_sites = ['Dongsi', 'Shunyi', 'Aotizhongxin', 'Nongzhanguan']
    print(f"Selected sites for analysis: {selected_sites}")
    print(f"- Dongsi: Urban commercial/residential area")
    print(f"- Shunyi: Suburban area")
    print(f"- Aotizhongxin: Rural area (Olympic Sports Center)")
    print(f"- Nongzhanguan: Industrial/hotspot (Agricultural exhibition center)")
    
    # Read & label selected datasets
    dfs = []
    for site in selected_sites:
        path = f'PRSA_Data_{site}_20130301-20170228.csv'
        
        # Add file existence safety check
        if not os.path.exists(path):
            print(f"Warning: File {path} does not exist. Skipping this site.")
            continue
            
        print(f"Importing data from {path}")
        df = pd.read_csv(path)
        df['site'] = site
        print(f"  - Shape: {df.shape}")
        dfs.append(df)
    
    # Check if we have at least one dataset
    if len(dfs) == 0:
        raise FileNotFoundError("No data files found for the selected sites. Please check file names.")
    
    # Merge datasets
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Display first few rows of the merged dataset
    print("\nFirst 5 rows of merged dataset:")
    print(merged_df.head())
    
    return merged_df

# ===== TASK 2a: Fundamental Data Understanding =====
def fundamental_data_understanding(df):
    """
    Task 2a: Fundamental Data Understanding
    - Demonstrate the understanding of the data to gain general insights
    - Covers: number of rows and columns, values in the data, data types, missing values
    """
    print("\n===== TASK 2a: Fundamental Data Understanding =====")
    
    # Display basic information about the dataset
    print("\nBasic Information:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Display column names and data types
    print("\nColumn Names and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")
    
    # Check for missing values
    print("\nMissing Values by Column:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df) * 100).round(2)
    
    for col, count in missing_values.items():
        if count > 0:
            print(f"  - {col}: {count} missing values ({missing_percent[col]}%)")
    
    # Summary statistics for numeric columns
    print("\nSummary Statistics for Key Pollutants:")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    print(df[pollutants].describe().round(2))
    
    # Summary statistics for meteorological data
    print("\nSummary Statistics for Meteorological Data:")
    meteo_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    print(df[meteo_vars].describe().round(2))
    
    # Distribution of data across sites
    print("\nData Distribution Across Sites:")
    site_counts = df['site'].value_counts()
    for site, count in site_counts.items():
        print(f"  - {site}: {count} records ({count/len(df)*100:.2f}%)")
    
    return df


# Execute Tasks
if __name__ == "__main__":
    # Task 1: Data Handling
    merged_data = load_and_merge_data()
    print("\nTask 1 completed successfully!")
    
    # Task 2a: Fundamental Data Understanding
    fundamental_data_understanding(merged_data)
    print("\nTask 2a completed successfully!")
