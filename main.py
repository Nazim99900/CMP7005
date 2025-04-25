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


# Execute Tasks
if __name__ == "__main__":
    # Task 1: Data Handling
    merged_data = load_and_merge_data()
    print("\nTask 1 completed successfully!")
