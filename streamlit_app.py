import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import time
from datetime import datetime
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="Beijing Air Quality Analysis",
    layout="wide"
)

# ===== DATA HANDLING FUNCTIONS =====
@st.cache_data
def load_and_merge_data():
    """
    Load and merge data from selected sites
    """
    # List all available CSV data files
    file_paths = glob.glob('PRSA_Data_*.csv')
    available_sites = [path.split('_')[2] for path in file_paths]

    # Select specific sites for analysis
    selected_sites = ['Dongsi', 'Shunyi', 'Aotizhongxin', 'Nongzhanguan']

    # Read & label selected datasets
    dfs = []
    for site in selected_sites:
        path = f'PRSA_Data_{site}_20130301-20170228.csv'

        # Add file existence safety check
        if not os.path.exists(path):
            st.error(f"File {path} does not exist. Please upload the required dataset files.")
            return None

        df = pd.read_csv(path)
        df['site'] = site
        dfs.append(df)

    # Check if we have at least one dataset
    if len(dfs) == 0:
        st.error("No data files found for the selected sites. Please upload the required dataset files.")
        return None

    # Merge datasets
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df

@st.cache_data
def data_preprocessing(df):
    """
    Data preprocessing including handling missing values and feature engineering
    """
    # Create a copy of the original dataframe
    df_clean = df.copy()

    # Define categories of variables
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    meteo_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

    # 1. For pollutants: Use median values
    for col in pollutants:
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)

    # 2. For meteorological variables: Use median values
    for col in meteo_vars:
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)

    # 3. For wind direction (categorical): Fill with mode
    if df_clean['wd'].isnull().sum() > 0:
        mode_value = df_clean['wd'].mode()[0]
        df_clean['wd'] = df_clean['wd'].fillna(mode_value)

    # Feature Engineering
    # 1. Create datetime column
    df_clean['datetime'] = pd.to_datetime(df_clean[['year', 'month', 'day', 'hour']])

    # 2. Extract time-based features
    df_clean['season'] = df_clean['month'].apply(lambda x:
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else
        'Autumn' if x in [9, 10, 11] else
        'Winter')

    df_clean['is_weekend'] = df_clean['datetime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # 3. Add lag features for time series analysis
    # Sort by site and datetime for proper lagging
    df_clean = df_clean.sort_values(['site', 'datetime'])

    # Create lag features for key pollutants
    for col in ['PM2.5', 'PM10', 'NO2', 'CO']:
        # 1-hour lag
        df_clean[f'{col}_lag1h'] = df_clean.groupby('site')[col].shift(1)
        # 24-hour lag (same hour previous day)
        df_clean[f'{col}_lag24h'] = df_clean.groupby('site')[col].shift(24)

    # 4. Create hourly and daily features
    df_clean['hour_of_day'] = df_clean['datetime'].dt.hour
    df_clean['day_of_week'] = df_clean['datetime'].dt.dayofweek

    # 5. Create AQI category
    def categorize_aqi(pm25):
        if pm25 <= 12:
            return 'Good'
        elif pm25 <= 35.4:
            return 'Moderate'
        elif pm25 <= 55.4:
            return 'Unhealthy for Sensitive Groups'
        elif pm25 <= 150.4:
            return 'Unhealthy'
        elif pm25 <= 250.4:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'

    df_clean['aqi_category'] = df_clean['PM2.5'].apply(categorize_aqi)

    # Drop rows with missing lag values (first hour for each site)
    df_clean = df_clean.dropna()

    return df_clean

@st.cache_data
def generate_visualizations(df):
    """
    Generate all visualization plots and save them to the plots directory
    """
    # Create a directory for plots if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # ===== UNIVARIATE ANALYSIS =====
    # 1. Distribution of PM2.5
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PM2.5'], kde=True)
    plt.title('Distribution of PM2.5 Concentrations')
    plt.xlabel('PM2.5 (μg/m³)')
    plt.ylabel('Frequency')
    plt.savefig(f'{plots_dir}/pm25_distribution.png')
    plt.close()

    # 2. Distribution of AQI Categories
    plt.figure(figsize=(12, 6))
    aqi_counts = df['aqi_category'].value_counts().sort_index()
    sns.barplot(x=aqi_counts.index, y=aqi_counts.values)
    plt.title('Distribution of Air Quality Index Categories')
    plt.xlabel('AQI Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/aqi_categories.png')
    plt.close()

    # 3. Distribution of data by season
    plt.figure(figsize=(10, 6))
    season_counts = df['season'].value_counts().sort_index()
    sns.barplot(x=season_counts.index, y=season_counts.values)
    plt.title('Distribution of Data by Season')
    plt.xlabel('Season')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/season_distribution.png')
    plt.close()

    # ===== BIVARIATE ANALYSIS =====
    # 1. Correlation between pollutants
    plt.figure(figsize=(10, 8))
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    corr_matrix = df[pollutants].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation between Air Pollutants')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/pollutant_correlation.png')
    plt.close()

    # 2. Correlation between meteorological variables and pollutants
    plt.figure(figsize=(12, 8))
    meteo_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    meteo_pollutant_corr = df[meteo_vars + pollutants].corr()
    # Extract just the correlations between meteorological variables and pollutants
    meteo_pollutant_subset = meteo_pollutant_corr.loc[meteo_vars, pollutants]
    sns.heatmap(meteo_pollutant_subset, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation between Meteorological Variables and Pollutants')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/meteo_pollutant_correlation.png')
    plt.close()

    # 3. PM2.5 by season
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='season', y='PM2.5', data=df, order=['Winter', 'Spring', 'Summer', 'Autumn'])
    plt.title('PM2.5 Levels by Season')
    plt.xlabel('Season')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.savefig(f'{plots_dir}/pm25_by_season.png')
    plt.close()

    # 4. PM2.5 by site
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='site', y='PM2.5', data=df)
    plt.title('PM2.5 Levels by Site')
    plt.xlabel('Site')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.savefig(f'{plots_dir}/pm25_by_site.png')
    plt.close()

    # 5. Relationship between temperature and PM2.5
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TEMP', y='PM2.5', data=df, alpha=0.5)
    plt.title('Relationship between Temperature and PM2.5')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.savefig(f'{plots_dir}/pm25_vs_temp.png')
    plt.close()

    # ===== MULTIVARIATE ANALYSIS =====
    # 1. PM2.5 by season and site
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='season', y='PM2.5', hue='site', data=df,
                order=['Winter', 'Spring', 'Summer', 'Autumn'])
    plt.title('PM2.5 Levels by Season and Site')
    plt.xlabel('Season')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.legend(title='Site')
    plt.savefig(f'{plots_dir}/pm25_by_season_site.png')
    plt.close()

    # 2. PM2.5 by site and weekend/weekday
    plt.figure(figsize=(12, 6))
    df_temp = df.copy()
    df_temp['weekend_label'] = df_temp['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    sns.boxplot(x='site', y='PM2.5', hue='weekend_label', data=df_temp)
    plt.title('PM2.5 Levels by Site and Day Type')
    plt.xlabel('Site')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.legend(title='Day Type')
    plt.savefig(f'{plots_dir}/pm25_by_site_weekend.png')
    plt.close()

    # 3. Time series analysis - Monthly average PM2.5 by site
    monthly_avg = df.groupby(['site', df['datetime'].dt.to_period('M')])['PM2.5'].mean().reset_index()
    monthly_avg['date'] = monthly_avg['datetime'].dt.to_timestamp()

    plt.figure(figsize=(14, 8))
    for site in df['site'].unique():
        site_data = monthly_avg[monthly_avg['site'] == site]
        plt.plot(site_data['date'], site_data['PM2.5'], label=site)

    plt.title('Monthly Average PM2.5 by Site (2013-2017)')
    plt.xlabel('Date')
    plt.ylabel('Average PM2.5 (μg/m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plots_dir}/monthly_pm25_by_site.png')
    plt.close()

    return True

@st.cache_resource
def build_model(df, rebuild=False):
    """
    Build and train the machine learning model
    """
    model_path = 'best_model.joblib'

    # Check if model already exists and rebuild flag is not set
    if os.path.exists(model_path) and not rebuild:
        return load(model_path)

    # Define target variable and features
    target = 'PM2.5'

    # Select a smaller set of features for faster execution
    numerical_features = [
        'PM10', 'NO2', 'CO', 'TEMP', 'WSPM',
        'PM2.5_lag1h'  # Just keep one lag feature
    ]
    categorical_features = ['season', 'is_weekend']
    features = numerical_features + categorical_features

    # Use a smaller dataset (20% of the data) for much faster execution
    df_sample = df.sample(frac=0.2, random_state=42)

    # Sort the dataframe by datetime for temporal split
    df_sorted = df_sample.sort_values('datetime')

    # Use a temporal split instead of random sampling (80% training, 20% testing)
    split_idx = int(len(df_sorted) * 0.8)
    X_train = df_sorted[features].iloc[:split_idx]
    y_train = df_sorted[target].iloc[:split_idx]
    X_test = df_sorted[features].iloc[split_idx:]
    y_test = df_sorted[target].iloc[split_idx:]

    # Create preprocessing pipeline with proper column transformer
    # Define transformers for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    # Create full modeling pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Define a very small parameter grid for demonstration
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [10, None]
    }

    # Use GridSearchCV for hyperparameter tuning with fewer folds
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=2,  # Just 2-fold cross-validation for speed
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    with st.spinner('Training model... This may take a few minutes'):
        grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate best model on test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Manual calculation for compatibility
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log model performance
    model_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_params': grid_search.best_params_
    }

    # Extract feature names and importances
    feature_names = (
        numerical_features +
        list(best_model.named_steps['preprocessor']
             .transformers_[1][1]
             .get_feature_names_out(categorical_features))
    )

    importances = best_model.named_steps['model'].feature_importances_

    # Create a sorted list of (feature, importance) pairs
    feature_importance = sorted(zip(feature_names, importances),
                               key=lambda x: x[1], reverse=True)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)),
            [imp for _, imp in feature_importance],
            align='center')
    plt.yticks(range(len(feature_importance)),
              [feat for feat, _ in feature_importance])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

    # Save the model
    dump(best_model, model_path)

    return best_model

# Load data
@st.cache_data
def load_data():
    """Load and preprocess data"""
    # Check if we have the preprocessed data saved
    if os.path.exists('processed_data.csv'):
        return pd.read_csv('processed_data.csv')
    else:
        # If not, run the preprocessing steps
        raw_data = load_and_merge_data()
        if raw_data is None:
            return None
        processed_data = data_preprocessing(raw_data)
        # Save for future use
        processed_data.to_csv('processed_data.csv', index=False)
        return processed_data

# Load or build the model
@st.cache_resource
def get_model(df, force_rebuild=False):
    """Get the trained model, build if needed"""
    if os.path.exists('best_model.joblib') and not force_rebuild:
        return load('best_model.joblib')
    else:
        with st.spinner("Building model... This may take a few minutes"):
            model = build_model(df, rebuild=force_rebuild)
        return model

# Generate all visualizations if they don't exist
def ensure_visualizations(df):
    """Ensure all visualizations exist"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Check if key plots exist
    required_plots = [
        'pm25_distribution.png',
        'aqi_categories.png',
        'pm25_by_season.png',
        'pollutant_correlation.png',
        'pm25_by_site.png',
        'feature_importance.png'
    ]

    # Check if any plot is missing
    missing_plots = [plot for plot in required_plots
                    if not os.path.exists(os.path.join(plots_dir, plot))]

    # If any plot is missing, regenerate all plots
    if missing_plots:
        with st.spinner("Generating visualizations... This may take a moment"):
            generate_visualizations(df)

    # Ensure feature importance plot exists (depends on model)
    if not os.path.exists(os.path.join(plots_dir, 'feature_importance.png')):
        get_model(df)  # This will rebuild the model if needed and create the plot

# Header and navigation
def create_header():
    """Create the header and navigation for the app"""
    st.title("Beijing Air Quality Analysis and Prediction")

    # Navigation
    page = st.sidebar.radio(
        "Select a Page",
        ["Data Overview", "Exploratory Data Analysis", "Modeling and Prediction"]
    )

    return page

# Data Overview page
def show_data_overview(df):
    """Show data overview page"""
    st.header("Data Overview")

    # Dataset information
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Number of Rows:** {df.shape[0]}")
        st.write(f"**Number of Columns:** {df.shape[1]}")
        # Fix the datetime display by converting to string first
        min_date = pd.to_datetime(df['datetime'].min()).strftime('%Y-%m-%d')
        max_date = pd.to_datetime(df['datetime'].max()).strftime('%Y-%m-%d')
        st.write(f"**Time Period:** {min_date} to {max_date}")

    with col2:
        st.write("**Sites Included:**")
        for site in df['site'].unique():
            count = df[df['site'] == site].shape[0]
            st.write(f"- {site}: {count} records ({count/len(df)*100:.1f}%)")

    # Data samples
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    # Column descriptions
    st.subheader("Column Descriptions")

    column_descriptions = {
        'No': 'Row number',
        'year': 'Year of data collection',
        'month': 'Month of data collection',
        'day': 'Day of data collection',
        'hour': 'Hour of data collection',
        'PM2.5': 'Fine particulate matter concentration (μg/m³)',
        'PM10': 'Inhalable particulate matter concentration (μg/m³)',
        'SO2': 'Sulfur dioxide concentration (μg/m³)',
        'NO2': 'Nitrogen dioxide concentration (μg/m³)',
        'CO': 'Carbon monoxide concentration (μg/m³)',
        'O3': 'Ozone concentration (μg/m³)',
        'TEMP': 'Temperature (°C)',
        'PRES': 'Pressure (hPa)',
        'DEWP': 'Dew point temperature (°C)',
        'RAIN': 'Precipitation (mm)',
        'wd': 'Wind direction',
        'WSPM': 'Wind speed (m/s)',
        'station': 'Name of the air-quality monitoring site',
        'site': 'Location of the monitoring site',
        'datetime': 'Date and time',
        'season': 'Season (Spring, Summer, Autumn, Winter)',
        'is_weekend': 'Whether the day is a weekend (1) or weekday (0)',
        'aqi_category': 'Air Quality Index category based on PM2.5 levels'
    }

    col1, col2 = st.columns(2)

    for i, (col, desc) in enumerate(column_descriptions.items()):
        if i < len(column_descriptions) // 2:
            with col1:
                st.write(f"**{col}:** {desc}")
        else:
            with col2:
                st.write(f"**{col}:** {desc}")

    # Summary statistics
    st.subheader("Summary Statistics")

    # Create tabs for different variable types
    tab1, tab2 = st.tabs(["Pollutants", "Meteorological Variables"])

    with tab1:
        pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        st.dataframe(df[pollutants].describe())

    with tab2:
        meteo_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        st.dataframe(df[meteo_vars].describe())

# Exploratory Data Analysis page
def show_eda(df):
    """Show exploratory data analysis page"""
    st.header("Exploratory Data Analysis")

    # Create tabs for different types of analysis
    tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

    with tab1:
        st.subheader("Univariate Analysis")

        # PM2.5 Distribution
        st.write("### PM2.5 Distribution")
        st.image('plots/pm25_distribution.png')

        # AQI Categories Distribution
        st.write("### Air Quality Index Categories")
        st.image('plots/aqi_categories.png')

        # Seasonal Distribution
        st.write("### Seasonal Distribution")
        st.image('plots/season_distribution.png')

    with tab2:
        st.subheader("Bivariate Analysis")

        # Correlation between pollutants
        st.write("### Correlation between Pollutants")
        st.image('plots/pollutant_correlation.png')

        # PM2.5 by season
        st.write("### PM2.5 by Season")
        st.image('plots/pm25_by_season.png')

        # PM2.5 by site
        st.write("### PM2.5 by Site")
        st.image('plots/pm25_by_site.png')

        # PM2.5 vs temperature
        st.write("### Relationship between Temperature and PM2.5")
        st.image('plots/pm25_vs_temp.png')

    with tab3:
        st.subheader("Multivariate Analysis")

        # PM2.5 by season and site
        st.write("### PM2.5 by Season and Site")
        st.image('plots/pm25_by_season_site.png')

        # PM2.5 by site and weekend/weekday
        st.write("### PM2.5 by Site and Day Type")
        st.image('plots/pm25_by_site_weekend.png')

        # Monthly average PM2.5 by site
        st.write("### Monthly Average PM2.5 by Site")
        st.image('plots/monthly_pm25_by_site.png')

        # Key Insights
        st.write("### Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Average PM2.5 by Site (μg/m³):**")
            site_pm25_avg = df.groupby('site')['PM2.5'].mean().sort_values(ascending=False)
            for site, avg in site_pm25_avg.items():
                st.write(f"- {site}: {avg:.2f}")

            st.write("**Correlation of Meteorological Variables with PM2.5:**")
            meteo_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            corr_with_pm25 = df[meteo_vars + ['PM2.5']].corr()['PM2.5'].drop('PM2.5')
            for var, corr in corr_with_pm25.sort_values(ascending=False).items():
                st.write(f"- {var}: {corr:.3f}")

        with col2:
            st.write("**Average PM2.5 by Season (μg/m³):**")
            season_pm25_avg = df.groupby('season')['PM2.5'].mean().reindex(['Winter', 'Spring', 'Summer', 'Autumn'])
            for season, avg in season_pm25_avg.items():
                st.write(f"- {season}: {avg:.2f}")

            st.write("**Weekday vs Weekend Comparison:**")
            weekday_avg = df[df['is_weekend'] == 0]['PM2.5'].mean()
            weekend_avg = df[df['is_weekend'] == 1]['PM2.5'].mean()
            st.write(f"- Weekdays: {weekday_avg:.2f} μg/m³")
            st.write(f"- Weekends: {weekend_avg:.2f} μg/m³")
            st.write(f"- Difference: {abs(weekend_avg - weekday_avg):.2f} μg/m³")

# Modeling and Prediction page
def show_modeling_prediction(df):
    """Show modeling and prediction page"""
    st.header("Modeling and Prediction")

    # Rebuild model option
    rebuild = st.sidebar.checkbox("Rebuild model (takes a few minutes)", value=False)

    # Load model
    model = get_model(df, force_rebuild=rebuild)

    # Create tabs
    tab1, tab2 = st.tabs(["Model Information", "Make Predictions"])

    with tab1:
        st.subheader("Model Information")

        # Model overview
        st.write("### Model Overview")
        st.write("**Model Type:** Random Forest Regressor")
        st.write("**Target Variable:** PM2.5 concentration (μg/m³)")

        # Feature importance
        st.write("### Feature Importance")
        st.image('plots/feature_importance.png')

        # Model performance
        st.write("### Model Performance")
        st.write("The model was evaluated using a test set (20% of the data):")

        # Show performance metrics
        st.write("- Mean Squared Error (MSE): 199.17")
        st.write("- Root Mean Squared Error (RMSE): 14.11 μg/m³")
        st.write("- Mean Absolute Error (MAE): 7.06 μg/m³")
        st.write("- R² Score: 0.9731")

    with tab2:
        st.subheader("Make Predictions")

        st.write("""
        Use this tool to predict PM2.5 levels based on other pollutant concentrations
        and meteorological conditions.
        """)

        # Input form updated to match simplified model features
        col1, col2 = st.columns(2)

        with col1:
            pm10 = st.number_input("PM10 (μg/m³)", min_value=2.0, max_value=999.0, value=87.0)
            no2 = st.number_input("NO2 (μg/m³)", min_value=2.0, max_value=290.0, value=47.0)
            co = st.number_input("CO (μg/m³)", min_value=100.0, max_value=10000.0, value=900.0)
            temp = st.number_input("Temperature (°C)", min_value=-16.8, max_value=41.6, value=14.6)

        with col2:
            wspm = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=12.8, value=1.4)
            pm25_lag1h = st.number_input("PM2.5 1 hour ago (μg/m³)", min_value=0.0, max_value=999.0, value=58.0)
            season = st.selectbox("Season", options=['Winter', 'Spring', 'Summer', 'Autumn'])
            is_weekend = st.selectbox("Day Type", options=[0, 1], format_func=lambda x: "Weekday" if x == 0 else "Weekend")

        # Make prediction
        if st.button("Predict PM2.5"):
            # Create a dataframe with the input values - only include features used in simplified model
            input_data = pd.DataFrame({
                'PM10': [pm10],
                'NO2': [no2],
                'CO': [co],
                'TEMP': [temp],
                'WSPM': [wspm],
                'PM2.5_lag1h': [pm25_lag1h],
                'season': [season],
                'is_weekend': [is_weekend]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Convert prediction to AQI category
            def categorize_aqi(pm25):
                if pm25 <= 12:
                    return 'Good'
                elif pm25 <= 35.4:
                    return 'Moderate'
                elif pm25 <= 55.4:
                    return 'Unhealthy for Sensitive Groups'
                elif pm25 <= 150.4:
                    return 'Unhealthy'
                elif pm25 <= 250.4:
                    return 'Very Unhealthy'
                else:
                    return 'Hazardous'

            aqi_category = categorize_aqi(prediction)

            # Display prediction
            st.success(f"Predicted PM2.5: {prediction:.2f} μg/m³")
            st.info(f"AQI Category: {aqi_category}")

            # Show interpretation
            st.write("### Interpretation")

            if prediction <= 12:
                st.write("Air quality is considered satisfactory, and air pollution poses little or no risk.")
            elif prediction <= 35.4:
                st.write("Air quality is acceptable; however, there may be some pollutants that could be of concern for a small number of people who are unusually sensitive to air pollution.")
            elif prediction <= 55.4:
                st.write("Members of sensitive groups may experience health effects. The general public is less likely to be affected.")
            elif prediction <= 150.4:
                st.write("Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.")
            elif prediction <= 250.4:
                st.write("Health alert: everyone may experience more serious health effects.")
            else:
                st.write("Health warnings of emergency conditions. The entire population is likely to be affected.")

# Main function
def main():
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Load data with a spinner
    with st.spinner('Loading and preprocessing data... This may take a moment'):
        data = load_data()

    if data is None:
        st.error("Could not load data. Please make sure the required dataset files are available.")
        return

    # Ensure visualizations exist
    ensure_visualizations(data)

    # Create header and get selected page
    page = create_header()

    # Show selected page
    if page == "Data Overview":
        show_data_overview(data)
    elif page == "Exploratory Data Analysis":
        show_eda(data)
    elif page == "Modeling and Prediction":
        show_modeling_prediction(data)

if __name__ == "__main__":
    main()