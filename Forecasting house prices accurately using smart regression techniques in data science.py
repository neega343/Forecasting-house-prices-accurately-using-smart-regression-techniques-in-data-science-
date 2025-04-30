# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# 2. Data Preprocessing
# Handle missing values
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)

# Handle outliers using IQR method
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

df['median_house_value'] = cap_outliers(df['median_house_value'])
df['total_rooms'] = cap_outliers(df['total_rooms'])

# Log transform target variable
df['median_house_value'] = np.log1p(df['median_house_value'])

# Drop ocean_proximity (categorical) as document mentions no categorical variables used
df = df.drop('ocean_proximity', axis=1)

# 3. Feature Engineering
# Create new features
df['rooms_per_household'] = df['total_rooms'] / df['households']

# Approximate distance to coast (simplified using longitude and latitude)
df['distance_to_coast'] = np.abs(df['longitude'] + 122)  # Simplified approximation

# Create polynomial features for median_income
poly = PolynomialFeatures(degree=2, include_bias=False)
median_income_poly = poly.fit_transform(df[['median_income']])
df['median_income_poly1'] = median_income_poly[:, 1]

# 4. Exploratory Data Analysis (EDA)
# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Scatter plot: median_income vs median_house_value
plt.figure(figsize=(8, 6))
plt.scatter(df['median_income'], df['median_house_value'], alpha=0.5)
plt.xlabel('Median Income')
plt.ylabel('Log Median House Value')
plt.title('Median Income vs House Value')
plt.savefig('income_vs_house_value.png')
plt.close()

# Geographical plot
fig = px.scatter(df, x='longitude', y='latitude', color='median_house_value',
                 title='House Prices by Location')
fig.write('geo_plot.html')

# 5. Prepare data for modeling
# Define features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model Building and Evaluation
# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train and evaluate models
models = {'Linear Regression': lr, 'Random Forest': rf}
results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    
    # Visualizations for Random Forest
    if name == 'Random Forest':
        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot (Random Forest)')
        plt.savefig('rf_residual_plot.png')
        plt.close()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance (Random Forest)')
        plt.savefig('rf_feature_importance.png')
        plt.close()
        
        # Predicted vs Actual plot
