#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv(r'C:\Users\sauma\Documents\propertyvaluation\csvdata.csv', names=['ID', 'City', 'Price', 'Area', 'Location', 'Bedrooms'])

# Convert 'Area' and 'Bedrooms' to numeric, dropping any rows with non-numeric values
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with NaN values
df = df.dropna(subset=['Area', 'Bedrooms', 'Price'])

# Basic data exploration
print(df.describe())

# Prepare data for prediction
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Print the model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Area Coefficient: {model.coef_[0]:.2f}")
print(f"Bedrooms Coefficient: {model.coef_[1]:.2f}")

# Function to predict price
def predict_price(area, bedrooms):
    return model.predict([[area, bedrooms]])[0]

# Example predictions
print("\nExample Predictions:")
print(f"Predicted price for a 1500 sq ft, 3-bedroom house: {predict_price(1500, 3):.2f}")
print(f"Predicted price for a 2000 sq ft, 4-bedroom house: {predict_price(2000, 4):.2f}")

# Visualization: Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Visualization: Price vs Area with prediction line
plt.figure(figsize=(10, 6))
plt.scatter(df['Area'], df['Price'], alpha=0.5)
area_range = np.linspace(df['Area'].min(), df['Area'].max(), 100)
price_pred = model.predict(pd.DataFrame({'Area': area_range, 'Bedrooms': [df['Bedrooms'].median()] * 100}))
plt.plot(area_range, price_pred, 'r', lw=2)
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Price vs Area with Prediction Line')
plt.show()

#%%
#CODE FOR ML MODELS USED AND FORCASTING
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data from CSV file
df = pd.read_csv(r'C:\Users\sauma\Documents\propertyvaluation\csvdata.csv', names=['ID', 'City', 'Price', 'Area', 'Location', 'Bedrooms'])
df = df.dropna()

# Convert numeric columns to appropriate types
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')

# Drop rows with NaN values after conversion
df = df.dropna()

# Feature engineering
df['Price_per_sqft'] = df['Price'] / df['Area']

# Prepare data for prediction
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate models
print("Linear Regression Performance:")
print(f"R2 Score: {r2_score(y_test, lr_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")

print("\nRandom Forest Performance:")
print(f"R2 Score: {r2_score(y_test, rf_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")

# Feature importance (for Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Predictions
def predict_price(model, area, bedrooms):
    return model.predict([[area, bedrooms]])[0]

print("\nPrice Predictions:")
print(f"Linear Regression - 1500 sqft, 3 bedrooms: {predict_price(lr_model, 1500, 3):.2f}")
print(f"Random Forest - 1500 sqft, 3 bedrooms: {predict_price(rf_model, 1500, 3):.2f}")

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(df['Area'], df['Price'], alpha=0.5)
plt.xlabel('Area (sqft)')
plt.ylabel('Price')
plt.title('Price vs Area')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['Bedrooms'], df['Price'], alpha=0.5)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Price vs Number of Bedrooms')
plt.show()

# Price trends by location
location_avg_price = df.groupby('Location')['Price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
location_avg_price.plot(kind='bar')
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset

# Create a synthetic time series dataset from the property data
# Assuming the data is in a list of dictionaries format
property_data = [
    {"Price": 8025000, "Area": 1433, "Bedrooms": 3},
    {"Price": 17300000, "Area": 1408, "Bedrooms": 2},
    {"Price": 4671000, "Area": 2494, "Bedrooms": 4},
    # ... (include more data points)
]

# Convert to DataFrame
df = pd.DataFrame(property_data)

# Create a date range
date_range = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

# Add the date range to the DataFrame
df['Date'] = date_range

# Set Date as index and sort
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Calculate daily average price
price_data = df['Price'].resample('D').mean()

# Fill NaN values with forward fill method
price_data = price_data.fillna(method='ffill')

# Fit ARIMA model
model = ARIMA(price_data, order=(1,1,1))
results = model.fit()

# Make predictions
last_date = price_data.index[-1]
future_dates = [last_date + DateOffset(days=x) for x in range(1, 31)]  # Predict next 30 days
future_datest_df = pd.DataFrame(index=future_dates, columns=['Price'])
future_df = pd.concat([price_data.to_frame(name='Price'), future_datest_df])

future_df['Forecast'] = results.predict(start=len(price_data), end=len(price_data)+29, dynamic=True)

# Visualize forecast
plt.figure(figsize=(12,6))
plt.plot(future_df.index, future_df['Price'], label='Historical')
plt.plot(future_df.index, future_df['Forecast'], label='Forecast', color='red')
plt.title('Bangalore Real Estate Price Forecast')
plt.xlabel('Date')
plt.ylabel('Average Price (INR)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print forecast values
print(future_df['Forecast'].tail())
# %%