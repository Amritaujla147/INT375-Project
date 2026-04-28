##import pandas as pd
##import numpy as np
##import matplotlib.pyplot as plt
##import seaborn as sns
##
### ---------------- LOAD DATA ----------------
##df = pd.read_csv("car_dataset_2000_rows.csv")
##
##print(df.head())
##print("\nColumns:", df.columns)
##print("Total Rows:", len(df))
##
### ---------------- DATA CLEANING ----------------
##
### Fix column names (remove spaces)
##df.columns = df.columns.str.strip()
##
### Missing values
##print("\nMissing Values:\n", df.isnull().sum())
##
##df = df.ffill()
##df.drop_duplicates(inplace=True)
##
### ---------------- EDA ----------------
##print("\nStatistical Summary:\n", df.describe())
##
### ---------------- VISUALIZATIONS ----------------
##
### 1 BAR CHART (Average numeric values)
##
##
##
##fuel_price = df.groupby('fuel_type')['price'].mean().sort_values()
##
##plt.figure(figsize=(8,5))
##fuel_price.plot(kind='bar', color='skyblue')
##
##plt.title("Average Car Price by Fuel Type")
##plt.xlabel("Fuel Type")
##plt.ylabel("Average Price")
##plt.xticks(rotation=45)
##plt.grid(axis='y')
##
##plt.show()
##
### 2 HISTOGRAM
##plt.hist(df['price'], bins=30)
##plt.title("Price Distribution")
##plt.xlabel("Price")
##plt.ylabel("Frequency")
##plt.show()
##
### 3 SCATTER PLOT
####plt.scatter(df['mileage'], df['price'])
####plt.xlabel("Mileage")
####plt.ylabel("Price")
####plt.title("Mileage vs Price")
####plt.show()
##plt.figure(figsize=(8,5))
##
##sns.scatterplot(x='mileage', y='price', hue='fuel_type', data=df)
##
##sns.regplot(x='mileage', y='price', data=df,
##            scatter=False, color='red')
##
##plt.title("Mileage vs Price by Fuel Type")
##plt.xlabel("Mileage")
##plt.ylabel("Price")
##
##plt.show()
##
### 4 BOX / WHISKER PLOT
####plt.boxplot(df['price'])
####plt.title("Price Box Plot")
####plt.show()
##
##
##
##top10 = df['brand'].value_counts().head(10).index
##
##plt.figure(figsize=(12,6))
##sns.boxplot(x='brand', y='price', data=df[df['brand'].isin(top10)])
##
##plt.xticks(rotation=45)
##plt.title("Price Distribution by Brand")
##
##plt.show()
##
### 5 HEATMAP (IMPORTANT 🔥)
##numeric_df = df[['year','engine_size','mileage','price']]
##sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
##plt.title("Correlation Heatmap")
##plt.show()
##
### 6 PIE CHART (Fuel Type Distribution)
##fuel_counts = df['fuel_type'].value_counts()
##
##plt.pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%')
##plt.title("Fuel Type Distribution")
##plt.show()
##
### 7 LINE GRAPH
##plt.plot(df['price'].head(100))
##plt.title("Price Trend")
##plt.xlabel("Index")
##plt.ylabel("Price")
##plt.show()
##
### 8 STACKED BAR CHART
####grouped = df.groupby('fuel_type')[['price','mileage']].mean()
####grouped.plot(kind='bar', stacked=True)
####plt.title("Fuel Type vs Price & Mileage")
####plt.show()
##
##
##stacked = df.groupby('fuel_type')[['price','mileage']].mean()
##
##stacked.plot(kind='bar', stacked=True, figsize=(8,5))
##
##plt.title("Fuel Type vs Price & Mileage (Stacked)")
##plt.xlabel("Fuel Type")
##plt.ylabel("Values")
##plt.xticks(rotation=45)
##
##plt.show()
##
### ---------------- ADVANCED VISUALIZATION ----------------
##
### Boxplot with category
##sns.boxplot(x='fuel_type', y='price', data=df)
##plt.title("Fuel Type vs Price")
##plt.show()
##
### Countplot
##sns.countplot(x='transmission', data=df)
##plt.title("Transmission Count")
##plt.show()
##
### ---------------- FEATURE ENGINEERING ----------------
##
### Convert categorical to numeric
##df_encoded = pd.get_dummies(df, drop_first=True)
##
### ---------------- MODEL BUILDING ----------------
##from sklearn.model_selection import train_test_split
##from sklearn.linear_model import LinearRegression
##from sklearn.metrics import mean_absolute_error, r2_score
##
### Target variable
##y = df_encoded['price']
##X = df_encoded.drop('price', axis=1)
##
### Split
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
##
### Train model
##model = LinearRegression()
##model.fit(X_train, y_train)
##
### Prediction
##y_pred = model.predict(X_test)
##
### Evaluation
##print("\nModel Performance:")
##print("MAE:", mean_absolute_error(y_test, y_pred))
##print("R2 Score:", r2_score(y_test, y_pred))
##
###
##from scipy.stats import ttest_ind
##
### Separate groups
##petrol = df[df['fuel_type'] == 'Petrol']['price']
##diesel = df[df['fuel_type'] == 'Diesel']['price']
##
### Perform t-test
##t_stat, p_value = ttest_ind(petrol, diesel)
##
##print("T-Statistic:", t_stat)
##print("P-Value:", p_value)
##
### Interpretation
##if p_value < 0.05:
##    print("Significant difference in prices between Petrol and Diesel cars")
##else:
##    print("No significant difference")
##
### ---------------- INNOVATION ----------------
##
##def price_category(p):
##    if p > df['price'].mean():
##        return "High"
##    else:
##        return "Low"
##
##df['Price_Category'] = df['price'].apply(price_category)
##
##print(df[['price','Price_Category']].head())
##
##print("\n✅ PROJECT COMPLETED SUCCESSFULLY")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD DATA ----------------
df = pd.read_csv("car_dataset_2000_rows.csv")

print(df.head())
print("\nColumns:", df.columns)
print("Total Rows:", len(df))

# ---------------- DATA CLEANING ----------------

# Fix column names
df.columns = df.columns.str.strip()

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Remove missing values safely
df = df.dropna()

# Remove duplicates
df.drop_duplicates(inplace=True)

# ---------------- EDA ----------------
print("\nStatistical Summary:\n", df.describe())

# ---------------- VISUALIZATIONS ----------------

# 1. BAR CHART (Average Price by Fuel Type)
fuel_price = df.groupby('fuel_type')['price'].mean().sort_values()

plt.figure(figsize=(8,5))
fuel_price.plot(kind='bar')
plt.title("Average Car Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 2. HISTOGRAM
plt.hist(df['price'], bins=30)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# 3. SCATTER PLOT (Mileage vs Price)
plt.figure(figsize=(8,5))
sns.scatterplot(x='mileage', y='price', hue='fuel_type', data=df)
sns.regplot(x='mileage', y='price', data=df, scatter=False)
plt.title("Mileage vs Price")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.show()

# 4. BOXPLOT (Top 10 Brands)
top10 = df['brand'].value_counts().head(10).index

plt.figure(figsize=(12,6))
sns.boxplot(x='brand', y='price', data=df[df['brand'].isin(top10)])
plt.xticks(rotation=45)
plt.title("Price Distribution by Brand")
plt.show()

# 5. HEATMAP
numeric_df = df[['year','engine_size','mileage','price']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 6. PIE CHART
fuel_counts = df['fuel_type'].value_counts()
plt.pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%')
plt.title("Fuel Type Distribution")
plt.show()

# 7. LINE GRAPH
plt.plot(df['price'].head(100))
plt.title("Price Trend")
plt.xlabel("Index")
plt.ylabel("Price")
plt.show()

# ---------------- SIMPLE LINEAR REGRESSION ----------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ONLY ONE FEATURE
X = df[['mileage']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nSimple Linear Regression Results:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Equation
print("Intercept:", model.intercept_)
print("Slope (Mileage):", model.coef_[0])

# ---------------- T-TEST ----------------

from scipy.stats import ttest_ind

# Groups
petrol = df[df['fuel_type'] == 'Petrol']['price']
diesel = df[df['fuel_type'] == 'Diesel']['price']

# T-test
t_stat, p_value = ttest_ind(petrol, diesel)

print("\nT-Test Results:")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Significant difference in prices between Petrol and Diesel cars")
else:
    print("No significant difference in prices")

# ---------------- INNOVATION ----------------

def price_category(p):
    if p < df['price'].quantile(0.33):
        return "Low"
    elif p < df['price'].quantile(0.66):
        return "Medium"
    else:
        return "High"

df['Price_Category'] = df['price'].apply(price_category)

print("\nPrice Categories Sample:")
print(df[['price','Price_Category']].head())




































