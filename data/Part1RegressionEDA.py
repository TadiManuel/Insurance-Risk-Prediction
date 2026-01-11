import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("insurance.csv")

# EDA
print(df.describe())
print(df.isnull().sum())  # Check missing values

# Check dimensions
print("\nDataset shape:", df.shape)

# Data types and missing values
print("\nData types and missing values:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe(include='all'))

# Boxplots for numerical features
numerical_cols = ['age', 'bmi', 'children', 'charges']
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, y=col)
plt.tight_layout()
plt.show()
plt.savefig('BoxPlotNumerical_EDA1.png', format='png', dpi=300)


# Histograms/KDE plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Check skewness of 'charges'
print("\nSkewness of charges:", df['charges'].skew())
if df['charges'].skew() > 1:
    print("Log transformation may be needed for 'charges'.")
    

# Count plots for categorical features
categorical_cols = ['sex', 'smoker', 'region']
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=df, x=col)
    plt.title(f'Count of {col}')
plt.tight_layout()
plt.show()

# Charges by categorical features
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df, x=col, y='charges')
    plt.title(f'Charges by {col}')
plt.tight_layout()
plt.show()


# Correlation heatmap (numerical features)
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Scatter plots for key relationships
sns.pairplot(df[['age', 'bmi', 'charges', 'smoker']], hue='smoker')
plt.show()

# Print key observations
print("\nKey EDA Observations:")
print("- Missing values: None (complete dataset)")
print("- Outliers: Present in 'bmi' and 'charges' (right-skewed)")
print("- Skewness: 'charges' is highly right-skewed (skewness = {:.2f})".format(df['charges'].skew()))
print("- Smokers have significantly higher charges (visible in boxplots)")
print("- Weak correlation between 'age' and 'charges' (r = {:.2f})".format(corr_matrix.loc['age', 'charges']))





#Feature Selection: 
#1.) Selecting best features and training the data


from sklearn.model_selection import train_test_split

# Load data (no dropping)
dfModel = pd.read_csv("insurance.csv")

dfModel['sex'] = dfModel['sex'].map({'female': 0, 'male': 1})
dfModel['smoker'] = dfModel['smoker'].map({'no': 0, 'yes': 1})

# Multi-category ('region') → label encode as integers
#df['region'] = df['region'].astype('category').cat.codes

print(dfModel.head())

# One-hot encode categorical variables
df_encoded = pd.get_dummies(dfModel, columns=['region'], drop_first=False)
df_encoded = df_encoded.astype(int)
print(df_encoded.head())
print(df_encoded.dtypes)

# Define features (X) and target (y)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']


# Split the data into 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Selecting the Best Features


# Add intercept term (constant)
X_train_sm = sm.add_constant(X_train)

# Fit OLS model
model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())

import seaborn as sns
import matplotlib.pyplot as plt

# Get correlation matrix
corr = df_encoded.corr()

# Plot heatmap for correlation with 'charges'
plt.figure(figsize=(10, 6))
sns.heatmap(corr[['charges']].sort_values(by='charges', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation of features with 'charges'")
plt.show()


#Backward Elimination - Feature Selection
# Add constant (intercept) to model
X_with_const = sm.add_constant(X)

# Initial full model
model = sm.OLS(y, X_with_const).fit()

# Loop for backward elimination
while True:
    p_values = model.pvalues.drop("const")  # exclude intercept
    max_pval = p_values.max()
    if max_pval > 0.05:
        worst_feature = p_values.idxmax()
        print(f"Removing '{worst_feature}' with p-value = {max_pval:.4f}")
        X_with_const = X_with_const.drop(columns=[worst_feature])
        model = sm.OLS(y, X_with_const).fit()
    else:
        break

print(model.summary())






#Training model on The Chosen HyperParameters:
# 'age', 'bmi', 'children', 'smoker',
#'region_northeast', 'region_northwest', 'region_southeast',
#'region_southwest'




from sklearn.metrics import mean_squared_error, r2_score

# 1. Define the final set of features (excluding 'sex')
selected_features = ['age', 'bmi', 'children', 'smoker', 
                     'region_northeast', 'region_northwest', 
                     'region_southeast', 'region_southwest']

# 2. Add constant to test data
X_train_selected = sm.add_constant(X_train[selected_features])
X_test_selected = sm.add_constant(X_test[selected_features])

# 3. Fit OLS model on training data
ols_model = sm.OLS(y_train, X_train_selected).fit()

print(ols_model.summary())


# 4. Predict on test data
y_pred = ols_model.predict(X_test_selected)





#Evaluating The Model----------------------------------------------

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R-squared: {r2:.4f}")



# Residuals
residuals = y_test - y_pred

# Histogram of residuals
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()



# Q-Q plot (check normality)
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# Residuals vs Predicted
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.show()


from statsmodels.stats.diagnostic import het_breuschpagan

# Test for heteroskedasticity
bp_test = het_breuschpagan(residuals, X_test_selected)
labels = ['LM Stat', 'LM p-value', 'F Stat', 'F p-value']

print(dict(zip(labels, bp_test)))


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_selected.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_selected.values, i) 
                   for i in range(X_train_selected.shape[1])]

print(vif_data)



