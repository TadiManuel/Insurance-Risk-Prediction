# Insurance Risk Claim Prediction

## 📌 Project Overview
This project involves a machine learning prediction analysis of medical insurance claims. The goal is to build a predictive model that accurately estimates individual medical charges based on demographic and lifestyle attributes.

The dataset includes individual-level information such as **age, sex, BMI, number of children, smoking status, and residential region**. Given the continuous nature of the target variable (`charges`), **Linear Regression** was selected as the primary modelling technique to ensure interpretability and predictive power.

## 📊 Key Findings & Results
* **Model Performance:** The final model achieved an **$R^2$ score of 0.77**, meaning that **77%** of the variability in medical insurance charges is explained by the model's features.
* **Top Predictor:** **Smoking status** was found to have the most substantial impact on insurance charges, followed by **Age** and **BMI**.
* **Model Reliability:** Diagnostic tests confirmed that the model satisfies key assumptions:
    * **Linearity:** Confirmed via residual analysis.
    * **Homoscedasticity:** Variance of residuals is constant.
    * **No Multicollinearity:** Validated to ensure independent features.

## 🛠️ Methodology
The analysis followed a structured data science workflow:

1.  **Exploratory Data Analysis (EDA):**
    * Analyzed data structure to check for missing values and outliers.
    * Visualized distributions and relationships using correlation heatmaps and boxplots.
2.  **Feature Engineering & Selection:**
    * Utilized **Backward Elimination** based on p-values to remove statistically insignificant features.
    * Applied domain knowledge to refine the feature set.
3.  **Preprocessing:**
    * Converted categorical variables (Sex, Smoker, Region) into numeric formats using **One-Hot Encoding**.
4.  **Model Building:**
    * Developed an **Ordinary Least Squares (OLS)** regression model using the `statsmodels` library.
5.  **Evaluation:**
    * Assessed performance using **RMSE** (Root Mean Squared Error) and **$R^2$**.
    * Conducted residual analysis to verify normality and model fit.

## 📂 Project Structure
```text
Medical-Insurance-Prediction/
├── data/               # Raw dataset (insurance.csv)
├── notebooks/          # Jupyter Notebook with cleaning, EDA, and modelling code
├── images/             # Generated plots (Heatmaps, Residuals, Boxplots)
├── reports/            # Full analysis report (PDF)
└── README.md           # Project documentation