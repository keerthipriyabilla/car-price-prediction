# ================== IMPORTS ==================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import re
import joblib

# ================== LOAD DATA ==================
df = pd.read_csv("car_details.csv")

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Check column names
print(df.columns.tolist())

# Drop unwanted columns
df.drop(["Fuel","RTO","Transmission .1","other_features","Engine"], axis=1, inplace=True)

# ================== REGISTRATION YEAR ==================
df["Registration Year"] = pd.to_numeric(df["Registration Year"], errors="coerce")
df["Registration Year"].fillna(df["Registration Year"].median(), inplace=True)
sns.boxplot(df["Registration Year"])

# Outlier handling
Q1 = df["Registration Year"].quantile(0.25)
Q3 = df["Registration Year"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df["Registration Year"] = df["Registration Year"].clip(lower, upper)
sns.boxplot(df["Registration Year"])

# ================== YEAR OF MANUFACTURE ==================
df["Year of Manufacture"] = pd.to_numeric(df["Year of Manufacture"], errors="coerce")
df["Year of Manufacture"].fillna(df["Year of Manufacture"].median(), inplace=True)
sns.boxplot(df["Year of Manufacture"])

Q1 = df["Year of Manufacture"].quantile(0.25)
Q3 = df["Year of Manufacture"].quantile(0.75)
IQR = Q3 - Q1
df["Year of Manufacture"] = df["Year of Manufacture"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
sns.boxplot(df["Year of Manufacture"])

# ================== SEATS CLEANING ==================
def parse_seats(x):
    if pd.isnull(x):
        return np.nan
    s = str(x)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return np.nan

df["Seats"] = df["Seats"].apply(parse_seats)
df["Seats"].fillna(df["Seats"].median(), inplace=True)
sns.boxplot(df["Seats"])
df["Seats"] = df["Seats"].clip(lower=3, upper=8)
sns.boxplot(df["Seats"])

# ================== KMS DRIVEN ==================
def clean_numeric(x):
    if pd.isnull(x): return np.nan
    x = str(x).replace(",", "").replace("Kms", "").strip()
    try: return float(x)
    except: return np.nan

df["Kms Driven"] = df["Kms Driven"].apply(clean_numeric)
df["Kms Driven"].fillna(df["Kms Driven"].median(), inplace=True)
sns.boxplot(df["Kms Driven"])

Q1, Q3 = df["Kms Driven"].quantile(0.25), df["Kms Driven"].quantile(0.75)
IQR = Q3 - Q1
df["Kms Driven"] = df["Kms Driven"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
sns.boxplot(df["Kms Driven"])

# ================== ENGINE DISPLACEMENT ==================
df["Engine Displacement"] = df["Engine Displacement"].astype(str).str.replace("cc", "").str.strip()
df["Engine Displacement"] = pd.to_numeric(df["Engine Displacement"], errors="coerce")
df["Engine Displacement"].fillna(df["Engine Displacement"].median(), inplace=True)
sns.boxplot(df["Engine Displacement"])

Q1, Q3 = df["Engine Displacement"].quantile(0.25), df["Engine Displacement"].quantile(0.75)
IQR = Q3 - Q1
df["Engine Displacement"] = df["Engine Displacement"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
sns.boxplot(df["Engine Displacement"])

# ================== POWER ==================
df["Power"] = df["Power"].astype(str).str.replace("bhp", "").str.strip()
df["Power"] = pd.to_numeric(df["Power"], errors="coerce")
df["Power"].fillna(df["Power"].median(), inplace=True)
sns.boxplot(df["Power"])

Q1, Q3 = df["Power"].quantile(0.25), df["Power"].quantile(0.75)
IQR = Q3 - Q1
df["Power"] = df["Power"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
sns.boxplot(df["Power"])

# ================== MILEAGE ==================
df["Mileage"] = df["Mileage"].astype(str).str.replace("kmpl", "").str.strip()
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Mileage"].fillna(df["Mileage"].median(), inplace=True)
sns.boxplot(df["Mileage"])

Q1, Q3 = df["Mileage"].quantile(0.25), df["Mileage"].quantile(0.75)
IQR = Q3 - Q1
df["Mileage"] = df["Mileage"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
sns.boxplot(df["Mileage"])

# ================== PRICE CLEANING ==================
def simple_price_parser(x):
    if pd.isnull(x):
        return np.nan
    s = str(x).replace(",", "").lower().strip()

    if "cr" in s or "crore" in s:
        m = re.search(r"([\d\.]+)", s)
        if m: return float(m.group(1)) * 1e7

    if "lakh" in s or "lac" in s:
        m = re.search(r"([\d\.]+)", s)
        if m: return float(m.group(1)) * 1e5

    if "thousand" in s or "k" in s:
        m = re.search(r"([\d\.]+)", s)
        if m: return float(m.group(1)) * 1e3

    m = re.search(r"([\d\.]+)", s)
    if m: return float(m.group(1))
    return np.nan

df["new_vehical_price"] = df["new_vehical_price"].apply(simple_price_parser)
df["new_vehical_price"].fillna(df["new_vehical_price"].median(), inplace=True)

df["vehical_price"] = df["vehical_price"].apply(simple_price_parser)
df["vehical_price"].fillna(df["vehical_price"].median(), inplace=True)

# ================== OWNERSHIP ==================
df['Ownership'] = df['Ownership'].fillna(method='bfill')

# ================== DRIVE TYPE ==================
def clean_drive_type(x):
    if pd.isnull(x):
        return "Unknown"
    x = str(x).strip().upper()

    if x in ["FWD"]:
        return "FWD"
    elif x.startswith("RWD"):
        return "RWD"
    elif x in ["4X4", "4WD", "FOUR WHELL DRIVE"]:
        return "4WD"
    elif x in ["2WD", "2 WD", "4X2"]:
        return "2WD"
    elif "AWD" in x or "ALL WHEEL DRIVE" in x or "PERMANENT AWD" in x:
        return "AWD"
    else:
        return "Unknown"

df["Drive Type"] = df["Drive Type"].apply(clean_drive_type)

# ================== INSURANCE ==================
df["Insurance"] = df["Insurance"].astype(str)
df["Insurance"].replace("nan", np.nan, inplace=True)
df["Insurance"] = df["Insurance"].replace("-", "Unknown")
df["Insurance"].fillna(df["Insurance"].mode()[0], inplace=True)

# ================== CORRELATION HEATMAP ==================
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap of Car Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ================== MODEL TRAINING ==================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

x = df[["Registration Year","Year of Manufacture","Seats","Kms Driven","Engine Displacement","Power","Mileage","new_vehical_price"]]
y = df[["vehical_price"]]
z = df[["Ownership","Fuel Type","Transmission","Drive Type","vehical_name","Insurance"]]

# Scale numeric
numeric_cols = x.columns.tolist()
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x[numeric_cols])
x_scaled_df = pd.DataFrame(x_scaled, columns=numeric_cols)
joblib.dump(scaler_x, "scaler_x.pkl")

# Encode categorical
encoders = {}
z_encoded = z.copy()
for col in z.columns:
    le = LabelEncoder()
    z_encoded[col] = le.fit_transform(z[col])
    encoders[col] = le

joblib.dump(encoders, "encoder.pkl")

X = pd.concat([x_scaled_df, z_encoded], axis=1)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
joblib.dump(scaler_y, "scaler_y.pkl")

x_train, x_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# ================== MODEL SELECTION ==================
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

models_params = {
  "RandomForest": (RandomForestRegressor(random_state=42), {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [5, 10]
    })
}

results = []

for name, (model, params) in models_params.items():
    print(f"\nTraining {name}...")

    grid = GridSearchCV(model, params, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    y_train_pred = best_model.predict(x_train)
    y_test_pred = best_model.predict(x_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    results.append({
        "Model": name,
        "Best Params": best_params,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Test RMSE": rmse
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.sort_values(by="Test R2", ascending=False))

# ================== FINAL MODEL ==================
preferred_model_name = "RandomForest"
best_row = results_df[results_df["Model"] == preferred_model_name].iloc[0]
print(f"\nSelected Model: {preferred_model_name}")
print(best_row)

model, params = models_params[preferred_model_name]
grid = GridSearchCV(model, params, cv=3, scoring="r2", n_jobs=-1)
grid.fit(x_train, y_train)
final_model = grid.best_estimator_

joblib.dump(final_model, f"{preferred_model_name}_model.pkl")
print(f"{preferred_model_name} model saved successfully ✅")

