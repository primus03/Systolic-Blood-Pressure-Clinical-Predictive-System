import numpy as np
import pandas as pd

# Load the PRO.csv dataset
df = pd.read_csv('PRO.csv')

print("Original Dataset:")
print(df.head(10))

# 1. Remove duplicate Patient IDs
df = df.drop_duplicates(subset=['patient_id'], keep='first')

# 2. Clean patient_id column
df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')

# 3. Clean Age column (Keep age between 18 and 100)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df.loc[(df['age'] < 18) | (df['age'] > 100), 'age'] = np.nan

median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age).astype(int)

# 4. Clean BMI column (Remove unrealistic values)
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df.loc[(df['bmi'] < 10) | (df['bmi'] > 60), 'bmi'] = np.nan

median_bmi = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(median_bmi)

# 5. Clean Daily Sodium Intake
df['daily_sodium_g'] = pd.to_numeric(
    df['daily_sodium_g'], errors='coerce'
)

df.loc[
    (df['daily_sodium_g'] < 0) |
    (df['daily_sodium_g'] > 15),
    'daily_sodium_g'
] = np.nan

median_sodium = df['daily_sodium_g'].median()
df['daily_sodium_g'] = df['daily_sodium_g'].fillna(median_sodium)

# 6. Clean Resting Heart Rate
df['resting_heart_rate'] = pd.to_numeric(
    df['resting_heart_rate'], errors='coerce'
)

df.loc[
    (df['resting_heart_rate'] < 30) |
    (df['resting_heart_rate'] > 220),
    'resting_heart_rate'
] = np.nan

median_heart_rate = df['resting_heart_rate'].median()
df['resting_heart_rate'] = df['resting_heart_rate'].fillna(
    median_heart_rate
).astype(int)

# 7. Clean Family History of Hypertension
df['family_history_hypertension'] = pd.to_numeric(
    df['family_history_hypertension'], errors='coerce'
)

df.loc[
    ~df['family_history_hypertension'].isin([0, 1]),
    'family_history_hypertension'
] = np.nan

# Fill missing family history values with mode
mode_family_history = df['family_history_hypertension'].mode()[0]

df['family_history_hypertension'] = (
    df['family_history_hypertension']
    .fillna(mode_family_history)
    .astype(int)
)

# 8. Clean Systolic Blood Pressure
df['systolic_bp'] = pd.to_numeric(
    df['systolic_bp'], errors='coerce'
)

df.loc[
    (df['systolic_bp'] < 60) |
    (df['systolic_bp'] > 250),
    'systolic_bp'
] = np.nan

median_bp = df['systolic_bp'].median()
df['systolic_bp'] = df['systolic_bp'].fillna(median_bp).astype(int)

# 9. Export cleaned dataset
df.to_csv('cleaned_patient.csv', index=False)

print("\nCleaned Dataset:")
print(df.head(10))

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset saved as cleaned_patient.csv")