import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Paths
CSV_FILE = "Customer Experience Dataset.csv"
PKL_FILE = "models/test_model.pkl"

# Ensure dataset exists
if not os.path.exists(CSV_FILE):
    print(f"❌ Dataset not found at: {CSV_FILE}")
    exit(1)

# Load dataset
df = pd.read_csv(CSV_FILE, encoding="latin1")
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)

# Rename columns
col_map = {
    'which_product_categories_do_you_frequently_buy_select_all_that_apply': 'frequent_categories',
    'which_categories_do_you_prefer_to_buy_during_offers_select_all_that_apply': 'offer_categories',
    'which_offer_types_do_you_prefer__select_all_that_apply': 'offer_types',
    'do_you_consider_ecofriendliness_when_choosing_a_product': 'eco_friendly',
    'do_you_use_any_price_comparison_or_shopping_apps_before_buying': 'price_comparison'
}
missing_cols = [col for col in col_map if col not in df.columns]
if missing_cols:
    print("❌ Missing columns:", missing_cols)
    exit(1)
df.rename(columns=col_map, inplace=True)

# Parse multi-select inputs
def parse(val):
    return [v.strip().lower() for v in str(val).split(",") if v.strip()]

df.dropna(subset=['frequent_categories', 'offer_categories', 'offer_types', 'eco_friendly'], inplace=True)
df['frequent_categories'] = df['frequent_categories'].apply(parse)
df['offer_categories'] = df['offer_categories'].apply(parse)
df['offer_types'] = df['offer_types'].apply(parse)

# Vectorize inputs
mlb1 = MultiLabelBinarizer()
mlb2 = MultiLabelBinarizer()
mlb3 = MultiLabelBinarizer()
X1 = mlb1.fit_transform(df['frequent_categories'])
X2 = mlb2.fit_transform(df['offer_categories'])
X3 = mlb3.fit_transform(df['offer_types'])
eco_map = {'yes': 1, 'no': 0}
eco = df['eco_friendly'].str.lower().map(eco_map).fillna(0.5)

# Final features
import numpy as np
X = np.concatenate([X1, X2, X3, eco.values.reshape(-1, 1)], axis=1)
y = df['frequent_categories'].apply(lambda x: 1 if 'snacks' in x else 0)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open(PKL_FILE, "wb") as f:
    pickle.dump({"model": model}, f)

print(f"✅ Model saved to {PKL_FILE}")
