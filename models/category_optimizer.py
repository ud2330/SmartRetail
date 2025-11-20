import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_PATH = "models/category_trend_model.pkl"
DATA_PATH = "data/survey.csv"

def train_category_model():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH, encoding='latin1')
    except Exception as e:
        print("❌ Failed to load dataset:", str(e))
        return

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    column_map = {
        'which_product_categories_do_you_frequently_buy_select_all_that_apply': 'frequent_categories',
        'which_categories_do_you_prefer_to_buy_during_offers_select_all_that_apply': 'offer_categories',
        'which_offer_types_do_you_prefer__select_all_that_apply': 'offer_types',
        'do_you_consider_ecofriendliness_when_choosing_a_product': 'eco_friendly',
        'do_you_use_any_price_comparison_or_shopping_apps_before_buying': 'price_comparison'
    }

    missing_cols = [col for col in column_map if col not in df.columns]
    if missing_cols:
        print(f"❌ Required columns not found: {missing_cols}")
        return

    df.rename(columns=column_map, inplace=True)
    print("✅ Columns cleaned and renamed successfully.")

    def parse_input(val):
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        elif isinstance(val, list):
            return val
        return []

    df.dropna(subset=list(column_map.values()), inplace=True)
    df['frequent_categories'] = df['frequent_categories'].apply(parse_input)
    df['offer_categories'] = df['offer_categories'].apply(parse_input)
    df['offer_types'] = df['offer_types'].apply(parse_input)

    mlb_freq = MultiLabelBinarizer()
    mlb_offer = MultiLabelBinarizer()
    mlb_type = MultiLabelBinarizer()

    X_freq = mlb_freq.fit_transform(df['frequent_categories'])
    X_offer = mlb_offer.fit_transform(df['offer_categories'])
    X_type = mlb_type.fit_transform(df['offer_types'])

    df['eco_friendly'] = df['eco_friendly'].str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0.5)

    import numpy as np
    X_final = pd.concat([
        pd.DataFrame(X_freq, columns=["freq_" + c for c in mlb_freq.classes_]),
        pd.DataFrame(X_offer, columns=["offer_" + c for c in mlb_offer.classes_]),
        pd.DataFrame(X_type, columns=["type_" + c for c in mlb_type.classes_]),
        df['eco_friendly'].reset_index(drop=True)
    ], axis=1)

    y = df['frequent_categories'].apply(lambda cats: int('snacks' in [c.lower() for c in cats]))

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": clf,
        "mlb_freq": mlb_freq,
        "mlb_offer": mlb_offer,
        "mlb_type": mlb_type,
        "features": list(X_final.columns),
        "target": "frequent_snack_buyer"
    }

    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"✅ Model and components saved to: {MODEL_PATH}")
    except Exception as e:
        print("❌ Failed to save model:", str(e))


def generate_category_insights(user_inputs):
    insights = []

    if user_inputs.get("frequent_categories"):
        frequent = user_inputs["frequent_categories"]
        insights.append(f"You frequently purchase: {', '.join(frequent)}.")
        if "Snacks" in frequent or "Beverages" in frequent:
            insights.append("To save on frequent items like Snacks or Beverages, consider bulk buying or subscriptions.")
        if "Personal Care" in frequent:
            insights.append("Combo packs or loyalty programs could help you save in Personal Care.")

    if user_inputs.get("offer_categories"):
        offer_cats = user_inputs["offer_categories"]
        insights.append(f"You look for offers in: {', '.join(offer_cats)}.")
        if len(offer_cats) >= 3:
            insights.append("You explore many deal categories—try smart carts to combine offers and save more.")

    offer_types = user_inputs.get("offer_types", [])
    if offer_types:
        insights.append(f"Preferred offer types: {', '.join(offer_types)}.")
        if "Buy 1 Get 1 Free" in offer_types:
            insights.append("BOGO offers are great for Snacks or high-use products—keep an eye out.")

    if user_inputs.get("price_comparison") == "Yes":
        insights.append("You compare prices already—enable price alerts for favorite items.")
    else:
        insights.append("Try using price comparison apps to save more on recurring purchases.")

    if user_inputs.get("eco_friendly") == "Yes":
        insights.append("You care about sustainability—try eco-friendly or refillable options.")
    else:
        insights.append("Explore greener options like eco-labeled or refillable products.")

    insights.append("Review your top 3 product categories monthly to improve shopping strategy.")

    return insights
