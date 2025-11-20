# trend_forecaster.py ──────────────────────────────────────────────────────────
# SMARTSTOCK Realistic Trend Forecasting Engine (Final Version) using pickle only

import os
import pickle
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Prevent GUI backend issues
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = "Customer Experience Dataset.csv"
MODEL_PATH = "models/trend_model.pkl"
CHART_PATH = "static/trend_chart.png"
ENCODING = "cp1252"

FEATURES = {
    "shopping_frequency": "How often do you shop at supermarkets?",
    "product_categories": "Which product categories do you frequently buy? (Select all that apply)",
    "price_comparison": "Do you use any price comparison or shopping apps before buying?",
    "shop_offer": "Do you shop more frequently during offers?",
    "purchase_influence": "What influences your purchase the most?",
    "shopping_time": "When do you usually shop?"
}

LABELS = {
    "shopping_frequency": "Shopping Frequency",
    "product_categories": "Product Categories",
    "price_comparison": "Price Comparison Apps",
    "shop_offer": "Offer Sensitivity",
    "purchase_influence": "Shopping Influence",
    "shopping_time": "Shopping Time"
}

GAP_SCORES = {
    "On-Trend": 15,
    "Partial-Trend": 10,
    "Ahead of Trend": 20,
    "Lagging Trend": 7,
    "Off-Trend": 0
}

INTERPRETATIONS = {
    "On-Trend": "Aligns closely with typical consumer behavior.",
    "Partial-Trend": "Some overlap with market behavior — you're moderately aligned.",
    "Ahead of Trend": "You show progressive shopping behavior. Future-focused!",
    "Lagging Trend": "Your habits lag behind modern preferences.",
    "Off-Trend": "Your habits diverge significantly from the average shopper."
}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def find_column_ignore_case(target_col, all_columns):
    target = target_col.strip().lower()
    for col in all_columns:
        if col.strip().lower() == target:
            return col
    return None

def _gap_classify(user, market):
    user = str(user).lower().strip()
    market = str(market).lower().strip()
    if user == market:
        return "On-Trend"
    elif any(w in user for w in ["early", "innovative", "organic", "smart", "new"]):
        return "Ahead of Trend"
    elif any(w in user for w in ["late", "once", "rare", "never", "no", "not"]):
        return "Lagging Trend"
    elif set(user.split(", ")).intersection(set(market.split(", "))):
        return "Partial-Trend"
    else:
        return "Off-Trend"

# ─── Training Function ─────────────────────────────────────────────────────────
def train_trend_model():
    df = pd.read_csv(DATA_PATH, encoding=ENCODING)
    actual_cols = {k: find_column_ignore_case(v, df.columns) for k, v in FEATURES.items()}
    if None in actual_cols.values():
        raise ValueError("❌ One or more required columns are missing in the CSV.")

    df[actual_cols["product_categories"]] = df[actual_cols["product_categories"]].fillna("").apply(
        lambda x: ", ".join(sorted([i.strip() for i in str(x).split(",") if i.strip()]))
    )
    df.dropna(subset=list(actual_cols.values()), inplace=True)

    def segment_logic(row):
        offer = row[actual_cols["shop_offer"]].lower().strip()
        influence = row[actual_cols["purchase_influence"]].lower().strip()
        if offer == "yes" and "price" in influence:
            return "Deal-Seeker"
        elif "quality" in influence or "advertisement" in influence:
            return "Quality-Conscious"
        elif "brand" in influence:
            return "Brand-Driven"
        elif offer == "no" and "reviews" in influence:
            return "Neutral"
        else:
            return "Offer-Hunter"

    df["Trend Segment"] = df.apply(segment_logic, axis=1)

    X = df[list(actual_cols.values())]
    y = df["Trend Segment"]

    encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)])
    model = Pipeline([("encoder", encoder), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Trend Model Trained and Saved at", MODEL_PATH)

# ─── Prediction & Analysis Function ────────────────────────────────────────────
def predict_trend_segment(user_input: dict):
    df = pd.read_csv(DATA_PATH, encoding=ENCODING)
    actual_cols = {k: find_column_ignore_case(v, df.columns) for k, v in FEATURES.items()}

    df[actual_cols["product_categories"]] = df[actual_cols["product_categories"]].fillna("").apply(
        lambda x: ", ".join(sorted([i.strip() for i in str(x).split(",") if i.strip()]))
    )
    df.dropna(subset=list(actual_cols.values()), inplace=True)

    comparison_rows = []
    user_score = 0
    detailed_reasons = []

    for key in FEATURES:
        col = actual_cols[key]
        market_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
        user_val = user_input.get(key, "")
        if isinstance(user_val, list):
            user_val = ", ".join(sorted([i.strip() for i in user_val]))
        gap = _gap_classify(user_val, market_val)
        score = GAP_SCORES.get(gap, 0)
        comparison_rows.append({
            "feature": LABELS[key],
            "you": user_val or "—",
            "market": market_val or "—",
            "gap": gap,
            "score": score,
            "interpretation": INTERPRETATIONS.get(gap, "Unclassified")
        })
        user_score += score

        if gap in ["Off-Trend", "Lagging Trend"] and len(detailed_reasons) < 3:
            detailed_reasons.append(f"{LABELS[key]} — '{user_val}' vs Market: '{market_val}'\n{INTERPRETATIONS.get(gap)}")

    predicted_segment = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        input_df = pd.DataFrame([{
            actual_cols[k]: ", ".join(user_input[k]) if isinstance(user_input[k], list) else user_input[k]
            for k in FEATURES
        }])
        predicted_segment = model.predict(input_df)[0]

    # ─── Plot ────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    sns.set(style="whitegrid", palette="pastel", font_scale=1.0)
    labels = [row["feature"] for row in comparison_rows]
    user_scores = [row["score"] for row in comparison_rows]

    market_scores = []
    for key in FEATURES:
        col = actual_cols[key]
        mode_val = df[col].mode().iloc[0]
        freq = df[col].value_counts(normalize=True).get(mode_val, 0)
        score = round(freq * 15)
        market_scores.append(score)

    x = np.arange(len(labels))
    plt.plot(x, user_scores, label="You", marker='o', linewidth=3, color="#74a5ee")
    plt.plot(x, market_scores, label="Market Avg", marker='D', linestyle='--', linewidth=2, color='#999999')

    plt.xticks(x, labels, rotation=20, ha="right", fontsize=10)
    plt.ylabel("Trend Match Score", fontsize=11)
    plt.title("Your Shopping Profile vs Market Trends", fontsize=13, fontweight='bold')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
    plt.savefig(CHART_PATH, dpi=100)
    plt.close()

    # ─── Insights ───────────────────────────────────────────────────────────
    trend_score = f"{user_score} / 100"
    if user_score >= 85:
        summary = "You're deeply in tune with prevailing trends."
        advice = "Maintain your proactive shopping behavior to stay ahead."
    elif user_score >= 70:
        summary = "You're mostly aligned with market trends."
        advice = "You're well-aligned! Explore new categories to stay updated."
    else:
        summary = "You are missing some key trends."
        advice = "Try adapting to trending preferences and offers for better deals."

    insights = [
        f"Predicted Segment: {predicted_segment}",
        f"Your Trend Score: {trend_score}",
        f"Diagnosis: {summary}",
        *detailed_reasons,
        f"Advice: {advice}"
    ]

    return predicted_segment, trend_score, advice, CHART_PATH, comparison_rows, insights