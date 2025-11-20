from sklearn.calibration import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify # type: ignore
import pandas as pd
import pickle
from models.upload_model import RetailAnalyticsModel
from pymongo import MongoClient # type: ignore
import numpy as np
import os # Import os module for path operations
from models.trend_forecaster import train_trend_model, predict_trend_segment
from models.category_optimizer import train_category_model, generate_category_insights



app = Flask(__name__)
app.secret_key = 'passwords' # IMPORTANT: Change this to a strong, random key in production!

# MongoDB connection
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["smartretail"]
    users_collection = db["users"]
    print("MongoDB connected successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None
    db = None
    users_collection = None

analytics_model = RetailAnalyticsModel()

# Load ML models
stockout_model = None
segmentation_model = None
sales_model = None
offer_model = None
retention_model = None
trend_model = None

# MultiLabelBinarizer for offer recommendations (to inverse transform predictions)
offer_mlb = None

# Global variables to store the expected feature names *after* the ColumnTransformer (final features for classifier)
EXPECTED_STOCKOUT_CLASSIFIER_FEATURES = None
EXPECTED_SALES_REGRESSOR_FEATURES = None
EXPECTED_OFFER_MODEL_FEATURES = None # Global for offer model's expected features
SEGMENTATION_MODEL_INPUT_FEATURES_ORDER = None # NEW: To store the exact feature order from pipeline


# --- Segment Descriptions (from your provided cluster_info) ---
SEGMENT_DESCRIPTIONS = {
    0: {
        "description": "This segment consists of **Budget-Conscious Shoppers** who prioritize value and are highly influenced by offers. They are moderately likely to switch to online shopping if it provides better deals. Stock-outs are a concern, and they might seek alternatives or switch stores.",
        "characteristics": {
            "Spending Habits": "Moderate to Low Monthly Grocery Spend, High Offer Spend Increase",
            "Shopping Preferences": "Price-sensitive, utilize shopping apps, influenced by offers",
            "Stock-Out Reaction": "Might buy alternatives or switch stores, concerned about stock-outs",
            "Online Shopping Tendency": "Moderate to High likelihood to switch online"
        }
    },
    1: {
        "description": "This segment represents **Convenience-Driven Loyalists**. They spend moderately on groceries, value convenience, and are less concerned about offers. They are less likely to switch to online shopping and prefer their current store even if stock-outs occur, often waiting for restock or buying alternatives within the same store.",
        "characteristics": {
            "Spending Habits": "Moderate Monthly Grocery Spend, Low Offer Spend Increase",
            "Shopping Preferences": "Value convenience, less influenced by offers, might not use shopping apps",
            "Stock-Out Reaction": "Patient, wait for restock or buy alternatives within store",
            "Online Shopping Tendency": "Low likelihood to switch online"
        }
    },
    2: {
        "description": "This segment is made up of **Efficiency-Focused Shoppers**. They are efficient with their shopping time, have a moderate household size, and are somewhat influenced by offers. They may use shopping apps and are pragmatic about stock-outs, often buying alternatives. Their online shopping likelihood is moderate.",
        "characteristics": {
            "Spending Habits": "Moderate Monthly Grocery Spend, Moderate Offer Spend Increase",
            "Shopping Preferences": "Efficient shopping trips, may use shopping apps, moderately influenced by offers",
            "Stock-Out Reaction": "Pragmatic, often buy alternative products",
            "Online Shopping Tendency": "Moderate likelihood to switch online"
        }
    },
    3: {
        "description": "This segment are **High-Value Explorers**. They have higher grocery spending and are open to new products or categories, especially during offers. They are technologically inclined, using shopping apps, and are highly likely to switch to online shopping for better experiences. Stock-outs are a significant frustration, leading them to other stores or online.",
        "characteristics": {
            "Spending Habits": "High Monthly Grocery Spend, High Offer Spend Increase",
            "Shopping Preferences": "Exploratory, use shopping apps, highly influenced by offers",
            "Stock-Out Reaction": "High frustration, switch stores or order online",
            "Online Shopping Tendency": "High likelihood to switch online"
        }
    },
    4: {
        "description": "This segment consists of **Occasional Shoppers with Diverse Needs**. They might shop less frequently but are interested in a variety of categories. Their spending and offer influence can vary, and they may be exploring different shopping options.",
        "characteristics": {
            "Spending Habits": "Varied Monthly Grocery Spend",
            "Shopping Preferences": "Diverse category interests, moderate offer influence",
            "Stock-Out Reaction": "Flexible, might seek alternatives or adapt",
            "Online Shopping Tendency": "Moderate, open to trying new channels"
        }
    }
}

# Global variable for average monthly grocery spend
AVERAGE_MONTHLY_GROCERY_SPEND = 0.0

# Define all possible categories for multi-selects to ensure consistent column creation
# These lists MUST match the ones used in your training scripts (e.g., offer_recommendation_model.py)
ALL_FREQUENT_CATEGORIES_OPTIONS = ['Groceries', 'Snacks', 'Dairy', 'Beverages', 'Personal Care', 'Household Items', 'Fruits & Vegetables', 'Packaged Food', 'Other']
ALL_STOCKOUT_CATEGORIES_OPTIONS = ['Groceries', 'Snacks', 'Dairy', 'Beverages', 'Personal Care', 'Household Items', 'Fruits', 'Vegetables', 'Other']
ALL_OFFER_TYPES_PREFERRED_OPTIONS = ['Flat Discount', 'Buy 1 Get 1 Free', 'Cashback', 'Loyalty Points', 'Free Gifts']
ALL_CATEGORIES_BUY_OFFERS_OPTIONS = ['Snacks', 'Beverages', 'Personal Care', 'Household Items', 'Groceries', 'Fruits & Vegetables', 'Other']


try:
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found. Please ensure your models are placed there.")
        raise FileNotFoundError(f"Directory '{models_dir}' does not exist.")

    # Load dataset to calculate average monthly grocery spend
    try:
        df_full = pd.read_csv('Customer Experience Dataset.csv')
        column_mapping_full = {
            'How much do you typically spend on groceries per month? (Numeric input)': 'Monthly_Grocery_Spend'
        }
        df_full.rename(columns=column_mapping_full, inplace=True)
        df_full['Monthly_Grocery_Spend'] = pd.to_numeric(df_full['Monthly_Grocery_Spend'], errors='coerce')
        
        # Ensure AVERAGE_MONTHLY_GROCERY_SPEND is a valid number, converting NaN to 0.0
        AVERAGE_MONTHLY_GROCERY_SPEND = df_full['Monthly_Grocery_Spend'].mean()
        AVERAGE_MONTHLY_GROCERY_SPEND = np.nan_to_num(AVERAGE_MONTHLY_GROCERY_SPEND, nan=0.0)
        
        print(f"Calculated average monthly grocery spend from dataset: ₹{AVERAGE_MONTHLY_GROCERY_SPEND:.2f}")
    except FileNotFoundError:
        print("Error: Customer Experience Dataset.csv not found. Cannot calculate average spend.")
        # If file not found, AVERAGE_MONTHLY_GROCERY_SPEND remains 0.0
    except Exception as e:
        print(f"Error calculating average monthly grocery spend: {e}")
        # If any other error, AVERAGE_MONTHLY_GROCERY_SPEND remains 0.0

    # --- Load Stock-Out Model (Critical) ---
    try:
        stockout_model_path = os.path.join(models_dir, 'stockout_model.pkl')
        stockout_model = pickle.load(open(stockout_model_path, 'rb'))
        print("stockout_model.pkl loaded.")

        preprocessor_stockout = stockout_model.named_steps['preprocessor']
        
        # Dynamically infer expected features for the preprocessor's input.
        columns_for_stockout_preprocessor_input_df = [
            'Shopping_Frequency', 'Monthly_Grocery_Spend', 'Stockout_Frequency',
            'Stockout_Reaction', 'Check_Online_Availability', 'Willing_Preorder',
            'Acceptable_Restock_Wait_Time', 'Switch_Store_Stockout'
        ]
        # Append expanded multi-selects based on the new options and naming convention
        columns_for_stockout_preprocessor_input_df.extend([f'Frequent_Categories_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")}' for cat in ALL_FREQUENT_CATEGORIES_OPTIONS])
        columns_for_stockout_preprocessor_input_df.extend([f'Stockout_Categories_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")}' for cat in ALL_STOCKOUT_CATEGORIES_OPTIONS])

        dummy_data_stockout = {}
        # Initialize numericals
        dummy_data_stockout['Monthly_Grocery_Spend'] = 0.0
        dummy_data_stockout['Acceptable_Restock_Wait_Time'] = 0.0
        # Initialize expanded multi-selects to 0
        for cat in ALL_FREQUENT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
            dummy_data_stockout[f'Frequent_Categories_{safe_name}'] = 0
        for cat in ALL_STOCKOUT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
            dummy_data_stockout[f'Stockout_Categories_{safe_name}'] = 0
        # Initialize categoricals
        dummy_data_stockout['Shopping_Frequency'] = 'Daily'
        dummy_data_stockout['Stockout_Frequency'] = 'Not Applicable'
        dummy_data_stockout['Stockout_Reaction'] = 'Not Applicable'
        dummy_data_stockout['Check_Online_Availability'] = 'No'
        dummy_data_stockout['Willing_Preorder'] = 'No'
        dummy_data_stockout['Switch_Store_Stockout'] = 'No'

        # Fill any missing columns in the dummy_data_stockout that might be in the list
        for col in columns_for_stockout_preprocessor_input_df:
            if col not in dummy_data_stockout:
                if any(x in col for x in ['Monthly_Grocery_Spend', 'Acceptable_Restock_Wait_Time', 'Frequent_Categories_', 'Stockout_Categories_']):
                    dummy_data_stockout[col] = 0.0
                else: # Default for other single categoricals if not explicitly set
                    dummy_data_stockout[col] = 'None' # Or a suitable default string

        dummy_df_for_stockout_features = pd.DataFrame([dummy_data_stockout], columns=columns_for_stockout_preprocessor_input_df)
        
        _ = preprocessor_stockout.transform(dummy_df_for_stockout_features) # Transform to ensure all features are generated
        
        EXPECTED_STOCKOUT_CLASSIFIER_FEATURES = preprocessor_stockout.get_feature_names_out()
        print(f"Expected final features for stock-out classifier initialized successfully ({len(EXPECTED_STOCKOUT_CLASSIFIER_FEATURES)} features).")

    except FileNotFoundError:
        print(f"Error: stockout_model.pkl not found in '{models_dir}/'. Stock-out prediction will not be available.")
        stockout_model = None
        EXPECTED_STOCKOUT_CLASSIFIER_FEATURES = None
    except Exception as e:
        print(f"An unexpected error occurred during stockout model loading or feature extraction: {e}")
        import traceback
        traceback.print_exc()
        stockout_model = None
        EXPECTED_STOCKOUT_CLASSIFIER_FEATURES = None

    # --- Load Sales Model ---
    try:
        sales_model_path = os.path.join(models_dir, 'sales_rf_model.pkl')
        sales_model = pickle.load(open(sales_model_path, 'rb'))
        print("sales_rf_model.pkl loaded.")

        preprocessor_sales = sales_model.named_steps['preprocessor']

        # Determine features that went into the sales preprocessor during training
        sales_preprocessor_input_cols = [
            'Age_Group', 'City_Town_Residence', 'Monthly_Income_Range',
            'Household_Size', 'Shopping_Frequency', 'Shopping_Trip_Duration',
            'Prefer_Offers', 'Offers_Influence_Purchase',
            'Offer_Spend_Increase_Percentage', 'Purchase_Influence', 'Likely_Switch_Online_Shopping'
        ]
        # Append expanded multi-selects based on the new options and naming convention
        sales_preprocessor_input_cols.extend([f'Frequent_Categories_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()}' for cat in ALL_FREQUENT_CATEGORIES_OPTIONS])
        sales_preprocessor_input_cols.extend([f'Offer_Types_Preferred_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()}' for cat in ALL_OFFER_TYPES_PREFERRED_OPTIONS])
        sales_preprocessor_input_cols.extend([f'Categories_Buy_Offers_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()}' for cat in ALL_CATEGORIES_BUY_OFFERS_OPTIONS])

        dummy_data_sales = {}
        # Initialize all numerical columns to 0.0 or 0
        numerical_cols_sales_for_dummy = [
            'Household_Size', 'Shopping_Trip_Duration', 'Offer_Spend_Increase_Percentage',
            'Likely_Switch_Online_Shopping'
        ]
        for col in numerical_cols_sales_for_dummy:
            dummy_data_sales[col] = 0.0
        
        # Initialize expanded multi-select columns to 0
        for cat in ALL_FREQUENT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            dummy_data_sales[f'Frequent_Categories_{safe_name}'] = 0
        for cat in ALL_OFFER_TYPES_PREFERRED_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            dummy_data_sales[f'Offer_Types_Preferred_{safe_name}'] = 0
        for cat in ALL_CATEGORIES_BUY_OFFERS_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            dummy_data_sales[f'Categories_Buy_Offers_{safe_name}'] = 0

        # Initialize categorical columns with placeholder strings
        dummy_data_sales['Age_Group'] = '18–25'
        dummy_data_sales['City_Town_Residence'] = 'Chennai'
        dummy_data_sales['Monthly_Income_Range'] = 'Below ₹20,000'
        dummy_data_sales['Shopping_Frequency'] = 'Daily'
        dummy_data_sales['Prefer_Offers'] = 'No'
        dummy_data_sales['Offers_Influence_Purchase'] = 'Neutral'
        dummy_data_sales['Purchase_Influence'] = 'Price'

        # Ensure all columns expected by the preprocessor are in the dummy_data_sales dict
        for col in sales_preprocessor_input_cols:
            if col not in dummy_data_sales:
                if any(x in col for x in numerical_cols_sales_for_dummy):
                    dummy_data_sales[col] = 0.0
                elif any(x in col for x in ['Frequent_Categories_', 'Offer_Types_Preferred_', 'Categories_Buy_Offers_']):
                    dummy_data_sales[col] = 0
                else:
                    dummy_data_sales[col] = 'None' # Default for other categoricals

        dummy_df_for_sales_features = pd.DataFrame([dummy_data_sales], columns=sales_preprocessor_input_cols)
        
        _ = preprocessor_sales.transform(dummy_df_for_sales_features) # Transform to ensure all features are generated
        
        EXPECTED_SALES_REGRESSOR_FEATURES = preprocessor_sales.get_feature_names_out()
        print(f"Expected final features for sales regressor initialized successfully ({len(EXPECTED_SALES_REGRESSOR_FEATURES)} features).")

    except FileNotFoundError:
        print(f"Error: sales_rf_model.pkl not found in '{models_dir}/'. Associated functionality will not be available.")
        sales_model = None
        EXPECTED_SALES_REGRESSOR_FEATURES = None
    except Exception as e:
        print(f"An unexpected error occurred during sales model loading or feature extraction: {e}")
        import traceback
        traceback.print_exc()

    # --- Load Offer Recommendation Model ---
    try:
        offer_model_path = os.path.join(models_dir, 'offer_rf_model.pkl')
        mlb_path = os.path.join(models_dir, 'offer_mlb.pkl')

        offer_model = pickle.load(open(offer_model_path, 'rb'))
        offer_mlb = pickle.load(open(mlb_path, 'rb'))
        print("offer_rf_model.pkl and offer_mlb.pkl loaded.")

        preprocessor_offer = offer_model.named_steps['preprocessor']
        
        # Define the exact input columns for the offer model's preprocessor
        # These should match the X.columns after manual expansion in offer_recommendation_model.py
        offer_preprocessor_input_cols = [
            'Household_Size', 'Monthly_Grocery_Spend', 'Shopping_Trip_Duration',
            'Offer_Spend_Increase_Percentage', 'Likely_Switch_Online_Shopping', 'Brand_Loyalty',
            'Prefer_Offers', 'Offers_Influence_Purchase', 'Shopping_Style', 'Eco_Friendliness'
        ]
        # Dynamically add the expanded multi-select columns
        for cat in ALL_FREQUENT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            offer_preprocessor_input_cols.append(f'Frequent_Categories_{safe_name}')
        for cat in ALL_CATEGORIES_BUY_OFFERS_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            offer_preprocessor_input_cols.append(f'Categories_Buy_Offers_{safe_name}')

        # Create a dummy DataFrame with these exact columns to get the feature names out
        dummy_data_offer = {}
        # Initialize numericals and expanded binary columns
        for col in [c for c in offer_preprocessor_input_cols if c in ['Household_Size', 'Monthly_Grocery_Spend', 'Shopping_Trip_Duration', 'Offer_Spend_Increase_Percentage', 'Likely_Switch_Online_Shopping', 'Brand_Loyalty'] or c.startswith(('Frequent_Categories_', 'Categories_Buy_Offers_'))]:
            dummy_data_offer[col] = 0.0
        # Initialize categoricals
        for col in [c for c in offer_preprocessor_input_cols if c not in dummy_data_offer]:
            if col == 'Prefer_Offers': dummy_data_offer[col] = 'No'
            elif col == 'Offers_Influence_Purchase': dummy_data_offer[col] = 'Neutral'
            elif col == 'Shopping_Style': dummy_data_offer[col] = 'Planner (I make a list and stick to it)'
            elif col == 'Eco_Friendliness': dummy_data_offer[col] = 'No'
            else: dummy_data_offer[col] = 'Unknown' # Fallback
        
        dummy_df_offer = pd.DataFrame([dummy_data_offer], columns=offer_preprocessor_input_cols)
        
        _ = preprocessor_offer.transform(dummy_df_offer) # Transform to ensure all features are generated

        EXPECTED_OFFER_MODEL_FEATURES = preprocessor_offer.get_feature_names_out()
        print(f"Expected final features for offer model initialized successfully ({len(EXPECTED_OFFER_MODEL_FEATURES)} features).")

    except FileNotFoundError:
        print(f"Error: offer_rf_model.pkl or offer_mlb.pkl not found in '{models_dir}/'. Offer recommendation functionality will not be available.")
        offer_model = None
        offer_mlb = None
        EXPECTED_OFFER_MODEL_FEATURES = None
    except Exception as e:
        print(f"An unexpected error occurred during offer model loading or feature extraction: {e}")
        import traceback
        traceback.print_exc()

    # --- Load Segmentation Model ---
    try:
        segmentation_model_path = os.path.join(models_dir, 'segmentation_kmeans_model.pkl')
        segmentation_model = pickle.load(open(segmentation_model_path, 'rb'))
        print("segmentation_kmeans_model.pkl loaded.")
    except FileNotFoundError:
        print(f"Error: segmentation_kmeans_model.pkl not found in '{models_dir}/'. Segmentation functionality will not be available.")
        segmentation_model = None
    except Exception as e:
        print(f"An unexpected error occurred during segmentation model loading: {e}")
        segmentation_model = None


    # --- Load Other Models (Independent, similar graceful handling) ---
    model_files = {
        'retention_rf_model.pkl': 'retention_model',
        'trend_model.pkl': 'trend_model'
    }

    for filename, var_name in model_files.items():
        try:
            model_path = os.path.join(models_dir, filename)
            globals()[var_name] = pickle.load(open(model_path, 'rb'))
            print(f"{filename} loaded.")
        except FileNotFoundError:
            print(f"Error: {filename} not found in '{models_dir}/'. Associated functionality will not be available.")
            globals()[var_name] = None
        except Exception as e:
            print(f"An unexpected error occurred during loading {filename}: {e}")
            import traceback
            traceback.print_exc()

except FileNotFoundError as e:
    print(f"Error: Models directory '{models_dir}' not found. Please ensure your models are placed there. {e}")
except Exception as e:
    print(f"An unexpected error occurred during initial model directory check: {e}")
    import traceback
    traceback.print_exc()


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles user registration."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        phone_number = request.form['phone_number']

        if users_collection is not None:
            existing_user = users_collection.find_one({'email': email})
            if existing_user:
                flash("Email already registered. Please use a different email.", "error")
                return redirect(url_for('signup'))

            user_data = {
                "email": email,
                "password": password,
                "full_name": full_name,
                "phone_number": phone_number
            }
            users_collection.insert_one(user_data)

            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("Database connection error. Please try again later.", "error")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if users_collection is not None:
            user = users_collection.find_one({"email": email, "password": password})
            
            if user:
                session['user_id'] = str(user["_id"])
                session['full_name'] = user.get('full_name', 'Guest')
                session['email'] = user.get('email', 'No Email')
                
                flash(f"Welcome, {session['full_name']}!", "success")
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid email or password.", "error")
                return redirect(url_for('login'))
        else:
            flash("Database connection error. Please try again later.", "error")

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """Renders the user dashboard, requires login."""
    if 'user_id' not in session:
        flash("Please login to access the dashboard.", "error")
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/stockout_form_page')
def stockout_form_page():
    """Renders the HTML form for stock-out prediction."""
    if 'user_id' not in session:
        flash("Please login to access this tool.", "error")
        return redirect(url_for('login'))
    return render_template('predict_stockout.html')

@app.route('/predict_stockout', methods=['POST'])
def predict_stockout():
    """
    Receives user input from the HTML form, preprocesses it,
    and returns a stock-out prediction. This is a POST-only endpoint.
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized access. Please log in.'}), 401

    if stockout_model is None or EXPECTED_STOCKOUT_CLASSIFIER_FEATURES is None:
        return jsonify({'error': 'Stock-out prediction model or its expected features are not loaded. Please check server logs for details.'}), 500

    try:
        user_input_raw = request.json
        
        preprocessor_input_dict = {}

        # 1. Handle single-value categorical features
        preprocessor_input_dict['Shopping_Frequency'] = user_input_raw.get('Shopping_Frequency', None)
        preprocessor_input_dict['Stockout_Frequency'] = user_input_raw.get('Stockout_Frequency', None)
        preprocessor_input_dict['Stockout_Reaction'] = user_input_raw.get('Stockout_Reaction', None)
        preprocessor_input_dict['Check_Online_Availability'] = user_input_raw.get('Check_Online_Availability', None)
        preprocessor_input_dict['Willing_Preorder'] = user_input_raw.get('Willing_Preorder', None)
        preprocessor_input_dict['Switch_Store_Stockout'] = user_input_raw.get('Switch_Store_Stockout', None)

        # 2. Handle numerical features
        for num_col in ['Monthly_Grocery_Spend', 'Acceptable_Restock_Wait_Time']:
            val = user_input_raw.get(num_col, '')
            try:
                preprocessor_input_dict[num_col] = float(val) if val != '' else np.nan
            except ValueError:
                preprocessor_input_dict[num_col] = np.nan # Handle non-numeric input

        # 3. Handle manually expanded multi-select features (0/1 binary columns)
        # Initialize all potential multi-select columns to 0
        for cat in ALL_FREQUENT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
            preprocessor_input_dict[f'Frequent_Categories_{safe_name}'] = 0

        for cat in ALL_STOCKOUT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
            preprocessor_input_dict[f'Stockout_Categories_{safe_name}'] = 0

        # Now populate based on user's actual selections for multi-selects
        selected_frequent = user_input_raw.get('Frequent_Categories', [])
        if isinstance(selected_frequent, str): # Handle case where only one is selected and not an array
            selected_frequent = [selected_frequent]
        for cat in selected_frequent:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
            col_name = f'Frequent_Categories_{safe_name}'
            if col_name in preprocessor_input_dict: # Ensure the column name was pre-initialized
                preprocessor_input_dict[col_name] = 1

        selected_stockout = user_input_raw.get('Stockout_Categories', [])
        if isinstance(selected_stockout, str): # Handle case where only one is selected and not an array
            selected_stockout = [selected_stockout]
        for cat in selected_stockout:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
            col_name = f'Stockout_Categories_{safe_name}'
            if col_name in preprocessor_input_dict: # Ensure the column name was pre-initialized
                preprocessor_input_dict[col_name] = 1
        
        # 4. Handle conditional logic for 'Faced_Stockouts_For_Conditional'
        if user_input_raw.get('Faced_Stockouts_For_Conditional') == 'No':
            preprocessor_input_dict['Stockout_Frequency'] = 'Not Applicable'
            preprocessor_input_dict['Stockout_Reaction'] = 'Not Applicable'
            preprocessor_input_dict['Acceptable_Restock_Wait_Time'] = 0.0

        columns_for_stockout_preprocessor_input_df_local = [
            'Shopping_Frequency', 'Monthly_Grocery_Spend', 'Stockout_Frequency',
            'Stockout_Reaction', 'Check_Online_Availability', 'Willing_Preorder',
            'Acceptable_Restock_Wait_Time', 'Switch_Store_Stockout'
        ] + [f'Frequent_Categories_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")}' for cat in ALL_FREQUENT_CATEGORIES_OPTIONS] \
          + [f'Stockout_Categories_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")}' for cat in ALL_STOCKOUT_CATEGORIES_OPTIONS]

        filtered_preprocessor_input_dict = {
            col: preprocessor_input_dict[col] for col in columns_for_stockout_preprocessor_input_df_local if col in preprocessor_input_dict
        }
        
        input_df_for_preprocessor = pd.DataFrame([filtered_preprocessor_input_dict], columns=columns_for_stockout_preprocessor_input_df_local)
        
        preprocessed_data = stockout_model.named_steps['preprocessor'].transform(input_df_for_preprocessor)

        # Ensure preprocessed_data is a dense NumPy array before creating DataFrame
        if isinstance(preprocessed_data, tuple):
            preprocessed_data_dense = np.hstack(preprocessed_data)
        elif hasattr(preprocessed_data, 'toarray'):
            preprocessed_data_dense = preprocessed_data.toarray()
        else:
            preprocessed_data_dense = preprocessed_data

        final_input_for_classifier = pd.DataFrame(preprocessed_data_dense,
                                                    columns=EXPECTED_STOCKOUT_CLASSIFIER_FEATURES)
        
        prediction = stockout_model.named_steps['classifier'].predict(final_input_for_classifier)[0]
        probability_of_stockout = stockout_model.named_steps['classifier'].predict_proba(final_input_for_classifier)[0][1] * 100

        result_text = "LIKELY to face stock-outs." if prediction == 1 else "UNLIKELY to face stock-outs."
        
        if prediction == 1:
            insight = (
                f"Based on the provided information, the customer has a high likelihood of facing stock-outs. "
                f"This suggests they frequently encounter out-of-stock items or their shopping habits align "
                f"with patterns observed in customers who face stock-outs."
            )
        else:
            insight = (
                f"The customer is unlikely to face significant stock-outs based on their profile. "
                f"Their shopping frequency and preferences suggest good alignment with available inventory, "
                f"or they employ strategies to avoid out-of-stock situations."
            )

        return jsonify({
            'prediction': int(prediction),
            'result_text': result_text,
            'probability': round(probability_of_stockout, 2),
            'insight': insight,
            'visualization_data': {
                'type': 'bar',
                'label': 'Stock-Out Likelihood (%)',
                'value': round(probability_of_stockout, 2),
                'max_value': 100
            }
        })

    except Exception as e:
        print(f"Error during stock-out prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An error occurred during stock-out prediction: {e}"}), 500

@app.route('/sales')
def sales():
    """Renders the sales analytics dashboard."""
    if 'user_id' not in session:
        flash("Please login to access this tool.", "error")
        return redirect(url_for('login'))
    return render_template('sales_dashboard.html')

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    """
    Receives user input for sales prediction, preprocesses it,
    and returns a predicted monthly grocery spend.
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized access. Please log in.'}), 401

    if sales_model is None or EXPECTED_SALES_REGRESSOR_FEATURES is None:
        return jsonify({'error': 'Sales prediction model is not loaded. Please check server logs.'}), 500

    try:
        user_input_raw = request.json
        # print(f"Received raw user input for sales: {user_input_raw}")

        preprocessor_input_dict = {}

        # Handle numerical features for sales model
        numerical_sales_cols_from_form = [
            'Household_Size', 'Shopping_Trip_Duration', 'Offer_Spend_Increase_Percentage',
            'Likely_Switch_Online_Shopping'
        ]
        for col in numerical_sales_cols_from_form:
            val = user_input_raw.get(col, '')
            try:
                preprocessor_input_dict[col] = float(val) if val != '' else np.nan
            except ValueError:
                preprocessor_input_dict[col] = np.nan

        # Handle single-value categorical features for sales model
        categorical_sales_cols_from_form = [
            'Age_Group', 'City_Town_Residence', 'Monthly_Income_Range',
            'Shopping_Frequency', 'Prefer_Offers', 'Offers_Influence_Purchase',
            'Purchase_Influence'
        ]
        for col in categorical_sales_cols_from_form:
            preprocessor_input_dict[col] = user_input_raw.get(col, None)

        # Handle manually expanded multi-select features (0/1 binary columns)
        for cat in ALL_FREQUENT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            preprocessor_input_dict[f'Frequent_Categories_{safe_name}'] = 0
        for cat in ALL_OFFER_TYPES_PREFERRED_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            preprocessor_input_dict[f'Offer_Types_Preferred_{safe_name}'] = 0
        for cat in ALL_CATEGORIES_BUY_OFFERS_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            preprocessor_input_dict[f'Categories_Buy_Offers_{safe_name}'] = 0

        selected_frequent = user_input_raw.get('Frequent_Categories', [])
        if isinstance(selected_frequent, str): selected_frequent = [selected_frequent]
        for cat in selected_frequent:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            col_name = f'Frequent_Categories_{safe_name}'
            if col_name in preprocessor_input_dict:
                preprocessor_input_dict[col_name] = 1

        selected_offer_types = user_input_raw.get('Offer_Types_Preferred', [])
        if isinstance(selected_offer_types, str): selected_offer_types = [selected_offer_types]
        for cat in selected_offer_types:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            col_name = f'Offer_Types_Preferred_{safe_name}'
            if col_name in preprocessor_input_dict:
                preprocessor_input_dict[col_name] = 1
        
        selected_categories_offers = user_input_raw.get('Categories_Buy_Offers', [])
        if isinstance(selected_categories_offers, str): selected_categories_offers = [selected_categories_offers]
        for cat in selected_categories_offers:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            col_name = f'Categories_Buy_Offers_{safe_name}'
            if col_name in preprocessor_input_dict:
                preprocessor_input_dict[col_name] = 1

        # Reconstruct the list of columns expected by the sales preprocessor
        columns_for_sales_preprocessor_input_df = [
            'Age_Group', 'City_Town_Residence', 'Monthly_Income_Range',
            'Household_Size', 'Shopping_Frequency', 'Shopping_Trip_Duration',
            'Prefer_Offers', 'Offers_Influence_Purchase',
            'Offer_Spend_Increase_Percentage', 'Purchase_Influence', 'Likely_Switch_Online_Shopping'
        ]
        columns_for_sales_preprocessor_input_df.extend([f'Frequent_Categories_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()}' for cat in ALL_FREQUENT_CATEGORIES_OPTIONS])
        columns_for_sales_preprocessor_input_df.extend([f'Offer_Types_Preferred_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()}' for cat in ALL_OFFER_TYPES_PREFERRED_OPTIONS])
        columns_for_sales_preprocessor_input_df.extend([f'Categories_Buy_Offers_{cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()}' for cat in ALL_CATEGORIES_BUY_OFFERS_OPTIONS])

        filtered_sales_input_dict = {
            col: preprocessor_input_dict.get(col) for col in columns_for_sales_preprocessor_input_df
        }

        input_df_for_sales_preprocessor = pd.DataFrame([filtered_sales_input_dict], columns=columns_for_sales_preprocessor_input_df)
        
        preprocessed_sales_data = sales_model.named_steps['preprocessor'].transform(input_df_for_sales_preprocessor)
        
        # Ensure preprocessed_sales_data is a dense NumPy array before creating DataFrame
        if isinstance(preprocessed_sales_data, tuple):
            preprocessed_sales_data_dense = np.hstack(preprocessed_sales_data)
        elif hasattr(preprocessed_sales_data, 'toarray'):
            preprocessed_sales_data_dense = preprocessed_sales_data.toarray()
        else:
            preprocessed_sales_data_dense = preprocessed_sales_data

        final_input_for_regressor = pd.DataFrame(preprocessed_sales_data_dense, columns=EXPECTED_SALES_REGRESSOR_FEATURES)
        
        predicted_spend = sales_model.named_steps['regressor'].predict(final_input_for_regressor)[0]

        return jsonify({
            'predicted_spend': round(float(predicted_spend), 2),
            'average_spend': round(float(AVERAGE_MONTHLY_GROCERY_SPEND), 2), # Pass the average spend
            'insight': f"Based on the provided customer profile, the predicted monthly grocery expenditure is approximately ₹{round(float(predicted_spend), 2)}."
        })

    except Exception as e:
        print(f"Error during sales prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An error occurred during sales prediction: {e}"}), 500


@app.route('/segmentation')
def segmentation():
    """Renders the customer segmentation page."""
    if 'user_id' not in session:
        flash("Please login to access this tool.", "error")
        return redirect(url_for('login'))
    return render_template('segmentation.html')

@app.route('/predict-segment', methods=['POST'])
def predict_segment():
    if segmentation_model is None:
        return jsonify({'error': 'Segmentation model not loaded.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # 1. Create DataFrame from input
    input_df = pd.DataFrame([data])

    # 2. One-hot encode product categories
    all_categories = [
        'Groceries', 'Snacks', 'Dairy', 'Beverages', 'Personal Care',
        'Household Items', 'Fruits & Vegetables', 'Packaged Food', 'Other'
    ]

    multi_select_col = 'Which product categories do you frequently buy? (Select all that apply)'
    if multi_select_col in input_df.columns:
        if isinstance(input_df[multi_select_col][0], str):
            selected = [i.strip() for i in input_df[multi_select_col][0].split(',') if i.strip()]
        elif isinstance(input_df[multi_select_col][0], list):
            selected = input_df[multi_select_col][0]
        else:
            selected = []
    else:
        selected = []

    for cat in all_categories:
        input_df[cat] = int(cat in selected)

    if multi_select_col in input_df.columns:
        input_df = input_df.drop(columns=[multi_select_col])

    # 3. Predict
    try:
        cluster = int(segmentation_model.predict(input_df)[0])
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # 4. Detailed output for each cluster
    SEGMENT_DETAILS = {
    0: {
        "name": "Value Seekers",
        "description": "You’re a smart shopper who always finds the best deals! You love saving money and making the most of every offer. Retailers value your keen eye for discounts and your loyalty to savings.",
        "characteristics": [
            "You never miss a good offer or sale.",
            "Your loyalty cards are always ready for extra rewards.",
            "Your household enjoys a well-stocked pantry—without breaking the bank.",
            "You inspire others to shop wisely and spend thoughtfully."
        ]
    },
    1: {
        "name": "Brand Loyalists",
        "description": "You know what you love and you stick with it! Quality and trust matter most to you, and your favorite brands appreciate your unwavering loyalty. Your shopping choices set trends for others.",
        "characteristics": [
            "You’re always first to try new launches from your favorite brands.",
            "Your shopping basket reflects refined taste and consistency.",
            "You value long-term relationships with trusted retailers.",
            "Your confidence in your choices inspires those around you."
        ]
    },
    2: {
        "name": "Impulse Buyers",
        "description": "You bring excitement to every shopping trip! Your spontaneous choices add color and variety to your life—and to the store shelves. Retailers love your adventurous spirit and openness to new experiences.",
        "characteristics": [
            "You enjoy discovering new snacks and treats.",
            "Shopping is a joyful, in-the-moment experience for you.",
            "Your curiosity leads you to try the latest products.",
            "Your energy makes every store visit memorable."
        ]
    },
    3: {
        "name": "Online Switchers",
        "description": "You’re always ahead of the curve, embracing convenience and technology. Your forward-thinking approach means you get the best of both in-store and online worlds. Retailers value your adaptability and digital savvy.",
        "characteristics": [
            "You love seamless shopping—anytime, anywhere.",
            "Digital payment and smart offers are your go-to.",
            "You’re quick to explore new online shopping trends.",
            "Your feedback helps shape the future of retail."
        ]
    }
}


    details = SEGMENT_DETAILS.get(cluster, {
        "name": "Unknown Segment",
        "description": "No details available for this segment.",
        "characteristics": []
    })

    return jsonify({
        'cluster': cluster,
        'segment_name': details['name'],
        'segment_description': details['description'],
        'characteristics': details['characteristics']
    })


@app.route('/recommendation')
def recommendation():
    """Renders the offer recommendation page."""
    if 'user_id' not in session:
        flash("Please login to access this tool.", "error")
        return redirect(url_for('login'))
    return render_template('recommendation.html')


@app.route('/predict_recommendation', methods=['POST'])
def predict_recommendation():
    """
    Receives user input from the HTML form, preprocesses it,
    and returns a recommended offer type along with probabilities.
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized access. Please log in.'}), 401

    if offer_model is None or offer_mlb is None or EXPECTED_OFFER_MODEL_FEATURES is None:
        return jsonify({'error': 'Offer recommendation model, MLB, or expected features are not loaded. Please check server logs.'}), 500

    try:
        user_input_raw = request.json

        preprocessor_input_dict = {}

        # 1. Handle Numerical features for offer model (using the reduced list from offer_recommendation_model.py)
        numerical_offer_cols_from_form = [
            'Household_Size', 'Monthly_Grocery_Spend', 'Shopping_Trip_Duration',
            'Offer_Spend_Increase_Percentage', 'Likely_Switch_Online_Shopping', 'Brand_Loyalty'
        ]
        for col in numerical_offer_cols_from_form:
            val = user_input_raw.get(col, '')
            try:
                # Special handling for Brand_Loyalty if it comes as 'X star'
                if col == 'Brand_Loyalty':
                    # Extract numeric part, handle potential 'X (Text)' format
                    parsed_val = str(val).split(' ')[0] if isinstance(val, str) else val
                    preprocessor_input_dict[col] = float(parsed_val) if parsed_val != '' else np.nan
                else:
                    preprocessor_input_dict[col] = float(val) if val != '' else np.nan
            except ValueError:
                preprocessor_input_dict[col] = np.nan # Handle non-numeric input

        # 2. Handle single-value Categorical features for offer model
        categorical_offer_cols_from_form = [
            'Prefer_Offers', 'Offers_Influence_Purchase', 'Shopping_Style', 'Eco_Friendliness'
        ]
        for col in categorical_offer_cols_from_form:
            preprocessor_input_dict[col] = user_input_raw.get(col, None) # None will be imputed by preprocessor

        # 3. Handle manually expanded multi-select features (0/1 binary columns)
        # Initialize all relevant multi-select columns to 0 first based on ALL_..._OPTIONS
        for cat in ALL_FREQUENT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            preprocessor_input_dict[f'Frequent_Categories_{safe_name}'] = 0
        for cat in ALL_CATEGORIES_BUY_OFFERS_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            preprocessor_input_dict[f'Categories_Buy_Offers_{safe_name}'] = 0

        # Now populate with 1s based on user's actual selections
        selected_frequent_categories = user_input_raw.get('Frequent_Categories', [])
        if isinstance(selected_frequent_categories, str): selected_frequent_categories = [selected_frequent_categories]
        for cat in selected_frequent_categories:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            col_name = f'Frequent_Categories_{safe_name}'
            if col_name in preprocessor_input_dict: # Double-check existence (should always be there due to initialization)
                preprocessor_input_dict[col_name] = 1

        selected_categories_buy_offers = user_input_raw.get('Categories_Buy_Offers', [])
        if isinstance(selected_categories_buy_offers, str): selected_categories_buy_offers = [selected_categories_buy_offers]
        for cat in selected_categories_buy_offers:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            col_name = f'Categories_Buy_Offers_{safe_name}'
            if col_name in preprocessor_input_dict: # Double-check existence
                preprocessor_input_dict[col_name] = 1
        
        # Define the full list of columns that the offer model's preprocessor expects as input
        # This list MUST match the X.columns *before* preprocessing in offer_recommendation_model.py
        offer_preprocessor_input_cols_for_df = [
            'Household_Size', 'Monthly_Grocery_Spend', 'Shopping_Trip_Duration',
            'Offer_Spend_Increase_Percentage', 'Likely_Switch_Online_Shopping', 'Brand_Loyalty',
            'Prefer_Offers', 'Offers_Influence_Purchase', 'Shopping_Style', 'Eco_Friendliness'
        ]
        # Extend with all potential expanded multi-select columns
        for cat in ALL_FREQUENT_CATEGORIES_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            offer_preprocessor_input_cols_for_df.append(f'Frequent_Categories_{safe_name}')
        for cat in ALL_CATEGORIES_BUY_OFFERS_OPTIONS:
            safe_name = cat.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
            offer_preprocessor_input_cols_for_df.append(f'Categories_Buy_Offers_{safe_name}')

        # Create the DataFrame for the preprocessor, ensuring all columns are present and in correct order
        input_df_for_preprocessor = pd.DataFrame([preprocessor_input_dict], columns=offer_preprocessor_input_cols_for_df)
        
        # Transform the input data using the model's preprocessor
        preprocessed_data = offer_model.named_steps['preprocessor'].transform(input_df_for_preprocessor)

        # Ensure preprocessed_data is a dense NumPy array before creating DataFrame
        if isinstance(preprocessed_data, tuple):
            preprocessed_data_dense = np.hstack(preprocessed_data)
        elif hasattr(preprocessed_data, 'toarray'):
            preprocessed_data_dense = preprocessed_data.toarray()
        else:
            preprocessed_data_dense = preprocessed_data

        # Convert to DataFrame with correct column names for the classifier
        final_input_for_classifier = pd.DataFrame(preprocessed_data_dense,
                                                    columns=EXPECTED_OFFER_MODEL_FEATURES)
        
        # Make predictions (binary array indicating preferred offer types)
        # The classifier in the pipeline is now accessed via named_steps
        predicted_offer_binary = offer_model.named_steps['classifier'].predict(final_input_for_classifier)[0]
        
        # Get probabilities for all offer types
        # Each element of predict_proba is an array [proba_class_0, proba_class_1]
        predicted_probabilities = offer_model.named_steps['classifier'].predict_proba(final_input_for_classifier)
        
        # Extract the probability of the positive class (1) for each offer type
        offer_probabilities = {}
        for i, offer_name in enumerate(offer_mlb.classes_):
            # Corrected access for probability: predicted_probabilities[i][0][1]
            if isinstance(predicted_probabilities[i], np.ndarray) and predicted_probabilities[i].shape == (1, 2):
                offer_probabilities[offer_name] = round(predicted_probabilities[i][0][1] * 100, 2)
            else:
                # Fallback for unexpected format (e.g., if a classifier only predicts one class)
                offer_probabilities[offer_name] = 0.0


        # Use MultiLabelBinarizer to convert binary predictions back to original labels
        recommended_offers = offer_mlb.inverse_transform(predicted_offer_binary.reshape(1, -1))
        
        # Corrected: Flatten the list of lists for display, explicitly convert tuple to list
        recommended_offers_list = list(recommended_offers[0]) if recommended_offers else []

        if not recommended_offers_list:
            recommendation_text = "Based on the provided profile, no specific offer types are strongly recommended at this time. Consider exploring general offers."
        else:
            recommendation_text = "Based on your preferences, the following offer types are recommended:"

        # Prepare detailed offers with probabilities
        detailed_offers = []
        for offer_name in offer_mlb.classes_: # Iterate through all known offer types learned by MLB
            is_recommended = offer_name in recommended_offers_list
            probability = offer_probabilities.get(offer_name, 0.0)
            detailed_offers.append({
                'name': offer_name,
                'recommended': is_recommended,
                'probability': probability
            })


        return jsonify({
            'recommended_offers': recommended_offers_list, # Still include for backward compatibility if needed
            'recommendation_text': recommendation_text,
            'detailed_offers': detailed_offers # New detailed list with probabilities
        })

    except Exception as e:
        print(f"Error during offer recommendation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An error occurred during offer recommendation: {e}"}), 500
    
# -------------------------
# Category Optimization Route
# -------------------------
@app.route("/category", methods=["GET", "POST"])
def category():
    if request.method == "POST":
        # Collect user input from the form
        user_inputs = {
            "frequent_categories": request.form.getlist("frequent_categories"),
            "offer_categories": request.form.getlist("offer_categories"),
            "offer_types": request.form.getlist("offer_types"),
            "price_comparison": request.form.get("price_comparison"),
            "eco_friendly": request.form.get("eco_friendly")
        }

        # Generate insights
        insights = generate_category_insights(user_inputs)

        # Render result template with insights
        return render_template("category_result.html", insights=insights)

    # For GET request, render the input form
    return render_template("category.html")

@app.route('/retention_risk')
def retention_risk():
    """Renders the retention risk prediction page."""
    if 'user_id' not in session:
        flash("Please login to access this tool.", "error")
        return redirect(url_for('login'))
    return render_template('retention_risk.html')

@app.route('/category_optimization')
def category_optimization():
    return render_template("category_optimization.html")

@app.route("/trend", methods=["GET", "POST"])
def trend():
    if request.method == "POST":
        try:
            # ─── Get input ─────────────────────────────────────────
            shopping_frequency = request.form.get("shopping_frequency")
            product_categories = request.form.getlist("product_categories[]")
            product_categories_other = request.form.get("product_categories_other", "").strip()
            price_comparison = request.form.get("price_comparison")
            shop_offer = request.form.get("shop_offer")
            purchase_influence = request.form.get("purchase_influence")
            shopping_time = request.form.get("shopping_time")

            if product_categories_other:
                product_categories.append(product_categories_other)

            if not all([shopping_frequency, price_comparison, shop_offer, purchase_influence, shopping_time]) or not product_categories:
                flash("⚠ Please fill all required fields and select at least one category.", "warning")
                return render_template("trend.html")

            user_input = {
                "shopping_frequency": shopping_frequency,
                "product_categories": ", ".join(product_categories),
                "price_comparison": price_comparison,
                "shop_offer": shop_offer,
                "purchase_influence": purchase_influence,
                "shopping_time": shopping_time
            }

            # ─── Predict Using Model ───────────────────────────────
            segment, trend_score, advice, chart_path, comparison_rows, insights = predict_trend_segment(user_input)

            return render_template("trend_result.html",
                                   segment=segment,
                                   trend_score=trend_score,
                                   advice=advice,
                                   chart_path=chart_path,
                                   comparison_rows=comparison_rows,
                                   insights=insights)

        except Exception as e:
            print("[ERROR]", str(e))
            flash("❌ Something went wrong while processing the trend analysis.", "danger")
            return render_template("trend.html")

    return render_template("trend.html")


@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and initial data preprocessing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            file_content = file.read()
            # Load and preprocess the data. The analytics_model instance will store processed_df.
            available_features = analytics_model.load_and_preprocess_data(file_content)

            # Store only what's necessary in the session (e.g., encoder classes for potential later use)
            # Do NOT store the entire DataFrame in session as it can exceed cookie limits.
            session['label_encoders_classes'] = {k: list(v.classes_) for k, v in analytics_model.label_encoders.items()}
            session['processed_columns'] = available_features # Store column names if needed for frontend logic


            return jsonify({
                "message": "File uploaded and processed successfully! Data is ready for exploration.",
                "features": available_features # Still useful for displaying columns if needed
            })
        except ValueError as e:
            app.logger.warning("ValueError during file upload: %s", str(e))
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.error("Error during file upload: %s", str(e), exc_info=True)
            return jsonify({"error": f"An unexpected error occurred during processing: {e}"}), 500
    return jsonify({"error": "Something went wrong"}), 500

@app.route('/get_data_summary', methods=['GET'])
def get_data_summary():
    """Returns a summary of the uploaded and processed data."""
    # Directly check if the processed_df exists in the global analytics_model instance
    if analytics_model.processed_df is None:
        return jsonify({"error": "No data available. Please upload a dataset first."}), 400
    
    try:
        summary = {
            "rows": len(analytics_model.processed_df),
            "columns": len(analytics_model.processed_df.columns),
            # Convert dtypes to string before sending to JSON
            "data_types": analytics_model.processed_df.dtypes.astype(str).to_dict(),
            "first_5_rows": analytics_model.processed_df.head().to_dict(orient='records')
        }
        return jsonify({"summary": summary})
    except Exception as e:
        app.logger.error("Error getting data summary: %s", str(e), exc_info=True)
        return jsonify({"error": f"An unexpected error occurred while summarizing data: {e}"}), 500

@app.route('/get_correlations', methods=['GET'])
def get_correlations():
    """Returns the correlation matrix for numerical features in the processed data."""
    # Directly check if the processed_df exists in the global analytics_model instance
    if analytics_model.processed_df is None:
        return jsonify({"error": "No data available. Please upload a dataset first."}), 400
    
    try:
        numerical_df = analytics_model.processed_df.select_dtypes(include=np.number)
        if numerical_df.empty:
            return jsonify({"correlations": {}, "message": "No numerical features for correlation analysis."})

        correlation_matrix = numerical_df.corr().to_dict()
        return jsonify({"correlations": correlation_matrix})
    except Exception as e:
        app.logger.error("Error getting correlations: %s", str(e), exc_info=True)
        return jsonify({"error": f"An unexpected error occurred while calculating correlations: {e}"}), 500

@app.route('/get_feature_details', methods=['GET'])
def get_feature_details():
    """
    Returns detailed information about all processed features,
    including their type and categories for encoded ones.
    """
    if analytics_model.processed_df is None:
        return jsonify({"error": "No data available. Please upload a dataset first."}), 400

    # Reload label encoders from session, as they are needed for category lookup
    if 'label_encoders_classes' in session:
        analytics_model.label_encoders = {k: LabelEncoder().fit(v) for k, v in session['label_encoders_classes'].items()}

    feature_details = []
    # Use the columns from the processed DataFrame directly
    for feature in analytics_model.processed_df.columns.tolist():
        feature_type = 'unknown' # Default
        original_name = feature
        categories = []

        if feature.endswith('_timestamp'):
            feature_type = 'datetime'
            original_name = feature.replace('_timestamp', '')
        elif feature.endswith('_encoded'):
            original_name = feature.replace('_encoded', '')
            if original_name in analytics_model.label_encoders:
                feature_type = 'categorical'
                categories = list(analytics_model.label_encoders[original_name].classes_)
            else:
                feature_type = 'encoded_numerical' # Fallback if encoder not found for some reason
        elif feature in analytics_model.processed_df.columns:
            if analytics_model.processed_df[feature].dtype == 'int64':
                feature_type = 'integer'
            elif analytics_model.processed_df[feature].dtype == 'float64':
                feature_type = 'float'
            elif analytics_model.processed_df[feature].dtype == 'object':
                feature_type = 'string'
            elif analytics_model.processed_df[feature].dtype == 'bool':
                feature_type = 'boolean'

        feature_details.append({
            "name": original_name,
            "processed_name": feature,
            "type": feature_type,
            "categories": categories
        })
    return jsonify({"feature_details": feature_details})


# Error handlers for common HTTP errors
@app.errorhandler(404)
def not_found_error(error):
    app.logger.error("404 Not Found: %s", request.url)
    return jsonify({"error": "Resource not found. Please check the URL."}), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    app.logger.error("405 Method Not Allowed: %s for %s", request.method, request.url)
    return jsonify({"error": "Method not allowed for this URL. Please check the request method (e.g., GET vs. POST)."}), 405


@app.route('/logout')
def logout():
    """Logs out the user."""
    session.pop('user_id', None)
    session.pop('full_name', None)
    session.pop('email', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
