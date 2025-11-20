import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
import os
from sklearn import set_config # Import set_config

# Set global output for transformers to pandas DataFrames
set_config(transform_output="pandas")

# --- Configuration for Sales Prediction Model ---
TARGET_COLUMN_SALES = 'Monthly_Grocery_Spend' # Target variable for sales prediction

# Define all possible categories for multi-selects to ensure consistent column creation
ALL_FREQUENT_CATEGORIES_OPTIONS = ['Groceries', 'Snacks', 'Dairy', 'Beverages', 'Personal Care', 'Household Items', 'Fruits & Vegetables', 'Packaged Food', 'Other']
ALL_OFFER_TYPES_PREFERRED_OPTIONS = ['Flat Discount', 'Buy 1 Get 1 Free', 'Cashback', 'Loyalty Points', 'Free Gifts']
ALL_CATEGORIES_BUY_OFFERS_OPTIONS = ['Snacks', 'Beverages', 'Personal Care', 'Household Items', 'Groceries', 'Fruits & Vegetables', 'Other']


try:
    # --- Step 1: Load the Dataset ---
    df = pd.read_csv('Customer Experience Dataset.csv')
    print("Dataset loaded successfully!")

    # --- Step 2: Rename Columns for Easier Access ---
    column_mapping = {
        'Timestamp': 'Timestamp',
        'Age Group': 'Age_Group',
        'Gender': 'Gender',
        'City or Town of Residence': 'City_Town_Residence',
        'Monthly Household Income Range': 'Monthly_Income_Range',
        'Household Size (Number input)': 'Household_Size',
        'Access to Vehicle': 'Access_Vehicle',
        'How often do you shop at supermarkets?': 'Shopping_Frequency',
        'When do you usually shop?': 'Shopping_Time',
        'How much do you typically spend on groceries per month? (Numeric input)': 'Monthly_Grocery_Spend',
        'Preferred payment method': 'Payment_Method',
        'Do you use any price comparison or shopping apps before buying?': 'Use_Shopping_Apps',
        'Are you a loyalty card member for any store?': 'Loyalty_Card_Member',
        'How much time do you usually spend during one shopping trip? (in minutes) (Numeric input)': 'Shopping_Trip_Duration',
        'Which product categories do you frequently buy? (Select all that apply)': 'Frequent_Categories',
        'Have you faced stock-outs while shopping?': 'Faced_Stockouts',
        '(If yes) How frequently do you find your preferred items out of stock?': 'Stockout_Frequency',
        '(If yes) How do you react when an item is out of stock?': 'Stockout_Reaction',
        '(If yes) Which product categories are often out of stock in your experience? (Select all that apply)': 'Stockout_Categories',
        'Do you check online for availability before visiting a store?': 'Check_Online_Availability',
        'Are you willing to pre-order if items are out of stock?': 'Willing_Preorder',
        'What is the acceptable wait time for restocking (in days)? (Numeric input)': 'Acceptable_Restock_Wait_Time',
        'Would you switch to another store if your preferred items are frequently out of stock?': 'Switch_Store_Stockout',
        'Do you prefer shopping when offers are available?': 'Prefer_Offers',
        'Which offer types do you prefer? (Select all that apply)': 'Offer_Types_Preferred',
        'Do offers influence your purchase decision?': 'Offers_Influence_Purchase',
        'By what percentage do you typically increase the amount of money spent compared to your usual purchases? (For example, enter 20 if you spend 20% more during offers.)': 'Offer_Spend_Increase_Percentage',
        'Do you shop more frequently during offers?': 'Shop_More_During_Offers',
        'Which categories do you prefer to buy during offers? (Select all that apply)': 'Categories_Buy_Offers',
        'What influences your purchase the most?': 'Purchase_Influence',
        'How would you describe your shopping style?': 'Shopping_Style',
        'On a scale of 1–5, how loyal are you to your preferred brands? (Ratings: 1 = Not at all, 5 = Very loyal)': 'Brand_Loyalty',
        'Do you consider eco-friendliness when choosing a product?': 'Eco_Friendliness',
        'How likely are you to switch to online shopping in the next 12 months? (% probability) (Numeric input: 0 to 100)': 'Likely_Switch_Online_Shopping'
    }
    df.rename(columns=column_mapping, inplace=True)

    # --- Step 3: Feature Engineering / Data Cleaning specific to Sales Model ---

    # Convert 'Brand_Loyalty' to numeric, handling potential non-numeric entries or ranges
    def parse_brand_loyalty(value):
        if pd.isna(value):
            return np.nan
        try:
            return int(str(value).split(' ')[0]) # Extract the number part
        except ValueError:
            return np.nan
    df['Brand_Loyalty'] = df['Brand_Loyalty'].apply(parse_brand_loyalty)

    # Convert 'Likely_Switch_Online_Shopping' to numeric, handling potential non-numeric entries
    df['Likely_Switch_Online_Shopping'] = pd.to_numeric(df['Likely_Switch_Online_Shopping'], errors='coerce')


    # 3.1 Manually expand multiselect columns into binary (0/1) features
    multiselect_cols_to_expand = {
        'Frequent_Categories': ALL_FREQUENT_CATEGORIES_OPTIONS,
        'Offer_Types_Preferred': ALL_OFFER_TYPES_PREFERRED_OPTIONS,
        'Categories_Buy_Offers': ALL_CATEGORIES_BUY_OFFERS_OPTIONS
    }
    expanded_multi_select_cols = []

    for col, options_list in multiselect_cols_to_expand.items():
        if col in df.columns:
            df[col] = df[col].fillna('') # Fill NaN for consistent splitting
            print(f"\nExpanding multiselect column: '{col}'.")
            for option in options_list:
                safe_option_name = option.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip() # Added hyphen, parenthesis, etc.
                new_col_name = f'{col}_{safe_option_name}'
                df[new_col_name] = df[col].apply(lambda x: 1 if option in x else 0)
                expanded_multi_select_cols.append(new_col_name)
            df.drop(columns=[col], inplace=True) # Drop original multiselect column

    # --- Step 4: Separate Features (X) and Target (y) ---
    # Ensure target column is numeric
    df[TARGET_COLUMN_SALES] = pd.to_numeric(df[TARGET_COLUMN_SALES], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN_SALES], inplace=True) # Drop rows where target is NaN

    # MODIFIED: Reduced set of features based on importance analysis
    X_features_to_keep_original_names = [
        'Age_Group', 'City_Town_Residence', 'Monthly_Income_Range',
        'Household_Size', 'Shopping_Frequency', 'Shopping_Trip_Duration',
        'Frequent_Categories', # Will be expanded
        'Prefer_Offers', 'Offer_Types_Preferred', # Will be expanded
        'Offers_Influence_Purchase', 'Offer_Spend_Increase_Percentage',
        'Categories_Buy_Offers', # Will be expanded
        'Purchase_Influence', 'Likely_Switch_Online_Shopping'
    ]

    # Dynamically select columns present in the DataFrame based on the refined list
    # This ensures that even if 'Frequent_Categories' etc. are already dropped and expanded,
    # the new binary columns are handled correctly.
    X_selected_columns = []
    for col_name in X_features_to_keep_original_names:
        if col_name in df.columns:
            X_selected_columns.append(col_name)
        # Add the expanded binary columns if they exist and are relevant to the original multi-selects
        elif col_name in multiselect_cols_to_expand.keys():
            for option in multiselect_cols_to_expand[col_name]:
                safe_option_name = option.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
                expanded_col_name = f'{col_name}_{safe_option_name}'
                if expanded_col_name in df.columns and expanded_col_name not in X_selected_columns:
                    X_selected_columns.append(expanded_col_name)

    # Ensure all manually expanded columns are explicitly included if they are not already.
    # This covers the case where original multi-select columns were already dropped.
    for col in expanded_multi_select_cols:
        if col in df.columns and col not in X_selected_columns:
            X_selected_columns.append(col)

    X = df[X_selected_columns].copy()
    y = df[TARGET_COLUMN_SALES]
    
    # Identify numerical and categorical features for the ColumnTransformer
    # Numerical columns include Household_Size, Shopping_Trip_Duration, Offer_Spend_Increase_Percentage,
    # Likely_Switch_Online_Shopping, AND all expanded multi-select binary columns.
    numerical_cols = [
        'Household_Size', 'Shopping_Trip_Duration', 'Offer_Spend_Increase_Percentage',
        'Likely_Switch_Online_Shopping'
    ] + [col for col in X.columns if col.startswith(('Frequent_Categories_', 'Offer_Types_Preferred_', 'Categories_Buy_Offers_'))]
    
    numerical_cols = [col for col in numerical_cols if col in X.columns] # Filter to existing columns

    categorical_cols = [col for col in X.columns if col not in numerical_cols]
    
    print(f"\nFeatures identified for sales model:")
    print(f"Numerical: {numerical_cols}")
    print(f"Categorical: {categorical_cols}")
    print(f"Total features for X before preprocessing: {len(X.columns)}")


    # --- Step 5: Define Preprocessing Steps with ColumnTransformer ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False is crucial here
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False # Ensures feature names are propagated
    ).set_output(transform="pandas") # Explicitly set output to pandas DataFrame

    # --- Step 6: Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # No stratify for regression
    print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --- Step 7: Create the Model Pipeline ---
    sales_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
    ]).set_output(transform="pandas") # Also set output for the entire pipeline

    # --- Step 8: Train the Model ---
    print("\nTraining the Random Forest Regressor for Sales Prediction...")
    sales_model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- Step 9: Evaluate the Model ---
    train_score = sales_model_pipeline.score(X_train, y_train) # R2 score
    test_score = sales_model_pipeline.score(X_test, y_test) # R2 score
    print(f"\nTraining R-squared (Sales): {train_score:.4f}")
    print(f"Test R-squared (Sales): {test_score:.4f}")

    # --- Step 10: Calculate and Display Feature Importances ---
    print("\nCalculating Feature Importances...")
    feature_importances = sales_model_pipeline.named_steps['regressor'].feature_importances_
    feature_names = sales_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances_series = pd.Series(feature_importances, index=feature_names)
    sorted_importances = importances_series.sort_values(ascending=False)

    print("\nTop 15 Feature Importances for Sales Prediction (after feature reduction):")
    print(sorted_importances.head(15)) # Display top 15 features for more insight


    # --- Step 11: Save the Trained Model ---
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_filename = os.path.join(models_dir, 'sales_rf_model.pkl')
    with open(model_filename, 'wb') as file:
        pickle.dump(sales_model_pipeline, file)
    print(f"\nSales prediction model saved as '{model_filename}'")

except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e}. Please ensure 'Customer Experience Dataset.csv' is in the script directory.")
except Exception as e:
    print(f"An unexpected error occurred during model training: {e}")
    import traceback
    traceback.print_exc()
