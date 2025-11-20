import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# --- Configuration for stock-out model (REFINED FEATURES) ---
TARGET_COLUMN_STOCKOUT = 'Have you faced stock-outs while shopping?'

# Refined list of features for stock-out prediction
# These are the ORIGINAL GForm questions.
REFINED_STOCKOUT_FEATURES_ORIGINAL = [
    'How often do you shop at supermarkets?',
    'How much do you typically spend on groceries per month? (Numeric input)',
    'Which product categories do you frequently buy? (Select all that apply)',
    'Have you faced stock-outs while shopping?', # Target, but used for conditional logic
    '(If yes) How frequently do you find your preferred items out of stock?',
    '(If yes) How do you react when an item is out of stock?',
    '(If yes) Which product categories are often out of stock in your experience? (Select all that apply)',
    'Do you check online for availability before visiting a store?',
    'Are you willing to pre-order if items are out of stock?',
    'What is the acceptable wait time for restocking (in days)? (Numeric input)',
    'Would you switch to another store if your preferred items are frequently out of stock?'
]

try:
    # --- Step 1: Load the Dataset ---
    # Ensure 'Customer Experience Dataset.csv' is in the same directory as this script.
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
        'Have you faced stock-outs while shopping?': 'Faced_Stockouts', # Target variable
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

    # Filter to only the relevant features for the stockout model
    adjusted_refined_stockout_features = [column_mapping[col] for col in REFINED_STOCKOUT_FEATURES_ORIGINAL if col in column_mapping]
    df_stockout_refined = df[adjusted_refined_stockout_features].copy()

    # --- Step 3: Preprocessing for Stock-Out Prediction Model ---

    # 3.1 Handle the target variable: 'Faced_Stockouts'
    df_stockout_refined['Faced_Stockouts'] = df_stockout_refined['Faced_Stockouts'].map({'Yes': 1, 'No': 0})
    print(f"\nValue counts for target variable 'Faced_Stockouts':\n{df_stockout_refined['Faced_Stockouts'].value_counts()}")

    # 3.2 Handle conditional missing values for stock-out related features.
    conditional_features_info = {
        'Stockout_Frequency': 'category',
        'Stockout_Reaction': 'category',
        'Stockout_Categories': 'multiselect', # Still needs initial processing here
        'Acceptable_Restock_Wait_Time': 'numeric'
    }

    # Ensure all multi-select columns are handled consistently for the options list
    # This list must match the one in app.py
    ALL_FREQUENT_CATEGORIES_OPTIONS = ['Groceries', 'Snacks', 'Dairy', 'Beverages', 'Personal Care', 'Household Items', 'Fruits & Vegetables', 'Packaged Food', 'Other']
    ALL_STOCKOUT_CATEGORIES_OPTIONS = ['Groceries', 'Snacks', 'Dairy', 'Beverages', 'Personal Care', 'Household Items', 'Fruits', 'Vegetables', 'Other']

    # Update df_stockout_refined based on conditional logic *before* expanding multi-selects
    # This ensures "Not Applicable" is correctly filled before processing
    for col, col_type in conditional_features_info.items():
        if col in df_stockout_refined.columns:
            if col_type == 'category' or col_type == 'multiselect':
                # If 'Faced_Stockouts' is No, set corresponding conditional features to 'Not Applicable'
                df_stockout_refined.loc[df_stockout_refined['Faced_Stockouts'] == 0, col] = 'Not Applicable'
                # For any remaining NaNs (which would be in 'Yes' cases), fill with mode
                df_stockout_refined[col] = df_stockout_refined[col].fillna(df_stockout_refined[col].mode()[0])
            elif col_type == 'numeric':
                # If 'Faced_Stockouts' is No, set numerical conditional feature to 0
                df_stockout_refined.loc[df_stockout_refined['Faced_Stockouts'] == 0, col] = 0
                # For any remaining NaNs (which would be in 'Yes' cases), fill with median
                df_stockout_refined[col] = df_stockout_refined[col].fillna(df_stockout_refined[col].median())


    # 3.3 Manually expand multiselect columns into binary (0/1) features
    # These new columns will be added to `numerical_cols_final` to prevent double-encoding.
    multiselect_cols_to_expand = {
        'Frequent_Categories': ALL_FREQUENT_CATEGORIES_OPTIONS,
        'Stockout_Categories': ALL_STOCKOUT_CATEGORIES_OPTIONS
    }
    expanded_multi_select_cols = [] # To store names of newly created binary columns

    for col, options_list in multiselect_cols_to_expand.items():
        if col in df_stockout_refined.columns:
            df_stockout_refined[col] = df_stockout_refined[col].fillna('') # Fill NaN for consistent splitting
            
            print(f"\nExpanding multiselect column: '{col}'. Using options: {options_list}")
            for option in options_list:
                safe_option_name = option.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_")
                new_col_name = f'{col}_{safe_option_name}'
                # Check if the option is present in the comma-separated string
                df_stockout_refined[new_col_name] = df_stockout_refined[col].apply(lambda x: 1 if option in x else 0)
                expanded_multi_select_cols.append(new_col_name) # Add to our list
            df_stockout_refined.drop(columns=[col], inplace=True) # Drop original multiselect column

    # --- Step 4: Separate Features (X) and Target (y) ---
    X = df_stockout_refined.drop(columns='Faced_Stockouts')
    y = df_stockout_refined['Faced_Stockouts']

    # --- Step 5: Identify Final Numerical and Categorical Features for the Pipeline ---
    # IMPORTANT: Include the newly expanded multi-select columns as NUMERICAL features.
    numerical_cols_final = [
        'Monthly_Grocery_Spend',
        'Acceptable_Restock_Wait_Time'
    ] + expanded_multi_select_cols # ADDING EXPANDED BINARY FEATURES HERE

    numerical_cols_final = [col for col in numerical_cols_final if col in X.columns]

    # Categorical columns are now ONLY the single-value categorical features that haven't been manually expanded.
    categorical_cols_final = [col for col in X.columns if col not in numerical_cols_final]

    print(f"\nFinal numerical features for preprocessing (including expanded multi-selects): {numerical_cols_final}")
    print(f"Final categorical features for preprocessing (single-value only): {categorical_cols_final}")
    print(f"Total features for X after expansion and selection: {len(X.columns)}")

    # --- Step 6: Define Preprocessing Steps with ColumnTransformer ---
    # Numerical transformer: handles imputation for all numerical features (including the 0/1 multi-selects)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical transformer: handles imputation and one-hot encoding for the remaining single-value categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # MODIFIED: Set sparse_output=False for OneHotEncoder
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols_final),
            ('cat', categorical_transformer, categorical_cols_final)
        ],
        remainder='drop', # Drop any columns not explicitly handled
        verbose_feature_names_out=False # Ensure feature names are propagated (default in recent scikit-learn)
    ).set_output(transform="pandas") # Explicitly set output to pandas DataFrame

    # --- Step 7: Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --- Step 8: Create the Model Pipeline ---
    stockout_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ]).set_output(transform="pandas") # Also set output for the entire pipeline

    # --- Step 9: Train the Model ---
    print("Training the Random Forest Classifier for Stock-Out Prediction with refined features...")
    stockout_model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- Step 10: Evaluate the Model ---
    train_accuracy = stockout_model_pipeline.score(X_train, y_train)
    test_accuracy = stockout_model_pipeline.score(X_test, y_test)
    print(f"\nTraining Accuracy (refined features): {train_accuracy:.4f}")
    print(f"Test Accuracy (refined features): {test_accuracy:.4f}")

    # --- Step 11: Save the Trained Model ---
    if not os.path.exists('models'):
        os.makedirs('models')
    model_filename = 'models/stockout_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(stockout_model_pipeline, file)
    print(f"Refined Stock-out prediction model saved as '{model_filename}'")

except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e}. Please ensure 'Customer Experience Dataset.csv' is in the script directory.")
except Exception as e:
    print(f"An unexpected error occurred during model training: {e}")
    import traceback
    traceback.print_exc()
