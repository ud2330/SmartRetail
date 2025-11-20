import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import pickle
import os

# --- Configuration for Offer Recommendation Model ---
# The original GForm question name for the target, used for identification
TARGET_COLUMN_OFFERS_PHRASE = 'Which offer types do you prefer?'
TARGET_COLUMN_OFFERS_RENAMED = 'Offer_Types_Preferred'

# List of features to be used for the offer recommendation model (REDUCED SET)
# These are the ORIGINAL GForm questions based on feature importance analysis.
OFFER_RECOMMENDATION_FEATURES_ORIGINAL = [
    'Household Size (Number input)',
    'How much do you typically spend on groceries per month? (Numeric input)',
    'How much time do you usually spend during one shopping trip? (in minutes) (Numeric input)',
    'Which product categories do you frequently buy? (Select all that apply)',
    'Do you prefer shopping when offers are available?',
    'Do offers influence your purchase decision?',
    'By what percentage do you typically increase the amount of money spent compared to your usual purchases? (For example, enter 20 if you spend 20% more during offers.)',
    'Which categories do you prefer to buy during offers? (Select all that apply)',
    'How would you describe your shopping style?',
    'On a scale of 1–5, how loyal are you to your preferred brands? (Ratings: 1 = Not at all, 5 = Very loyal)',
    'Do you consider eco-friendliness when choosing a product?',
    'How likely are you to switch to online shopping in the next 12 months? (% probability) (Numeric input: 0 to 100)'
]

try:
    # --- Step 1: Load the Dataset ---
    df = pd.read_csv('Customer Experience Dataset.csv')
    print("Dataset loaded successfully!")

    # --- Step 2: Clean and Rename Columns for Easier Access ---
    # Strip whitespace from column names to handle potential inconsistencies
    df.columns = df.columns.str.strip()

    # Dynamically find the exact target column name after stripping
    actual_target_column_in_df = None
    for col in df.columns:
        if col.startswith(TARGET_COLUMN_OFFERS_PHRASE):
            actual_target_column_in_df = col
            break
    
    if actual_target_column_in_df is None:
        raise KeyError(f"Target column starting with '{TARGET_COLUMN_OFFERS_PHRASE}' not found in DataFrame. Please check the CSV column names.")

    # Create a mapping from cleaned original GForm question names to internal renamed names
    column_rename_map = {
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
        # Use the dynamically found column name for the target
        actual_target_column_in_df: TARGET_COLUMN_OFFERS_RENAMED,
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

    # Build the rename dictionary based on columns actually present in the DataFrame
    final_rename_dict = {}
    for original_col_name, new_col_name in column_rename_map.items():
        if original_col_name in df.columns: # Check against the stripped column names
            final_rename_dict[original_col_name] = new_col_name
        else:
            print(f"Warning: Original column '{original_col_name}' not found in DataFrame. Skipping renaming for this column.")

    df.rename(columns=final_rename_dict, inplace=True)

    # Ensure the target column exists *after* renaming
    if TARGET_COLUMN_OFFERS_RENAMED not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN_OFFERS_RENAMED}' (original: '{actual_target_column_in_df}') not found in DataFrame after renaming. This should not happen if previous checks passed.")

    # Filter the DataFrame to only include relevant features and the target
    # Ensure that only features that were successfully renamed and are in the dataframe are selected.
    # Updated to use the REFINED list for feature selection
    adjusted_features_for_selection = [column_rename_map[col] for col in OFFER_RECOMMENDATION_FEATURES_ORIGINAL if col in column_rename_map and column_rename_map[col] in df.columns]
    
    # Add the target column, ensuring it's in the dataframe
    if TARGET_COLUMN_OFFERS_RENAMED in df.columns:
        adjusted_features_for_selection.append(TARGET_COLUMN_OFFERS_RENAMED)
    
    df_offer_reco = df[adjusted_features_for_selection].copy()

    # --- Step 3: Preprocessing for Offer Recommendation Model ---

    # 3.1 Handle Numerical and Categorical columns for input features (X)
    # Update these lists to reflect only the selected features from OFFER_RECOMMENDATION_FEATURES_ORIGINAL
    numerical_features = [
        'Household_Size',
        'Monthly_Grocery_Spend',
        'Shopping_Trip_Duration',
        'Offer_Spend_Increase_Percentage',
        'Likely_Switch_Online_Shopping',
        'Brand_Loyalty' # Kept as numerical
    ]
    categorical_features = [
        'Prefer_Offers',
        'Offers_Influence_Purchase',
        'Shopping_Style',
        'Eco_Friendliness'
    ]
    
    # Filter numerical features to only include those present after initial df selection
    numerical_features = [col for col in numerical_features if col in df_offer_reco.columns]
    categorical_features = [col for col in categorical_features if col in df_offer_reco.columns]


    # Convert 'Brand_Loyalty' to numeric, handling potential non-numeric entries or ranges
    def parse_brand_loyalty(value):
        if pd.isna(value):
            return np.nan
        try:
            return int(str(value).split(' ')[0]) # Extract the number part
        except ValueError:
            return np.nan
    
    if 'Brand_Loyalty' in df_offer_reco.columns:
        df_offer_reco.loc[:, 'Brand_Loyalty'] = df_offer_reco['Brand_Loyalty'].apply(parse_brand_loyalty)


    # 3.2 Handle multi-select columns by creating binary (0/1) features
    # These will be treated as numerical columns in the ColumnTransformer later
    # Update this list based on the reduced features
    multiselect_cols_to_expand_X = [
        'Frequent_Categories', # This was 'Which product categories do you frequently buy?'
        'Categories_Buy_Offers' # This was 'Which categories do you prefer to buy during offers?'
    ]
    expanded_multi_select_cols_X = []

    for col in multiselect_cols_to_expand_X:
        if col in df_offer_reco.columns:
            # Fill NaN values with empty string to avoid errors during split
            df_offer_reco.loc[:, col] = df_offer_reco[col].fillna('')
            # Get all unique options by splitting comma-separated strings
            all_options = df_offer_reco[col].str.split(', ').explode().unique()
            all_options = [opt for opt in all_options if opt and opt != 'Not Applicable'] # Filter out empty and 'Not Applicable'
            for option in all_options:
                safe_option_name = option.replace(" ", "_").replace("&", "and").replace(",", "").replace("/", "_").replace("–", "_").replace("(", "").replace(")", "").replace("e.g.", "").replace("%", "").strip()
                new_col_name = f'{col}_{safe_option_name}'
                df_offer_reco.loc[:, new_col_name] = df_offer_reco[col].apply(lambda x: 1 if option in x else 0)
                expanded_multi_select_cols_X.append(new_col_name) # Add to our list
            df_offer_reco.drop(columns=[col], inplace=True) # Drop original multi-select column

    # Add the newly created binary multi-select columns to numerical features
    numerical_features.extend(expanded_multi_select_cols_X)

    # Ensure all numerical features are coerced to numeric, filling NaNs with median
    for col in numerical_features:
        if col in df_offer_reco.columns:
            df_offer_reco.loc[:, col] = pd.to_numeric(df_offer_reco[col], errors='coerce')
            df_offer_reco.loc[:, col] = df_offer_reco[col].fillna(df_offer_reco[col].median())

    # Ensure all categorical features are treated as string and fill NaNs with 'Unknown'
    for col in categorical_features:
        if col in df_offer_reco.columns:
            df_offer_reco.loc[:, col] = df_offer_reco[col].astype(str).fillna('Unknown')


    # --- Step 4: Prepare the Target Variable (y) ---
    # The target `Offer_Types_Preferred` is a multi-select, so we'll use MultiLabelBinarizer.
    df_offer_reco.loc[:, TARGET_COLUMN_OFFERS_RENAMED] = df_offer_reco[TARGET_COLUMN_OFFERS_RENAMED].fillna('')
    # Split the comma-separated strings into lists of individual preferences
    df_offer_reco.loc[:, 'Offer_Types_Preferred_List'] = df_offer_reco[TARGET_COLUMN_OFFERS_RENAMED].apply(
        lambda x: [item.strip() for item in x.split(',')] if x else []
    )
    # Remove empty strings from the lists, which might result from `fillna('')`
    df_offer_reco.loc[:, 'Offer_Types_Preferred_List'] = df_offer_reco['Offer_Types_Preferred_List'].apply(
        lambda x: [item for item in x if item]
    )

    # Initialize MultiLabelBinarizer on the full dataset to ensure it learns all possible classes
    mlb = MultiLabelBinarizer()
    y_mlb = mlb.fit_transform(df_offer_reco['Offer_Types_Preferred_List'])
    y = pd.DataFrame(y_mlb, columns=mlb.classes_, index=df_offer_reco.index)
    print(f"\nTarget variable transformed into {len(mlb.classes_)} binary columns: {mlb.classes_.tolist()}")

    # --- Step 5: Separate Features (X) and Target (y) ---
    # X includes all processed input features, y includes the binarized offer types.
    X = df_offer_reco.drop(columns=[TARGET_COLUMN_OFFERS_RENAMED, 'Offer_Types_Preferred_List'])
    
    # Filter X to ensure only the features we've prepared are present
    X_cols_final = [col for col in numerical_features + categorical_features if col in X.columns]
    X = X[X_cols_final]

    print(f"\nShape of X: {X.shape}, Shape of y: {y.shape}")
    print(f"X columns after manual expansion and selection: {X.columns.tolist()}")


    # --- Step 6: Define Preprocessing Steps with ColumnTransformer for X ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # handle_unknown='ignore' is crucial here for deployment to handle unseen categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any columns not explicitly handled
    )

    # --- Step 7: Create the Model Pipeline ---
    # Use MultiOutputClassifier with RandomForestClassifier as the estimator
    offer_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42, n_estimators=100)))
    ])

    # --- Step 8: Split Data into Training and Testing Sets ---
    # No stratification needed for multi-label, but random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --- Step 9: Train the Model ---
    print("Training the Multi-Output Random Forest Classifier for Offer Recommendation...")
    offer_model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- Step 10: Evaluate the Model (Optional, but good practice) ---
    # For multi-label, evaluation metrics are more complex (e.g., Jaccard, Hamming loss, F1-score)
    # Simple accuracy can be misleading. Let's just predict for demonstration.
    train_pred_proba = offer_model_pipeline.predict_proba(X_train)
    test_pred_proba = offer_model_pipeline.predict_proba(X_test)

    print("\nModel trained. Ready for saving.")

    # --- Step 11: Calculate and Display Feature Importances ---
    print("\nCalculating Feature Importances for Offer Recommendation Model:")
    fitted_preprocessor = offer_model_pipeline.named_steps['preprocessor']

    dummy_input_for_feature_names = X_train.head(1)
    transformed_dummy = fitted_preprocessor.transform(dummy_input_for_feature_names)
    preprocessed_feature_names = fitted_preprocessor.get_feature_names_out()

    classifier = offer_model_pipeline.named_steps['classifier']
    summed_importances = np.zeros(len(preprocessed_feature_names))

    for estimator in classifier.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            summed_importances += estimator.feature_importances_
        else:
            print("Warning: An estimator in MultiOutputClassifier does not have 'feature_importances_'. Skipping.")

    feature_importances_series = pd.Series(summed_importances, index=preprocessed_feature_names)
    sorted_feature_importances = feature_importances_series.sort_values(ascending=False)

    print("\nTop 20 Feature Importances for Offer Recommendation (Summed Across Outputs):")
    print(sorted_feature_importances.head(20))


    # --- Step 12: Save the Trained Model and MultiLabelBinarizer ---
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_filename = os.path.join(models_dir, 'offer_rf_model.pkl')
    mlb_filename = os.path.join(models_dir, 'offer_mlb.pkl') # Save the MLB to transform predictions back

    with open(model_filename, 'wb') as file:
        pickle.dump(offer_model_pipeline, file)
    print(f"Offer recommendation model saved as '{model_filename}'")

    with open(mlb_filename, 'wb') as file:
        pickle.dump(mlb, file)
    print(f"MultiLabelBinarizer for offers saved as '{mlb_filename}'")

except FileNotFoundError:
    print("Error: 'Customer Experience Dataset.csv' not found. Please ensure the file is uploaded correctly.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
