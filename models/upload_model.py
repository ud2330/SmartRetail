import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import io

class RetailAnalyticsModel:
    def __init__(self):
        self.processed_df = None
        self.label_encoders = {}
        self.target_encoder = None # Specific encoder for the target column if it's categorical
        self.model = None
        self.scaler = None
        self.X_columns = None
        self.is_classification = False
        self.target_column = None

    def load_and_preprocess_data(self, file_content):
        """
        Loads the dataset from file content (bytes), performs initial preprocessing,
        and stores the processed DataFrame and encoders.
        """
        print("Loading and preprocessing data...")
        try:
            # Use io.StringIO to read string content as a file
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        except Exception as e:
            raise ValueError(f"An error occurred while loading the dataset: {e}")

        print("\n--- Original Dataset Info ---")
        df.info()
        print("\nFirst 5 rows of the dataset:")
        print(df.head())

        columns_to_drop_after_processing = []
        current_label_encoders = {} # Local encoders for this processing run

        # Handle Date/Time columns first
        print("\n--- Handling Date/Time Columns ---")
        for column in df.columns:
            temp_series = pd.to_datetime(df[column], errors='coerce')

            if not temp_series.isna().all() and df[column].dtype == 'object' and temp_series.count() / len(df) > 0.5:
                # Only consider it a datetime column if more than 50% successfully convert
                print(f"Detected potential date/time column: '{column}'. Converting to Unix timestamp.")
                df[f'{column}_timestamp'] = temp_series.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
                columns_to_drop_after_processing.append(column)
                if df[f'{column}_timestamp'].isnull().any():
                    median_timestamp = df[f'{column}_timestamp'].median()
                    df[f'{column}_timestamp'].fillna(median_timestamp, inplace=True)
                    print(f"Filled missing timestamps in '{column}_timestamp' with median: {median_timestamp}")
            else:
                print(f"'{column}' is not identified as a date/time column or is already numerical/boolean or has too many NaT values.")

        # Handle missing values: Fill numerical with median, categorical with mode
        print("\n--- Handling Missing Values ---")
        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype in ['int64', 'float64']:
                    median_val = df[column].median()
                    df[column].fillna(median_val, inplace=True)
                    print(f"Filled missing values in numerical column '{column}' with median: {median_val}")
                elif df[column].dtype == 'object':
                    mode_val = df[column].mode()[0]
                    df[column].fillna(mode_val, inplace=True)
                    print(f"Filled missing values in categorical column '{column}' with mode: {mode_val}")

        # Encode remaining categorical features
        print("\n--- Encoding Categorical Features ---")
        for column in df.select_dtypes(include='object').columns:
            if df[column].nunique() < 50:
                le = LabelEncoder()
                df[f'{column}_encoded'] = le.fit_transform(df[column])
                current_label_encoders[column] = le
                print(f"Encoded categorical column: '{column}'")
                columns_to_drop_after_processing.append(column)
            else:
                print(f"Skipping encoding for '{column}' as it has too many unique values ({df[column].nunique()}).")

        df = df.drop(columns=columns_to_drop_after_processing, errors='ignore')

        print("\n--- Processed Dataset Info ---")
        df.info()
        print("\nFirst 5 rows of processed dataset:")
        print(df.head())

        self.processed_df = df
        self.label_encoders = current_label_encoders
        # Reset model, scaler etc. as new data is loaded
        self.model = None
        self.scaler = None
        self.X_columns = None
        self.is_classification = False
        self.target_column = None
        self.target_encoder = None

        return df.columns.tolist() # Return available features for target selection

    def train_model(self, target_column):
        """
        Trains a RandomForest model based on the processed DataFrame and selected target.
        Stores the trained model, scaler, and column information.
        """
        if self.processed_df is None:
            raise ValueError("No data loaded. Please upload a dataset first.")

        if target_column not in self.processed_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in processed data.")

        self.target_column = target_column

        X = self.processed_df.drop(columns=[target_column])
        y = self.processed_df[target_column]

        # Handle scaling for numerical features in X
        self.scaler = StandardScaler()
        numerical_cols_X = X.select_dtypes(include=np.number).columns
        if not numerical_cols_X.empty:
            X[numerical_cols_X] = self.scaler.fit_transform(X[numerical_cols_X])
            print(f"Scaled numerical features in X for training: {list(numerical_cols_X)}")
        else:
            print("No numerical features to scale in X for training.")

        # Determine model type based on target variable
        self.is_classification = False
        if y.dtype == 'object' or (y.dtype in ['int64', 'float64'] and y.nunique() < 0.1 * len(y) and y.nunique() <= 50):
            print("Target variable appears to be categorical/discrete. Using RandomForestClassifier.")
            self.is_classification = True
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)
                print(f"Encoded target column '{target_column}'.")
        else:
            print("Target variable appears to be numerical. Using RandomForestRegressor.")

        # Split data into training and testing sets
        if len(X) < 2:
            raise ValueError("Not enough samples in dataset to train a model.")
        
        # Ensure that y has at least two unique classes for classification if it's classification
        if self.is_classification and y.nunique() < 2:
            raise ValueError(f"Target column '{target_column}' has less than two unique classes. Cannot train a classification model.")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nData split: Training samples={len(X_train)}, Testing samples={len(X_test)}")

        if self.is_classification:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        print("Training the model...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        if self.is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy on Test Set: {accuracy:.4f}")
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"Model RMSE on Test Set: {rmse:.4f}")

        self.X_columns = X.columns.tolist()
        return {
            "model_ready": True,
            "prediction_features": self.X_columns,
            "is_classification": self.is_classification
        }

    def predict_new_data(self, new_input_data_dict):
        """
        Makes a prediction using the trained model on new input data.
        """
        if self.model is None or self.scaler is None or self.X_columns is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Convert input dictionary to DataFrame
        # Ensure all expected columns are present, fill missing with 0 (or a more sophisticated imputation)
        new_input_df = pd.DataFrame([new_input_data_dict])
        
        # Identify numerical columns that need scaling based on the original X
        numerical_cols_X_trained = [col for col in self.X_columns if self.processed_df[col].dtype in ['int64', 'float64']]


        # Process categorical and timestamp features in the new input
        processed_input_data = {}
        for col_name in self.X_columns:
            if col_name.endswith('_timestamp'):
                original_col = col_name.replace('_timestamp', '')
                try:
                    date_val = new_input_data_dict.get(original_col) # Get original date string input
                    if date_val:
                        dt_obj = pd.to_datetime(date_val, errors='coerce')
                        if pd.notna(dt_obj):
                            processed_input_data[col_name] = dt_obj.timestamp()
                        else:
                            # If date parsing fails, use median from training data if available
                            median_val = self.processed_df[col_name].median() if col_name in self.processed_df.columns else 0
                            processed_input_data[col_name] = median_val
                            print(f"Warning: Could not parse date for '{original_col}'. Using median timestamp: {median_val}")
                    else:
                        # If date value is missing, use median from training data
                        median_val = self.processed_df[col_name].median() if col_name in self.processed_df.columns else 0
                        processed_input_data[col_name] = median_val
                        print(f"Warning: Missing date for '{original_col}'. Using median timestamp: {median_val}")
                except Exception as e:
                    median_val = self.processed_df[col_name].median() if col_name in self.processed_df.columns else 0
                    processed_input_data[col_name] = median_val
                    print(f"Error processing date for '{original_col}': {e}. Using median timestamp: {median_val}")
            elif col_name.endswith('_encoded'):
                original_col = col_name.replace('_encoded', '')
                if original_col in self.label_encoders:
                    le = self.label_encoders[original_col]
                    input_val = new_input_data_dict.get(original_col)
                    if input_val is not None and input_val in le.classes_:
                        processed_input_data[col_name] = le.transform([input_val])[0]
                    else:
                        # If input value is not in known classes or missing, use a default (e.g., first class)
                        default_val = le.transform([le.classes_[0]])[0] if len(le.classes_) > 0 else 0
                        processed_input_data[col_name] = default_val
                        print(f"Warning: Input '{input_val}' for '{original_col}' not recognized. Using default: {le.inverse_transform([default_val])[0] if len(le.classes_) > 0 else 'N/A'}")
                else:
                    # If encoder not found, assume 0 or handle error
                    processed_input_data[col_name] = 0
            else:
                # For direct numerical or boolean columns
                processed_input_data[col_name] = new_input_data_dict.get(col_name, 0) # Default to 0 if not provided

        processed_input_df = pd.DataFrame([processed_input_data])
        processed_input_df = processed_input_df[self.X_columns] # Ensure column order


        # Apply scaling to numerical features in the new input
        if not numerical_cols_X_trained == []: # Check if the list is not empty
            processed_input_df[numerical_cols_X_trained] = self.scaler.transform(processed_input_df[numerical_cols_X_trained])
            print("Scaled numerical features in new input for prediction.")
        else:
            print("No numerical features to scale in new input for prediction.")

        prediction = self.model.predict(processed_input_df)

        if self.is_classification:
            # Inverse transform if target was encoded
            if self.target_encoder:
                predicted_value = self.target_encoder.inverse_transform(prediction.round().astype(int))[0]
            else:
                predicted_value = prediction[0]
            return {"prediction": str(predicted_value), "type": "classification"}
        else:
            return {"prediction": float(prediction[0]), "type": "regression"}
