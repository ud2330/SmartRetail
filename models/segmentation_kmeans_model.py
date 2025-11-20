import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# 1. Load data
df = pd.read_csv('Customer Experience Dataset.csv')

# 2. Feature selection (15 features)
features = [
    'Age Group',
    'Gender',
    'Monthly Household Income Range',
    'Household Size (Number input)',
    'Access to Vehicle',
    'How often do you shop at supermarkets?',
    'How much do you typically spend on groceries per month? (Numeric input)',
    'Preferred payment method',
    'Are you a loyalty card member for any store?',
    'Do you prefer shopping when offers are available?',
    'Do offers influence your purchase decision?',
    'By what percentage do you typically increase the amount of money spent compared to your usual purchases? (For example, enter 20 if you spend 20% more during offers.)',
    'How would you describe your shopping style?',
    'On a scale of 1–5, how loyal are you to your preferred brands? (Ratings: 1 = Not at all, 5 = Very loyal)',
    'How likely are you to switch to online shopping in the next 12 months? (% probability) (Numeric input: 0 to 100)',
    'Which product categories do you frequently buy? (Select all that apply)'
]

# 3. Preprocessing
df_selected = df[features].copy()

# Handle multi-select column: 'Which product categories do you frequently buy?'
mlb = MultiLabelBinarizer()
df_selected['Which product categories do you frequently buy? (Select all that apply)'] = (
    df_selected['Which product categories do you frequently buy? (Select all that apply)']
    .fillna('')
    .apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])
)
categories = mlb.fit_transform(df_selected['Which product categories do you frequently buy? (Select all that apply)'])
cat_df = pd.DataFrame(categories, columns=mlb.classes_, index=df_selected.index)
df_selected = df_selected.drop('Which product categories do you frequently buy? (Select all that apply)', axis=1)
df_selected = pd.concat([df_selected, cat_df], axis=1)

# Identify categorical and numeric columns
categorical_cols = [
    'Age Group',
    'Gender',
    'Monthly Household Income Range',
    'Access to Vehicle',
    'How often do you shop at supermarkets?',
    'Preferred payment method',
    'Are you a loyalty card member for any store?',
    'Do you prefer shopping when offers are available?',
    'Do offers influence your purchase decision?',
    'How would you describe your shopping style?'
]
numeric_cols = [
    'Household Size (Number input)',
    'How much do you typically spend on groceries per month? (Numeric input)',
    'By what percentage do you typically increase the amount of money spent compared to your usual purchases? (For example, enter 20 if you spend 20% more during offers.)',
    'On a scale of 1–5, how loyal are you to your preferred brands? (Ratings: 1 = Not at all, 5 = Very loyal)',
    'How likely are you to switch to online shopping in the next 12 months? (% probability) (Numeric input: 0 to 100)'
]
category_cols = list(mlb.classes_)

# Fill missing numeric values with median
for col in numeric_cols:
    df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
    df_selected[col] = df_selected[col].fillna(df_selected[col].median())

# Fill missing categorical values with mode
for col in categorical_cols:
    df_selected[col] = df_selected[col].fillna(df_selected[col].mode()[0])

# 4. Column Transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # category_cols are already binary
)

# 5. Pipeline: Preprocessing + PCA + KMeans
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=10, random_state=42)),  # Reduce to 10 principal components
    ('kmeans', KMeans(n_clusters=4, random_state=42))  # Set n_clusters as needed
])

# 6. Fit pipeline
pipeline.fit(df_selected)

# 7. Predict clusters
clusters = pipeline.named_steps['kmeans'].labels_
df['Cluster'] = clusters

# 8. Output segmented data
df.to_csv('segmented_customers.csv', index=False)
print("Segmentation complete. Results saved to segmented_customers.csv.")

# 9. Save the pipeline as a .pkl file
with open('segmentation_kmeans_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Model pipeline saved as segmentation_kmeans_model.pkl.")

# Optional: To see cluster counts
print(df['Cluster'].value_counts())
