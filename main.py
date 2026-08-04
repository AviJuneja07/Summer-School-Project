import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# today's four new tools, all from scikit-learn
from sklearn.model_selection import GroupKFold, train_test_split   # the split
from sklearn.pipeline import Pipeline                   # the idiom
from sklearn.compose import ColumnTransformer           # per-column prep
from sklearn.preprocessing import OneHotEncoder         # categories -> 0/1
from sklearn.linear_model import LinearRegression       # the model
from sklearn.metrics import root_mean_squared_error     # the error measure
from sklearn.impute import SimpleImputer                 # missing values
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Load Data
df = pd.read_csv('athens.csv.gz', compression='gzip')

# 2. Check Size Floor (Brief requirement: >= 5,000 listings & hundreds of multi-listing hosts)
print(f"Total listings: {len(df)}")
host_counts = df["host_id"].value_counts()
print(f"Hosts with >1 listing: {(host_counts > 1).sum()}")

# 3. Clean Price Column safely
df['price_clean'] = (
    df['price']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True) # Strips $, €, commas, and non-numeric chars
    .astype(float)
)

# 4. Define and Apply Exclusion Rules (Day 1 Milestone)
# Example rule: Remove €0 listings 
fix_data = df[df['host_location'] == 'Athens, Greece']
exclusion_rule = (df['price_clean'] > 0) & (df['price_clean'] <= 1000) & (df['host_location'] == 'Athens, Greece') #exclusion rule: Remove €0 listings and cap at €1000 and only include listings from Athens, Greece
df_filtered = df[exclusion_rule].copy()
# 5. Compute Do-Nothing Baseline (Mean & Log Mean)
mean_baseline = df_filtered['price_clean'].mean()
median_baseline = df_filtered['price_clean'].median()
print(f"Total filtered listings: {len(df_filtered)}")
print(f"Cleaned Do-Nothing Baseline (Mean): €{mean_baseline:.2f}")
print(f"Median Price: €{median_baseline:.2f}")
print(f"Max Price in Filtered Set: €{df_filtered['price_clean'].max():.2f}")

categorical_features = [
    'property_type',
    'room_type'
]
numeric_features = [
    'hosts_time_as_user_months',
    'hosts_time_as_host_months',
    'host_is_superhost',
    'host_has_profile_pic',
    'host_identity_verified',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'minimum_nights',
    'maximum_nights',
    'host_listings_count',
    'review_scores_rating'
    
]
df_filtered1 = df_filtered.dropna(subset = numeric_features)
X = df_filtered1[['hosts_time_as_user_months', 'hosts_time_as_host_months', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights', 'maximum_nights','review_scores_rating', 'host_listings_count']]
y = df_filtered1['price_clean']
gk5 = GroupKFold(n_splits=5)
groups = df_filtered1['host_id']
gk5.split(X, y, groups) 


for fold, (train_idx, test_idx) in enumerate(gk5.split(X, y, groups)):
    print(f"Fold {fold + 1}")
    print(f"Training listings: {len(train_idx)}")
    print(f"Testing listings: {len(test_idx)}")




categorical_features = [
    'property_type',
    'room_type'
]
bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']

numeric_features = [
    'hosts_time_as_user_months',
    'hosts_time_as_host_months',
    'host_is_superhost',
    'host_has_profile_pic',
    'host_identity_verified',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'minimum_nights',
    'maximum_nights',
    'host_listings_count',
    'review_scores_rating'
    
]

for col in bool_cols:
    df_filtered[col] = df_filtered[col].map({'t': 1, 'f': 0})

# rebuild X so it picks up the converted columns

df_filtered1 = df_filtered.dropna(subset = numeric_features)
X = df_filtered1[numeric_features + categorical_features]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(df_filtered['price_clean'], bins=50, color='skyblue', edgecolor='black')
ax1.set_title('Athens Price Distribution (Levels)')
ax1.set_xlabel('Price (€)')
ax1.set_ylabel('Number of Listings')
ax1.grid(axis='y', alpha=0.3)
ax2.hist(np.log1p(df_filtered['price_clean']), bins=50, color='salmon', edgecolor='black')
ax2.set_title('Log-Transformed Price Distribution')
ax2.set_xlabel('log(Price + 1)')
ax2.set_ylabel('Number of Listings')
ax2.grid(axis='y', alpha=0.3)
#plt.tight_layout()
#plt.show()

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
X_transformed = preprocessor.fit_transform(X)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

fold_rmses = []
baseline_rmses = []

for fold, (train_idx, test_idx) in enumerate(gk5.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- model ---
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    fold_rmses.append(rmse)

    # --- do-nothing baseline for this fold ---
    train_mean = y_train.mean()                      # only look at training data, same as the model would
    baseline_preds = np.full_like(y_test, train_mean, dtype=float)  # predict this mean for every test row
    baseline_rmse = root_mean_squared_error(y_test, baseline_preds)
    baseline_rmses.append(baseline_rmse)

    print(f"Fold {fold + 1}: Model RMSE = €{rmse:.2f} | Baseline RMSE = €{baseline_rmse:.2f} | Improvement = €{baseline_rmse - rmse:.2f}")

print(f"\nMean Model RMSE:    €{np.mean(fold_rmses):.2f}")
print(f"Mean Baseline RMSE: €{np.mean(baseline_rmses):.2f}")
