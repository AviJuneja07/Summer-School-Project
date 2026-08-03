import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# today's four new tools, all from scikit-learn
from sklearn.model_selection import train_test_split   # the split
from sklearn.pipeline import Pipeline                   # the idiom
from sklearn.compose import ColumnTransformer           # per-column prep
from sklearn.preprocessing import OneHotEncoder         # categories -> 0/1
from sklearn.linear_model import LinearRegression       # the model
from sklearn.metrics import root_mean_squared_error     # the error measure

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
exclusion_rule = (df['price_clean'] > 0) & (df['price_clean'] <= 1000) #exclusion rule: Remove €0 listings and cap at €1000
df_filtered = df[exclusion_rule].copy()
# 5. Compute Do-Nothing Baseline (Mean & Log Mean)
mean_baseline = df_filtered['price_clean'].mean()
median_baseline = df_filtered['price_clean'].median()

print(f"Cleaned Do-Nothing Baseline (Mean): €{mean_baseline:.2f}")
print(f"Median Price: €{median_baseline:.2f}")
print(f"Max Price in Filtered Set: €{df_filtered['price_clean'].max():.2f}")

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
plt.tight_layout()
plt.show()
