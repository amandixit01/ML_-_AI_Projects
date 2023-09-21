import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('fast_food_data.csv')

# Check the first few rows of the data
print(df.head())

# Check for missing values
print(df.isnull().sum())

# If there are missing values, fill them with appropriate method, here we are using forward fill method
df.fillna(method='ffill', inplace=True)

# Select features to include in the model
features = ['Age', 'Income', 'Frequency_of_visits', 'Average_spend_per_visit']

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)

# Add the cluster labels to the original data
df['Segment'] = kmeans.labels_

# Profile the segments
profile = df.groupby('Segment').mean()

print(profile)
