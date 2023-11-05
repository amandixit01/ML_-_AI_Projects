import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Perform KMeans clustering, here we're assuming the optimal number of clusters is 3
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(scaled_features)

# Add the cluster labels to the original data
df['Segment'] = cluster_labels

# Profile the segments
profile = df.groupby('Segment').mean()

print(profile)
