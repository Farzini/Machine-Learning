import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data into a DataFrame
data = pd.read_csv('data.csv')

# Drop any rows with missing values
data = data.dropna()

# Encode any categorical features using a label encoder
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a random forest classifier
clf = RandomForestClassifier()

# Train the classifier on the training data
clf = clf.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = clf.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
