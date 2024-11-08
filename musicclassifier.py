import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("D:\intern project\genre.csv")  # Replace with your actual dataset path

# Inspect the dataset
print(data.head())
print(data.info())

# Check for missing values
print("Check for missing values")
print(data.isnull().sum())

# Fill missing values with the median (or another strategy suitable for your data)
data.fillna(data.median(), inplace=True)

print("Check after replacing missing values")
print(data.isnull().sum())

# Select features and target
features = data[['Feature1', 'Feature2', 'Feature3', 'Feature4']]
target = data['Genre']

# Encode target variable
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_encoded, test_size=0.2, random_state=0)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'Support Vector Classifier': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Assuming 'target_names' are correctly defined or use classes from label_encoder
classes = label_encoder.classes_

# Train and evaluate models
results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred, labels=range(len(classes)), target_names=classes),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # Print performance metrics
    print(f"\n{name} Accuracy: {accuracy}")
    print(f"{name} Classification Report:\n{results[name]['report']}")


# Example new customer data
new_customers = pd.DataFrame({
    'Feature1': [1.5, 2.5],
    'Feature2': [4.0, 5.5],
    'Feature3': [0.5, 0.8],
    'Feature4': [6.0, 7.5]
})

# Standardize new customer data
new_customers_scaled = scaler.transform(new_customers)

# Predict genres for new customer data with each model
for name, model in models.items():
    predicted_genres = model.predict(new_customers_scaled)
    new_customers[f'{name} PredictedGenre'] = label_encoder.inverse_transform(predicted_genres)

print("\nNew Customer Predictions:\n", new_customers)

# Plot confusion matrices
for name, model in models.items():
    cm = results[name]['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Extract model names and their corresponding accuracies
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

# Create a bar plot for model accuracies
plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1)  # Accuracy is between 0 and 1
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
