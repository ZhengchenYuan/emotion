import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from mord import OrdinalRidge
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import numpy as np

# 1. Load the dataset
file_path = 'eng.csv'  # 请确保数据集路径正确

data = pd.read_csv(file_path)
print("Dataset Loaded Successfully!")
print(data.head())

# 2. Define emotions to predict
target_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# 3. Split data into training and test sets (90% training, 10% test)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# 4. Use TF-IDF to vectorize the text column
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

# 5. Train and evaluate an Ordinal Regression model for each emotion
results = {}
comparison_results = {}

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")
    
    # Define target variable
    y_train = train_data[emotion]
    y_test = test_data[emotion]
    
    # Train an Ordinal Regression model
    model = OrdinalRidge()
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_rounded = np.round(y_pred).clip(min=0, max=3)  # Ensure values are in 0-3 range
    
    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred_rounded)
    report = classification_report(y_test, y_pred_rounded, zero_division=0)
    results[emotion] = {'Accuracy': accuracy, 'Report': report}
    
    # Combine predictions with true values for comparison
    comparison = pd.DataFrame({
        'Text': test_data['text'],
        'True_Label': y_test,
        'Predicted_Label': y_pred_rounded
    })
    comparison_results[emotion] = comparison
    
    # Check for predicted values of 2 and 3
    has_2_or_3 = comparison[comparison['Predicted_Label'].isin([2, 3])]
    print(f"\nExamples where Predicted_Label is 2 or 3 for {emotion}:")
    print(has_2_or_3.head(5))

# 6. Display overall results
for emotion in target_emotions:
    print(f"\nResults for {emotion}:")
    print(f"Accuracy: {results[emotion]['Accuracy']}")
    print("Classification Report:")
    print(results[emotion]['Report'])
    
    print("Sample Predictions:")
    print(comparison_results[emotion].head(10))
