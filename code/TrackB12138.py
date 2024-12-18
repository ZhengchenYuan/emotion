import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
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

# 5. Train a two-stage model with data augmentation for minority classes
results = {}
comparison_results = {}

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")
    
    # Define target variable
    y_train = train_data[emotion]
    y_test = test_data[emotion]
    
    # Stage 1: Predict binary outcome (0 vs >0)
    y_train_binary = (y_train > 0).astype(int)
    
    # Resample the data to balance classes
    X_resampled, y_resampled = resample(X_train, y_train_binary, random_state=42)
    
    stage1_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    stage1_model.fit(X_resampled, y_resampled)
    y_pred_binary = stage1_model.predict(X_test)
    
    # Stage 2: Predict intensities (1, 2, 3) for non-zero samples
    X_train_stage2 = X_train[y_train > 0]
    y_train_stage2 = y_train[y_train > 0]
    X_test_stage2 = X_test[y_pred_binary > 0]

    if X_test_stage2.shape[0] > 0:
        stage2_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        stage2_model.fit(X_train_stage2, y_train_stage2)
        y_pred_stage2 = stage2_model.predict(X_test_stage2)
        
        # Combine results
        final_predictions = np.zeros(len(y_test))
        final_predictions[y_pred_binary > 0] = y_pred_stage2
    else:
        final_predictions = np.zeros(len(y_test))
    
    # Evaluate results
    accuracy = accuracy_score(y_test, final_predictions)
    report = classification_report(y_test, final_predictions, zero_division=0)
    results[emotion] = {'Accuracy': accuracy, 'Report': report}
    
    # Combine predictions for inspection
    comparison = pd.DataFrame({
        'Text': test_data['text'],
        'True_Label': y_test,
        'Predicted_Label': final_predictions
    })
    comparison_results[emotion] = comparison
    
    # Show examples with 2 and 3 predictions
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
