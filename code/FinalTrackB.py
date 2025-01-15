import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load the dataset
data = pd.read_csv(r"C:\Users\ziyue\nlp\emotion\eng(b).csv")
print("Dataset Loaded Successfully!")

# Define emotions to predict
target_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Split data into training, validation, and test sets
train_data = data.iloc[:1600]  # First 1600 entries as training data
validation_data = data.iloc[1600:2000].reset_index(drop=True)  # Next 400 as validation data
test_data = data.iloc[2000:].reset_index(drop=True)  # Remaining as test data

# Use TF-IDF to vectorize the text column
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_validation = vectorizer.transform(validation_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

# Train a two-stage model with hyperparameter tuning
final_predictions_df = test_data[['text']].copy()
overall_true_labels = []
overall_pred_labels = []

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")
    
    # Stage 1: Binary classification (0 vs >0)
    y_train_binary = (train_data[emotion] > 0).astype(int)
    y_validation_binary = (validation_data[emotion] > 0).astype(int)
    y_test = test_data[emotion]
    
    # Hyperparameter tuning for Stage 1
    param_grid_stage1 = {
        'max_depth': [5, 10, 20, None],
        'n_estimators': [50, 100, 200],
        'class_weight': ['balanced'],
        'random_state': [42]
    }
    stage1_model = GridSearchCV(RandomForestClassifier(), param_grid_stage1, scoring='f1_macro', cv=3, verbose=2, n_jobs=-1)
    stage1_model.fit(X_train, y_train_binary)
    print(f"Best parameters for Stage 1 ({emotion}): {stage1_model.best_params_}")
    
    # Evaluate on validation data
    y_pred_validation_binary = stage1_model.predict(X_validation)
    print(f"Validation Metrics for Stage 1 ({emotion}):")
    print(classification_report(y_validation_binary, y_pred_validation_binary))
    
    # Predict binary outcomes on test data
    y_pred_binary = stage1_model.predict(X_test)
    
    # Stage 2: Multi-class classification (1, 2, 3) for non-zero samples
    X_train_stage2 = X_train[train_data[emotion] > 0]
    y_train_stage2 = train_data[emotion][train_data[emotion] > 0]
    X_validation_stage2 = X_validation[validation_data[emotion] > 0]
    y_validation_stage2 = validation_data[emotion][validation_data[emotion] > 0]
    X_test_stage2 = X_test[y_pred_binary > 0]

    final_predictions = np.zeros(len(y_test), dtype=int)
    if X_test_stage2.shape[0] > 0:
        # Hyperparameter tuning for Stage 2
        param_grid_stage2 = {
            'max_depth': [5, 10, 20, None],
            'n_estimators': [50, 100, 200],
            'class_weight': ['balanced'],
            'random_state': [42]
        }
        stage2_model = GridSearchCV(RandomForestClassifier(), param_grid_stage2, scoring='f1_macro', cv=3, verbose=2, n_jobs=-1)
        stage2_model.fit(X_train_stage2, y_train_stage2)
        print(f"Best parameters for Stage 2 ({emotion}): {stage2_model.best_params_}")
        
        # Evaluate on validation data for Stage 2
        if X_validation_stage2.shape[0] > 0:
            y_pred_validation_stage2 = stage2_model.predict(X_validation_stage2)
            print(f"Validation Metrics for Stage 2 ({emotion}):")
            print(classification_report(y_validation_stage2, y_pred_validation_stage2))
        
        y_pred_stage2 = stage2_model.predict(X_test_stage2)
        final_predictions[y_pred_binary > 0] = y_pred_stage2.astype(int)
    
    overall_true_labels.append(y_test.values.astype(int))
    overall_pred_labels.append(final_predictions.astype(int))

    # Calculate metrics for the current emotion
    print(f"Metrics for {emotion}:")
    print(classification_report(y_test.values.astype(int), final_predictions))

# Combine true and predicted labels for display
final_predictions_df['True_Labels'] = ["[" + ", ".join(map(str, labels)) + "]" for labels in zip(*overall_true_labels)]
final_predictions_df['Predicted_Labels'] = ["[" + ", ".join(map(str, labels)) + "]" for labels in zip(*overall_pred_labels)]

# Calculate the match rate
matches = sum([1 for true, pred in zip(final_predictions_df['True_Labels'], final_predictions_df['Predicted_Labels']) if true == pred])
total_samples = len(final_predictions_df)
match_rate = matches / total_samples * 100
print(f"\nNumber of samples where predicted labels match true labels: {matches}/{total_samples}")
print(f"Match rate: {match_rate:.2f}%")

# Save the results to a CSV file
output_file_path = 'emotion_predictions_cleaned_results.csv'
final_predictions_df.to_csv(output_file_path, index=False)
print(f"Results saved to {output_file_path}")
