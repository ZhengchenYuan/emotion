import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
train_data = data.iloc[:1600]  # First 1600 entries for training
val_data = data.iloc[1600:2000]  # Next 400 entries for validation
test_data = data.iloc[2000:].reset_index(drop=True)  # Remaining entries for testing

# Use TF-IDF to vectorize the text column
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_val = vectorizer.transform(val_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

final_predictions_df = test_data[['text']].copy()
overall_true_labels = []
overall_pred_labels = []

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")

    # Binary Classification (Stage 1)
    y_train_binary = (train_data[emotion] > 0).astype(int)
    y_val_binary = (val_data[emotion] > 0).astype(int)
    y_test_binary = (test_data[emotion] > 0).astype(int)

    # Hyperparameter tuning for Stage 1
    param_grid_stage1 = {'max_depth': [5, 10, None], 'n_estimators': [50, 100, 200]}
    grid_stage1 = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                               param_grid_stage1, scoring='f1_macro', cv=3, n_jobs=-1)
    grid_stage1.fit(X_train, y_train_binary)
    print(f"Best Stage 1 Params for {emotion}: {grid_stage1.best_params_}")

    stage1_model = grid_stage1.best_estimator_
    y_pred_binary = stage1_model.predict(X_test)

    # Multi-class Classification (Stage 2)
    X_train_stage2 = X_train[train_data[emotion] > 0]
    y_train_stage2 = train_data[emotion][train_data[emotion] > 0]
    X_val_stage2 = X_val[val_data[emotion] > 0]
    y_val_stage2 = val_data[emotion][val_data[emotion] > 0]
    X_test_stage2 = X_test[y_pred_binary > 0]

    final_predictions = np.zeros(len(test_data), dtype=int)
    if len(X_train_stage2) > 0 and len(X_test_stage2) > 0:
        param_grid_stage2 = {'max_depth': [5, 10, None], 'n_estimators': [50, 100, 200]}
        grid_stage2 = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                                   param_grid_stage2, scoring='f1_macro', cv=3, n_jobs=-1)
        grid_stage2.fit(X_train_stage2, y_train_stage2)
        print(f"Best Stage 2 Params for {emotion}: {grid_stage2.best_params_}")

        stage2_model = grid_stage2.best_estimator_
        y_pred_stage2 = stage2_model.predict(X_test_stage2)
        final_predictions[y_pred_binary > 0] = y_pred_stage2

    overall_true_labels.append(test_data[emotion].values)
    overall_pred_labels.append(final_predictions)

    # Evaluate for the current emotion
    print(f"\nClassification Report for {emotion}:")
    print(classification_report(test_data[emotion], final_predictions))

# Combine results and calculate match rate
final_predictions_df['True_Labels'] = ["[" + ", ".join(map(str, labels)) + "]" for labels in zip(*overall_true_labels)]
final_predictions_df['Predicted_Labels'] = ["[" + ", ".join(map(str, labels)) + "]" for labels in zip(*overall_pred_labels)]
matches = sum([1 for true, pred in zip(final_predictions_df['True_Labels'], final_predictions_df['Predicted_Labels']) if true == pred])
print(f"\nMatch Rate: {matches / len(final_predictions_df) * 100:.2f}%")
