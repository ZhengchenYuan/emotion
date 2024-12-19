import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Load the dataset
file_path = 'eng(b).csv'  # 请确保数据集路径正确
data = pd.read_csv(file_path)
print("Dataset Loaded Successfully!")
print(data.head())

# 2. Define emotions to predict
target_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# 3. Split data into training and test sets
train_data = data.iloc[:2000]  # 前 2000 条数据作为训练集
test_data = data.iloc[2000:].reset_index(drop=True)  # 剩余数据作为测试集，并重置索引

# 4. Use TF-IDF to vectorize the text column
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

# 5. Train a two-stage model with data augmentation for minority classes
final_predictions_df = test_data[['text']].copy()
true_labels = []
predicted_labels = []

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
        final_predictions = np.zeros(len(y_test), dtype=int)
        final_predictions[y_pred_binary > 0] = y_pred_stage2.astype(int)
    else:
        final_predictions = np.zeros(len(y_test), dtype=int)
    
    # Append true and predicted labels
    true_labels.append(y_test.values.astype(int))
    predicted_labels.append(final_predictions.astype(int))

# Combine true and predicted labels for display
final_predictions_df['True_Labels'] = list(map(list, zip(*true_labels)))
final_predictions_df['Predicted_Labels'] = list(map(list, zip(*predicted_labels)))

# 6. Display final results
print("\nFinal comparison of predictions and true labels for all emotions:")
# Convert True_Labels and Predicted_Labels into list format (if they are strings)
final_predictions_df['True_Labels'] = final_predictions_df['True_Labels'].apply(lambda x: [int(i) for i in x.strip('[]').split(',')] if isinstance(x, str) else x)
final_predictions_df['Predicted_Labels'] = final_predictions_df['Predicted_Labels'].apply(lambda x: [int(i) for i in x.strip('[]').split(',')] if isinstance(x, str) else x)

# Ensure True_Labels and Predicted_Labels are formatted as comma-separated strings
final_predictions_df['True_Labels'] = final_predictions_df['True_Labels'].apply(lambda x: ','.join(map(str, x)))
final_predictions_df['Predicted_Labels'] = final_predictions_df['Predicted_Labels'].apply(lambda x: ','.join(map(str, x)))

# Save the cleaned results to a CSV file
output_file_path = 'emotion_predictions_cleaned_results.csv'
final_predictions_df[['text', 'True_Labels', 'Predicted_Labels']].to_csv(output_file_path, index=False)

print(f"Results saved to {output_file_path}")

# 7. Calculate the number of matching labels between True_Labels and Predicted_Labels
matches = 0  # 计数完全匹配的样本
total_samples = len(final_predictions_df)

# 遍历每一行，检查 True_Labels 和 Predicted_Labels 是否完全相等
for _, row in final_predictions_df.iterrows():
    true = list(map(int, row['True_Labels'].split(',')))
    pred = list(map(int, row['Predicted_Labels'].split(',')))
    if true == pred:
        matches += 1

# 计算匹配率
match_rate = matches / total_samples * 100

# 打印结果
print(f"\nNumber of samples where predicted labels match true labels: {matches}/{total_samples}")
print(f"Match rate: {match_rate:.2f}%")

# 添加到结果文件
final_predictions_df['Match'] = final_predictions_df.apply(
    lambda row: 1 if list(map(int, row['True_Labels'].split(','))) == list(map(int, row['Predicted_Labels'].split(','))) else 0,
    axis=1
)

# 保存更新后的结果文件
final_predictions_df.to_csv(output_file_path, index=False)
print(f"Updated results with match information saved to {output_file_path}")


