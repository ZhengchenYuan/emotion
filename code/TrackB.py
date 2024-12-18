from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 加载数据集
data = pd.read_csv('eng(b).csv')

# 目标情感
target_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# 分割训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
test_data = test_data.sample(50, random_state=42).reset_index(drop=True)

# 文本特征提取
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

# 存储最终结果
final_predictions_df = test_data[['text']].copy()
true_labels = []
predicted_labels = []

# 引入 Bootstrap 技术
bootstrap_iterations = 5  # 每个阶段的 Bootstrap 迭代次数

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")
    
    # 第一阶段：预测是否存在该情感
    y_train = train_data[emotion]
    y_test = test_data[emotion]
    y_train_binary = (y_train > 0).astype(int)
    
    # 使用 Bootstrap 进行训练
    stage1_predictions = np.zeros((bootstrap_iterations, len(y_test)))
    for i in range(bootstrap_iterations):
        # Bootstrap 重采样
        X_resampled, y_resampled = resample(X_train, y_train_binary, replace=True, random_state=i)
        stage1_model = RandomForestClassifier(n_estimators=100, random_state=i, class_weight='balanced')
        stage1_model.fit(X_resampled, y_resampled)
        stage1_predictions[i] = stage1_model.predict(X_test)
    
    # 聚合第一阶段结果（投票方式）
    y_pred_binary = (stage1_predictions.mean(axis=0) > 0.5).astype(int)
    
    # 第二阶段：预测强度值
    X_train_stage2 = X_train[y_train > 0]
    y_train_stage2 = y_train[y_train > 0]
    X_test_stage2 = X_test[y_pred_binary > 0]
    
    if X_test_stage2.shape[0] > 0:
        stage2_predictions = np.zeros((bootstrap_iterations, X_test_stage2.shape[0]))
        for i in range(bootstrap_iterations):
            # Bootstrap 重采样
            X_train_resampled, y_train_resampled = resample(X_train_stage2, y_train_stage2, replace=True, random_state=i)
            stage2_model = RandomForestClassifier(n_estimators=200, random_state=i, class_weight='balanced')
            stage2_model.fit(X_train_resampled, y_train_resampled)
            stage2_predictions[i] = stage2_model.predict(X_test_stage2)
        
        # 聚合第二阶段结果（取平均或多数投票）
        y_pred_stage2 = np.round(stage2_predictions.mean(axis=0)).astype(int)
        final_predictions = np.zeros(len(y_test), dtype=int)
        final_predictions[y_pred_binary > 0] = y_pred_stage2
    else:
        final_predictions = np.zeros(len(y_test), dtype=int)
    
    # 存储真实值和预测值
    true_labels.append(y_test.values.astype(int))
    predicted_labels.append(final_predictions.astype(int))

# 整理并显示最终结果
final_predictions_df['True_Labels'] = list(map(list, zip(*true_labels)))
final_predictions_df['Predicted_Labels'] = list(map(list, zip(*predicted_labels)))

print("\nFinal comparison of predictions and true labels for all emotions:")
print(final_predictions_df[['text', 'True_Labels', 'Predicted_Labels']])
