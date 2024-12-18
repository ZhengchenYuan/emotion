import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import entropy, ttest_ind
from scipy.stats import poisson
import numpy as np

# 1. Load the dataset
file_path = 'eng(b).csv'  # 请确保数据集路径正确
data = pd.read_csv(file_path)
print("Dataset Loaded Successfully!")
print(data.head())

# 2. Define emotions to predict
target_emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# 3. Analyze emotion intensity distribution
def analyze_distribution(data, emotions):
    print("\nEmotion Intensity Distribution Analysis:")
    for emotion in emotions:
        print(f"\n{emotion}强度分布:")
        print(data[emotion].value_counts())
        
        # 拟合Poisson分布
        mu = data[emotion].mean()
        print(f"拟合Poisson分布的参数 μ={mu:.2f}")
        x = np.arange(0, 4)
        print(f"理论分布: {poisson.pmf(x, mu)}")

analyze_distribution(data, target_emotions)

# 4. Split data into training and test sets (90% training, 10% test)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
test_data = test_data.sample(50, random_state=42).reset_index(drop=True)  # Select 50 test samples

# 5. Use TF-IDF to vectorize the text column
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

# 6. Train a two-stage model with statistical integration
final_predictions_df = test_data[['text']].copy()
true_labels = []
predicted_labels = []
bootstrap_results = []

for emotion in target_emotions:
    print(f"\nTraining and evaluating for emotion: {emotion}")
    
    # Define target variable
    y_train = train_data[emotion]
    y_test = test_data[emotion]
    
    # Stage 1: Predict binary outcome (0 vs >0)
    y_train_binary = (y_train > 0).astype(int)
    stage1_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    stage1_model.fit(X_train, y_train_binary)
    y_pred_binary = stage1_model.predict(X_test)
    
    # Stage 2: Predict intensities (1, 2, 3) for non-zero samples
    X_train_stage2 = X_train[y_train > 0]
    y_train_stage2 = y_train[y_train > 0]
    X_test_stage2 = X_test[y_pred_binary > 0]

    if X_test_stage2.shape[0] > 0:
        stage2_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        stage2_model.fit(X_train_stage2, y_train_stage2)
        y_pred_stage2 = stage2_model.predict(X_test_stage2)
        
        final_predictions = np.zeros(len(y_test), dtype=int)
        final_predictions[y_pred_binary > 0] = y_pred_stage2.astype(int)
    else:
        final_predictions = np.zeros(len(y_test), dtype=int)
    
    # KL散度计算
    true_distribution = y_test.value_counts(normalize=True).sort_index()
    pred_distribution = pd.Series(final_predictions).value_counts(normalize=True).sort_index()
    pred_distribution = pred_distribution.reindex(true_distribution.index, fill_value=0)
    kl_div = entropy(pred_distribution, true_distribution)
    print(f"KL散度 (真实 vs 预测): {kl_div:.4f}")
    
    # Bootstrap置信区间
    bootstrap_preds = []
    for i in range(500):
        X_boot, y_boot = resample(X_train, y_train, random_state=i)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_boot, y_boot)
        bootstrap_preds.append(model.predict(X_test).mean())
    lower, upper = np.percentile(bootstrap_preds, [2.5, 97.5])
    print(f"Bootstrap置信区间: [{lower:.3f}, {upper:.3f}]")
    
    # Append results
    true_labels.append(y_test.values.astype(int))
    predicted_labels.append(final_predictions.astype(int))
    bootstrap_results.append((lower, upper))

# Combine true and predicted labels for display
final_predictions_df['True_Labels'] = list(map(list, zip(*true_labels)))
final_predictions_df['Predicted_Labels'] = list(map(list, zip(*predicted_labels)))

# 7. Display final results
print("\nFinal comparison of predictions and true labels for all emotions:")
print(final_predictions_df[['text', 'True_Labels', 'Predicted_Labels']])

# 8. T-test for significant differences
def t_test_emotions(train_data, emotion1, emotion2):
    t_stat, p_val = ttest_ind(train_data[emotion1], train_data[emotion2])
    print(f"\nT-test between {emotion1} and {emotion2}: T-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print(f"{emotion1} 和 {emotion2} 之间的强度分布显著不同！")
    else:
        print(f"{emotion1} 和 {emotion2} 之间的强度分布没有显著差异。")

t_test_emotions(train_data, 'Joy', 'Sadness')

t_test_emotions(train_data, 'Anger', 'Fear')
