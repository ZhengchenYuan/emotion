import pandas as pd
from sklearn.model_selection import train_test_split
from ganbert import Ganbert
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 加载数据
file_path = r"C:\Users\Shu Han\nlp\emotion\eng.a.csv"
data = pd.read_csv(file_path)

# 定义情感列
emotion_columns = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# 提取标签函数
def extract_label(row):
    for emotion in emotion_columns:
        if row[emotion] > 0:
            return f"{emotion}_{row[emotion]}"
    return "neutral"

# 生成标签列
data['label'] = data.apply(extract_label, axis=1)
data['label'] = data['label'].str.strip()  # 去除首尾空格

# 动态生成标签列表
label_list = sorted(data['label'].unique())
print(f"标签列表: {label_list}")

# 数据划分
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

# 转换数据格式为 [(text, label)] 形式
labeled_train_data = [(row['text'], row['label']) for _, row in train_data.iterrows()]
labeled_val_data = [(row['text'], row['label']) for _, row in val_data.iterrows()]
test_data = [(row['text'], row['label']) for _, row in test_data.iterrows()]

# 初始化模型
model = Ganbert(
    label_list=label_list,
    model="bert-base-cased"
)

# 定义结果目录
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, "output.txt")

# 开始训练
model.train(
    labeled_data=labeled_train_data,
    unlabeled_data=None,
    test_data=test_data,
    label_list=label_list,
    outfile_name=output_file,
    tgt_label="emotion_intensity",
    num_train_epochs=3
)

# 生成预测函数
def predict_labels(test_data, model):
    """
    使用模型预测标签
    """
    y_true = [label for _, label in test_data]
    y_pred = []
    for text, _ in test_data:
        pred_label = model.predict(text)  # 假设 Ganbert 类有一个 predict 方法
        y_pred.append(pred_label)
    return y_true, y_pred

# 生成报告函数
def generate_report(y_true, y_pred, label_list, output_dir):
    """
    生成分类报告和混淆矩阵
    """
    # 分类报告
    report = classification_report(y_true, y_pred, target_names=label_list, digits=3)

    # 保存报告到文件
    report_file = os.path.join(output_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    # 打印报告
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_list, yticklabels=label_list, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print(f"报告已生成并保存在 {output_dir}")

# 使用模型进行预测并生成报告
y_true, y_pred = predict_labels(test_data, model)
generate_report(y_true, y_pred, label_list, results_dir)
