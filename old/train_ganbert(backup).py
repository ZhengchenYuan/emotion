import pandas as pd
from sklearn.model_selection import train_test_split
from ganbert import Ganbert
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 加载数据
file_path = r"C:\Users\Shu Han\nlp\emotion\eng.csv"
data = pd.read_csv(file_path)

emotion_columns = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
def extract_label(row):
    for emotion in emotion_columns:
        if row[emotion] > 0:
            return f"{emotion}_{row[emotion]}"
    return "neutral"

data['label'] = data.apply(extract_label, axis=1)
# 清除多余空格并规范标签
data['label'] = data['label'].str.strip()  # 去除首尾空格
data['label'] = data['label'].str.lower()  # 如果需要小写

data = data[data['label'] != 'Fear_3']
data = data[data['label'] != 'Fear_2']
data = data[data['label'] != 'Anger_1']
data = data[data['label'] != 'Fear_1']
data = data[data['label'] != 'Anger_2']


# 数据划分
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

labeled_train_data = [(row['text'], row['label']) for _, row in train_data.iterrows()]
labeled_val_data = [(row['text'], row['label']) for _, row in val_data.iterrows()]
test_data = [(row['text'], row['label']) for _, row in test_data.iterrows()]

# 动态生成标签列表
label_list = sorted(data['label'].unique())
print(f"标签列表: {label_list}")

# 初始化模型
model = Ganbert(
    label_list=label_list, 
    model="bert-base-cased"
)

# 定义输出路径
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

# 生成报告函数
def generate_report(test_data, model, output_dir):
    """
    生成报告，包含分类报告和混淆矩阵
    """
    # 测试集预测
    y_true = [label for _, label in test_data]
    y_pred = []
    for text, _ in test_data:
        pred_label = model.predict(text)  # 假设 Ganbert 类有一个 predict 方法
        y_pred.append(pred_label)

    # 分类报告
    report = classification_report(y_true, y_pred, target_names=label_list, digits=3)

    # 保存报告到文件
    report_file = os.path.join(output_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

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

# 调用生成报告
generate_report(test_data, model, results_dir)
