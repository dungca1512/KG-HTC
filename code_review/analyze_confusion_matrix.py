import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

# Đọc file kết quả
with open('dataset/multilabel_results_v2.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

def get_label_set(results, key):
    label_set = set()
    for item in results:
        label_set.update(item.get(key, []))
    return sorted(label_set)

def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Đã lưu confusion matrix: {filename}")

def analyze_level(level_name, true_key, pred_key):
    print(f"\n=== Confusion Matrix cho {level_name} ===")
    # Lấy tất cả nhãn xuất hiện
    all_labels = sorted(list(set(get_label_set(results, true_key)) | set(get_label_set(results, pred_key))))
    # Tạo vector nhị phân cho từng sample
    y_true = []
    y_pred = []
    for item in results:
        true_labels = set(item.get(true_key, []))
        pred_labels = set(item.get(pred_key, []))
        y_true.append([1 if label in true_labels else 0 for label in all_labels])
        y_pred.append([1 if label in pred_labels else 0 for label in all_labels])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Tính confusion matrix cho từng nhãn
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for idx, label in enumerate(all_labels):
        print(f"\nConfusion matrix for label: {label}")
        print(pd.DataFrame(mcm[idx], index=['True Neg','True Pos'], columns=['Pred Neg','Pred Pos']))
        # Vẽ heatmap cho từng nhãn (nếu muốn)
        # plot_confusion_matrix(mcm[idx], ['Neg','Pos'], f'{level_name} - {label}', f'code_review/dataset/cm_{level_name}_{label}.png')
    # Tổng hợp confusion matrix (micro)
    cm_sum = np.zeros((2,2), dtype=int)
    for mat in mcm:
        cm_sum += mat
    print(f"\nTổng hợp confusion matrix (micro) cho {level_name}:")
    print(pd.DataFrame(cm_sum, index=['True Neg','True Pos'], columns=['Pred Neg','Pred Pos']))
    # Vẽ confusion matrix tổng hợp
    plot_confusion_matrix(cm_sum, ['Neg','Pos'], f'{level_name} (micro sum)', f'dataset/cm_{level_name}_micro.png')

if __name__ == "__main__":
    analyze_level('Type', 'true_types', 'pred_types')
    analyze_level('Category', 'true_categories', 'pred_categories')
    analyze_level('Tag', 'true_tags', 'pred_tags') 