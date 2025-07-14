import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

def analyze_multilabel_results():
    """Analyze multi-label classification results"""
    
    try:
        # Load results
        with open("dataset/multilabel_results.json", "r", encoding='utf-8') as f:
            results = json.load(f)
        
        print("📊 Multi-label Classification Analysis")
        print("=" * 60)
        
        # Filter valid results
        valid_results = [r for r in results if r.get('pred_types', ['error'])[0] != 'error']
        print(f"Total samples: {len(results)}")
        print(f"Valid samples: {len(valid_results)}")
        print(f"Error samples: {len(results) - len(valid_results)}")
        
        if not valid_results:
            print("❌ No valid results to analyze!")
            return
        
        # Analyze multi-label distribution
        print(f"\n🏷️ Multi-label Distribution Analysis:")
        print("-" * 40)
        
        # Types
        type_counts = Counter()
        multi_type_samples = 0
        for r in valid_results:
            true_types = r['true_types']
            type_counts[len(true_types)] += 1
            if len(true_types) > 1:
                multi_type_samples += 1
        
        print(f"Type label distribution:")
        for num_labels, count in sorted(type_counts.items()):
            print(f"  {num_labels} label(s): {count} samples ({count/len(valid_results)*100:.1f}%)")
        print(f"Multi-type samples: {multi_type_samples}/{len(valid_results)} ({multi_type_samples/len(valid_results)*100:.1f}%)")
        
        # Categories
        category_counts = Counter()
        multi_category_samples = 0
        for r in valid_results:
            true_categories = r['true_categories']
            category_counts[len(true_categories)] += 1
            if len(true_categories) > 1:
                multi_category_samples += 1
        
        print(f"\nCategory label distribution:")
        for num_labels, count in sorted(category_counts.items()):
            print(f"  {num_labels} label(s): {count} samples ({count/len(valid_results)*100:.1f}%)")
        print(f"Multi-category samples: {multi_category_samples}/{len(valid_results)} ({multi_category_samples/len(valid_results)*100:.1f}%)")
        
        # Performance by label count
        print(f"\n📈 Performance by Number of Labels:")
        print("-" * 40)
        
        # Type performance
        print(f"Type prediction performance:")
        for num_labels in sorted(type_counts.keys()):
            samples_with_n_labels = [r for r in valid_results if len(r['true_types']) == num_labels]
            if not samples_with_n_labels:
                continue
                
            exact_matches = sum(1 for r in samples_with_n_labels 
                              if set(r['true_types']) == set(r['pred_types']))
            
            # Calculate F1 for this group
            total_f1 = 0
            for r in samples_with_n_labels:
                true_set = set(r['true_types'])
                pred_set = set(r['pred_types'])
                
                if len(pred_set) > 0:
                    precision = len(true_set & pred_set) / len(pred_set)
                else:
                    precision = 0
                    
                if len(true_set) > 0:
                    recall = len(true_set & pred_set) / len(true_set)
                else:
                    recall = 1 if len(pred_set) == 0 else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                    
                total_f1 += f1
            
            avg_f1 = total_f1 / len(samples_with_n_labels)
            exact_match_rate = exact_matches / len(samples_with_n_labels)
            
            print(f"  {num_labels} label(s): Exact Match={exact_match_rate:.3f}, Avg F1={avg_f1:.3f} ({len(samples_with_n_labels)} samples)")
        
        # Most common label combinations
        print(f"\n🔗 Most Common Label Combinations:")
        print("-" * 40)
        
        # True type combinations
        true_type_combos = Counter()
        pred_type_combos = Counter()
        
        for r in valid_results:
            true_combo = ",".join(sorted(r['true_types']))
            pred_combo = ",".join(sorted(r['pred_types']))
            true_type_combos[true_combo] += 1
            pred_type_combos[pred_combo] += 1
        
        print(f"Top 5 true type combinations:")
        for combo, count in true_type_combos.most_common(5):
            print(f"  '{combo}': {count} samples ({count/len(valid_results)*100:.1f}%)")
        
        print(f"\nTop 5 predicted type combinations:")
        for combo, count in pred_type_combos.most_common(5):
            print(f"  '{combo}': {count} samples ({count/len(valid_results)*100:.1f}%)")
        
        # Mixed sentiment analysis
        print(f"\n🎭 Mixed Sentiment Analysis:")
        print("-" * 40)
        
        # Find samples that should be mixed (have both positive and negative)
        mixed_sentiment_samples = []
        for r in valid_results:
            true_types = set(r['true_types'])
            if 'positive' in true_types and 'negative' in true_types:
                mixed_sentiment_samples.append(r)
        
        print(f"True mixed sentiment samples: {len(mixed_sentiment_samples)}")
        
        if mixed_sentiment_samples:
            # Check how well we predict mixed sentiment
            correct_mixed_predictions = 0
            for r in mixed_sentiment_samples:
                pred_types = set(r['pred_types'])
                if 'positive' in pred_types and 'negative' in pred_types:
                    correct_mixed_predictions += 1
            
            mixed_accuracy = correct_mixed_predictions / len(mixed_sentiment_samples)
            print(f"Mixed sentiment detection accuracy: {mixed_accuracy:.3f}")
            
            # Show examples
            print(f"\nExample mixed sentiment predictions:")
            for i, r in enumerate(mixed_sentiment_samples[:3]):
                print(f"  {i+1}. Text: {r['text'][:80]}...")
                print(f"     True: {r['true_types']} | Pred: {r['pred_types']}")
                match = set(r['true_types']) == set(r['pred_types'])
                print(f"     Match: {match}")
        
        # Error analysis
        print(f"\n❌ Error Analysis:")
        print("-" * 40)
        
        # Type prediction errors
        type_errors = []
        for r in valid_results:
            if set(r['true_types']) != set(r['pred_types']):
                type_errors.append(r)
        
        print(f"Type prediction errors: {len(type_errors)}/{len(valid_results)} ({len(type_errors)/len(valid_results)*100:.1f}%)")
        
        if type_errors:
            # Common error patterns
            error_patterns = Counter()
            for r in type_errors:
                true_str = ",".join(sorted(r['true_types']))
                pred_str = ",".join(sorted(r['pred_types']))
                error_patterns[f"{true_str} → {pred_str}"] += 1
            
            print(f"Top 3 error patterns:")
            for pattern, count in error_patterns.most_common(3):
                print(f"  {pattern}: {count} times")
        
        # Create summary statistics
        print(f"\n📋 Summary Statistics:")
        print("-" * 40)
        
        # Load metrics if available
        try:
            # Thử mở file metrics v2 trước, nếu không có thì mở file cũ
            metrics_path = None
            for path in ["dataset/multilabel_metrics_v2.json", "dataset/multilabel_metrics.json"]:
                if os.path.exists(path):
                    metrics_path = path
                    break
            if metrics_path is None:
                print("Metrics file not found. Run multilabel_classification.py first.")
            else:
                with open(metrics_path, "r", encoding='utf-8') as f:
                    metrics = json.load(f)
                # Ưu tiên lấy metrics sklearn nếu có
                if 'type_metrics_sklearn' in metrics:
                    type_metrics = metrics['type_metrics_sklearn']
                    print(f"Type Classification (macro/micro, sklearn):")
                    print(f"  Macro F1: {type_metrics['macro_f1']:.3f}")
                    print(f"  Micro F1: {type_metrics['micro_f1']:.3f}")
                    print(f"  Macro Precision: {type_metrics['macro_precision']:.3f}")
                    print(f"  Micro Precision: {type_metrics['micro_precision']:.3f}")
                    print(f"  Macro Recall: {type_metrics['macro_recall']:.3f}")
                    print(f"  Micro Recall: {type_metrics['micro_recall']:.3f}")
                elif 'type_metrics' in metrics:
                    type_metrics = metrics['type_metrics']
                    print(f"Type Classification:")
                    print(f"  Exact Match: {type_metrics['exact_match_ratio']:.3f}")
                    print(f"  Avg F1:      {type_metrics['avg_f1']:.3f}")
                    print(f"  Avg Precision: {type_metrics['avg_precision']:.3f}")
                    print(f"  Avg Recall:    {type_metrics['avg_recall']:.3f}")
                else:
                    print("No type metrics found in metrics file.")
        except Exception as e:
            print(f"❌ Error analyzing metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # --- Bổ sung xuất confusion matrix và sample sai cho file v2 ---
        if metrics_path and metrics_path.endswith("_v2.json"):
            import numpy as np
            from sklearn.preprocessing import MultiLabelBinarizer
            from sklearn.metrics import multilabel_confusion_matrix
            # Chuẩn bị dữ liệu
            def get_true_pred_lists(field):
                true_list = [r[f'true_{field}'] for r in valid_results]
                pred_list = [r[f'pred_{field}'] for r in valid_results]
                mlb = MultiLabelBinarizer()
                mlb.fit(true_list + pred_list)
                y_true = mlb.transform(true_list)
                y_pred = mlb.transform(pred_list)
                return y_true, y_pred, mlb
            # Confusion matrix cho từng trường
            for field in ['types', 'categories', 'tags']:
                y_true, y_pred, mlb = get_true_pred_lists(field)
                cm = multilabel_confusion_matrix(y_true, y_pred)
                # Lưu từng nhãn thành bảng riêng, nối dọc vào 1 file CSV
                with open(f"dataset/confusion_matrix_{field}_v2.csv", "w", encoding='utf-8') as f:
                    f.write("label,tn,fp,fn,tp\n")
                    for i, label in enumerate(mlb.classes_):
                        tn, fp, fn, tp = cm[i].ravel()
                        f.write(f"{label},{tn},{fp},{fn},{tp}\n")
            print("Đã lưu confusion matrix cho type, category, tag vào dataset/confusion_matrix_*_v2.csv")
            # Lưu sample sai
            wrong_samples = []
            for r in valid_results:
                type_wrong = set(r['true_types']) != set(r['pred_types'])
                cat_wrong = set(r['true_categories']) != set(r['pred_categories'])
                tag_wrong = set(r['true_tags']) != set(r['pred_tags'])
                if type_wrong or cat_wrong or tag_wrong:
                    wrong_samples.append({
                        'text': r['text'],
                        'true_types': r['true_types'], 'pred_types': r['pred_types'],
                        'true_categories': r['true_categories'], 'pred_categories': r['pred_categories'],
                        'true_tags': r['true_tags'], 'pred_tags': r['pred_tags'],
                        'type_wrong': type_wrong, 'category_wrong': cat_wrong, 'tag_wrong': tag_wrong
                    })
            with open("dataset/wrong_samples_v2.json", "w", encoding='utf-8') as f:
                json.dump(wrong_samples, f, indent=4, ensure_ascii=False)
            print(f"Đã lưu {len(wrong_samples)} sample sai vào dataset/wrong_samples_v2.json")
        
        # --- Bổ sung trực quan hóa confusion matrix cho file v2 ---
        if metrics_path and metrics_path.endswith("_v2.json"):
            import numpy as np
            from sklearn.preprocessing import MultiLabelBinarizer
            from sklearn.metrics import multilabel_confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            # Chuẩn bị dữ liệu
            def get_true_pred_lists(field):
                true_list = [r[f'true_{field}'] for r in valid_results]
                pred_list = [r[f'pred_{field}'] for r in valid_results]
                mlb = MultiLabelBinarizer()
                mlb.fit(true_list + pred_list)
                y_true = mlb.transform(true_list)
                y_pred = mlb.transform(pred_list)
                return y_true, y_pred, mlb
            # Confusion matrix cho từng trường
            for field in ['types', 'categories', 'tags']:
                y_true, y_pred, mlb = get_true_pred_lists(field)
                cm = multilabel_confusion_matrix(y_true, y_pred)
                # Lưu từng nhãn thành bảng riêng, nối dọc vào 1 file CSV
                with open(f"dataset/confusion_matrix_{field}_v2.csv", "w", encoding='utf-8') as f:
                    f.write("label,tn,fp,fn,tp\n")
                    for i, label in enumerate(mlb.classes_):
                        tn, fp, fn, tp = cm[i].ravel()
                        f.write(f"{label},{tn},{fp},{fn},{tp}\n")
                # Vẽ heatmap cho từng nhãn (nếu số nhãn <= 20)
                if len(mlb.classes_) <= 20:
                    # Tạo ma trận tổng hợp: mỗi nhãn là một confusion matrix 2x2, ta chỉ vẽ TP/FP/FN cho từng nhãn
                    matrix = np.array([[cm[i][1,1], cm[i][0,1], cm[i][1,0]] for i in range(len(mlb.classes_))])
                    # TP, FP, FN
                    plt.figure(figsize=(max(8, len(mlb.classes_)*0.7), 5))
                    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                                yticklabels=mlb.classes_, xticklabels=['TP','FP','FN'])
                    plt.title(f'Confusion Matrix for {field.capitalize()} (per label)')
                    plt.ylabel('Label')
                    plt.xlabel('Metric')
                    plt.tight_layout()
                    plt.savefig(f"dataset/confusion_matrix_{field}_v2.png")
                    plt.close()
                else:
                    # Nếu quá nhiều nhãn, chỉ vẽ top 20 nhãn xuất hiện nhiều nhất
                    true_flat = [l for sub in [r[f'true_{field}'] for r in valid_results] for l in sub]
                    top_labels = [l for l,_ in Counter(true_flat).most_common(20)]
                    idxs = [i for i,l in enumerate(mlb.classes_) if l in top_labels]
                    matrix = np.array([[cm[i][1,1], cm[i][0,1], cm[i][1,0]] for i in idxs])
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                                yticklabels=[mlb.classes_[i] for i in idxs], xticklabels=['TP','FP','FN'])
                    plt.title(f'Confusion Matrix for {field.capitalize()} (Top 20 labels)')
                    plt.ylabel('Label')
                    plt.xlabel('Metric')
                    plt.tight_layout()
                    plt.savefig(f"dataset/confusion_matrix_{field}_v2.png")
                    plt.close()
            print("Đã lưu confusion matrix cho type, category, tag vào dataset/confusion_matrix_*_v2.csv và PNG")
        
        print(f"\n✅ Analysis completed!")
        
    except FileNotFoundError:
        print("❌ Results file not found!")
        print("Please run multilabel_classification.py first to generate results.")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_multilabel_results()