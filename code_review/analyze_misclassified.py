import json
import pandas as pd
import numpy as np
import os
from collections import defaultdict

def analyze_misclassifications():
    """Phân tích các trường hợp bị gán nhãn sai"""
    
    # Đọc file kết quả
    with open('dataset/multilabel_results_v2.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Tổng số samples: {len(results)}")
    
    # Phân tích từng cấp độ
    levels = [
        ('Type', 'true_types', 'pred_types'),
        ('Category', 'true_categories', 'pred_categories'), 
        ('Tag', 'true_tags', 'pred_tags')
    ]
    
    misclassified_cases = {}
    
    for level_name, true_key, pred_key in levels:
        print(f"\n{'='*60}")
        print(f"PHÂN TÍCH MISCLASSIFICATION CHO {level_name.upper()}")
        print(f"{'='*60}")
        
        misclassified = []
        error_types = defaultdict(int)
        
        for i, item in enumerate(results):
            true_labels = set(item.get(true_key, []))
            pred_labels = set(item.get(pred_key, []))
            
            # Kiểm tra có misclassification không
            if true_labels != pred_labels:
                misclassified.append({
                    'sample_index': i,
                    'review_text': item.get('text', item.get('Tag_clean', 'N/A')),
                    'true_labels': list(true_labels),
                    'pred_labels': list(pred_labels),
                    'missing_labels': list(true_labels - pred_labels),  # Thiếu
                    'extra_labels': list(pred_labels - true_labels),    # Thừa
                    'correct_labels': list(true_labels & pred_labels)   # Đúng
                })
                
                # Phân loại loại lỗi
                if true_labels and not pred_labels:
                    error_types['False Negative (Missed all)'] += 1
                elif pred_labels and not true_labels:
                    error_types['False Positive (Extra all)'] += 1
                elif true_labels - pred_labels:
                    error_types['Partial Miss (Missing some)'] += 1
                if pred_labels - true_labels:
                    error_types['Partial Extra (Extra some)'] += 1
        
        # Thống kê
        total_samples = len(results)
        misclassified_count = len(misclassified)
        accuracy = (total_samples - misclassified_count) / total_samples * 100
        
        print(f"Tổng số samples: {total_samples}")
        print(f"Số samples bị misclassify: {misclassified_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Error rate: {100-accuracy:.2f}%")
        
        print(f"\nPhân loại lỗi:")
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")
        
        # Hiển thị các trường hợp misclassified
        if misclassified:
            print(f"\nChi tiết {min(10, len(misclassified))} trường hợp đầu tiên:")
            for i, case in enumerate(misclassified[:10]):
                print(f"\n{i+1}. Sample {case['sample_index']}")
                print(f"   Review: {case['review_text'][:200]}{'...' if len(case['review_text']) > 200 else ''}")
                print(f"   True: {case['true_labels']}")
                print(f"   Pred: {case['pred_labels']}")
                if case['missing_labels']:
                    print(f"   ❌ Missing: {case['missing_labels']}")
                if case['extra_labels']:
                    print(f"   ❌ Extra: {case['extra_labels']}")
                if case['correct_labels']:
                    print(f"   ✅ Correct: {case['correct_labels']}")
        
        misclassified_cases[level_name] = misclassified
    
    # Lưu kết quả phân tích
    analysis_results = {
        'summary': {
            'total_samples': len(results),
            'levels_analyzed': [level[0] for level in levels]
        },
        'misclassified_cases': misclassified_cases
    }
    
    # Tạo thư mục dataset nếu chưa có
    os.makedirs('dataset', exist_ok=True)
    
    with open('dataset/misclassification_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("KẾT QUẢ ĐÃ ĐƯỢC LƯU VÀO: dataset/misclassification_analysis.json")
    
    # Tạo Excel file với các trường hợp misclassified
    create_misclassified_excel(misclassified_cases, len(results))
    
    # Tạo CSV file với các trường hợp misclassified
    create_misclassified_csv(misclassified_cases)
    
    return misclassified_cases

def create_misclassified_excel(misclassified_cases, total_samples):
    """Tạo file Excel với các trường hợp misclassified"""
    
    # Tạo thư mục dataset nếu chưa có
    os.makedirs('dataset', exist_ok=True)
    
    with pd.ExcelWriter('dataset/misclassified_cases.xlsx', engine='openpyxl') as writer:
        
        for level_name, cases in misclassified_cases.items():
            if cases:
                # Tạo DataFrame cho level này
                df_data = []
                for case in cases:
                    df_data.append({
                        'Sample_Index': case['sample_index'],
                        'Review_Text': case['review_text'],
                        'True_Labels': ', '.join(case['true_labels']),
                        'Predicted_Labels': ', '.join(case['pred_labels']),
                        'Missing_Labels': ', '.join(case['missing_labels']),
                        'Extra_Labels': ', '.join(case['extra_labels']),
                        'Correct_Labels': ', '.join(case['correct_labels']),
                        'Error_Type': get_error_type(case)
                    })
                
                df = pd.DataFrame(df_data)
                df.to_excel(writer, sheet_name=f'{level_name}_Misclassified', index=False)
        
        # Tạo sheet tổng hợp
        summary_data = []
        for level_name, cases in misclassified_cases.items():
            summary_data.append({
                'Level': level_name,
                'Total_Samples': len(cases),
                'Error_Rate': f"{len(cases)/total_samples*100:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print("EXCEL FILE ĐÃ ĐƯỢC TẠO: dataset/misclassified_cases.xlsx")

def create_misclassified_csv(misclassified_cases):
    """Tạo file CSV với các trường hợp misclassified"""
    
    # Tạo thư mục dataset nếu chưa có
    os.makedirs('dataset', exist_ok=True)
    
    # Tạo DataFrame tổng hợp cho tất cả các level
    all_misclassified_data = []
    
    for level_name, cases in misclassified_cases.items():
        for case in cases:
            all_misclassified_data.append({
                'Level': level_name,
                'Sample_Index': case['sample_index'],
                'Review_Text': case['review_text'],
                'True_Labels': ', '.join(case['true_labels']),
                'Predicted_Labels': ', '.join(case['pred_labels']),
                'Missing_Labels': ', '.join(case['missing_labels']),
                'Extra_Labels': ', '.join(case['extra_labels']),
                'Correct_Labels': ', '.join(case['correct_labels']),
                'Error_Type': get_error_type(case)
            })
    
    if all_misclassified_data:
        # Tạo DataFrame tổng hợp
        combined_df = pd.DataFrame(all_misclassified_data)
        
        # Lưu file CSV tổng hợp
        combined_df.to_csv('dataset/misclassified_cases_combined.csv', 
                          index=False, encoding='utf-8')
        print("CSV FILE TỔNG HỢP ĐÃ ĐƯỢC TẠO: dataset/misclassified_cases_combined.csv")
        
        # Lưu file CSV riêng cho từng level
        for level_name, cases in misclassified_cases.items():
            if cases:
                level_data = [item for item in all_misclassified_data if item['Level'] == level_name]
                level_df = pd.DataFrame(level_data)
                level_df.to_csv(f'dataset/misclassified_cases_{level_name.lower()}.csv', 
                               index=False, encoding='utf-8')
                print(f"CSV FILE {level_name} ĐÃ ĐƯỢC TẠO: dataset/misclassified_cases_{level_name.lower()}.csv")
    else:
        print("Không có trường hợp misclassified nào để lưu vào CSV")

def get_error_type(case):
    """Xác định loại lỗi cho một case"""
    if not case['true_labels'] and case['pred_labels']:
        return "False Positive (Extra all)"
    elif case['true_labels'] and not case['pred_labels']:
        return "False Negative (Missed all)"
    elif case['missing_labels'] and case['extra_labels']:
        return "Mixed (Missing + Extra)"
    elif case['missing_labels']:
        return "Partial Miss"
    elif case['extra_labels']:
        return "Partial Extra"
    else:
        return "Unknown"

def find_common_error_patterns(misclassified_cases):
    """Tìm các pattern lỗi phổ biến"""
    
    print(f"\n{'='*60}")
    print("PHÂN TÍCH PATTERN LỖI PHỔ BIẾN")
    print(f"{'='*60}")
    
    for level_name, cases in misclassified_cases.items():
        print(f"\n{level_name} Level:")
        
        # Thống kê labels bị miss nhiều nhất
        missing_labels = defaultdict(int)
        extra_labels = defaultdict(int)
        
        for case in cases:
            for label in case['missing_labels']:
                missing_labels[label] += 1
            for label in case['extra_labels']:
                extra_labels[label] += 1
        
        if missing_labels:
            print(f"  Labels bị miss nhiều nhất:")
            for label, count in sorted(missing_labels.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {label}: {count} lần")
        
        if extra_labels:
            print(f"  Labels bị predict thừa nhiều nhất:")
            for label, count in sorted(extra_labels.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {label}: {count} lần")

if __name__ == "__main__":
    misclassified_cases = analyze_misclassifications()
    find_common_error_patterns(misclassified_cases) 