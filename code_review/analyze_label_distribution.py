import pandas as pd

# Đọc file processed
input_path = 'code_review/dataset/multilabel_review_processed.csv'
df = pd.read_csv(input_path)

print("=== Phân bổ nhãn Type ===")
type_dist = df['Type_clean'].value_counts(normalize=True).round(3) * 100
print(type_dist)

type_dist.to_csv('code_review/dataset/type_distribution.csv')

print("\n=== Phân bổ nhãn Category ===")
category_dist = df['Category_clean'].value_counts(normalize=True).round(3) * 100
print(category_dist)

category_dist.to_csv('code_review/dataset/category_distribution.csv')

print("\n=== Phân bổ nhãn Tag (top 20) ===")
tag_dist = df['Tag_clean'].value_counts(normalize=True).round(3) * 100
print(tag_dist.head(20))

tag_dist.to_csv('code_review/dataset/tag_distribution.csv')

print("\nĐã xuất file phân bổ nhãn vào code_review/dataset/") 