import sys
import os
import json

# Thêm đường dẫn cha vào sys.path để import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db import VectorDB

# Sửa đường dẫn cho giống init_review.py
DATABASE_PATH = 'database/custom_review'  # Đường dẫn relative từ code_review/
COLLECTION_NAME = 'custom_review'

# In ra đường dẫn tuyệt đối của vector DB khi dump
print("VECTOR DB PATH (dump):", os.path.abspath(DATABASE_PATH))

# Khởi tạo vector DB
vector_db = VectorDB(DATABASE_PATH, COLLECTION_NAME)

# Lấy toàn bộ dữ liệu
all_data = vector_db._collection.get()

# In ra màn hình
print(json.dumps(all_data, indent=2, ensure_ascii=False))

# Lưu ra file
os.makedirs('dataset', exist_ok=True)  # Tạo thư mục dataset nếu chưa có
output_path = 'dataset/chromadb_dump.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f'Đã lưu toàn bộ dữ liệu ChromaDB vào file {output_path}')