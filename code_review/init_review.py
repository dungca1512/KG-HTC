import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from src.vector_db import VectorDB
from src.graph_db import GraphDB

load_dotenv()

def init_multilabel_data():
    config = {
        "data_name": "custom_review",
        "data_path": "/Users/dungca/KG-HTC/sample_data.xlsx",
        "vectdb_path": "database/custom_review",
        "template": {
            "sys": "prompts/system/custom/llm_graph.txt",
            "user": "prompts/user/custom/llm_graph.txt"
        },
        "query_params": {
            "l2_top_k": 10,
            "l3_top_k": 40
        }
    }

    # Đọc dữ liệu từ Excel
    print("Reading Excel file...")
    
    df_reviews = pd.read_excel(config["data_path"], sheet_name="Raw Review")
    df_tags = pd.read_excel(config["data_path"], sheet_name="Tag list")
    
    print(f"Raw Review shape: {df_reviews.shape}")
    print(f"Tag list shape: {df_tags.shape}")
    
    # Lọc dữ liệu hợp lệ
    print("\nFiltering valid data...")
    review_text_col = 'Content translate'
    df_reviews_filtered = df_reviews.dropna(subset=[review_text_col, 'Type', 'Category', 'Tag'])
    
    print(f"Valid reviews after filtering: {len(df_reviews_filtered)}")
    
    if len(df_reviews_filtered) == 0:
        print("No valid data found!")
        return
    
    # Xử lý multi-label: KHÔNG expand, giữ nguyên comma-separated values
    processed_rows = []
    
    for _, row in df_reviews_filtered.iterrows():
        review_text = row[review_text_col]
        type_val = str(row['Type']).strip() if pd.notna(row['Type']) else ""
        category_val = str(row['Category']).strip() if pd.notna(row['Category']) else ""
        tag_val = str(row['Tag']).strip() if pd.notna(row['Tag']) else ""
        
        if type_val and category_val and tag_val:
            processed_rows.append({
                'review_text': review_text,
                'type': type_val,  # Giữ nguyên "positive,negative" 
                'category': category_val,  # Giữ nguyên "cat1,cat2"
                'tag': tag_val,  # Giữ nguyên "tag1,tag2"
                'app': row.get('App', ''),
                'platform': row.get('Platform', ''),
                'country': row.get('Country', '')
            })
    
    print(f"Total processed rows: {len(processed_rows)}")
    
    if len(processed_rows) == 0:
        print("No processed data available!")
        return
    
    # Tạo DataFrame từ processed data
    df_processed = pd.DataFrame(processed_rows)
    
    # Lấy unique individual labels cho graph và vector DB
    all_types = set()
    all_categories = set()
    all_tags = set()
    
    for _, row in df_processed.iterrows():
        # Split để lấy individual labels cho graph
        types = [t.strip().lower().replace(' ', '').replace('\'', '').replace('\"', '') 
                for t in str(row['type']).split(',')]
        categories = [c.strip().lower().replace(' ', '').replace('\'', '').replace('\"', '') 
                     for c in str(row['category']).split(',')]
        tags = [t.strip().lower().replace(' ', '').replace('\'', '').replace('\"', '') 
               for t in str(row['tag']).split(',')]
        
        all_types.update(types)
        all_categories.update(categories)
        all_tags.update(tags)
    
    all_types = list(all_types)
    all_categories = list(all_categories)
    all_tags = list(all_tags)
    
    print(f"Unique individual types: {all_types}")
    print(f"Unique individual categories: {len(all_categories)}")
    print(f"Unique individual tags: {len(all_tags)}")
    
    # Clear and rebuild graph database với individual labels
    print("\nClearing and rebuilding Graph Database...")
    graph_db = GraphDB()
    
    # Clear existing graph
    clear_query = "MATCH (n) DETACH DELETE n"
    try:
        graph_db.create_database(clear_query)
        print("✓ Graph cleared successfully")
    except Exception as e:
        print(f"Error clearing graph: {e}")
    
    # Rebuild graph với all possible combinations
    query_create_graph = """
    MERGE (level1:Category1 {name: $l1})
    MERGE (level2:Category2 {name: $l2})
    MERGE (level3:Category3 {name: $l3})
    MERGE (level1)-[:contains]->(level2)
    MERGE (level2)-[:contains]->(level3)
    """
    
    # Tạo combinations từ data thực tế
    combinations = set()
    for _, row in df_processed.iterrows():
        types = [t.strip().lower().replace(' ', '').replace('\'', '').replace('\"', '') 
                for t in str(row['type']).split(',')]
        categories = [c.strip().lower().replace(' ', '').replace('\'', '').replace('\"', '') 
                     for c in str(row['category']).split(',')]
        tags = [t.strip().lower().replace(' ', '').replace('\'', '').replace('\"', '') 
               for t in str(row['tag']).split(',')]
        
        # Tạo all combinations
        for t in types:
            for c in categories:
                for tag in tags:
                    if t and c and tag:
                        combinations.add((t, c, tag))
    
    print(f"Creating {len(combinations)} graph relationships...")
    
    # Tạo các mối quan hệ trong Neo4j
    for l1, l2, l3 in tqdm(combinations, desc="Creating graph relationships"):
        try:
            graph_db.create_database(query_create_graph, l1=l1, l2=l2, l3=l3)
        except Exception as e:
            print(f"Error creating graph for {l1}->{l2}->{l3}: {e}")
            continue
    
    # Clear and rebuild vector database
    print("\nClearing and rebuilding Vector Database...")
    
    # Remove existing vector database
    import shutil
    import os
    try:
        if os.path.exists(config["vectdb_path"]):
            shutil.rmtree(config["vectdb_path"])
        print("✓ Old vector database removed")
    except Exception as e:
        print(f"Error removing old vector DB: {e}")
    
    # Create new vector database
    vector_db = VectorDB(
        database_path=config["vectdb_path"],
        collection_name=config["data_name"]
    )
    
    # In ra đường dẫn tuyệt đối của vector DB
    print("VECTOR DB PATH (init):", os.path.abspath(config["vectdb_path"]))
    
    # Sử dụng descriptions từ Tag list nếu có
    print("Processing tag descriptions...")
    
    # Tạo mapping từ tag/category name -> description
    tag_desc_map = {}
    category_desc_map = {}
    type_desc_map = {}
    
    for _, row in df_tags.iterrows():
        tag_key = str(row['Tag']).lower().replace(' ', '').replace('\'', '').replace('\"', '')
        category_key = str(row['Category']).lower().replace(' ', '').replace('\'', '').replace('\"', '')
        type_key = str(row['Type']).lower().replace(' ', '').replace('\'', '').replace('\"', '')
        
        if pd.notna(row['Detail']):
            tag_desc_map[tag_key] = f"{row['Tag']}: {row['Detail']}"
            category_desc_map[category_key] = f"{row['Category']}: {row['Detail']}"
            type_desc_map[type_key] = f"{row['Type']}: {row['Detail']}"
    
    # Tạo texts cho embeddings với descriptions nếu có
    l1_texts = [type_desc_map.get(tag, tag) for tag in all_types]
    l2_texts = [category_desc_map.get(tag, tag) for tag in all_categories]
    l3_texts = [tag_desc_map.get(tag, tag) for tag in all_tags]
    
    print(f"Adding {len(all_types)} level 1 labels...")
    vector_db.batch_add(l1_texts, metadatas=[{"level": "Category1"}] * len(all_types))
    print("Số lượng document hiện tại sau L1:", len(vector_db._collection.get()["documents"]))
    
    print(f"Adding {len(all_categories)} level 2 labels...")
    vector_db.batch_add(l2_texts, metadatas=[{"level": "Category2"}] * len(all_categories))
    print("Số lượng document hiện tại sau L2:", len(vector_db._collection.get()["documents"]))
    
    print(f"Adding {len(all_tags)} level 3 labels...")
    vector_db.batch_add(l3_texts, metadatas=[{"level": "Category3"}] * len(all_tags))
    print("Số lượng document hiện tại sau L3:", len(vector_db._collection.get()["documents"]))
    
    print("VectorDB created successfully!")
    
    # Lưu processed data với multi-labels
    processed_data = df_processed[['review_text', 'type', 'category', 'tag', 'app', 'platform', 'country']].copy()
    processed_data.columns = ['text', 'type', 'category', 'tag', 'app', 'platform', 'country']
    
    # Tạo thư mục dataset nếu chưa có
    os.makedirs("dataset", exist_ok=True)
    
    processed_data.to_csv("dataset/multilabel_review_processed.csv", index=False)
    
    print(f"\nProcessed data saved to: dataset/multilabel_review_processed.csv")
    print(f"Total samples: {len(processed_data)}")
    print(f"Sample multi-labels:")
    for i in range(min(3, len(processed_data))):
        row = processed_data.iloc[i]
        print(f"  {i+1}. Type: {row['type']} | Category: {row['category']} | Tag: {row['tag']}")
    
    return config

if __name__ == "__main__":
    config = init_multilabel_data()