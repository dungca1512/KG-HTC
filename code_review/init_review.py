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
        "data_path": "dataset/label.csv",
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

    print("Reading CSV file...")
    df = pd.read_csv(config["data_path"])
    print(f"Label CSV shape: {df.shape}")

    # Chuẩn hóa dữ liệu
    df = df.dropna(subset=['Type', 'Category', 'Tag'])
    df['Type_clean'] = df['Type'].str.strip().str.lower().str.replace(' ', '').str.replace("'", '').str.replace('"', '')
    df['Category_clean'] = df['Category'].str.strip().str.lower().str.replace(' ', '').str.replace("'", '').str.replace('"', '')
    df['Tag_clean'] = df['Tag'].str.strip().str.lower().str.replace(' ', '').str.replace("'", '').str.replace('"', '')

    # Lấy unique cho vector DB
    all_types = df['Type_clean'].unique().tolist()
    all_categories = df['Category_clean'].unique().tolist()
    all_tags = df['Tag_clean'].unique().tolist()

    print(f"Unique types: {len(all_types)} | categories: {len(all_categories)} | tags: {len(all_tags)}")

    # Clear and rebuild graph database
    print("\nClearing and rebuilding Graph Database...")
    graph_db = GraphDB()
    clear_query = "MATCH (n) DETACH DELETE n"
    try:
        graph_db.create_database(clear_query)
        print("✓ Graph cleared successfully")
    except Exception as e:
        print(f"Error clearing graph: {e}")

    query_create_graph = """
    MERGE (level1:Category1 {name: $l1})
    MERGE (level2:Category2 {name: $l2})
    MERGE (level3:Category3 {name: $l3})
    MERGE (level1)-[:contains]->(level2)
    MERGE (level2)-[:contains]->(level3)
    """

    print(f"Creating {len(df)} graph relationships...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating graph relationships"):
        l1 = row['Type_clean']
        l2 = row['Category_clean']
        l3 = row['Tag_clean']
        try:
            graph_db.create_database(query_create_graph, l1=l1, l2=l2, l3=l3)
        except Exception as e:
            print(f"Error creating graph for {l1}->{l2}->{l3}: {e}")
            continue

    # Clear and rebuild vector database
    print("\nClearing and rebuilding Vector Database...")
    import shutil, os
    try:
        if os.path.exists(config["vectdb_path"]):
            shutil.rmtree(config["vectdb_path"])
        print("✓ Old vector database removed")
    except Exception as e:
        print(f"Error removing old vector DB: {e}")

    vector_db = VectorDB(
        database_path=config["vectdb_path"],
        collection_name=config["data_name"]
    )

    print(f"Adding {len(all_types)} level 1 labels...")
    vector_db.batch_add(all_types, metadatas=[{"level": "Category1"}] * len(all_types))
    print(f"Adding {len(all_categories)} level 2 labels...")
    vector_db.batch_add(all_categories, metadatas=[{"level": "Category2"}] * len(all_categories))
    print(f"Adding {len(all_tags)} level 3 labels...")
    vector_db.batch_add(all_tags, metadatas=[{"level": "Category3"}] * len(all_tags))
    print("VectorDB created successfully!")

    # Lưu lại file đã chuẩn hóa (nếu cần)
    df[['Type_clean', 'Category_clean', 'Tag_clean']].to_csv("dataset/multilabel_review_processed.csv", index=False)
    print(f"Processed data saved to: dataset/multilabel_review_processed.csv")

    return config

if __name__ == "__main__":
    config = init_multilabel_data()