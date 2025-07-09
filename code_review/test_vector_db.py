import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.vector_db import VectorDB
from dotenv import load_dotenv

load_dotenv()

def test_vector_database():
    """Test vector database to see what's wrong"""
    
    config = {
        "data_name": "custom_review",
        "vectdb_path": "database/custom_review",
    }
    
    print("Testing Vector Database...")
    
    try:
        # Khởi tạo vector database
        vector_db = VectorDB(config["vectdb_path"], config["data_name"])
        print("✓ Vector DB initialized successfully")
        
        # Kiểm tra collection có dữ liệu không
        collection_data = vector_db._collection.get()
        print(f"✓ Collection has {len(collection_data['documents'])} documents")
        
        if len(collection_data['documents']) == 0:
            print("❌ ERROR: Vector database is empty!")
            print("You need to run init_review.py first to populate the database")
            return
        
        # Hiển thị sample data
        print(f"\nSample documents:")
        for i in range(min(5, len(collection_data['documents']))):
            print(f"  {i}: {collection_data['documents'][i]}")
            print(f"     Metadata: {collection_data['metadatas'][i]}")
        
        # Test query với sample text
        sample_text = "I had given up on learning Japanese, but with this app, it's so much fun"
        print(f"\nTesting query with: '{sample_text}'")
        
        # Test L2 query
        try:
            l2_result = vector_db.query_l2(sample_text, 5)
            print(f"✓ L2 query successful")
            print(f"  L2 documents: {l2_result['documents']}")
            print(f"  L2 metadatas: {l2_result['metadatas']}")
            print(f"  L2 distances: {l2_result['distances']}")
        except Exception as e:
            print(f"❌ L2 query failed: {e}")
            
        # Test L3 query
        try:
            l3_result = vector_db.query_l3(sample_text, 5)
            print(f"✓ L3 query successful")
            print(f"  L3 documents: {l3_result['documents']}")
            print(f"  L3 metadatas: {l3_result['metadatas']}")
            print(f"  L3 distances: {l3_result['distances']}")
        except Exception as e:
            print(f"❌ L3 query failed: {e}")
            
        # Test pipeline query_related_nodes
        print(f"\nTesting pipeline query_related_nodes...")
        sys.path.append('.')
        from src.llm_openai import LLM
        from src.pipeline import Pipeline
        
        pipeline_config = {
            "data_name": "custom_review",
            "vectdb_path": "database/custom_review",
            "template": {
                "sys": "prompts/system/custom/llm_graph.txt",
                "user": "prompts/user/custom/llm_graph.txt"
            },
            "query_params": {
                "l2_top_k": 5,
                "l3_top_k": 5
            }
        }
        
        llm = LLM(model="gpt-3.5-turbo", temperature=0.1)
        pipeline = Pipeline(llm, pipeline_config)
        
        try:
            retrieved_nodes = pipeline.query_related_nodes(sample_text)
            print(f"✓ Pipeline query successful")
            print(f"  Retrieved L2: {retrieved_nodes['l2']}")
            print(f"  Retrieved L3: {retrieved_nodes['l3']}")
            print(f"  L2 type: {type(retrieved_nodes['l2'])}")
            print(f"  L3 type: {type(retrieved_nodes['l3'])}")
            
            if isinstance(retrieved_nodes['l2'], list) and len(retrieved_nodes['l2']) > 0:
                print(f"  L2 first element: {retrieved_nodes['l2'][0]}")
            else:
                print(f"  ❌ L2 is empty or not a list!")
                
            if retrieved_nodes['l3'] is not None:
                if isinstance(retrieved_nodes['l3'], list) and len(retrieved_nodes['l3']) > 0:
                    print(f"  L3 first element: {retrieved_nodes['l3'][0]}")
                else:
                    print(f"  ❌ L3 is empty or not a list!")
            else:
                print(f"  L3 is None")
                
        except Exception as e:
            print(f"❌ Pipeline query failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Vector DB initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_database()