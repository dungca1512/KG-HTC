import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.graph_db import GraphDB
from dotenv import load_dotenv

load_dotenv()

def test_graph_database():
    """Test graph database to see what's wrong"""
    
    print("Testing Graph Database...")
    
    try:
        graph_db = GraphDB()
        print("✓ Graph DB initialized successfully")
        
        # Test với một số sample data
        sample_l1 = "positive"
        sample_l2 = "usersfeeling"  
        sample_l3_descriptions = [
            "Fun: Người dùng thấy vui vẻ, thú vị, thoải mái khi học",
            "Simple: Người dùng thấy app đơn giản",
            "Useful: Hữu ích cho việc học"
        ]
        
        print(f"\n=== Testing Graph Queries ===")
        
        # Test 1: Query L2 from L1
        print(f"1. Testing query_l2_from_l1('{sample_l1}')...")
        try:
            l2_children = graph_db.query_l2_from_l1(sample_l1)
            print(f"   ✓ Found {len(l2_children)} L2 children: {l2_children}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 2: Query L3 from L2
        print(f"2. Testing query_l3_from_l2('{sample_l2}')...")
        try:
            l3_children = graph_db.query_l3_from_l2(sample_l2)
            print(f"   ✓ Found {len(l3_children)} L3 children: {l3_children}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 3: Query L1 from L2
        print(f"3. Testing query_l1_from_l2('{sample_l2}')...")
        try:
            l1_parent = graph_db.query_l1_from_l2(sample_l2)
            print(f"   ✓ Found L1 parent: {l1_parent}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 4: Query L2 from L3 (với description)
        print(f"4. Testing query_l2_from_l3 with descriptions...")
        for desc in sample_l3_descriptions:
            try:
                # Thử với full description
                l2_parent = graph_db.query_l2_from_l3(desc)
                print(f"   ✓ '{desc[:30]}...' -> {l2_parent}")
            except Exception as e:
                print(f"   ❌ '{desc[:30]}...' -> Error: {e}")
                
                # Thử với tag name only (trước dấu :)
                tag_name = desc.split(':')[0].lower().replace(' ', '')
                try:
                    l2_parent = graph_db.query_l2_from_l3(tag_name)
                    print(f"   ✓ '{tag_name}' -> {l2_parent}")
                except Exception as e2:
                    print(f"   ❌ '{tag_name}' -> Error: {e2}")
        
        # Test 5: Kiểm tra pipeline build_linked_labels
        print(f"\n5. Testing build_linked_labels...")
        try:
            from src.llm_openai import LLM
            from src.pipeline import Pipeline
            
            config = {
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
            pipeline = Pipeline(llm, config)
            
            # Test với sample L3 và L2
            sample_l3 = sample_l3_descriptions[:2]  # 2 descriptions đầu
            sample_l2 = ["usersfeeling", "learningmaterials"]
            
            print(f"   Input L3: {sample_l3}")
            print(f"   Input L2: {sample_l2}")
            
            try:
                subgraph = pipeline.build_linked_labels(sample_l3, sample_l2)
                print(f"   ✓ Built subgraph: {subgraph}")
            except Exception as e:
                print(f"   ❌ build_linked_labels failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Debug step by step
                print(f"   Debug: Testing individual L3 nodes...")
                for l3_node in sample_l3:
                    try:
                        # Test với full description
                        l2_node = graph_db.query_l2_from_l3(l3_node)
                        print(f"     '{l3_node[:20]}...' -> {l2_node}")
                    except Exception as e_detail:
                        print(f"     '{l3_node[:20]}...' -> Error: {e_detail}")
                        
                        # Test với tag name only
                        tag_only = l3_node.split(':')[0].lower().replace(' ', '')
                        try:
                            l2_node = graph_db.query_l2_from_l3(tag_only)
                            print(f"     '{tag_only}' -> {l2_node}")
                        except Exception as e_tag:
                            print(f"     '{tag_only}' -> Error: {e_tag}")
            
        except Exception as e:
            print(f"   ❌ Pipeline initialization failed: {e}")
        
        # Test 6: Kiểm tra tất cả nodes trong graph
        print(f"\n6. Checking all graph nodes...")
        try:
            # Query all L1 nodes
            query_all_l1 = "MATCH (n:Category1) RETURN n.name LIMIT 10"
            result = graph_db._query_database(query_all_l1)
            l1_nodes = [record.get("n.name") for record in result.records]
            print(f"   Sample L1 nodes: {l1_nodes}")
            
            # Query all L2 nodes  
            query_all_l2 = "MATCH (n:Category2) RETURN n.name LIMIT 10"
            result = graph_db._query_database(query_all_l2)
            l2_nodes = [record.get("n.name") for record in result.records]
            print(f"   Sample L2 nodes: {l2_nodes}")
            
            # Query all L3 nodes
            query_all_l3 = "MATCH (n:Category3) RETURN n.name LIMIT 10"
            result = graph_db._query_database(query_all_l3)
            l3_nodes = [record.get("n.name") for record in result.records]
            print(f"   Sample L3 nodes: {l3_nodes}")
            
        except Exception as e:
            print(f"   ❌ Error querying all nodes: {e}")
            
    except Exception as e:
        print(f"❌ Graph DB initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graph_database()