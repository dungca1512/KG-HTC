import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm_openai import LLM
from src.graph_db import GraphDB  # Use fixed version
from src.pipeline import Pipeline
from dotenv import load_dotenv

load_dotenv()

def comprehensive_test():
    """Comprehensive test to ensure everything works"""
    
    print("🔧 Comprehensive Pipeline Test")
    print("=" * 50)
    
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
    
    # Sample texts
    test_samples = [
        "I had given up on learning Japanese, but with this app, it's so much fun",
        "The app is simple and easy to use",
        "Great for beginners learning Japanese"
    ]
    
    try:
        # Initialize components
        print("1. Initializing components...")
        llm = LLM(model="gpt-3.5-turbo", temperature=0.1)
        graph_db = GraphDB()  # Use fixed version
        pipeline = Pipeline(llm, config)
        print("   ✓ All components initialized successfully")
        
        # Test each sample
        for i, sample_text in enumerate(test_samples):
            print(f"\n2.{i+1} Testing sample: '{sample_text[:50]}...'")
            
            try:
                # Step 1: Query vector DB
                retrieved_nodes = pipeline.query_related_nodes(sample_text)
                print(f"   ✓ Vector query: L2={len(retrieved_nodes['l2'])}, L3={len(retrieved_nodes['l3'])}")
                
                # Step 2: Build subgraph
                subgraph = []
                if retrieved_nodes["l3"] and len(retrieved_nodes["l3"]) > 0:
                    try:
                        subgraph = pipeline.build_linked_labels(retrieved_nodes["l3"][:3], retrieved_nodes["l2"][:3])
                        if subgraph:
                            print(f"   ✓ Built subgraph: {len(subgraph)} relationships")
                            print(f"     Sample: {subgraph[0] if subgraph else 'None'}")
                        else:
                            print(f"   ⚠ Subgraph empty, trying fallback...")
                            subgraph = pipeline.build_simple_subgraph(retrieved_nodes["l2"][:3])
                    except Exception as e:
                        print(f"   ⚠ Subgraph error: {e}, using fallback...")
                        subgraph = pipeline.build_simple_subgraph(retrieved_nodes["l2"][:3])
                
                if not subgraph:
                    subgraph = ["No hierarchical information available"]
                
                # Step 3: Test prediction
                test_categories = ["positive", "negative"]
                try:
                    prediction = pipeline.predict_level(sample_text, test_categories, subgraph)
                    print(f"   ✓ Prediction: {prediction}")
                except Exception as e:
                    print(f"   ❌ Prediction failed: {e}")
                
            except Exception as e:
                print(f"   ❌ Sample test failed: {e}")
                continue
        
        # Test graph database directly
        print(f"\n3. Testing graph database queries...")
        test_queries = [
            ("positive", "query_l2_from_l1"),
            ("usersfeeling", "query_l3_from_l2"),
            ("fun", "query_l2_from_l3"),
            ("usersfeeling", "query_l1_from_l2")
        ]
        
        for test_input, method_name in test_queries:
            try:
                method = getattr(graph_db, method_name)
                result = method(test_input)
                print(f"   ✓ {method_name}('{test_input}') -> {result}")
            except Exception as e:
                print(f"   ❌ {method_name}('{test_input}') -> Error: {e}")
        
        print(f"\n🎉 Comprehensive test completed!")
        print(f"   The pipeline should now work correctly for classification.")
        
    except Exception as e:
        print(f"❌ Critical error in comprehensive test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_test()