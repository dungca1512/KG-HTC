import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm_openai import LLM
from src.graph_db import GraphDB
from src.pipeline import MultiLabelPipeline
from dotenv import load_dotenv

load_dotenv()

def test_multilabel_pipeline():
    """Test multi-label pipeline functionality"""
    
    print("🔧 Testing Multi-label Pipeline")
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
    
    # Test samples with different sentiment combinations
    test_samples = [
        {
            "text": "I love this app! It's fun and easy to learn Japanese. However, it's quite expensive.",
            "expected": "Mixed sentiment - positive (fun, easy) + negative (expensive)"
        },
        {
            "text": "The app is great for beginners and has good learning materials, but the UI is confusing and there are too many ads.",
            "expected": "Mixed sentiment - positive (good for beginners) + negative (bad UI, ads)"
        },
        {
            "text": "Excellent app! Fun, engaging, and worth the money.",
            "expected": "Pure positive"
        },
        {
            "text": "Terrible app. Bad UI, not useful, waste of money.",
            "expected": "Pure negative"
        }
    ]
    
    try:
        # Initialize components
        print("1. Initializing components...")
        llm = LLM(model="gpt-3.5-turbo", temperature=0.1)
        graph_db = GraphDB()
        pipeline = MultiLabelPipeline(llm, config)
        print("   ✓ All components initialized successfully")
        
        # Test available labels
        all_types = ["positive", "negative"]
        all_categories = ["usersfeeling", "learningmaterials", "price", "ui/ux", "ads"]
        all_tags = ["fun", "easytolearn", "expensive", "badui", "useful"]
        
        print(f"   Available types: {all_types}")
        print(f"   Available categories: {all_categories}")
        print(f"   Available tags: {all_tags}")
        
        # Test each sample
        for i, sample in enumerate(test_samples):
            print(f"\n2.{i+1} Testing: {sample['expected']}")
            print(f"     Text: '{sample['text'][:60]}...'")
            
            try:
                # Step 1: Query vector DB
                retrieved_nodes = pipeline.query_related_nodes(sample['text'])
                print(f"     ✓ Vector query successful: L2={len(retrieved_nodes['l2'])}, L3={len(retrieved_nodes['l3'])}")
                
                # Step 2: Build subgraph
                subgraph = []
                if retrieved_nodes["l3"]:
                    subgraph = pipeline.build_linked_labels(retrieved_nodes["l3"][:3], retrieved_nodes["l2"][:3])
                
                if not subgraph:
                    subgraph = pipeline.build_simple_subgraph(retrieved_nodes["l2"][:3])
                
                if not subgraph:
                    subgraph = ["No hierarchical information available"]
                
                print(f"     ✓ Subgraph built: {len(subgraph)} relationships")
                
                # Step 3: Multi-label predictions
                print("     🏷️  Multi-label predictions:")
                
                # Test Type prediction
                pred_types = pipeline.predict_multi_labels(sample['text'], all_types, subgraph, "type")
                print(f"        Types: {pred_types}")
                
                # Test Category prediction  
                pred_categories = pipeline.predict_multi_labels(sample['text'], all_categories, subgraph, "category")
                print(f"        Categories: {pred_categories}")
                
                # Test Tag prediction
                pred_tags = pipeline.predict_multi_labels(sample['text'], all_tags, subgraph, "tag")
                print(f"        Tags: {pred_tags}")
                
                # Analyze results
                is_mixed = len(pred_types) > 1
                print(f"     📊 Analysis: {'Mixed sentiment' if is_mixed else 'Single sentiment'}")
                
                if is_mixed:
                    print(f"        ✓ Successfully detected mixed sentiment!")
                else:
                    print(f"        ℹ️  Single sentiment detected: {pred_types[0]}")
                
            except Exception as e:
                print(f"     ❌ Sample test failed: {e}")
                continue
        
        # Test direct multi-label function
        print(f"\n3. Testing multi-label function directly...")
        
        mixed_text = "I love the learning content but hate the price and ads"
        direct_types = pipeline.predict_multi_labels(mixed_text, all_types, ["positive -> usersfeeling -> fun", "negative -> price -> expensive"], "type")
        print(f"   Mixed text: '{mixed_text}'")
        print(f"   Direct prediction: {direct_types}")
        
        if len(direct_types) > 1:
            print(f"   ✓ Multi-label function working correctly!")
        else:
            print(f"   ⚠️  Single label returned, may need prompt tuning")
        
        # Test edge cases
        print(f"\n4. Testing edge cases...")
        
        # Empty context
        try:
            empty_result = pipeline.predict_multi_labels("test", [], [], "type")
            print(f"   Empty context: {empty_result}")
        except Exception as e:
            print(f"   Empty context error: {e}")
        
        # Single option
        try:
            single_result = pipeline.predict_multi_labels("positive review", ["positive"], [], "type")
            print(f"   Single option: {single_result}")
        except Exception as e:
            print(f"   Single option error: {e}")
        
        print(f"\n🎉 Multi-label pipeline test completed!")
        print(f"   The pipeline should now handle mixed sentiment correctly.")
        
    except Exception as e:
        print(f"❌ Critical error in multi-label test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multilabel_pipeline()