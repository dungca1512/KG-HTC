import sys
from pathlib import Path
import json

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm_openai import LLM
from src.graph_db import GraphDB
from src.pipeline import MultiLabelPipeline

def predict_review_labels(review_text):
    """
    Dự đoán nhãn cho một review mới
    """
    config = {
        "data_name": "custom_review",
        "vectdb_path": "database/custom_review",
        "template": {
            "sys": "prompts/system/custom/llm_graph.txt",
            "user": "prompts/user/custom/llm_graph.txt"
        },
        "query_params": {
            "l2_top_k": 15,
            "l3_top_k": 30
        }
    }
    
    try:
        # Khởi tạo components
        print("Initializing components...")
        llm = LLM(model="gpt-3.5-turbo", temperature=0.1)
        graph_db = GraphDB()
        pipeline = MultiLabelPipeline(llm, config)
        print("✓ Components initialized successfully")
        
        # Lấy tất cả nhãn có sẵn từ vector DB
        print("Loading available labels...")
        all_types = set()
        all_categories = set()
        all_tags = set()
        
        # Query vector DB để lấy tất cả nhãn
        l2_result = pipeline._vector_db.query_l2("", 1000)  # Lấy tất cả L2
        l3_result = pipeline._vector_db.query_l3("", 1000)  # Lấy tất cả L3
        
        if l2_result["documents"]:
            all_categories.update(l2_result["documents"][0])
        if l3_result["documents"]:
            all_tags.update(l3_result["documents"][0])
            
        # Lấy types từ graph DB
        try:
            # Query một số L1 nodes để lấy types
            for cat in list(all_categories)[:10]:  # Lấy từ 10 categories đầu
                try:
                    l1 = graph_db.query_l1_from_l2(cat)
                    all_types.add(l1)
                except:
                    continue
        except:
            # Fallback: dùng các types cơ bản
            all_types = {"positive", "negative"}
        
        all_types = list(all_types)
        all_categories = list(all_categories)
        all_tags = list(all_tags)
        
        print(f"Available labels - Types: {len(all_types)}, Categories: {len(all_categories)}, Tags: {len(all_tags)}")
        
        # Dự đoán
        print(f"\nAnalyzing review: {review_text[:100]}...")
        
        # Get related nodes from vector DB
        retrieved_nodes = pipeline.query_related_nodes(review_text)
        print(f"Retrieved L2: {len(retrieved_nodes['l2'])}, L3: {len(retrieved_nodes['l3'])}")
        
        # Build subgraph
        subgraph = []
        if retrieved_nodes["l3"]:
            subgraph = pipeline.build_linked_labels(retrieved_nodes["l3"], retrieved_nodes["l2"])
        
        if not subgraph:
            subgraph = pipeline.build_simple_subgraph(retrieved_nodes["l2"])
        
        if not subgraph:
            subgraph = ["No hierarchical information available"]
        
        print(f"Built subgraph with {len(subgraph)} relationships")
        
        # Multi-label prediction for Type
        print("Predicting types...")
        pred_types = pipeline.predict_multi_labels(review_text, all_types, subgraph, "type")
        print(f"Predicted types: {pred_types}")
        
        # Multi-label prediction for Category
        print("Predicting categories...")
        potential_categories = set()
        for pred_type in pred_types:
            try:
                type_children = graph_db.query_l2_from_l1(pred_type)
                potential_categories.update(type_children)
            except:
                pass
        
        potential_categories.update(retrieved_nodes["l2"])
        potential_categories = list(potential_categories)
        
        if not potential_categories:
            potential_categories = all_categories[:15]
        
        pred_categories = pipeline.predict_multi_labels(review_text, potential_categories, subgraph, "category")
        print(f"Predicted categories: {pred_categories}")
        
        # Multi-label prediction for Tags
        print("Predicting tags...")
        potential_tags = set()
        for pred_cat in pred_categories:
            try:
                cat_children = graph_db.query_l3_from_l2(pred_cat)
                potential_tags.update(cat_children)
            except:
                pass
        
        if retrieved_nodes["l3"]:
            extracted_tags = [pipeline._extract_tag_name(desc) for desc in retrieved_nodes["l3"]]
            potential_tags.update(extracted_tags)
        
        potential_tags = list(potential_tags)
        
        if not potential_tags:
            potential_tags = all_tags[:15]
        
        pred_tags = pipeline.predict_multi_labels(review_text, potential_tags, subgraph, "tag")
        print(f"Predicted tags: {pred_tags}")
        
        # Kết quả
        result = {
            "review_text": review_text,
            "predicted_types": pred_types,
            "predicted_categories": pred_categories,
            "predicted_tags": pred_tags,
            "subgraph_used": subgraph[:3]
        }
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Ví dụ sử dụng
    review_text = input("Nhập review cần phân tích: ")
    
    if not review_text.strip():
        print("Review không được để trống!")
        return
    
    result = predict_review_labels(review_text)
    
    if result:
        print("\n" + "="*50)
        print("KẾT QUẢ PHÂN TÍCH")
        print("="*50)
        print(f"Review: {result['review_text'][:100]}...")
        print(f"Types: {result['predicted_types']}")
        print(f"Categories: {result['predicted_categories']}")
        print(f"Tags: {result['predicted_tags']}")
        print(f"Subgraph: {result['subgraph_used']}")
        
        # Lưu kết quả
        with open("code_review/dataset/single_review_result.json", "w", encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print("\nKết quả đã được lưu vào: code_review/dataset/single_review_result.json")
    else:
        print("Có lỗi xảy ra trong quá trình phân tích!")

if __name__ == "__main__":
    main() 