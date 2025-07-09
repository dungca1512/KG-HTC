import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import json
import os

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm_openai import LLM
from src.graph_db import GraphDB
from src.pipeline import MultiLabelPipeline


def create_multilabel_prompt_templates():
    """Tạo các template prompts cho multi-label classification"""
    
    os.makedirs("prompts/system/custom", exist_ok=True)
    os.makedirs("prompts/user/custom", exist_ok=True)
    
    # System prompt template cho multi-label
    system_prompt = """Classify the app review text into categories from: {category_text}. 

The review may contain multiple aspects - include ALL relevant categories.

Here is the partial knowledge graph showing hierarchical relationships: 
\"\"\"
{knowledge}
\"\"\"

Instructions:
1. Analyze the review for ALL applicable categories
2. If multiple categories apply, output them separated by commas  
3. Use the knowledge graph to understand relationships
4. Output only category names, no extra characters"""
    
    # User prompt template
    user_prompt = """App Review Text: 
\"\"\"
{text}
\"\"\"

Categories:"""
    
    # Lưu templates
    with open("prompts/system/custom/llm_graph.txt", "w", encoding='utf-8') as f:
        f.write(system_prompt)
    
    with open("prompts/user/custom/llm_graph.txt", "w", encoding='utf-8') as f:
        f.write(user_prompt)

def parse_multilabel_ground_truth(label_string):
    """Parse comma-separated ground truth labels"""
    if pd.isna(label_string):
        return []
    
    labels = [label.strip().lower().replace(' ', '').replace('\'', '').replace('\"', '') 
              for label in str(label_string).split(',')]
    return [label for label in labels if label]

def calculate_multilabel_metrics(true_labels_list, pred_labels_list):
    """Calculate multi-label classification metrics"""
    if len(true_labels_list) != len(pred_labels_list):
        return {}
    
    exact_matches = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for true_labels, pred_labels in zip(true_labels_list, pred_labels_list):
        true_set = set(true_labels)
        pred_set = set(pred_labels)
        
        # Exact match
        if true_set == pred_set:
            exact_matches += 1
        
        # Precision, Recall, F1 for this sample
        if len(pred_set) > 0:
            precision = len(true_set & pred_set) / len(pred_set)
        else:
            precision = 0
            
        if len(true_set) > 0:
            recall = len(true_set & pred_set) / len(true_set)
        else:
            recall = 1 if len(pred_set) == 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    n_samples = len(true_labels_list)
    
    return {
        "exact_match_ratio": exact_matches / n_samples,
        "avg_precision": total_precision / n_samples,
        "avg_recall": total_recall / n_samples,
        "avg_f1": total_f1 / n_samples
    }

def run_multilabel_classification():
    """Chạy multi-label classification"""
    
    # Tạo prompt templates
    create_multilabel_prompt_templates()
    
    config = {
        "data_name": "custom_review",
        "data_path": "dataset/multilabel_review_processed.csv",
        "output_path": "dataset/multilabel_results.json",
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

    # Kiểm tra file dữ liệu
    if not os.path.exists(config["data_path"]):
        print(f"Error: {config['data_path']} not found!")
        print("Please run init_multilabel_review.py first")
        return
    
    # Đọc dữ liệu
    print("Loading processed data...")
    df = pd.read_csv(config["data_path"])
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse ground truth multi-labels
    print("\nParsing ground truth multi-labels...")
    df['true_types'] = df['type'].apply(parse_multilabel_ground_truth)
    df['true_categories'] = df['category'].apply(parse_multilabel_ground_truth)
    df['true_tags'] = df['tag'].apply(parse_multilabel_ground_truth)
    
    # Hiển thị thống kê
    print(f"\nMulti-label statistics:")
    multi_type_count = sum(1 for labels in df['true_types'] if len(labels) > 1)
    multi_category_count = sum(1 for labels in df['true_categories'] if len(labels) > 1)
    multi_tag_count = sum(1 for labels in df['true_tags'] if len(labels) > 1)
    
    print(f"Samples with multiple types: {multi_type_count}/{len(df)} ({multi_type_count/len(df)*100:.1f}%)")
    print(f"Samples with multiple categories: {multi_category_count}/{len(df)} ({multi_category_count/len(df)*100:.1f}%)")
    print(f"Samples with multiple tags: {multi_tag_count}/{len(df)} ({multi_tag_count/len(df)*100:.1f}%)")
    
    # Get all unique individual labels
    all_types = set()
    all_categories = set()
    all_tags = set()
    
    for labels_list in df['true_types']:
        all_types.update(labels_list)
    for labels_list in df['true_categories']:
        all_categories.update(labels_list)
    for labels_list in df['true_tags']:
        all_tags.update(labels_list)
    
    all_types = list(all_types)
    all_categories = list(all_categories)
    all_tags = list(all_tags)
    
    print(f"Unique types: {all_types}")
    print(f"Unique categories: {len(all_categories)}")
    print(f"Unique tags: {len(all_tags)}")
    
    # Lấy mẫu để test
    sample_size = min(30, len(df))
    df_sample = df.head(sample_size)
    ds = df_sample.to_dict(orient="records")
    
    # Khởi tạo components
    print("\nInitializing components...")
    try:
        llm = LLM(model="gpt-3.5-turbo", temperature=0.1)
        graph_db = GraphDB()
        pipeline = MultiLabelPipeline(llm, config)
        print("✓ Components initialized successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    # Load existing results
    try:
        with open(config["output_path"], "r", encoding='utf-8') as f:
            inference_list = json.load(f)
    except:
        inference_list = []

    print(f"\nStarting multi-label classification for {len(ds)} samples...")
    
    for idx in tqdm(range(len(ds)), desc="Multi-label classification"):
        if idx < len(inference_list):
            continue
            
        data = ds[idx].copy()
        review_text = data['text']
        
        try:
            print(f"\n--- Sample {idx}: {review_text[:100]}...")
            
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
            print("Predicting types (multi-label)...")
            pred_types = pipeline.predict_multi_labels(review_text, all_types, subgraph, "type")
            print(f"Predicted types: {pred_types}")
            print(f"True types: {data['true_types']}")
            
            # Multi-label prediction for Category
            print("Predicting categories (multi-label)...")
            # Get potential categories from graph based on predicted types
            potential_categories = set()
            for pred_type in pred_types:
                try:
                    type_children = graph_db.query_l2_from_l1(pred_type)
                    potential_categories.update(type_children)
                except:
                    pass
            
            # Add vector-retrieved categories
            potential_categories.update(retrieved_nodes["l2"])
            potential_categories = list(potential_categories)
            
            if not potential_categories:
                potential_categories = all_categories[:15]
            
            pred_categories = pipeline.predict_multi_labels(review_text, potential_categories, subgraph, "category")
            print(f"Predicted categories: {pred_categories}")
            print(f"True categories: {data['true_categories']}")
            
            # Multi-label prediction for Tags
            print("Predicting tags (multi-label)...")
            # Get potential tags from graph based on predicted categories
            potential_tags = set()
            for pred_cat in pred_categories:
                try:
                    cat_children = graph_db.query_l3_from_l2(pred_cat)
                    potential_tags.update(cat_children)
                except:
                    pass
            
            # Add vector-retrieved tags (extract names from descriptions)
            if retrieved_nodes["l3"]:
                extracted_tags = [pipeline._extract_tag_name(desc) for desc in retrieved_nodes["l3"]]
                potential_tags.update(extracted_tags)
            
            potential_tags = list(potential_tags)
            
            if not potential_tags:
                potential_tags = all_tags[:15]
            
            pred_tags = pipeline.predict_multi_labels(review_text, potential_tags, subgraph, "tag")
            print(f"Predicted tags: {pred_tags}")
            print(f"True tags: {data['true_tags']}")
            
            # Lưu kết quả
            data["pred_types"] = pred_types
            data["pred_categories"] = pred_categories
            data["pred_tags"] = pred_tags
            data["subgraph_used"] = subgraph[:3]
            
            print(f"✓ Completed sample {idx}")
            
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            data["pred_types"] = ["error"]
            data["pred_categories"] = ["error"]
            data["pred_tags"] = ["error"]
            data["error"] = str(e)
        
        inference_list.append(data)
        
        # Lưu kết quả sau mỗi sample
        with open(config["output_path"], "w", encoding='utf-8') as f:
            json.dump(inference_list, f, indent=4, ensure_ascii=False)

    print(f"\nMulti-label classification completed!")
    print(f"Results saved to: {config['output_path']}")
    
    # Tính toán metrics
    valid_results = [item for item in inference_list if item.get('pred_types', ['error'])[0] != 'error']
    
    if valid_results:
        print(f"\nCalculating multi-label metrics on {len(valid_results)} valid samples...")
        
        # Prepare data for metrics calculation
        true_types_list = [item['true_types'] for item in valid_results]
        pred_types_list = [item['pred_types'] for item in valid_results]
        
        true_categories_list = [item['true_categories'] for item in valid_results]
        pred_categories_list = [item['pred_categories'] for item in valid_results]
        
        true_tags_list = [item['true_tags'] for item in valid_results]
        pred_tags_list = [item['pred_tags'] for item in valid_results]
        
        # Calculate metrics
        type_metrics = calculate_multilabel_metrics(true_types_list, pred_types_list)
        category_metrics = calculate_multilabel_metrics(true_categories_list, pred_categories_list)
        tag_metrics = calculate_multilabel_metrics(true_tags_list, pred_tags_list)
        
        # Display results
        print(f"\n📊 Multi-label Classification Results:")
        print(f"=" * 60)
        
        print(f"\n🏷️  TYPE METRICS:")
        print(f"  Exact Match Ratio: {type_metrics['exact_match_ratio']:.3f}")
        print(f"  Average Precision: {type_metrics['avg_precision']:.3f}")
        print(f"  Average Recall:    {type_metrics['avg_recall']:.3f}")
        print(f"  Average F1:        {type_metrics['avg_f1']:.3f}")
        
        print(f"\n📁 CATEGORY METRICS:")
        print(f"  Exact Match Ratio: {category_metrics['exact_match_ratio']:.3f}")
        print(f"  Average Precision: {category_metrics['avg_precision']:.3f}")
        print(f"  Average Recall:    {category_metrics['avg_recall']:.3f}")
        print(f"  Average F1:        {category_metrics['avg_f1']:.3f}")
        
        print(f"\n🏆 TAG METRICS:")
        print(f"  Exact Match Ratio: {tag_metrics['exact_match_ratio']:.3f}")
        print(f"  Average Precision: {tag_metrics['avg_precision']:.3f}")
        print(f"  Average Recall:    {tag_metrics['avg_recall']:.3f}")
        print(f"  Average F1:        {tag_metrics['avg_f1']:.3f}")
        
        # Show some example predictions
        print(f"\n🔍 Sample Multi-label Predictions:")
        print(f"=" * 60)
        for i, item in enumerate(valid_results[:5]):
            print(f"\n{i+1}. Text: {item['text'][:80]}...")
            print(f"   True Types: {item['true_types']}")
            print(f"   Pred Types: {item['pred_types']}")
            print(f"   True Categories: {item['true_categories']}")
            print(f"   Pred Categories: {item['pred_categories']}")
            print(f"   True Tags: {item['true_tags']}")
            print(f"   Pred Tags: {item['pred_tags']}")
            
            # Check matches
            type_match = set(item['true_types']) == set(item['pred_types'])
            cat_match = set(item['true_categories']) == set(item['pred_categories'])
            tag_match = set(item['true_tags']) == set(item['pred_tags'])
            
            print(f"   Matches: Type={type_match}, Category={cat_match}, Tag={tag_match}")
        
        # Save metrics to file
        metrics_summary = {
            "total_samples": len(valid_results),
            "type_metrics": type_metrics,
            "category_metrics": category_metrics,
            "tag_metrics": tag_metrics,
            "sample_predictions": valid_results[:10]  # Save first 10 for reference
        }
        
        metrics_path = "dataset/multilabel_metrics.json"
        with open(metrics_path, "w", encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
        
        print(f"\n💾 Metrics saved to: {metrics_path}")
        
    else:
        print("❌ No valid results to calculate metrics!")

if __name__ == "__main__":
    run_multilabel_classification()