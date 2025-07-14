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
    os.makedirs("prompts/system/custom", exist_ok=True)
    os.makedirs("prompts/user/custom", exist_ok=True)
    system_prompt = """Classify the app review text into categories from: {category_text}. \n\nThe review may contain multiple aspects - include ALL relevant categories.\n\nHere is the partial knowledge graph showing hierarchical relationships: \n\"\"\"\n{knowledge}\n\"\"\"\n\nInstructions:\n1. Analyze the review for ALL applicable categories\n2. If multiple categories apply, output them separated by commas  \n3. Use the knowledge graph to understand relationships\n4. Output only category names, no extra characters"""
    user_prompt = """App Review Text: \n\"\"\"\n{text}\n\"\"\"\n\nCategories:"""
    with open("prompts/system/custom/llm_graph.txt", "w", encoding='utf-8') as f:
        f.write(system_prompt)
    with open("prompts/user/custom/llm_graph.txt", "w", encoding='utf-8') as f:
        f.write(user_prompt)

def parse_multilabel_ground_truth(label_string):
    if pd.isna(label_string):
        return []
    labels = [label.strip().lower().replace(' ', '').replace("'", '').replace('"', '') 
              for label in str(label_string).split(',')]
    return [label for label in labels if label]

def run_multilabel_classification():
    create_multilabel_prompt_templates()
    config = {
        "data_name": "custom_review",
        "data_path": "dataset/multilabel_review_processed.csv",
        "output_path": "dataset/multilabel_results_v2.json",
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
    if not os.path.exists(config["data_path"]):
        print(f"Error: {config['data_path']} not found!")
        print("Please run init_multilabel_review.py first")
        return
    print("Loading processed data...")
    df = pd.read_csv(config["data_path"])
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print("\nParsing ground truth multi-labels...")
    df['true_types'] = df['type'].apply(parse_multilabel_ground_truth)
    df['true_categories'] = df['category'].apply(parse_multilabel_ground_truth)
    df['true_tags'] = df['tag'].apply(parse_multilabel_ground_truth)
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
    sample_size = min(30, len(df))
    df_sample = df.head(sample_size)
    ds = df_sample.to_dict(orient="records")
    print("\nInitializing components...")
    try:
        llm = LLM(model="gpt-3.5-turbo", temperature=0.1)
        graph_db = GraphDB()
        pipeline = MultiLabelPipeline(llm, config)
        print("✓ Components initialized successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
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
            retrieved_nodes = pipeline.query_related_nodes(review_text)
            print(f"Retrieved L2: {len(retrieved_nodes['l2'])}, L3: {len(retrieved_nodes['l3'])}")
            subgraph = []
            if retrieved_nodes["l3"]:
                subgraph = pipeline.build_linked_labels(retrieved_nodes["l3"], retrieved_nodes["l2"])
            if not subgraph:
                subgraph = pipeline.build_simple_subgraph(retrieved_nodes["l2"])
            if not subgraph:
                subgraph = ["No hierarchical information available"]
            print(f"Built subgraph with {len(subgraph)} relationships")
            print("Predicting types (multi-label)...")
            pred_types = pipeline.predict_multi_labels(review_text, all_types, subgraph, "type")
            print(f"Predicted types: {pred_types}")
            print(f"True types: {data['true_types']}")
            print("Predicting categories (multi-label)...")
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
            print(f"True categories: {data['true_categories']}")
            print("Predicting tags (multi-label)...")
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
            print(f"True tags: {data['true_tags']}")
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
        with open(config["output_path"], "w", encoding='utf-8') as f:
            json.dump(inference_list, f, indent=4, ensure_ascii=False)
    print(f"\nMulti-label classification completed!")
    print(f"Results saved to: {config['output_path']}")
    valid_results = [item for item in inference_list if item.get('pred_types', ['error'])[0] != 'error']
    if valid_results:
        print(f"\nCalculating relaxed multi-label metrics (macro/micro F1, precision, recall) on {len(valid_results)} valid samples...")
        true_types_list = [item['true_types'] for item in valid_results]
        pred_types_list = [item['pred_types'] for item in valid_results]
        true_categories_list = [item['true_categories'] for item in valid_results]
        pred_categories_list = [item['pred_categories'] for item in valid_results]
        true_tags_list = [item['true_tags'] for item in valid_results]
        pred_tags_list = [item['pred_tags'] for item in valid_results]
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import f1_score, precision_score, recall_score
        def multilabel_sklearn_metrics(true_list, pred_list):
            mlb = MultiLabelBinarizer()
            mlb.fit(true_list + pred_list)
            y_true = mlb.transform(true_list)
            y_pred = mlb.transform(pred_list)
            return {
                'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
                'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'micro_precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
                'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'micro_recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
            }
        type_sklearn_metrics = multilabel_sklearn_metrics(true_types_list, pred_types_list)
        category_sklearn_metrics = multilabel_sklearn_metrics(true_categories_list, pred_categories_list)
        tag_sklearn_metrics = multilabel_sklearn_metrics(true_tags_list, pred_tags_list)
        print(f"\n📊 RELAXED Multi-label Classification Results (macro/micro F1, precision, recall):")
        print(f"=" * 60)
        print(f"\n🏷️  TYPE METRICS:")
        print(f"  Macro F1: {type_sklearn_metrics['macro_f1']:.3f}")
        print(f"  Micro F1: {type_sklearn_metrics['micro_f1']:.3f}")
        print(f"  Macro Precision: {type_sklearn_metrics['macro_precision']:.3f}")
        print(f"  Micro Precision: {type_sklearn_metrics['micro_precision']:.3f}")
        print(f"  Macro Recall: {type_sklearn_metrics['macro_recall']:.3f}")
        print(f"  Micro Recall: {type_sklearn_metrics['micro_recall']:.3f}")
        print(f"\n📁 CATEGORY METRICS:")
        print(f"  Macro F1: {category_sklearn_metrics['macro_f1']:.3f}")
        print(f"  Micro F1: {category_sklearn_metrics['micro_f1']:.3f}")
        print(f"  Macro Precision: {category_sklearn_metrics['macro_precision']:.3f}")
        print(f"  Micro Precision: {category_sklearn_metrics['micro_precision']:.3f}")
        print(f"  Macro Recall: {category_sklearn_metrics['macro_recall']:.3f}")
        print(f"  Micro Recall: {category_sklearn_metrics['micro_recall']:.3f}")
        print(f"\n🏆 TAG METRICS:")
        print(f"  Macro F1: {tag_sklearn_metrics['macro_f1']:.3f}")
        print(f"  Micro F1: {tag_sklearn_metrics['micro_f1']:.3f}")
        print(f"  Macro Precision: {tag_sklearn_metrics['macro_precision']:.3f}")
        print(f"  Micro Precision: {tag_sklearn_metrics['micro_precision']:.3f}")
        print(f"  Macro Recall: {tag_sklearn_metrics['macro_recall']:.3f}")
        print(f"  Micro Recall: {tag_sklearn_metrics['micro_recall']:.3f}")
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
        metrics_summary = {
            "total_samples": len(valid_results),
            "type_metrics_sklearn": type_sklearn_metrics,
            "category_metrics_sklearn": category_sklearn_metrics,
            "tag_metrics_sklearn": tag_sklearn_metrics,
            "sample_predictions": valid_results[:10]
        }
        metrics_path = "dataset/multilabel_metrics_v2.json"
        with open(metrics_path, "w", encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
        print(f"\n💾 Metrics saved to: {metrics_path}")
    else:
        print("❌ No valid results to calculate metrics!")

if __name__ == "__main__":
    run_multilabel_classification() 