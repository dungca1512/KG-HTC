import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import json
import random
# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm import LLM
from src.graph_db import GraphDB
from src.pipeline import Pipeline


config = {
    "data_name": "amazon",
    "data_path": f"dataset/amazon/amazon_val.csv",
    "output_path": "dataset/amazon/ablation_full_kg.json",
    "vectdb_path": "database/amazon",
    "template": {
        "sys": "prompts/system/amazon/llm_graph.txt",
        "user": "prompts/user/amazon/llm_graph.txt"
    },
    "query_params": {
        "l2_top_k": 62,
        "l3_top_k": 327
    }
}

# read csv file
df = pd.read_csv(config["data_path"])
df = df[df['Cat3'] != "unknown"]
df = df[df['Cat2'] != "unknown"]
df = df[df['Cat1'] != "unknown"]
df = df[df['Text'].notna()]
df = df[df['Title'].notna()]
df = df.sample(n=5000, random_state=42)
ds = df.to_dict(orient="records")

llm = LLM()
graph_db = GraphDB()
pipeline = Pipeline(llm, config)

try:
    with open(config["output_path"], "r") as f:
        inference_list = json.load(f)
except:
    inference_list = []

for idx in tqdm(range(len(ds))):
    if idx < len(inference_list):
        continue
    data = ds[idx].copy()

    query_txt_vecdb = f"{data['Title']}:\n{data['Text']}"
    retrieved_nodes = pipeline.query_related_nodes(query_txt_vecdb)
    sub_graph = pipeline.build_linked_labels(retrieved_nodes["l3"], retrieved_nodes["l2"])
    # print(sub_graph)

    potential_level1 = df["Cat1"].unique()
    pred_level1 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level1, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    child_level1 = graph_db.query_l2_from_l1(pred_level1)
    potential_level2 = list(set(child_level1 + retrieved_nodes["l2"]))
    pred_level2 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level2, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')
    # print(pred_level2)
    child_level2 = graph_db.query_l3_from_l2(pred_level2)
    
    potential_level3 = list(set(child_level2 + retrieved_nodes["l3"]))
    pred_level3 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level3, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')
    
    data["gpt3_graph_l1"], data["gpt3_graph_l2"], data["gpt3_graph_l3"] = pred_level1, pred_level2, pred_level3
    inference_list.append(data)
    # print(data)

    with open(config["output_path"], "w") as f:
        json.dump(inference_list, f, indent=4)

    # break