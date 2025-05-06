import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import json
import random
# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm_ollama import LLM
from src.graph_db import GraphDB
from src.vector_db import VectorDB
from src.pipeline import Pipeline


config = {
    "data_name": "wos",
    "model_type": "qwen2.5:7b",
    "data_path": f"dataset/wos/Meta-data/Data.xlsx",
    "output_path": "dataset/open_llm/wos_qwen7b.json",
    "vectdb_path": "database/wos",
    "template": {
        "sys": "prompts/system/wos/llm_graph.txt",
        "user": "prompts/user/wos/llm_graph.txt"
    },
    "query_params": {
        "l2_top_k": 20,
    }
}

# read csv file
df = pd.read_excel(config["data_path"])
df = df.sample(n=5000, random_state=42)
ds = df.to_dict(orient="records")

llm = LLM(model_type=config["model_type"])
graph_db = GraphDB()
pipeline = Pipeline(llm, config)

inference_list = []
for idx in tqdm(range(len(ds))):
    data = ds[idx].copy()

    query_txt_vecdb = data["Abstract"]
    retrieved_nodes = pipeline.query_related_nodes(query_txt_vecdb)
    sub_graph = []
    for l2 in retrieved_nodes["l2"]:
        l1 = graph_db.query_l1_from_l2(l2)
        sub_graph.append(f"{l1} -> {l2}")

    potential_level1 = df["Domain"].unique()
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
    
    data["qwen7b_l1"], data["qwen7b_l2"] = pred_level1, pred_level2
    inference_list.append(data)

    with open(config["output_path"], "w") as f:
        json.dump(inference_list, f, indent=4)
