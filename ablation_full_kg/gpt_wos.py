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
    "data_name": "wos",
    "data_path": f"dataset/wos/Meta-data/Data.xlsx",
    "output_path": "dataset/wos/ablation_full_kg.json",
    "vectdb_path": "database/wos",
    "template": {
        "sys": "prompts/system/wos/llm_graph.txt",
        "user": "prompts/user/wos/llm_graph.txt"
    },
    "query_params": {
        "l2_top_k": 141,
    }
}

# read csv file
df = pd.read_excel(config["data_path"])
df = df.sample(n=5000, random_state=42)
ds = df.to_dict(orient="records")

# print(len(df["Domain"].unique()))
# print(len(df["area"].unique()))


llm = LLM()
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
    # print(sub_graph)

    potential_level1 = df["Domain"].unique()
    pred_level1 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level1, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')
    # print(potential_level1)

    potential_level2 = df["area"].unique()
    pred_level2 = pipeline.predict_level(
        query_txt_vecdb, 
        potential_level2, 
        sub_graph
    ).lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')
    
    data["gpt3_graph_l1"], data["gpt3_graph_l2"] = pred_level1, pred_level2
    inference_list.append(data)
    # print(data)

    with open(config["output_path"], "w") as f:
        json.dump(inference_list, f, indent=4)

    # break