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
    "data_name": "dbpedia",
    "model_type": "qwen2.5:7b",
    "data_path": f"dataset/dbpedia/DBPEDIA_val.csv",
    "output_path": "dataset/open_llm/dbpedia_qwen7b_only.json",
    "vectdb_path": "database/dbpedia",
    "template": {
        "sys": "prompts/system/dbpedia/llm_only.txt",
        "user": "prompts/user/dbpedia/llm_only.txt"
    },
}
with open(config["template"]["sys"], "r") as f:
    system_template = f.read()
with open(config["template"]["user"], "r") as f:
    user_template = f.read()
# read csv file
df = pd.read_csv(config["data_path"])
df = df.sample(n=5000, random_state=42)
ds = df.to_dict(orient="records")

llm = LLM(model_type=config["model_type"])
potential_level1 = "**" + "**, **".join(df["l1"].unique()) + "**"
potential_level2 = "**" + "**, **".join(df["l2"].unique()) + "**"
potential_level3 = "**" + "**, **".join(df["l3"].unique()) + "**"

inference_list = []
for idx in tqdm(range(len(ds))):
    data = ds[idx].copy()

    query_txt_vecdb = data["text"]
    sys_msg = system_template.format(category_text=potential_level1)
    user_msg = user_template.format(text=query_txt_vecdb)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    response = llm.chat(messages)
    data["qwen7b_l1"] = response.lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    sys_msg = system_template.format(category_text=potential_level2)
    user_msg = user_template.format(text=query_txt_vecdb)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]   
    response = llm.chat(messages)
    data["qwen7b_l2"] = response.lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    sys_msg = system_template.format(category_text=potential_level3)
    user_msg = user_template.format(text=query_txt_vecdb)
    
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    response = llm.chat(messages)
    data["qwen7b_l3"] = response.lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    inference_list.append(data)
    with open(config["output_path"], "w") as f:
        json.dump(inference_list, f, indent=4)

    # break