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


config = {
    "data_name": "wos",
    "model_type": "qwen2.5:7b",
    "data_path": f"dataset/wos/Meta-data/Data.xlsx",
    "output_path": "dataset/open_llm/wos_qwen7b_only.json",
    "vectdb_path": "database/wos",
    "template": {
        "sys": "prompts/system/wos/llm_only.txt",
        "user": "prompts/user/wos/llm_only.txt"
    },
}

with open(config["template"]["sys"], "r") as f:
    system_template = f.read()
with open(config["template"]["user"], "r") as f:
    user_template = f.read()
# read csv file
df = pd.read_excel(config["data_path"])
df = df.sample(n=5000, random_state=42)
ds = df.to_dict(orient="records")

llm = LLM(model_type=config["model_type"])
potential_level1 = "**" + "**, **".join(df["Domain"].unique()) + "**"
potential_level2 = "**" + "**, **".join(df["area"].unique()) + "**"

inference_list = []
for idx in tqdm(range(len(ds))):
    data = ds[idx].copy()

    query_vectordb = data["Abstract"]
    sys_msg = system_template.format(category_text=potential_level1)
    user_msg = user_template.format(text=query_vectordb)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    response = llm.chat(messages)
    data["qwen7b_l1"] = response.lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    sys_msg = system_template.format(category_text=potential_level2)
    user_msg = user_template.format(text=query_vectordb)

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    response = llm.chat(messages)
    data["qwen7b_l2"] = response.lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')

    inference_list.append(data)
    with open(config["output_path"], "w") as f:
        json.dump(inference_list, f, indent=4)
