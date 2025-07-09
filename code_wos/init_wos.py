import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from src.vector_db import VectorDB
from src.graph_db import GraphDB


load_dotenv()

config = {
    "data_name": "ai_agent_wos",
    "data_path": f"dataset/ai_agent_wos/Meta-data/Data.xlsx",
    "output_path": "dataset/ai_agent_wos/ablation_full_kg.json",
    "vectdb_path": "database/ai_agent_wos",
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
ds = df.to_dict(orient="records")

graph_db = GraphDB()
# create a link in a graph db, the link is from l1 to l2 to l3
query_create_graph = """
MERGE (level1:Category1 {name: $l1})
MERGE (level2:Category2 {name: $l2})
MERGE (level1)-[:contains]->(level2)
"""
vector_db = VectorDB(
    database_path=config["vectdb_path"],
    collection_name=config["data_name"]
)

for data in tqdm(ds):
    graph_db.create_database(
        query_create_graph, l1=data["Domain"].lower().replace(' ', ''), 
        l2=data["area"].lower().replace(' ', '')
    )

# create vector db
df = pd.DataFrame(ds)
# Filter out None values and get unique categories
label_l1 = [cat.lower().replace(' ', '') for cat in df["Domain"].unique().tolist() if cat is not None]
label_l2 = [cat.lower().replace(' ', '') for cat in df["area"].unique().tolist() if cat is not None]

vector_db.batch_add(label_l1, metadatas=[{"level": "Category1"}] * len(label_l1))
vector_db.batch_add(label_l2, metadatas=[{"level": "Category2"}] * len(label_l2))
print("VectorDB created successfully.")