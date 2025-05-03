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
    "data_name": "amazon",
    "data_path": f"dataset/amazon/amazon_val.csv",
    "output_path": "dataset/amazon/llm_graph_gpt3.json",
    "vectdb_path": "database/amazon",
    "template": {
        "sys": "prompts/system/amazon/llm_graph.txt",
        "user": "prompts/user/amazon/llm_graph.txt"
    },
    "query_params": {
        "l2_top_k": 10,
        "l3_top_k": 40
    }
}

# read csv file
df = pd.read_csv(config["data_path"])
df = df[df['Cat3'] != "unknown"]
df = df[df['Cat2'] != "unknown"]
df = df[df['Cat1'] != "unknown"]
df = df[df['Text'].notna()]
df = df[df['Title'].notna()]
ds = df.to_dict(orient="records")

graph_db = GraphDB()
# create a link in a graph db, the link is from l1 to l2 to l3
query_create_graph = """
MERGE (level1:Category1 {name: $l1})
MERGE (level2:Category2 {name: $l2})
MERGE (level3:Category3 {name: $l3})
MERGE (level1)-[:contains]->(level2)
MERGE (level2)-[:contains]->(level3)
"""
vector_db = VectorDB(
    database_path=config["vectdb_path"],
    collection_name=config["data_name"]
)

for data in tqdm(ds):
    if data["Cat1"] != "unknown" and data["Cat2"] != "unknown" and data["Cat3"] != "unknown":
        graph_db.create_database(
            query_create_graph, l1=data["Cat1"].lower().replace(' ', ''), 
            l2=data["Cat2"].lower().replace(' ', ''), l3=data["Cat3"].lower().replace(' ', '')
        )

# create vector db
df = pd.DataFrame(ds)
# Filter out None values and get unique categories
label_l1 = [cat.lower().replace(' ', '') for cat in df["Cat1"].unique().tolist() if cat is not None]
label_l2 = [cat.lower().replace(' ', '') for cat in df["Cat2"].unique().tolist() if cat is not None]
label_l3 = [cat.lower().replace(' ', '') for cat in df["Cat3"].unique().tolist() if cat is not None]

# check if unknown is in the list
if "unknown" in label_l1:
    label_l1.remove("unknown")
if "unknown" in label_l2:
    label_l2.remove("unknown")
if "unknown" in label_l3:
    label_l3.remove("unknown")

vector_db.batch_add(label_l1, metadatas=[{"level": "Category1"}] * len(label_l1))
vector_db.batch_add(label_l2, metadatas=[{"level": "Category2"}] * len(label_l2))
vector_db.batch_add(label_l3, metadatas=[{"level": "Category3"}] * len(label_l3))
print("VectorDB created successfully.")