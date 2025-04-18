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

# read csv file
df = pd.read_csv('dataset/dbpedia/DBPEDIA_val.csv')
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
    database_path="database/dbpedia",
    collection_name="dbpedia"
)

for data in tqdm(ds):
    graph_db.create_database(
        query_create_graph, l1=data["l1"].lower().replace(' ', ''), 
        l2=data["l2"].lower().replace(' ', ''), l3=data["l3"].lower().replace(' ', '')
    )

# create vector db
df = pd.DataFrame(ds)
# Filter out None values and get unique categories
label_l1 = [cat.lower().replace(' ', '') for cat in df["l1"].unique().tolist() if cat is not None]
label_l2 = [cat.lower().replace(' ', '') for cat in df["l2"].unique().tolist() if cat is not None]
label_l3 = [cat.lower().replace(' ', '') for cat in df["l3"].unique().tolist() if cat is not None]

vector_db.batch_add(label_l1, metadatas=[{"level": "Category1"}] * len(label_l1))
vector_db.batch_add(label_l2, metadatas=[{"level": "Category2"}] * len(label_l2))
vector_db.batch_add(label_l3, metadatas=[{"level": "Category3"}] * len(label_l3))
print("VectorDB created successfully.")