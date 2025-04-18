import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import json
# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm import LLM
from src.graph_db import GraphDB
from src.vector_db import VectorDB


# read csv file
df = pd.read_csv('dataset/dbpedia/DBPEDIA_val.csv')
ds = df.to_dict(orient="records")

vector_db = VectorDB(
    database_path="database/dbpedia",
    collection_name="dbpedia"
)
graph_db = GraphDB()
llm = LLM()

with open("prompts/system/dbpedia/llm_graph.txt", "r") as f:
    sys_template = f.read()
with open("prompts/user/dbpedia/llm_graph.txt", "r") as f:
    user_template = f.read()

unique_category_1 = df["l1"].unique()
unique_category_1 = [cat.lower() for cat in unique_category_1]
category_text = "**" + "**, **".join(unique_category_1) + "**"

# read json file
with open("dataset/dbpedia/llm_graph_gpt3.json", "r") as f:
    inference_list = json.load(f)

query_graphdb_h2 = """
MATCH (level1:Category1)-[:contains]->(level2:Category2 {name: $l2}) 
RETURN level1
"""
query_graphdb_h3 = """
MATCH (level2:Category2)-[:contains]->(level3:Category3 {name: $l3}) 
RETURN level2
"""

for idx in tqdm(range(len(ds))):
    if idx < len(inference_list):
        continue
    data = ds[idx].copy()

    query_vectordb = data["text"]
    query_h2 = vector_db.query_by_text(
        query_vectordb,
        n_results=7,
        where={"level": "Category2"}
    )
    query_h3 = vector_db.query_by_text(
        query_vectordb,
        n_results=38,
        where={"level": "Category3"}
    )

    linked_labels = []
    for h3 in query_h3["documents"][0]:
        # print(graph_db.query_database(query_graphdb_h3, l3=h3))
        h2 = graph_db.query_database(query_graphdb_h3, l3=h3).records[0].get("level2").get("name")
        h1 = graph_db.query_database(query_graphdb_h2, l2=h2).records[0].get("level1").get("name")
        if h2 in query_h2["documents"][0]:
            linked_labels.append(f"{h1} -> {h2} -> {h3}")

    sys_message = sys_template.format(
        category_text=category_text,
        knowledge="\n".join(linked_labels)
    )
    user_message = user_template.format(
        text=query_vectordb
    )

    messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": user_message},
    ]
    try:
        response = llm.chat(messages).choices[0].message.content
        if response not in unique_category_1:
            retries = 0
            while response.lower() not in unique_category_1 and retries < 5:
                retries += 1
                rethinking_messages = [
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Please note in particular that the output must strictly match the full category names: {category_text}. And cannot be added, abbreviated or modified."},
                ]
                response = llm.chat(rethinking_messages).choices[0].message.content
        data["gpt3_graph_l1"] = response
    except Exception as e:
        data["gpt3_graph_l1"] = None

    inference_list.append(data)

    with open("dataset/dbpedia/llm_graph_gpt3.json", "w") as f:
        json.dump(inference_list, f, indent=4)

    # break