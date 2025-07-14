import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Thông tin kết nối
uri = "neo4j+s://841e1940.databases.neo4j.io"
username = "neo4j"
password = "dL0sJtPiMXxm0NRU8OH_tn3sQ5xS-KOrwR2if-UyYDw"

driver = GraphDatabase.driver(uri, auth=(username, password))

# Đọc dữ liệu
df = pd.read_excel('/Users/dungca/KG-HTC/dataset/wos/Meta-data/Data.xlsx')
df['id'] = df.index

# Tạo node Domain
domains = df['Domain'].unique().tolist()
with driver.session() as session:
    session.run("UNWIND $domains AS domain MERGE (d:Domain {name: domain})", domains=domains)

# Tạo node Area và mối quan hệ CONTAINS
area_domains = df[['Domain', 'area']].drop_duplicates().to_dict('records')
with driver.session() as session:
    session.run("""
        UNWIND $area_domains AS ad
        MATCH (d:Domain {name: ad.Domain})
        MERGE (a:Area {name: ad.area})
        MERGE (d)-[:CONTAINS]->(a)
    """, area_domains=area_domains)

# Tạo node Abstract và mối quan hệ BELONGS_TO
abstracts = df[['id', 'Abstract', 'Domain', 'area', 'keywords', 'Y1', 'Y2', 'Y']].to_dict('records')
with driver.session() as session:
    session.run("""
        UNWIND $abstracts AS abs_data
        MERGE (abs:Abstract {id: abs_data.id})
        ON CREATE SET abs.abstract = abs_data.Abstract,
                      abs.keywords = abs_data.keywords,
                      abs.y1 = abs_data.Y1,
                      abs.y2 = abs_data.Y2,
                      abs.y = abs_data.Y
        WITH abs, abs_data
        MATCH (a:Area {name: abs_data.area})
        MERGE (abs)-[:BELONGS_TO]->(a)
    """, abstracts=abstracts)

driver.close()
print("Dữ liệu đã được đẩy vào Neo4j thành công!")