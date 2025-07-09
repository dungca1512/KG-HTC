import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from dotenv import load_dotenv
from src.graph_db import GraphDB

load_dotenv()

def clear_and_rebuild_graph():
    """Clear existing graph and rebuild it properly"""
    
    # Read processed data
    df = pd.read_csv("dataset/custom_review_processed.csv")
    print(f"Loaded {len(df)} processed samples")
    
    graph_db = GraphDB()
    
    # Clear existing graph first
    print("Clearing existing graph...")
    clear_query = """
    MATCH (n) DETACH DELETE n
    """
    try:
        graph_db.create_database(clear_query)
        print("✓ Graph cleared successfully")
    except Exception as e:
        print(f"Error clearing graph: {e}")
    
    # Rebuild graph with clean data
    print("Rebuilding graph...")
    
    # Create query để tạo graph với 3 levels: type -> category -> tag
    query_create_graph = """
    MERGE (level1:Category1 {name: $l1})
    MERGE (level2:Category2 {name: $l2})
    MERGE (level3:Category3 {name: $l3})
    MERGE (level1)-[:contains]->(level2)
    MERGE (level2)-[:contains]->(level3)
    """
    
    # Get unique combinations để tránh duplicate
    unique_combinations = df[['type', 'category', 'tag']].drop_duplicates()
    print(f"Creating {len(unique_combinations)} unique relationships...")
    
    success_count = 0
    error_count = 0
    
    # Tạo các mối quan hệ trong Neo4j
    for _, row in tqdm(unique_combinations.iterrows(), desc="Creating relationships", total=len(unique_combinations)):
        try:
            # Clean data trước khi tạo
            l1_clean = str(row['type']).lower().replace(' ', '').replace('\'', '').replace('\"', '')
            l2_clean = str(row['category']).lower().replace(' ', '').replace('\'', '').replace('\"', '')  
            l3_clean = str(row['tag']).lower().replace(' ', '').replace('\'', '').replace('\"', '')
            
            # Skip if any is empty
            if not l1_clean or not l2_clean or not l3_clean:
                continue
                
            graph_db.create_database(
                query_create_graph, 
                l1=l1_clean, 
                l2=l2_clean, 
                l3=l3_clean
            )
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"Error creating relationship {row['type']}->{row['category']}->{row['tag']}: {e}")
    
    print(f"✓ Graph rebuilding completed!")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    
    # Verify the new graph
    print("\nVerifying new graph structure...")
    
    try:
        # Count nodes
        count_l1_query = "MATCH (n:Category1) RETURN count(n) as count"
        count_l2_query = "MATCH (n:Category2) RETURN count(n) as count"  
        count_l3_query = "MATCH (n:Category3) RETURN count(n) as count"
        
        l1_count = graph_db._query_database(count_l1_query).records[0].get("count")
        l2_count = graph_db._query_database(count_l2_query).records[0].get("count")
        l3_count = graph_db._query_database(count_l3_query).records[0].get("count")
        
        print(f"  L1 nodes: {l1_count}")
        print(f"  L2 nodes: {l2_count}")
        print(f"  L3 nodes: {l3_count}")
        
        # Sample nodes
        sample_l1_query = "MATCH (n:Category1) RETURN n.name LIMIT 5"
        sample_l2_query = "MATCH (n:Category2) RETURN n.name LIMIT 5"
        sample_l3_query = "MATCH (n:Category3) RETURN n.name LIMIT 5"
        
        l1_samples = [r.get("n.name") for r in graph_db._query_database(sample_l1_query).records]
        l2_samples = [r.get("n.name") for r in graph_db._query_database(sample_l2_query).records]
        l3_samples = [r.get("n.name") for r in graph_db._query_database(sample_l3_query).records]
        
        print(f"  Sample L1: {l1_samples}")
        print(f"  Sample L2: {l2_samples}")
        print(f"  Sample L3: {l3_samples}")
        
        # Test some relationships
        print(f"\nTesting relationships...")
        if l1_samples:
            try:
                children = graph_db.query_l2_from_l1(l1_samples[0])
                print(f"  {l1_samples[0]} -> {children[:3]}...")
            except Exception as e:
                print(f"  Error testing L1->L2: {e}")
        
        if l2_samples:
            try:
                children = graph_db.query_l3_from_l2(l2_samples[0])
                print(f"  {l2_samples[0]} -> {children[:3]}...")
            except Exception as e:
                print(f"  Error testing L2->L3: {e}")
                
        if l3_samples:
            try:
                parent = graph_db.query_l2_from_l3(l3_samples[0])
                print(f"  {l3_samples[0]} <- {parent}")
            except Exception as e:
                print(f"  Error testing L3->L2: {e}")
        
    except Exception as e:
        print(f"Error verifying graph: {e}")

if __name__ == "__main__":
    clear_and_rebuild_graph()