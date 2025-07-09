from neo4j import GraphDatabase
from neo4j._sync.driver import EagerResult
import os
import dotenv


dotenv.load_dotenv()

class GraphDB:
    def __init__(self):
        self._URI = os.getenv("NEO4J_URI")
        self._USERNAME = os.getenv("NEO4J_USERNAME")
        self._PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    def create_database(self, query_text: str, **kwargs):
        with GraphDatabase.driver(
            self._URI,
            auth=(self._USERNAME, self._PASSWORD),
        ) as driver:
            driver.execute_query(
                query_text,
                **kwargs
            )

    def _query_database(self, query_text: str, **kwargs) -> EagerResult:
        with GraphDatabase.driver(
            self._URI,
            auth=(self._USERNAME, self._PASSWORD),
        ) as driver:
            query_result = driver.execute_query(
                query_text,
                **kwargs
            )

            return query_result
        
    def query_l1_from_l2(self, l2: str) -> str:
        query_text = """
        MATCH (level1:Category1)-[:contains]->(level2:Category2 {name: $l2}) 
        RETURN level1
        """
        
        try:
            results = self._query_database(query_text, l2=l2).records
            if len(results) == 0:
                raise Exception(f"No L1 parent found for L2: {l2}")
            return results[0].get("level1").get("name")
        except Exception as e:
            raise Exception(f"Error querying L1 from L2 '{l2}': {e}")
    
    def query_l2_from_l3(self, l3: str) -> str:
        query_text = """
        MATCH (level2:Category2)-[:contains]->(level3:Category3 {name: $l3}) 
        RETURN level2
        """
        
        try:
            results = self._query_database(query_text, l3=l3).records
            if len(results) == 0:
                raise Exception(f"No L2 parent found for L3: {l3}")
            return results[0].get("level2").get("name")
        except Exception as e:
            raise Exception(f"Error querying L2 from L3 '{l3}': {e}")
    
    def query_l2_from_l1(self, l1: str) -> list[str]:
        query_text = """
        MATCH (level1:Category1 {name: $l1})-[:contains]->(level2:Category2) 
        RETURN level2
        """
        
        try:
            results = self._query_database(query_text, l1=l1).records
            return [record.get("level2").get("name") for record in results]
        except Exception as e:
            raise Exception(f"Error querying L2 children from L1 '{l1}': {e}")
    
    def query_l3_from_l2(self, l2: str) -> list[str]:
        query_text = """
        MATCH (level2:Category2 {name: $l2})-[:contains]->(level3:Category3) 
        RETURN level3
        """
        
        try:
            results = self._query_database(query_text, l2=l2).records
            return [record.get("level3").get("name") for record in results]
        except Exception as e:
            raise Exception(f"Error querying L3 children from L2 '{l2}': {e}")