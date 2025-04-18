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
        
    def query_l1_from_l2(self, l2: str) -> list[str]:
        query_text = """
        MATCH (level1:Category1)-[:contains]->(level2:Category2 {name: $l2}) 
        RETURN level1
        """

        return self._query_database(query_text, l2=l2)
    
    def query_l2_from_l3(self, l3: str) -> list[str]:
        query_text = """
        MATCH (level2:Category2)-[:contains]->(level3:Category3 {name: $l3}) 
        RETURN level2
        """

        return self._query_database(query_text, l3=l3)
        