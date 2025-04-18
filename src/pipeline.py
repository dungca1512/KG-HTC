from typing import List, Dict, Any
from src.llm import LLM
from src.graph_db import GraphDB
from src.vector_db import VectorDB


class Pipeline:
    def __init__(self, config: dict):
        self._config = config
        self._llm = LLM()
        self._graph_db = GraphDB()
        self._vector_db = VectorDB(
            database_path=self._config["vectdb_path"],
            collection_name=self._config["data_name"]
        )
        self._load_prompts()

    def _load_prompts(self):
        with open(self._config["template"]["sys"], "r") as f:
            self.system_template = f.read()
        with open(self._config["template"]["user"], "r") as f:
            self.user_template = f.read()

    def _format_category_text(self, categories: List[str]) -> str:
        return "**" + "**, **".join(categories) + "**"
    
    def query_related_nodes(self, text: str) -> Dict[str, Any]:
        return {
            "l2": self._vector_db.query_l2(text, self._config["query_params"]["l2_top_k"])["documents"][0],
            "l3": self._vector_db.query_l3(text, self._config["query_params"]["l3_top_k"])["documents"][0]
        }
    
    def build_linked_labels(self, l3_nodes: List[str], related_l2_nodes: List[str]) -> List[str]:
        labels = []
        for l3_node in l3_nodes:
            l2_node = self._graph_db.query_l2_from_l3(l3_node)
            l1_node = self._graph_db.query_l1_from_l2(l2_node)
            if l2_node in related_l2_nodes:
                labels.append(f"{l1_node} -> {l2_node} -> {l3_node}")
        return labels
    
    def predict_level(
            self, query_txt_vecdb: str, 
            context_nodes: List[str], 
            sub_graph: List[str]
        ) -> str:
        sys_msg = self.system_template.format(
            category_text=self._format_category_text(context_nodes),
            knowledge="\n".join(sub_graph)      
        )
        user_msg = self.user_template.format(text=query_txt_vecdb)
        messages = self._llm.construct_messages(sys_msg, user_msg)
        response = self._llm.chat(messages)
        
        return response
    