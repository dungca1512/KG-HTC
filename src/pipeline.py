from typing import List, Dict, Any
import random
import re
from src.llm_openai import LLM
from src.graph_db import GraphDB
from src.vector_db import VectorDB


class MultiLabelPipeline:
    def __init__(self, llm: LLM, config: dict):
        self._config = config
        self._llm = llm
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
    
    def _extract_tag_name(self, description: str) -> str:
        if ':' in description:
            return description.split(':')[0].strip().lower().replace(' ', '').replace('\'', '').replace('\"', '')
        return description.lower().replace(' ', '').replace('\'', '').replace('\"', '')
    
    def query_related_nodes(self, text: str) -> Dict[str, Any]:
        try:
            l2_result = self._vector_db.query_l2(text, self._config["query_params"]["l2_top_k"])
            l2_nodes = l2_result["documents"][0] if l2_result["documents"] and l2_result["documents"][0] else []
            
            l3_nodes = []
            if "l3_top_k" in self._config["query_params"]:
                l3_result = self._vector_db.query_l3(text, self._config["query_params"]["l3_top_k"])
                l3_nodes = l3_result["documents"][0] if l3_result["documents"] and l3_result["documents"][0] else []
            
            return {
                "l2": l2_nodes,
                "l3": l3_nodes
            }
        except Exception as e:
            print(f"Error in query_related_nodes: {e}")
            return {"l2": [], "l3": []}
    
    def build_linked_labels(self, l3_nodes: List[str], related_l2_nodes: List[str]) -> List[str]:
        labels = []
        
        if not l3_nodes or len(l3_nodes) == 0:
            print("[DEBUG] No l3_nodes provided to build_linked_labels")
            return labels
            
        for l3_node in l3_nodes:
            try:
                print(f"[DEBUG] l3_node: {l3_node}")
                tag_name = self._extract_tag_name(l3_node)
                print(f"[DEBUG] Extracted tag_name: {tag_name}")
                
                try:
                    l2_node = self._graph_db.query_l2_from_l3(tag_name)
                    print(f"[DEBUG] l2_node from tag_name '{tag_name}': {l2_node}")
                except Exception as e:
                    print(f"[ERROR] query_l2_from_l3 failed for tag_name '{tag_name}': {e}")
                    continue
                
                try:
                    l1_node = self._graph_db.query_l1_from_l2(l2_node)
                    print(f"[DEBUG] l1_node from l2_node '{l2_node}': {l1_node}")
                except Exception as e:
                    print(f"[ERROR] query_l1_from_l2 failed for l2_node '{l2_node}': {e}")
                    continue
                
                labels.append(f"{l1_node} -> {l2_node} -> {tag_name}")
                print(f"[DEBUG] Added label: {l1_node} -> {l2_node} -> {tag_name}")
                    
            except Exception as e:
                print(f"[ERROR] build_linked_labels failed for l3_node '{l3_node}': {e}")
                continue
                
        if not labels:
            print("[DEBUG] No linked labels could be built from provided l3_nodes.")
        return labels
    
    def build_simple_subgraph(self, l2_nodes: List[str]) -> List[str]:
        subgraph = []
        
        for l2_node in l2_nodes[:5]:
            try:
                l1_node = self._graph_db.query_l1_from_l2(l2_node)
                subgraph.append(f"{l1_node} -> {l2_node}")
            except Exception as e:
                continue
                
        return subgraph
    
    def predict_multi_labels(
            self, query_text: str, 
            context_nodes: List[str], 
            sub_graph: List[str],
            level_name: str = "category"
        ) -> List[str]:
        
        if not context_nodes or len(context_nodes) == 0:
            return ["unknown"]
        
        # Adaptive prompt based on level
        if level_name.lower() == "type":
            instruction = """Instructions:
1. If the review is purely positive, return: positive
2. If the review is purely negative, return: negative  
3. If the review contains BOTH positive and negative sentiments, return: positive,negative
4. Output only the type name(s), separated by comma if multiple
5. Do not add extra characters like quotes, asterisks, or explanations"""
        else:
            instruction = f"""Instructions:
1. Analyze the review for ALL applicable {level_name}s
2. The review may have multiple aspects - include ALL relevant ones
3. Output multiple {level_name}s if applicable, separated by comma
4. Be comprehensive - don't miss relevant aspects
5. Do not add extra characters like quotes, asterisks, or explanations"""
        
        # Multi-label system template
        multi_label_system_template = f"""You are analyzing app review text for {level_name} classification. The review may contain multiple aspects.

Available {level_name}s: {{category_text}}

Here is the hierarchical knowledge graph:
\"\"\"
{{knowledge}}
\"\"\"

{instruction}"""

        try:
            sys_msg = multi_label_system_template.format(
                category_text=self._format_category_text(context_nodes),
                knowledge="\n".join(sub_graph) if sub_graph else "No hierarchical information available"      
            )
            user_msg = self.user_template.format(text=query_text)
            messages = self._llm.construct_messages(sys_msg, user_msg)
            response = self._llm.chat(messages)

            if response is None:
                return [random.choice(context_nodes)]
            
            # Parse multiple labels
            clean_response = response.lower().replace(' ', '').replace('*', '').replace('\'', '').replace('\"', '')
            
            # Split by comma and validate each label
            predicted_labels = []
            if ',' in clean_response:
                labels = [label.strip() for label in clean_response.split(',')]
            else:
                labels = [clean_response]
            
            # Validate each label against available options
            clean_context = [c.lower().replace(' ', '').replace('\'', '').replace('\"', '') for c in context_nodes]
            
            for label in labels:
                if label in clean_context:
                    predicted_labels.append(label)
            
            # Fallback if no valid labels found
            if not predicted_labels:
                predicted_labels = [clean_context[0]]
                
            return predicted_labels
            
        except Exception as e:
            print(f"Error in predict_multi_labels: {e}")
            return [random.choice(context_nodes)] if context_nodes else ["unknown"]
    
    def predict_level(
            self, query_text: str, 
            context_nodes: List[str], 
            sub_graph: List[str]
        ) -> str:
        
        multi_labels = self.predict_multi_labels(query_text, context_nodes, sub_graph)
        return multi_labels[0] if multi_labels else "unknown"