import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.llm_openai import LLM
from src.graph_db import GraphDB
from src.pipeline import MultiLabelPipeline

# Page config
st.set_page_config(
    page_title="Review Labeling System",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        flex: 1;
        margin: 0 0.5rem;
        text-align: center;
    }
    .tag {
        background-color: #e1f5fe;
        color: #0277bd;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and components"""
    config = {
        "data_name": "custom_review",
        "vectdb_path": "database/custom_review",
        "template": {
            "sys": "prompts/system/custom/llm_graph.txt",
            "user": "prompts/user/custom/llm_graph.txt"
        },
        "query_params": {
            "l2_top_k": 15,
            "l3_top_k": 30
        }
    }
    
    try:
        llm = LLM(model="gpt-3.5-turbo-0125", temperature=0.1)
        graph_db = GraphDB()
        pipeline = MultiLabelPipeline(llm, config)
        
        # Load available labels
        all_types = set()
        all_categories = set()
        all_tags = set()
        
        # Query vector DB để lấy tất cả nhãn
        l2_result = pipeline._vector_db.query_l2("test", 1000)  # Lấy tất cả L2
        l3_result = pipeline._vector_db.query_l3("test", 1000)  # Lấy tất cả L3
        
        if l2_result["documents"]:
            all_categories.update(l2_result["documents"][0])
        if l3_result["documents"]:
            all_tags.update(l3_result["documents"][0])
            
        # Lấy types từ graph DB
        try:
            for cat in list(all_categories)[:10]:
                try:
                    l1 = graph_db.query_l1_from_l2(cat)
                    all_types.add(l1)
                except:
                    continue
        except:
            all_types = {"positive", "negative"}
        
        return pipeline, graph_db, list(all_types), list(all_categories), list(all_tags)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, [], [], []

def predict_labels(pipeline, graph_db, all_types, all_categories, all_tags, review_text):
    """Predict labels for a review"""
    try:
        # Get related nodes from vector DB
        retrieved_nodes = pipeline.query_related_nodes(review_text)
        
        # Build subgraph
        subgraph = []
        if retrieved_nodes["l3"]:
            subgraph = pipeline.build_linked_labels(retrieved_nodes["l3"], retrieved_nodes["l2"])
        
        if not subgraph:
            subgraph = pipeline.build_simple_subgraph(retrieved_nodes["l2"])
        
        if not subgraph:
            subgraph = ["No hierarchical information available"]
        
        # Multi-label prediction for Type
        pred_types = pipeline.predict_multi_labels(review_text, all_types, subgraph, "type")
        
        # Multi-label prediction for Category
        potential_categories = set()
        for pred_type in pred_types:
            try:
                type_children = graph_db.query_l2_from_l1(pred_type)
                potential_categories.update(type_children)
            except:
                pass
        
        potential_categories.update(retrieved_nodes["l2"])
        potential_categories = list(potential_categories)
        
        if not potential_categories:
            potential_categories = all_categories[:15]
        
        pred_categories = pipeline.predict_multi_labels(review_text, potential_categories, subgraph, "category")
        
        # Multi-label prediction for Tags
        potential_tags = set()
        for pred_cat in pred_categories:
            try:
                cat_children = graph_db.query_l3_from_l2(pred_cat)
                potential_tags.update(cat_children)
            except:
                pass
        
        if retrieved_nodes["l3"]:
            extracted_tags = [pipeline._extract_tag_name(desc) for desc in retrieved_nodes["l3"]]
            potential_tags.update(extracted_tags)
        
        potential_tags = list(potential_tags)
        
        if not potential_tags:
            potential_tags = all_tags[:15]
        
        pred_tags = pipeline.predict_multi_labels(review_text, potential_tags, subgraph, "tag")
        
        return {
            "types": pred_types,
            "categories": pred_categories,
            "tags": pred_tags,
            "subgraph": subgraph[:3],
            "retrieved_nodes": {
                "l2_count": len(retrieved_nodes["l2"]),
                "l3_count": len(retrieved_nodes["l3"])
            }
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏷️ Review Labeling System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system automatically labels app reviews with:
        - **Types**: Positive/Negative sentiment
        - **Categories**: User feelings, Features, etc.
        - **Tags**: Specific aspects mentioned
        """)
        
        st.header("📊 Model Info")
        st.info("Using GPT-3.5-turbo with Knowledge Graph integration")
        
        # Load model
        with st.spinner("Loading model..."):
            pipeline, graph_db, all_types, all_categories, all_tags = load_model()
        
        if pipeline:
            st.success("✅ Model loaded successfully!")
            st.metric("Available Types", len(all_types))
            st.metric("Available Categories", len(all_categories))
            st.metric("Available Tags", len(all_tags))
        else:
            st.error("❌ Failed to load model")
            return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Review")
        
        # Text input
        review_text = st.text_area(
            "Review Text",
            placeholder="Enter your app review here...",
            height=150,
            help="Enter the review text you want to analyze"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
            if review_text.strip():
                with st.spinner("Analyzing review..."):
                    result = predict_labels(pipeline, graph_db, all_types, all_categories, all_tags, review_text)
                
                if result:
                    # Display results
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Types Found", len(result["types"]))
                    with col2:
                        st.metric("Categories Found", len(result["categories"]))
                    with col3:
                        st.metric("Tags Found", len(result["tags"]))
                    
                    # Results in tables
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("🎯 Predicted Types")
                    if result["types"]:
                        types_df = pd.DataFrame({
                            "Type": result["types"],
                            "Count": [1] * len(result["types"])
                        })
                        st.dataframe(types_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No types predicted")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📁 Predicted Categories")
                    if result["categories"]:
                        categories_df = pd.DataFrame({
                            "Category": result["categories"],
                            "Count": [1] * len(result["categories"])
                        })
                        st.dataframe(categories_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No categories predicted")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("🏆 Predicted Tags")
                    if result["tags"]:
                        tags_df = pd.DataFrame({
                            "Tag": result["tags"],
                            "Count": [1] * len(result["tags"])
                        })
                        st.dataframe(tags_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No tags predicted")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Technical details (expandable)
                    with st.expander("🔧 Technical Details"):
                        st.write("**Knowledge Graph Relationships:**")
                        for rel in result["subgraph"]:
                            st.write(f"• {rel}")
                        
                        st.write(f"**Retrieved Nodes:** L2={result['retrieved_nodes']['l2_count']}, L3={result['retrieved_nodes']['l3_count']}")
                    
                    # Download results
                    result_data = {
                        "review_text": review_text,
                        "predicted_types": result["types"],
                        "predicted_categories": result["categories"],
                        "predicted_tags": result["tags"],
                        "subgraph": result["subgraph"]
                    }
                    
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(result_data, indent=2, ensure_ascii=False),
                        file_name="review_analysis.json",
                        mime="application/json"
                    )
                    
            else:
                st.warning("Please enter a review text to analyze.")
    
    with col2:
        st.header("📈 Quick Stats")
        
        # Sample reviews
        st.subheader("💡 Sample Reviews")
        sample_reviews = [
            "This app is amazing! Very easy to use and fun to learn with.",
            "The app crashes frequently and the UI is confusing.",
            "Great features but too expensive. The free version is limited.",
            "Love the gamification elements and the fact that it's free to use."
        ]
        
        for i, sample in enumerate(sample_reviews, 1):
            if st.button(f"Sample {i}", key=f"sample_{i}"):
                st.session_state.review_text = sample
                st.rerun()
        
        # Recent analyses (if any)
        if 'recent_analyses' not in st.session_state:
            st.session_state.recent_analyses = []
        
        if st.session_state.recent_analyses:
            st.subheader("🕒 Recent Analyses")
            for analysis in st.session_state.recent_analyses[-3:]:
                st.write(f"**{analysis['types']}** - {analysis['text'][:50]}...")

if __name__ == "__main__":
    main() 