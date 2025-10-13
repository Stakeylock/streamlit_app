import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pdfplumber
import re
from itertools import combinations
import json
import google.generativeai as genai
from adjustText import adjust_text
from io import BytesIO

class PDFProcessor:
    def __init__(self, uploaded_file):
        """Initializes with a Streamlit UploadedFile object."""
        self.uploaded_file = uploaded_file
        self.name_pattern = re.compile(r'\b[A-Z][a-z]{2,}\b') 
        self.common_words = {'The', 'A', 'An', 'Is', 'In', 'On', 'For', 'With', 'And', 'Was', 'From'}

    def extract_connections(self):
        """Extracts text and finds co-occurring names on each page."""
        connections = []
        try:
            with pdfplumber.open(self.uploaded_file) as pdf:
                if not pdf.pages:
                    st.warning("The uploaded PDF appears to be empty or unreadable.")
                    return []
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        potential_names = self.name_pattern.findall(text)
                        found_names = [name for name in set(potential_names) if name not in self.common_words]
                        if len(found_names) >= 2:
                            connections.extend(list(combinations(found_names, 2)))
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
            return None
        
        st.write(f"Found {len(connections)} potential connections across {len(pdf.pages)} pages.")
        return connections

class GeminiCategorizer:
    def __init__(self, api_key):
        """Initializes the Gemini model with a provided API key."""
        self.api_key = api_key
        self.model = self._configure_model()

    def _configure_model(self):
        """Configures and returns the Gemini generative model."""
        if not self.api_key:
            st.error("Error: Gemini API key is missing.")
            return None
        try:
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            st.error(f"Failed to configure Gemini AI: {e}")
            return None

    def categorize_nodes(self, nodes):
        """Sends nodes to the Gemini API for categorization into logical groups."""
        if not self.model:
            return {} 
        if not nodes:
            st.warning("No nodes were provided to categorize.")
            return {}
        
        unique_nodes = list(set(nodes))
        
        # Enhanced prompt for better, more structured JSON output
        prompt = f"""
        Analyze the following list of terms, which likely represent names of people, 
        technologies, companies, or concepts from a document. Group them into 4 to 6 logical categories.

        The terms are: {', '.join(unique_nodes)}.

        Provide the output as a single JSON array of objects. Each object must have two keys:
        1. "category_name": A descriptive string for the category's name.
        2. "terms": A list of strings from the input that belong to that category.
        
        Example format:
        [
            {{"category_name": "Technology Frameworks", "terms": ["React", "Angular"]}},
            {{"category_name": "Cloud Providers", "terms": ["Amazon", "Google"]}}
        ]
        """
        
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "category_name": {"type": "STRING"},
                        "terms": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "required": ["category_name", "terms"]
                }
            }
        )

        try:
            response = self.model.generate_content(prompt, generation_config=generation_config)
            response_json = json.loads(response.text)
            categories = {item['category_name']: item['terms'] for item in response_json}
            st.write(f"Successfully categorized nodes into {len(categories)} groups.")
            return categories
        except Exception as e:
            st.error(f"An error occurred while calling the Gemini API: {e}")
            return {}

class NetworkVisualizer:
    def __init__(self, connections):
        self.connections = connections
        self.graph = nx.Graph()

    def create_graph(self):
        if not self.connections:
            st.warning("No connections provided to create a graph.")
            return False
        self.graph.add_edges_from(self.connections)
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        return True

    def visualize(self, categories):
        if not self.graph.nodes():
            st.warning("Graph is empty, cannot visualize.")
            return None

        fig, ax = plt.subplots(figsize=(28, 28))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#0f0f0f')
        ax.set_facecolor('#0f0f0f')
        pos = nx.kamada_kawai_layout(self.graph)
        
        node_colors, color_map = self._get_node_colors(categories)
        node_sizes = [self.graph.degree(node) * 60 + 250 for node in self.graph.nodes()]
        
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, ax=ax, width=0.8, alpha=0.3, edge_color='#CCCCCC')
        texts = [ax.text(x, y, node, fontsize=9, color='white', ha='center', va='center') for node, (x, y) in pos.items()]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))
        
        if categories:
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                                          markerfacecolor=color, markersize=12) for cat, color in color_map.items()]
            ax.legend(handles=legend_handles, title="AI-Generated Categories", loc='upper right', 
                      fontsize='medium', title_fontsize='large')

        ax.set_title("PDF Content Network Visualization", size=24, pad=20, color='white')
        ax.axis('off')
        plt.tight_layout()
        
        return fig

    def _get_node_colors(self, categories):
        if not categories:
            return 'grey', {}

        category_colors = plt.get_cmap('Pastel1', len(categories))
        color_map = {category: category_colors(i) for i, category in enumerate(categories.keys())}
        
        node_to_category = {node: cat for cat, nodes in categories.items() for node in nodes}
        
        node_colors = [color_map.get(node_to_category.get(node), 'grey') for node in self.graph.nodes()]
        
        return node_colors, color_map


def main():
    st.set_page_config(layout="wide", page_title="PDF Network Visualizer", page_icon="🔗")

    st.title("🔗 PDF Content Network Visualizer")
    st.markdown("""
    This application transforms a PDF document into an interactive network graph.
    It identifies key terms, visualizes their connections, and uses **Google's Gemini AI** to
    intelligently group them into color-coded categories.
    """)
    st.divider()

    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("1. Upload your PDF", type="pdf")
        gemini_api_key = st.text_input("2. Enter your Gemini API Key", type="password", help="Your key is not stored.")
        
        st.divider()
        run_analysis = st.button("🚀 Generate Network Graph", type="primary", use_container_width=True)

    if run_analysis:
        if not uploaded_file:
            st.warning("Please upload a PDF file to begin.")
        elif not gemini_api_key:
            st.warning("Please enter your Gemini API key.")
        else:
            with st.spinner("The AI is working its magic... Please wait. ✨"):
                try:
                    st.subheader("1. Analyzing PDF Document")
                    processor = PDFProcessor(uploaded_file)
                    connections = processor.extract_connections()

                    if not connections:
                        st.error("No potential connections found. The PDF might be image-based or have no identifiable names. Please try another document.")
                        st.stop()

                    st.subheader("2. Building Network Graph")
                    visualizer = NetworkVisualizer(connections)
                    if not visualizer.create_graph():
                        st.error("Failed to create the graph from the connections found.")
                        st.stop()
                    st.write(f"Created a graph with **{visualizer.graph.number_of_nodes()}** nodes and **{visualizer.graph.number_of_edges()}** edges.")

                    st.subheader("3. AI-Powered Node Categorization")
                    all_nodes = list(visualizer.graph.nodes())
                    categorizer = GeminiCategorizer(gemini_api_key)
                    categories = categorizer.categorize_nodes(all_nodes)
                    
                    if not categories:
                        st.warning("Could not retrieve AI categories. The graph will be shown with default colors.")

                    st.subheader("4. Final Visualization")
                    graph_figure = visualizer.visualize(categories)

                    if graph_figure:
                        st.pyplot(graph_figure, use_container_width=True)
                        if categories:
                            with st.expander("🔍 View AI-Generated Categories"):
                                st.json(categories)
                    else:
                        st.error("Failed to generate the graph visualization.")

                except Exception as e:
                    st.error(f"A critical error occurred during the process: {e}")
    else:
        st.info("Upload a PDF and enter your API key, then click 'Generate' to see the network.")

if __name__ == '__main__':
    main()