from docx import Document
from langchain.text_splitter import CharacterTextSplitter
import os
# from dotenv import load_dotenv
from werkzeug.utils import secure_filename
# from typing import Tuple
# from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer  # Added import
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from pyvis.network import Network
import time
import logging
from langchain_core.documents import Document
from utils.llm_service import get_openai_chat

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n\n")
    basic = text_splitter.split_text(text)  # Splitting text into chunks
    return basic

def store_graph(graph_documents):
    start_time = time.time()
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships
   
    for node in nodes:
        net.add_node(node.id, label=node.id, title=str(node.type), color="skyblue")
   
    for relationship in relationships:
        net.add_edge(relationship.source.id, relationship.target.id, title=relationship.type, color="gray", arrows="to")
   
    net.repulsion()
   
    # Generate HTML file
    net.write_html("graph.html")
    logging.info('--- %s seconds to write into html file ---' % (time.time() - start_time))

# Create Knowledge Graph using openAI model
def create_knowledge_graph(text, usersession):
    start_time = time.time()
    # Preprocess the documents to convert lists to tuples
    documents = [Document(page_content=text)]

    print("#"*100)
    print(documents)
    print("#"*100)

    # Create Document objects from the text chunks
    # Use the centralized LLM service instead of hardcoded API key
    llm = get_openai_chat(model="gpt-4o")
    
    #llm=ChatOllama(model="llama3.1")
    #llm=ChatMistralAI(model_name="mistral-large-latest" , api_key="Vj3QDzzz3mm40VLUTojnw8Bdc2UWnRmR")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    logging.info("--- %s seconds to create knowledge graph ---" % (time.time() - start_time))
    logging.info(f'number of nodes and relationships: {len(graph_documents[0].nodes)}, {len(graph_documents[0].relationships)}')

    return graph_documents

# Function to handle file upload
def graph_from_document(session_name):
    start_time = time.time()

    raw_text = ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    name=session_name
    relative_path = os.path.join(session_name, 'content.txt')
    file_path = os.path.join(script_dir, relative_path)
    try:
            # Open the file in read mode
            with open(file_path, 'r') as file:
                    # Read the content of the file
                    raw_text = file.read()
                                       
                    # Print the content (or store it in a variable)
                    print(raw_text)
    except FileNotFoundError:
            print(f"The file at {file_path} was not found.")
    except Exception as e:
            print(f"An error occurred: {e}")
    retry_count = 2
    while raw_text:
        try:
            docs = create_knowledge_graph(raw_text, session_name)
            logging.info(f'data converted to graph in {time.time() - start_time} seconds')
            store_graph(docs)
            logging.info(f'graph stored in html in {time.time() - start_time} seconds')
            break # to break while loop
        except Exception as e:
            logging.info(f'count: {retry_count}, Error creating graph: {e}')
            # Delete graph.html and lib dir
            if os.path.exists('graph.html'):
                os.remove('graph.html')
            if os.path.exists('lib'):
                for root, dirs, files in os.walk('lib', topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir('lib')
            retry_count -= 1

    html_content = ""
    try: # Read html file
        with open('graph.html') as f:
            html_content = f.read()
        # Delete graph.html and lib dir
        '''
        if os.path.exists('graph.html'):
            os.remove('graph.html')
        if os.path.exists('lib'):
            for root, dirs, files in os.walk('lib', topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir('lib')'''
    except Exception as e:
        logging.info(f'error reading html file: {e}')
        html_content = f"<body>Error reading file: {e}</body>"

    logging.info(f"request processed successfully for graph in {time.time() - start_time} seconds")
    return html_content

# trial run
print("start")
graph_from_document("user")
print("end")