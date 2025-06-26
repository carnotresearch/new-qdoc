import os
import pickle
import json_repair
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import UnstructuredRelation
from langchain_core.messages import SystemMessage
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv
from collections import defaultdict
import logging
from config import Config
from app.services.llm_service import get_standard_llm

logger = logging.getLogger(__name__)
load_dotenv()

# Global variables
graph = None
llm = None
embedding_model = None

# SESSION-SPECIFIC entity memory for dense graph processing
session_entity_memory_dense = {}  # session_id -> entity_memory
SIMILARITY_THRESHOLD = 0.8

def initialize_globals():
    """Initialize global variables"""
    global graph, llm, embedding_model

    graph = Neo4jGraph(
        url=Config.NEO4J_URI,  
        username=Config.NEO4J_USERNAME,              
        password=Config.NEO4J_PASSWORD,       
    )
    
    # Use centralized LLM service for knowledge graph generation
    llm = get_standard_llm()
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_session_entity_memory_dense(session_id: str) -> Dict:
    """Get entity memory for a specific session in dense processing"""
    if session_id not in session_entity_memory_dense:
        session_entity_memory_dense[session_id] = {}
    return session_entity_memory_dense[session_id]

def clear_session_data_dense(session_id: str):
    """Clear entity memory for a specific session in dense processing"""
    if session_id in session_entity_memory_dense:
        del session_entity_memory_dense[session_id]

def create_session_scoped_id(entity_name: str, session_id: str) -> str:
    """Create a session-scoped unique ID for an entity"""
    return f"{session_id}_{entity_name}"

def resolve_entity(entity_name: str, entity_type: str, session_id: str) -> tuple:
    """Resolve entity name to canonical form using similarity within the same session"""
    entity_embedding = embedding_model.encode(entity_name, convert_to_tensor=True)
    
    # Get session-specific entity memory
    entity_memory = get_session_entity_memory_dense(session_id)
    
    # Only check entities from the same session
    for canonical, data in entity_memory.items():
        if data.get('session_id') == session_id:
            sim = util.cos_sim(entity_embedding, data['embedding']).item()
            if sim >= SIMILARITY_THRESHOLD and entity_type == data['type']:
                return canonical, True
    
    # Store with session_id in session-specific memory
    entity_memory[entity_name] = {
        'embedding': entity_embedding, 
        'type': entity_type,
        'session_id': session_id
    }
    return entity_name, False

def parse_chunked_file(file_path: str) -> List[Document]:
    """Parse a file with chunks separated by --- Chunk X --- markers"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    chunks = []
    current_chunk = []
    in_chunk = False

    for line in content.split('\n'):
        if line.startswith('--- Chunk'):
            if current_chunk and in_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            in_chunk = True
        elif in_chunk:
            current_chunk.append(line)

    if current_chunk and in_chunk:
        chunks.append('\n'.join(current_chunk))

    return [Document(page_content=chunk.strip(), metadata={}) for chunk in chunks if chunk.strip()]

def process_response(document: Document, i: int, j: int, metadata: Dict, session_id: str) -> GraphDocument:
    """Process a single document and extract entities/relationships with session isolation"""
    print(f"Processing document {i+1} out of {j} for session {session_id}")
    prompt = create_prompt()
    chain = prompt | llm
    raw_schema = chain.invoke({"input": document.page_content})
    parsed_json = json_repair.loads(raw_schema.content)

    nodes_set = set()
    relationships = []

    for rel in parsed_json:
        try:
            rel["head_type"] = rel.get("head_type") or "Unknown"
            rel["tail_type"] = rel.get("tail_type") or "Unknown"

            # Resolve entities within this session only
            head_resolved, head_matched = resolve_entity(rel["head"], rel["head_type"], session_id)
            tail_resolved, tail_matched = resolve_entity(rel["tail"], rel["tail_type"], session_id)

            # CREATE SESSION-SCOPED IDs for nodes
            head_session_id = create_session_scoped_id(head_resolved, session_id)
            tail_session_id = create_session_scoped_id(tail_resolved, session_id)

            source_node = Node(
                id=head_session_id,  
                type=rel["head_type"],
                properties={
                    "session_id": session_id,
                    "original_name": head_resolved,  # Store original name for display
                    "entity_name": head_resolved,    # For semantic queries
                    "source_file": metadata.get("source_file", "unknown")  # Add source file info
                }
            )
            target_node = Node(
                id=tail_session_id, 
                type=rel["tail_type"],
                properties={
                    "session_id": session_id,
                    "original_name": tail_resolved,  # Store original name for display
                    "entity_name": tail_resolved,    # For semantic queries
                    "source_file": metadata.get("source_file", "unknown")  # Add source file info
                }
            )

            # Create relationship with session-scoped node IDs
            relationships.append(Relationship(
                source=source_node,
                target=target_node,
                type=rel["relation"],
                properties={"session_id": session_id}
            ))

            # Handle "Same As" relationships for matched entities within the same session
            if head_matched and head_resolved != rel["head"]:
                original_head_id = create_session_scoped_id(rel['head'], session_id)
                relationships.append(Relationship(
                    source=Node(
                        id=original_head_id,
                        type=rel["head_type"],
                        properties={
                            "session_id": session_id,
                            "original_name": rel['head'],
                            "entity_name": rel['head'],
                            "source_file": metadata.get("source_file", "unknown")  # Add source file info
                        }
                    ),
                    target=source_node,
                    type="Same As",
                    properties={"session_id": session_id}
                ))
                nodes_set.add((original_head_id, str(rel["head_type"]), session_id, rel['head'], metadata.get("source_file", "unknown")))

            if tail_matched and tail_resolved != rel["tail"]:
                original_tail_id = create_session_scoped_id(rel['tail'], session_id)
                relationships.append(Relationship(
                    source=Node(
                        id=original_tail_id,
                        type=rel["tail_type"],
                        properties={
                            "session_id": session_id,
                            "original_name": rel['tail'],
                            "entity_name": rel['tail'],
                            "source_file": metadata.get("source_file", "unknown")  # Add source file info
                        }
                    ),
                    target=target_node,
                    type="Same As",
                    properties={"session_id": session_id}
                ))
                nodes_set.add((original_tail_id, str(rel["tail_type"]), session_id, rel['tail'], metadata.get("source_file", "unknown")))

            # Add nodes to set with session information
            nodes_set.add((head_session_id, str(rel["head_type"]), session_id, head_resolved, metadata.get("source_file", "unknown")))
            nodes_set.add((tail_session_id, str(rel["tail_type"]), session_id, tail_resolved, metadata.get("source_file", "unknown")))

        except Exception as e:
            print(f"Error processing relation: {rel}, error: {e}")

    # Build nodes list - only for current session
    nodes = []
    for el in nodes_set:
        if isinstance(el, tuple) and len(el) == 5:  # Updated to 5 elements to include file_name
            session_scoped_id, node_type, node_session_id, original_name, source_file = el
            # Ensure this node belongs to the current session
            if node_session_id == session_id:
                nodes.append(Node(
                    id=session_scoped_id,  # Session-scoped ID
                    type=node_type, 
                    properties={
                        "session_id": node_session_id,
                        "original_name": original_name,
                        "entity_name": original_name,
                        "source_file": source_file  # Include source file info
                    }
                ))
        else:
            print(f"Skipping malformed node: {el}")

    # All relationships are already session-scoped, no additional filtering needed
    return GraphDocument(nodes=nodes, relationships=relationships, source=document)

def update_nodes_with_session_id(session_id: str):
    """Update all nodes in the graph with a session_id property for a specific session"""
    query = """
    MATCH (n)
    WHERE NOT EXISTS(n.session_id)
    SET n.session_id = $session_id
    RETURN count(n) as nodes_updated
    """
    result = graph.query(query, params={"session_id": session_id})
    return result[0]["nodes_updated"]

def add_property_to_node(tx, node_id, property_name, property_value):
    """Add a property to a specific node"""
    query = """
    MATCH (n)
    WHERE id(n) = $node_id
    SET n[$property_name] = $property_value
    RETURN n
    """
    result = tx.run(query, node_id=node_id, property_name=property_name, property_value=property_value)
    return result.single()

def process_text_file_dense(input_file: str, session_id: str, file_name: str, meta: Optional[Dict] = None) -> List[GraphDocument]:
    """
    Main function to process a text file and store results in Neo4j with complete session isolation
    
    Args:
        input_file: Path to the input text file
        session_id: Unique session identifier
        file_name: Original name of the source file
        meta: Optional metadata dictionary
    Returns:
        list: The graph data that was stored in Neo4j
    """

    logger.info(f"Creating Dense Graph")
    
    if graph is None or llm is None or embedding_model is None:
        initialize_globals()
    
    clear_session_data_dense(session_id)
    
    metadata = flatten_json(meta) if meta else {}
    metadata['source_file'] = file_name  # Add file_name to metadata
    cache_file = f"output_dense_{session_id}.pkl" 

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            text_summaries = pickle.load(f)
    else:
        documents = parse_chunked_file(input_file)
        text_summaries = [process_response(doc, i, len(documents), metadata, session_id)
                         for i, doc in enumerate(documents)]
        with open(cache_file, "wb") as f:
            pickle.dump(text_summaries, f)

    graph.add_graph_documents(text_summaries, baseEntityLabel=True, include_source=True)
    
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    return text_summaries

def create_prompt():
    """Create the prompt template for entity extraction (dense version)"""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one type.",
        "Extract as many entities and relations as you can from the text to create "
        "a comprehensive and dense knowledge graph.",
        "Maintain Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
        "- Extract all possible entities and relationships for a dense graph.",
        "- Include both explicit and implicit relationships."
    ]
    
    system_prompt = "\n".join(filter(None, base_string_parts))
    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)
    
    examples = [
        {
            "text": (
                "Adam is a software engineer in Microsoft since 2009, "
                "and last year he got an award as the Best Talent. "
                "Adam has worked on several projects including Project X."
            ),
            "head": "Adam",
            "head_type": "Person",
            "relation": "WORKS_FOR",
            "tail": "Microsoft",
            "tail_type": "Company",
        },
        {
            "text": (
                "Adam is a software engineer in Microsoft since 2009, "
                "and last year he got an award as the Best Talent"
            ),
            "head": "Adam",
            "head_type": "Person",
            "relation": "HAS_AWARD",
            "tail": "Best Talent",
            "tail_type": "Award",
        }
    ]

    human_prompt = PromptTemplate(
        template="""Based on the following example, extract entities and
        relations from the provided text. Extract as many relationships as possible for a dense graph.\n\n

        Below are a number of examples of text and their extracted entities and relationships.
        {examples}

        For the following text or table, extract entities and relations as in the provided example. Table is in HTML format.
        {format_instructions}\nText: {input}
        IMPORTANT NOTES:
        - Each key must have a valid value, 'null' is not allowed. 
        - Don't add any explanation and text. 
        - Extract all possible entities and relationships.
        - Include implicit relationships and connections.""",
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message_prompt])
    return chat_prompt

def flatten_json(data: Dict) -> Dict:
    """Flatten a nested dictionary"""
    out = {}
    
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '*')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '*')
        else:
            out[name[:-1]] = x
    
    flatten(data)
    return out