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
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer, util
import torch
from config import Config
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)


class Element(BaseModel):
    type: str
    text: Any

# Global variables
graph = None
llm = None
embedding_model = None

# SESSION-SPECIFIC entity memory and frequencies  
session_entity_memory = {}  # session_id -> entity_memory
session_entity_frequencies = {}  # session_id -> entity_frequencies

SIMILARITY_THRESHOLD = 0.8  # For entity resolution
MIN_FREQUENCY_THRESHOLD = 3  # Higher threshold for high-level graph
MIN_RELATIONSHIP_COUNT = 2  # Minimum relationships per node to be included

def initialize_globals():
    """Initialize global variables"""
    global graph, llm, embedding_model

    
    
    graph = Neo4jGraph(
        url=Config.NEO4J_URI,  
        username=Config.NEO4J_USERNAME,              
        password=Config.NEO4J_PASSWORD,       
    )
    
    openai_api_key = Config.OPENAI_API_KEY
    
    llm = ChatOpenAI(
        temperature=0.2,
        max_tokens=1024,
        openai_api_key=openai_api_key,
        stop=None
    )
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_session_scoped_id(entity_name: str, session_id: str) -> str:
    """Create a session-scoped unique ID for an entity"""
    return f"{session_id}_{entity_name}"

def get_session_entity_memory(session_id: str) -> Dict:
    """Get entity memory for a specific session"""
    if session_id not in session_entity_memory:
        session_entity_memory[session_id] = {}
    return session_entity_memory[session_id]

def get_session_entity_frequencies(session_id: str) -> defaultdict:
    """Get entity frequencies for a specific session"""
    if session_id not in session_entity_frequencies:
        session_entity_frequencies[session_id] = defaultdict(int)
    return session_entity_frequencies[session_id]

def create_prompt():
    """Create the prompt template for entity extraction - focused on high-level entities"""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a HIGH-LEVEL knowledge graph. Your task is to identify "
        "the MOST IMPORTANT entities and relations from a given text. Focus on:",
        "1. Main subjects, key people, organizations, locations, and concepts",
        "2. High-level relationships that show major connections and dependencies", 
        "3. Entities that appear multiple times or have significant importance",
        "You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type".',
        "FOCUS ON HIGH-LEVEL ENTITIES ONLY:",
        "- Include: Main characters, key organizations, important locations, major concepts, significant events",
        "- Exclude: Minor details, adjectives, descriptive phrases, temporary states",
        "Maintain Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
        "- Focus ONLY on high-level, important entities.",
        "- Prioritize entities that appear multiple times or have clear significance.",
        "- Avoid extracting minor details or temporary descriptors."
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
        template="""Based on the following example, extract ONLY HIGH-LEVEL entities and
        relations from the provided text. Focus on the most important elements.\n\n

        Below are a number of examples of text and their extracted entities and relationships.
        {examples}

        For the following text or table, extract ONLY high-level entities and relations as in the provided example. Table is in HTML format.
        {format_instructions}\nText: {input}
        IMPORTANT NOTES:
        - Each key must have a valid value, 'null' is not allowed. 
        - Don't add any explanation and text. 
        - Extract ONLY high-level, important entities (main subjects, key people, organizations, major concepts).
        - Avoid minor details, descriptive adjectives, or temporary states.
        - Focus on entities that have clear significance or appear multiple times.""",
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

def resolve_entity(entity_name: str, entity_type: str, session_id: str) -> Tuple[str, bool]:
    """Resolve entity name to canonical form using similarity within the same session only"""
    entity_embedding = embedding_model.encode(entity_name, convert_to_tensor=True)
    
    entity_memory = get_session_entity_memory(session_id)
    entity_frequencies = get_session_entity_frequencies(session_id)
    
    entity_frequencies[entity_name] += 1
    
    for canonical, data in entity_memory.items():
        sim = util.cos_sim(entity_embedding, data['embedding']).item()
        if sim >= SIMILARITY_THRESHOLD and entity_type == data['type']:
            entity_frequencies[canonical] += 1
            return canonical, True
    
    entity_memory[entity_name] = {'embedding': entity_embedding, 'type': entity_type}
    return entity_name, False

def is_significant_entity(entity_name: str, session_id: str) -> bool:
    """Check if an entity appears frequently enough to be significant within a session"""
    entity_frequencies = get_session_entity_frequencies(session_id)
    return entity_frequencies.get(entity_name, 0) >= MIN_FREQUENCY_THRESHOLD

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

def filter_high_level_nodes(nodes: List[Node], relationships: List[Relationship]) -> Tuple[List[Node], List[Relationship]]:
    """
    Filter nodes to keep only high-level, well-connected entities
    """
    # Count connections for each node
    node_degrees = defaultdict(int)
    node_relationship_types = defaultdict(set)
    
    for rel in relationships:
        node_degrees[rel.source.id] += 1
        node_degrees[rel.target.id] += 1
        node_relationship_types[rel.source.id].add(rel.type)
        node_relationship_types[rel.target.id].add(rel.type)
    
    # Filter nodes based on multiple criteria for high-level graph
    filtered_nodes = []
    for node in nodes:
        node_id = node.id
        frequency = node.properties.get("frequency", 1)
        connections = node_degrees.get(node_id, 0)
        relationship_diversity = len(node_relationship_types.get(node_id, set()))
        
        # High-level criteria: frequency >= 3 OR (connections >= 2 AND frequency >= 2)
        is_high_level = (
            frequency >= MIN_FREQUENCY_THRESHOLD or 
            (connections >= MIN_RELATIONSHIP_COUNT and frequency >= 2) or
            relationship_diversity >= 3  # Nodes with diverse relationships are important
        )
        
        if is_high_level:
            filtered_nodes.append(node)
    
    # Keep only relationships between filtered nodes
    filtered_node_ids = {node.id for node in filtered_nodes}
    filtered_relationships = [
        rel for rel in relationships 
        if rel.source.id in filtered_node_ids and rel.target.id in filtered_node_ids
    ]
    
    print(f"Filtered from {len(nodes)} to {len(filtered_nodes)} high-level nodes")
    print(f"Filtered from {len(relationships)} to {len(filtered_relationships)} relationships")
    
    return filtered_nodes, filtered_relationships

def process_response(document: Document, i: int, j: int, metadata: dict, session_id: str) -> GraphDocument:
    """
    Process a single document, extract entities and relationships,
    ensuring complete session isolation with session-scoped node IDs.
    """
    print(f"Processing document {i+1} out of {j} for session {session_id}")
    prompt = create_prompt()
    chain = prompt | llm
    raw_schema = chain.invoke({"input": document.page_content})
    parsed_json = json_repair.loads(raw_schema.content)

    entity_frequencies = get_session_entity_frequencies(session_id)
    
    nodes_set = set()
    relationships = []

    for rel in parsed_json:
        try:
            rel["head_type"] = rel.get("head_type") or "Unknown"
            rel["tail_type"] = rel.get("tail_type") or "Unknown"

            head_resolved, head_matched = resolve_entity(rel["head"], rel["head_type"], session_id)
            tail_resolved, tail_matched = resolve_entity(rel["tail"], rel["tail_type"], session_id)

            # Include entities based on significance (will be further filtered later)
            if is_significant_entity(head_resolved, session_id) or is_significant_entity(tail_resolved, session_id):
                head_scoped_id = create_session_scoped_id(head_resolved, session_id)
                tail_scoped_id = create_session_scoped_id(tail_resolved, session_id)
                
                source_node = Node(
                    id=head_scoped_id,  
                    type=rel["head_type"],
                    properties={
                        "session_id": session_id,
                        "original_name": head_resolved,  # Display name
                        "entity_name": head_resolved,    # For semantic queries
                        "frequency": entity_frequencies.get(head_resolved, 1),
                        "source_file": metadata.get("source_file", "unknown")  # Add source file info
                    }
                )
                target_node = Node(
                    id=tail_scoped_id,  
                    type=rel["tail_type"],
                    properties={
                        "session_id": session_id,
                        "original_name": tail_resolved,  # Display name
                        "entity_name": tail_resolved,    # For semantic queries
                        "frequency": entity_frequencies.get(tail_resolved, 1),
                        "source_file": metadata.get("source_file", "unknown")  # Add source file info
                    }
                )

                relationships.append(Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel["relation"],
                    properties={"session_id": session_id}
                ))

                if head_matched and head_resolved != rel["head"]:
                    original_head_scoped_id = create_session_scoped_id(rel["head"], session_id)
                    relationships.append(Relationship(
                        source=Node(
                            id=original_head_scoped_id, 
                            type=rel["head_type"], 
                            properties={
                                "session_id": session_id,
                                "original_name": rel["head"],
                                "entity_name": rel["head"],
                                "source_file": metadata.get("source_file", "unknown")  # Add source file info
                            }
                        ),
                        target=Node(
                            id=head_scoped_id, 
                            type=rel["head_type"], 
                            properties={
                                "session_id": session_id,
                                "original_name": head_resolved,
                                "entity_name": head_resolved,
                                "source_file": metadata.get("source_file", "unknown")  # Add source file info
                            }
                        ),
                        type="SAME_AS",
                        properties={"session_id": session_id}
                    ))

                if tail_matched and tail_resolved != rel["tail"]:
                    original_tail_scoped_id = create_session_scoped_id(rel["tail"], session_id)
                    relationships.append(Relationship(
                        source=Node(
                            id=original_tail_scoped_id, 
                            type=rel["tail_type"], 
                            properties={
                                "session_id": session_id,
                                "original_name": rel["tail"],
                                "entity_name": rel["tail"],
                                "source_file": metadata.get("source_file", "unknown")  # Add source file info
                            }
                        ),
                        target=Node(
                            id=tail_scoped_id, 
                            type=rel["tail_type"], 
                            properties={
                                "session_id": session_id,
                                "original_name": tail_resolved,
                                "entity_name": tail_resolved,
                                "source_file": metadata.get("source_file", "unknown")  # Add source file info
                            }
                        ),
                        type="SAME_AS",
                        properties={"session_id": session_id}
                    ))

                nodes_set.add((head_scoped_id, str(rel["head_type"]), session_id, head_resolved, metadata.get("source_file", "unknown")))
                nodes_set.add((tail_scoped_id, str(rel["tail_type"]), session_id, tail_resolved, metadata.get("source_file", "unknown")))
                
                if head_matched and head_resolved != rel["head"]:
                    original_head_scoped_id = create_session_scoped_id(rel["head"], session_id)
                    nodes_set.add((original_head_scoped_id, str(rel["head_type"]), session_id, rel["head"], metadata.get("source_file", "unknown")))
                    
                if tail_matched and tail_resolved != rel["tail"]:
                    original_tail_scoped_id = create_session_scoped_id(rel["tail"], session_id)
                    nodes_set.add((original_tail_scoped_id, str(rel["tail_type"]), session_id, rel["tail"], metadata.get("source_file", "unknown")))

        except Exception as e:
            print(f"Error processing relation: {rel}, error: {e}")

    nodes = []
    for el in nodes_set:
        if isinstance(el, tuple) and len(el) == 5:  # Updated to include source_file
            session_scoped_id, node_type, node_session_id, original_name, source_file = el
            # Include entities that meet minimum threshold - will be filtered for high-level later
            if node_session_id == session_id and entity_frequencies.get(original_name, 0) >= 1:
                nodes.append(Node(
                    id=session_scoped_id,  # Session-scoped unique ID
                    type=node_type,
                    properties={
                        "session_id": node_session_id,
                        "original_name": original_name,
                        "entity_name": original_name,
                        "frequency": entity_frequencies.get(original_name, 1),
                        "source_file": source_file  # Include source file info
                    }
                ))
        else:
            print(f"Skipping malformed node: {el}")

    # Ensure session isolation
    nodes = [node for node in nodes if node.properties.get("session_id") == session_id]
    node_ids_in_session = {node.id for node in nodes}
    relationships = [
        rel for rel in relationships
        if rel.properties.get("session_id") == session_id and
           rel.source.id in node_ids_in_session and
           rel.target.id in node_ids_in_session
    ]

    # Apply high-level filtering
    nodes, relationships = filter_high_level_nodes(nodes, relationships)

    print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships for document {i+1}")
    return GraphDocument(nodes=nodes, relationships=relationships, source=document)


def clear_session_data(session_id: str):
    """Clear entity memory and frequencies for a specific session"""
    if session_id in session_entity_memory:
        del session_entity_memory[session_id]
    if session_id in session_entity_frequencies:
        del session_entity_frequencies[session_id]

def get_session_nodes(session_id: str) -> List[Dict]:
    """Query Neo4j for nodes belonging to a specific session"""
    if graph is None:
        initialize_globals()
    
    query = """
    MATCH (n)
    WHERE n.session_id = $session_id
    RETURN n.id as id, n.original_name as name, labels(n)[0] as type, n.frequency as frequency
    ORDER BY n.frequency DESC
    """
    
    result = graph.query(query, params={"session_id": session_id})
    return [dict(record) for record in result]

def get_session_relationships(session_id: str) -> List[Dict]:
    """Query Neo4j for relationships belonging to a specific session"""
    if graph is None:
        initialize_globals()
    
    query = """
    MATCH (a)-[r]->(b)
    WHERE r.session_id = $session_id
    RETURN a.original_name as source, type(r) as relationship, b.original_name as target
    """
    
    result = graph.query(query, params={"session_id": session_id})
    return [dict(record) for record in result]

def delete_session_data_from_neo4j(session_id: str):
    """Delete all nodes and relationships for a specific session from Neo4j"""
    if graph is None:
        initialize_globals()
    
    # Delete relationships first
    query_rels = """
    MATCH ()-[r]->()
    WHERE r.session_id = $session_id
    DELETE r
    """
    graph.query(query_rels, params={"session_id": session_id})
    
    # Then delete nodes
    query_nodes = """
    MATCH (n)
    WHERE n.session_id = $session_id
    DELETE n
    """
    graph.query(query_nodes, params={"session_id": session_id})
    
    print(f"Deleted all data for session: {session_id}")

def process_text_file(input_file: str, session_id: str, file_name: str, meta: Optional[Dict] = None) -> List[GraphDocument]:
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

    logging.info(f"Creating High Level Graph")
    if graph is None or llm is None or embedding_model is None:
        initialize_globals()
    
    clear_session_data(session_id)
    
    metadata = flatten_json(meta) if meta else {}
    metadata['source_file'] = file_name  
    cache_file = f"output_{session_id}.pkl"  

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            text_summaries = pickle.load(f)
    else:
        documents = parse_chunked_file(input_file)
        text_summaries = [process_response(doc, i, len(documents), metadata, session_id)
                         for i, doc in enumerate(documents)]
        with open(cache_file, "wb") as f:
            pickle.dump(text_summaries, f)

    # Store in Neo4j with session isolation
    graph.add_graph_documents(text_summaries, baseEntityLabel=True, include_source=True)
    
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    total_nodes = sum(len(doc.nodes) for doc in text_summaries)
    total_relationships = sum(len(doc.relationships) for doc in text_summaries)
    
    print(f"Successfully processed {len(text_summaries)} documents for session: {session_id}")
    print(f"Total nodes extracted: {total_nodes}")
    print(f"Total relationships extracted: {total_relationships}")
    return text_summaries