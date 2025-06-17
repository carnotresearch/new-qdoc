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
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv
from collections import defaultdict
from app.services.llm_service import get_standard_llm

load_dotenv()

class Element(BaseModel):
    type: str
    text: Any

graph = None
llm = None
embedding_model = None
entity_memory = {}
entity_frequencies = defaultdict(int)  
SIMILARITY_THRESHOLD = 0.8
MIN_FREQUENCY_THRESHOLD = 1

def initialize_globals():
    """Initialize global variables"""
    global graph, llm, embedding_model
    
    graph = Neo4jGraph()
    
    # Use centralized LLM service for knowledge graph generation
    llm = get_standard_llm()
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_prompt():
    """Create the prompt template for entity extraction"""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one type.",
        "Attempt to extract as many entities and relations as you can, but focus "
        "on entities that appear multiple times in the text as they are more significant.",
        "Maintain Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
        "- Focus on entities that appear multiple times in the text.",
        "- Only include relationships between significant entities (those that appear frequently)."
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
        relations from the provided text. Focus on entities that appear multiple times.\n\n

        Below are a number of examples of text and their extracted entities and relationships.
        {examples}

        For the following text or table, extract entities and relations as in the provided example. Table is in HTML format.
        {format_instructions}\nText: {input}
        IMPORTANT NOTES:
        - Each key must have a valid value, 'null' is not allowed. 
        - Don't add any explanation and text. 
        - Focus on entities that appear multiple times.
        - Only include relationships between significant entities.""",
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

def resolve_entity(entity_name: str, entity_type: str) -> Tuple[str, bool]:
    """Resolve entity name to canonical form using similarity and track frequency"""
    entity_embedding = embedding_model.encode(entity_name, convert_to_tensor=True)
    
    # Track frequency
    entity_frequencies[entity_name] += 1
    
    for canonical, data in entity_memory.items():
        sim = util.cos_sim(entity_embedding, data['embedding']).item()
        if sim >= SIMILARITY_THRESHOLD and entity_type == data['type']:
            # If we find a match, increment the frequency of the canonical entity
            entity_frequencies[canonical] += 1
            return canonical, True
    
    entity_memory[entity_name] = {'embedding': entity_embedding, 'type': entity_type}
    return entity_name, False

def is_significant_entity(entity_name: str) -> bool:
    """Check if an entity appears frequently enough to be significant"""
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

def process_response(document: Document, i: int, j: int, metadata: Dict, session_id: str) -> GraphDocument:
    """Process a single document and extract entities/relationships"""
    print(f"processing document {i+1} out of {j}")
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

            head_resolved, head_matched = resolve_entity(rel["head"], rel["head_type"])
            tail_resolved, tail_matched = resolve_entity(rel["tail"], rel["tail_type"])

            # Only include relationships where both entities are significant
            if is_significant_entity(head_resolved) and is_significant_entity(tail_resolved):
                # Create nodes with session_id property
                source_node = Node(
                    id=head_resolved, 
                    type=rel["head_type"],
                    properties={
                        "session_id": session_id,
                        "frequency": entity_frequencies.get(head_resolved, 1)
                    }  
                )
                target_node = Node(
                    id=tail_resolved, 
                    type=rel["tail_type"],
                    properties={
                        "session_id": session_id,
                        "frequency": entity_frequencies.get(tail_resolved, 1)
                    } 
                )

                relationships.append(Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel["relation"],
                    properties={"session_id": session_id}  
                ))

                if head_matched and head_resolved != rel["head"]:
                    relationships.append(Relationship(
                        source=Node(id=rel["head"], type=rel["head_type"], properties={"session_id": session_id}),
                        target=Node(id=head_resolved, type=rel["head_type"], properties={"session_id": session_id}),
                        type="Same As",
                        properties={"session_id": session_id}
                    ))

                if tail_matched and tail_resolved != rel["tail"]:
                    relationships.append(Relationship(
                        source=Node(id=rel["tail"], type=rel["tail_type"], properties={"session_id": session_id}),
                        target=Node(id=tail_resolved, type=rel["tail_type"], properties={"session_id": session_id}),
                        type="Same As",
                        properties={"session_id": session_id}
                    ))

                nodes_set.add((str(head_resolved), str(rel["head_type"]), session_id, entity_frequencies.get(head_resolved, 1)))
                nodes_set.add((str(tail_resolved), str(rel["tail_type"]), session_id, entity_frequencies.get(tail_resolved, 1)))

                if head_matched and head_resolved != rel["head"]:
                    nodes_set.add((str(rel["head"]), str(rel["head_type"]), session_id, 1))
                if tail_matched and tail_resolved != rel["tail"]:
                    nodes_set.add((str(rel["tail"]), str(rel["tail_type"]), session_id, 1))

        except Exception as e:
            print(f"Error processing relation: {rel}, error: {e}")

    nodes = []
    for el in nodes_set:
        if isinstance(el, tuple) and len(el) == 4:
            # Only include nodes that are significant
            if el[3] >= MIN_FREQUENCY_THRESHOLD:
                nodes.append(Node(
                    id=el[0], 
                    type=el[1], 
                    properties={
                        "session_id": el[2],
                        "frequency": el[3]
                    }
                ))
        else:
            print(f"Skipping malformed node: {el}")

    return GraphDocument(nodes=nodes, relationships=relationships, source=document)

def process_text_file(input_file: str, session_id: str, meta: Optional[Dict] = None) -> None:
    """
    Main function to process a text file and store results in Neo4j
    
    Args:
        input_file: Path to the input text file
        meta: Optional metadata dictionary
    Returns:
        dict: The graph data that was stored in Neo4j
    """
    if graph is None or llm is None or embedding_model is None:
        initialize_globals()
    
    metadata = flatten_json(meta) if meta else {}
    cache_file = "output.pkl"

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