
import json
import os
from neo4j import GraphDatabase
from typing import List,Dict,Optional
import json
import os
from neo4j import GraphDatabase
from typing import Dict

def fetch_neo4j_graph_data(
    session_id: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str
) -> Dict:
    """
    Fetch nodes and relationships strictly belonging to the provided session_id.
    """
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_username, neo4j_password)
    )

    # Query to get all nodes for this session only
    nodes_query = """
    MATCH (n {session_id: $session_id})
    RETURN 
        n.entity_name as display_name,
        n.original_name as original_name,
        properties(n) as properties, 
        labels(n) as labels,
        id(n) as neo4j_id
    """

    # Query to get relationships where both nodes and relationship have matching session_id
    rels_query = """
    MATCH (n {session_id: $session_id})-[r {session_id: $session_id}]->(m {session_id: $session_id})
    RETURN 
        n.entity_name as source_name,
        m.entity_name as target_name,
        properties(n) as source_props,
        labels(n) as source_labels,
        properties(m) as target_props,
        labels(m) as target_labels,
        type(r) as rel_type,
        properties(r) as rel_properties
    """

    graph_data = {'nodes': [], 'relationships': []}
    
    try:
        with driver.session() as session:
            # Fetch nodes
            nodes_result = session.run(nodes_query, session_id=session_id)
            node_map = {}
            
            for record in nodes_result:
                props = record['properties']
                if props.get('session_id') == session_id:
                    # Use display name for the graph
                    display_name = record['display_name'] or record['original_name']
                    node_map[display_name] = {
                        'id': display_name,  # Use entity name as display ID
                        'name': display_name,
                        'labels': record['labels'],
                        'properties': props,
                        'neo4j_id': record['neo4j_id']
                    }
                    graph_data['nodes'].append(node_map[display_name])

            # Fetch relationships
            rels_result = session.run(rels_query, session_id=session_id)
            
            for record in rels_result:
                source_name = record['source_name']
                target_name = record['target_name']
                source_props = record['source_props']
                target_props = record['target_props']
                rel_props = record['rel_properties']
                
                # Ensure all components belong to the same session
                if (source_props.get('session_id') == session_id and
                    target_props.get('session_id') == session_id and
                    rel_props.get('session_id') == session_id and
                    source_name in node_map and
                    target_name in node_map):
                    
                    graph_data['relationships'].append({
                        'source': source_name,  # Use display names
                        'target': target_name,
                        'type': record['rel_type'],
                        'properties': rel_props,
                        'source_labels': record['source_labels'],
                        'target_labels': record['target_labels']
                    })

    except Exception as e:
        raise Exception(f"Neo4j query failed: {str(e)}")
    finally:
        driver.close()
    
    return graph_data

def save_neo4j_graph(
    graph_data: Dict,
    output_path: str
) -> None:
    """
    Save Neo4j graph data to JSON file
    
    Args:
        graph_data: Dictionary containing nodes and relationships
        output_path: Full path to the output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)