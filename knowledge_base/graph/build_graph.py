#!/usr/bin/env python3
"""
Phase 3: Build NetworkX Graph

Build a NetworkX directed graph from extracted concepts and relations.
Outputs: kg.gpickle, graph_statistics.json
"""

import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import networkx as nx


def load_concepts(concepts_path: Path) -> pd.DataFrame:
    """Load concept nodes from parquet."""
    return pd.read_parquet(concepts_path)


def load_edges(edges_path: Path) -> pd.DataFrame:
    """Load semantic edges from parquet."""
    return pd.read_parquet(edges_path)


def load_evidence(evidence_path: Path) -> pd.DataFrame:
    """Load paper evidence from parquet."""
    return pd.read_parquet(evidence_path)


def build_knowledge_graph(
    concepts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    evidence_df: pd.DataFrame
) -> nx.DiGraph:
    """
    Build NetworkX directed graph from concepts and edges.
    
    Node attributes:
        - type: concept type (BrainRegion, CognitiveFunction, etc.)
        - display_name: human-readable name
        - description: concept description
        - frequency: number of papers mentioning this concept
    
    Edge attributes:
        - relation: relation type (SUPPORTS, CAUSES, etc.)
        - description: relation description
        - weight: evidence count
        - evidence_ids: list of paper IDs supporting this edge
    """
    G = nx.DiGraph()
    
    # Add concept nodes
    print("Adding concept nodes...")
    for _, row in concepts_df.iterrows():
        node_id = row['name']  # Use concept name as node ID
        G.add_node(
            node_id,
            type=row['type'],
            display_name=row['display_name'],
            description=row['description'],
            frequency=row['frequency'],
            aliases=row['aliases']
        )
    print(f"  Added {G.number_of_nodes()} nodes")
    
    # Add semantic edges
    print("Adding semantic edges...")
    for _, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        
        # Skip if nodes don't exist
        if source not in G.nodes or target not in G.nodes:
            print(f"  Skipping edge {source} -> {target}: node not found")
            continue
        
        G.add_edge(
            source,
            target,
            relation=row['relation'],
            description=row['description'],
            weight=row['weight'],
            evidence_count=row['evidence_count'],
            evidence_ids=row['evidence_ids']
        )
    print(f"  Added {G.number_of_edges()} edges")
    
    return G


def compute_graph_statistics(G: nx.DiGraph) -> dict:
    """Compute graph statistics."""
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_weakly_connected(G),
        'num_weakly_connected_components': nx.number_weakly_connected_components(G),
    }
    
    # Node statistics by type
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    stats['nodes_by_type'] = node_types
    
    # Edge statistics by relation
    edge_relations = {}
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'Unknown')
        edge_relations[relation] = edge_relations.get(relation, 0) + 1
    stats['edges_by_relation'] = edge_relations
    
    # Degree statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    stats['top_nodes_by_in_degree'] = sorted(
        in_degrees.items(), key=lambda x: x[1], reverse=True
    )[:10]
    
    stats['top_nodes_by_out_degree'] = sorted(
        out_degrees.items(), key=lambda x: x[1], reverse=True
    )[:10]
    
    # Find isolated nodes (no edges)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    stats['isolated_nodes'] = isolated
    stats['num_isolated_nodes'] = len(isolated)
    
    # Find most connected node pairs
    edge_weights = [(u, v, d['evidence_count']) for u, v, d in G.edges(data=True)]
    stats['strongest_edges'] = sorted(edge_weights, key=lambda x: x[2], reverse=True)[:10]
    
    return stats


def save_graph(G: nx.DiGraph, output_path: Path):
    """Save graph as pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved graph to {output_path}")


def save_statistics(stats: dict, output_path: Path):
    """Save graph statistics to JSON."""
    # Convert tuples to lists for JSON serialization
    stats_json = stats.copy()
    stats_json['top_nodes_by_in_degree'] = [
        {'node': n, 'in_degree': d} for n, d in stats['top_nodes_by_in_degree']
    ]
    stats_json['top_nodes_by_out_degree'] = [
        {'node': n, 'out_degree': d} for n, d in stats['top_nodes_by_out_degree']
    ]
    stats_json['strongest_edges'] = [
        {'source': u, 'target': v, 'evidence_count': w} 
        for u, v, w in stats['strongest_edges']
    ]
    
    with open(output_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"Saved statistics to {output_path}")


def visualize_graph_summary(G: nx.DiGraph, stats: dict):
    """Print graph summary."""
    print("\n" + "=" * 60)
    print("Knowledge Graph Summary")
    print("=" * 60)
    
    print(f"\nNodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Density: {stats['density']:.4f}")
    print(f"Weakly Connected: {stats['is_connected']}")
    print(f"Connected Components: {stats['num_weakly_connected_components']}")
    print(f"Isolated Nodes: {stats['num_isolated_nodes']}")
    
    print("\n--- Nodes by Type ---")
    for node_type, count in sorted(stats['nodes_by_type'].items(), key=lambda x: -x[1]):
        print(f"  {node_type}: {count}")
    
    print("\n--- Edges by Relation ---")
    for relation, count in sorted(stats['edges_by_relation'].items(), key=lambda x: -x[1]):
        print(f"  {relation}: {count}")
    
    print("\n--- Top 10 Nodes by In-Degree (most targeted) ---")
    for node, degree in stats['top_nodes_by_in_degree']:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"  {node} ({node_type}): {degree}")
    
    print("\n--- Top 10 Nodes by Out-Degree (most influential) ---")
    for node, degree in stats['top_nodes_by_out_degree']:
        node_type = G.nodes[node].get('type', 'Unknown')
        print(f"  {node} ({node_type}): {degree}")
    
    print("\n--- Strongest Edges (by evidence count) ---")
    for source, target, weight in stats['strongest_edges']:
        edge_data = G.edges[source, target]
        relation = edge_data.get('relation', 'Unknown')
        print(f"  {source} --{relation}--> {target}: {weight} papers")


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent
    
    concepts_path = base_dir / 'nodes' / 'concepts.parquet'
    edges_path = base_dir / 'edges' / 'semantic_edges.parquet'
    evidence_path = base_dir / 'evidence' / 'paper_evidence.parquet'
    graph_path = base_dir / 'kg.gpickle'
    stats_path = base_dir / 'graph_statistics.json'
    
    print("=" * 60)
    print("Phase 3: Build Knowledge Graph")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    concepts_df = load_concepts(concepts_path)
    edges_df = load_edges(edges_path)
    evidence_df = load_evidence(evidence_path)
    
    print(f"  Concepts: {len(concepts_df)}")
    print(f"  Edges: {len(edges_df)}")
    print(f"  Evidence papers: {len(evidence_df)}")
    
    # Build graph
    print("\nBuilding knowledge graph...")
    G = build_knowledge_graph(concepts_df, edges_df, evidence_df)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_graph_statistics(G)
    
    # Save outputs
    print("\nSaving outputs...")
    save_graph(G, graph_path)
    save_statistics(stats, stats_path)
    
    # Print summary
    visualize_graph_summary(G, stats)
    
    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
