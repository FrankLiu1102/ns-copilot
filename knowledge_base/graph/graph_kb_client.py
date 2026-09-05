#!/usr/bin/env python3
"""
GraphKBClient - Knowledge Graph Query Interface

Provides methods to query the neuroscience knowledge graph for QA tasks.
"""

import json
import pickle
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import pandas as pd


@dataclass
class PathResult:
    """Result of a path query between concepts."""
    source: str
    target: str
    path: list[str]           # List of node IDs in path
    edges: list[dict]         # List of edge data along path
    total_evidence: int       # Sum of evidence counts
    
    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'target': self.target,
            'path': self.path,
            'edges': self.edges,
            'total_evidence': self.total_evidence
        }


@dataclass
class SubgraphResult:
    """Result of a subgraph query."""
    nodes: list[dict]         # List of node data
    edges: list[dict]         # List of edge data
    center_node: str          # The query center
    
    def to_dict(self) -> dict:
        return {
            'center_node': self.center_node,
            'nodes': self.nodes,
            'edges': self.edges
        }


@dataclass
class ContextResult:
    """Result of context building for LLM."""
    context_text: str         # Formatted context string
    concepts: list[str]       # Concepts included
    edges: list[dict]         # Edges included
    evidence_papers: list[dict]  # Paper citations


class GraphKBClient:
    """
    Knowledge Graph Client for querying neuroscience concepts and relations.
    
    Usage:
        client = GraphKBClient()
        client.load()
        
        # Find paths between concepts
        paths = client.find_paths('dopamine', 'working_memory', max_hops=2)
        
        # Get neighbors of a concept
        neighbors = client.get_neighbors('prefrontal_cortex', hops=1)
        
        # Build context for LLM
        context = client.build_context(['dopamine', 'working_memory'])
    """
    
    def __init__(self, graph_dir: Optional[Path] = None):
        """
        Initialize GraphKBClient.
        
        Args:
            graph_dir: Path to graph directory. Defaults to this file's directory.
        """
        if graph_dir is None:
            graph_dir = Path(__file__).parent
        
        self.graph_dir = Path(graph_dir)
        self.graph: Optional[nx.DiGraph] = None
        self.concepts_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        self.evidence_df: Optional[pd.DataFrame] = None
        self.alias_index: dict[str, str] = {}
        self._loaded = False
    
    def load(self) -> None:
        """Load graph and related data into memory."""
        if self._loaded:
            return
        
        graph_path = self.graph_dir / 'kg.gpickle'
        concepts_path = self.graph_dir / 'nodes' / 'concepts.parquet'
        edges_path = self.graph_dir / 'edges' / 'semantic_edges.parquet'
        evidence_path = self.graph_dir / 'evidence' / 'paper_evidence.parquet'
        vocab_path = self.graph_dir / 'ontology' / 'concept_vocab.json'
        
        # Load graph
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Load dataframes
        self.concepts_df = pd.read_parquet(concepts_path)
        self.edges_df = pd.read_parquet(edges_path)
        self.evidence_df = pd.read_parquet(evidence_path)
        
        # Build alias index
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)['concepts']
        
        for concept_name, concept_data in vocab.items():
            for alias in concept_data.get('aliases', []):
                normalized = alias.lower().strip().replace(' ', '_')
                normalized = re.sub(r'[^a-z0-9_]', '', normalized)
                self.alias_index[normalized] = concept_name
                self.alias_index[alias.lower()] = concept_name
        
        self._loaded = True
    
    def _ensure_loaded(self):
        """Ensure graph is loaded."""
        if not self._loaded:
            self.load()
    
    def match_concept(self, text: str) -> Optional[str]:
        """
        Match text to a concept in the graph.
        
        Args:
            text: Text to match (can be alias or partial match)
            
        Returns:
            Canonical concept name or None if not found
        """
        self._ensure_loaded()
        
        # Normalize input
        normalized = text.lower().strip().replace(' ', '_')
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        
        # Direct match
        if normalized in self.graph.nodes:
            return normalized
        
        # Alias match
        if normalized in self.alias_index:
            return self.alias_index[normalized]
        
        # Try original text
        if text.lower() in self.alias_index:
            return self.alias_index[text.lower()]
        
        return None
    
    def match_concepts(self, text: str) -> list[str]:
        """
        Find all concepts mentioned in text.
        
        Args:
            text: Text to search for concepts
            
        Returns:
            List of matched concept names
        """
        self._ensure_loaded()
        
        found = set()
        text_lower = text.lower()
        
        # Check each alias
        for alias, concept in self.alias_index.items():
            if alias in text_lower or alias.replace('_', ' ') in text_lower:
                found.add(concept)
        
        return list(found)
    
    def get_node(self, concept: str) -> Optional[dict]:
        """
        Get node data for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            Node data dict or None
        """
        self._ensure_loaded()
        
        # Try to match concept
        matched = self.match_concept(concept)
        if matched and matched in self.graph.nodes:
            data = dict(self.graph.nodes[matched])
            data['id'] = matched
            return data
        
        return None
    
    def get_neighbors(
        self,
        concept: str,
        hops: int = 1,
        direction: str = 'both',
        relation_types: Optional[list[str]] = None
    ) -> SubgraphResult:
        """
        Get neighboring concepts within N hops.
        
        Args:
            concept: Center concept name
            hops: Number of hops (default 1)
            direction: 'in', 'out', or 'both'
            relation_types: Filter by relation types (optional)
            
        Returns:
            SubgraphResult with nodes and edges
        """
        self._ensure_loaded()
        
        matched = self.match_concept(concept)
        if not matched or matched not in self.graph.nodes:
            return SubgraphResult(nodes=[], edges=[], center_node=concept)
        
        # BFS to find neighbors
        visited = {matched}
        frontier = {matched}
        
        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                # Get successors (outgoing edges)
                if direction in ('out', 'both'):
                    for succ in self.graph.successors(node):
                        if succ not in visited:
                            edge_data = self.graph.edges[node, succ]
                            if relation_types is None or edge_data.get('relation') in relation_types:
                                new_frontier.add(succ)
                
                # Get predecessors (incoming edges)
                if direction in ('in', 'both'):
                    for pred in self.graph.predecessors(node):
                        if pred not in visited:
                            edge_data = self.graph.edges[pred, node]
                            if relation_types is None or edge_data.get('relation') in relation_types:
                                new_frontier.add(pred)
            
            visited.update(new_frontier)
            frontier = new_frontier
        
        # Build result
        nodes = []
        for node in visited:
            data = dict(self.graph.nodes[node])
            data['id'] = node
            nodes.append(data)
        
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in visited and v in visited:
                edge = dict(data)
                edge['source'] = u
                edge['target'] = v
                edges.append(edge)
        
        return SubgraphResult(nodes=nodes, edges=edges, center_node=matched)
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 2
    ) -> list[PathResult]:
        """
        Find all paths between two concepts.
        
        Args:
            source: Source concept
            target: Target concept
            max_hops: Maximum path length
            
        Returns:
            List of PathResult objects
        """
        self._ensure_loaded()
        
        source_matched = self.match_concept(source)
        target_matched = self.match_concept(target)
        
        if not source_matched or not target_matched:
            return []
        
        if source_matched not in self.graph.nodes or target_matched not in self.graph.nodes:
            return []
        
        results = []
        
        # Use undirected view for path finding
        undirected = self.graph.to_undirected()
        
        try:
            # Find all simple paths up to max_hops
            paths = list(nx.all_simple_paths(
                undirected, source_matched, target_matched, cutoff=max_hops
            ))
            
            for path in paths:
                edges = []
                total_evidence = 0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    
                    # Check both directions for edge
                    if self.graph.has_edge(u, v):
                        edge_data = dict(self.graph.edges[u, v])
                        edge_data['source'] = u
                        edge_data['target'] = v
                        edge_data['direction'] = 'forward'
                    elif self.graph.has_edge(v, u):
                        edge_data = dict(self.graph.edges[v, u])
                        edge_data['source'] = v
                        edge_data['target'] = u
                        edge_data['direction'] = 'reverse'
                    else:
                        continue
                    
                    edges.append(edge_data)
                    total_evidence += edge_data.get('evidence_count', 0)
                
                if edges:
                    results.append(PathResult(
                        source=source_matched,
                        target=target_matched,
                        path=path,
                        edges=edges,
                        total_evidence=total_evidence
                    ))
        
        except nx.NetworkXNoPath:
            pass
        
        # Sort by total evidence
        results.sort(key=lambda x: x.total_evidence, reverse=True)
        
        return results
    
    def get_edge_evidence(self, source: str, target: str) -> list[dict]:
        """
        Get paper evidence for an edge.
        
        Args:
            source: Source concept
            target: Target concept
            
        Returns:
            List of paper evidence dicts
        """
        self._ensure_loaded()
        
        source_matched = self.match_concept(source)
        target_matched = self.match_concept(target)
        
        if not self.graph.has_edge(source_matched, target_matched):
            return []
        
        edge_data = self.graph.edges[source_matched, target_matched]
        evidence_ids = json.loads(edge_data.get('evidence_ids', '[]'))
        
        papers = []
        for paper_id in evidence_ids:
            paper_row = self.evidence_df[self.evidence_df['id'] == paper_id]
            if not paper_row.empty:
                papers.append(paper_row.iloc[0].to_dict())
        
        return papers
    
    def build_context(
        self,
        concepts: list[str],
        max_hops: int = 2,
        max_edges: int = 15,
        max_papers: int = 10
    ) -> ContextResult:
        """
        Build LLM context from concepts.
        
        Args:
            concepts: List of concept names/aliases
            max_hops: Maximum hops for path finding
            max_edges: Maximum edges to include
            max_papers: Maximum papers to cite
            
        Returns:
            ContextResult with formatted context
        """
        self._ensure_loaded()
        
        # Match concepts
        matched_concepts = []
        for c in concepts:
            matched = self.match_concept(c)
            if matched:
                matched_concepts.append(matched)
        
        if not matched_concepts:
            return ContextResult(
                context_text="No matching concepts found in knowledge graph.",
                concepts=[],
                edges=[],
                evidence_papers=[]
            )
        
        # Collect relevant edges
        relevant_edges = []
        seen_edges = set()
        
        # Direct edges between matched concepts
        for i, c1 in enumerate(matched_concepts):
            for c2 in matched_concepts[i+1:]:
                # Check both directions
                if self.graph.has_edge(c1, c2):
                    edge_data = dict(self.graph.edges[c1, c2])
                    edge_data['source'] = c1
                    edge_data['target'] = c2
                    edge_key = (c1, c2)
                    if edge_key not in seen_edges:
                        relevant_edges.append(edge_data)
                        seen_edges.add(edge_key)
                
                if self.graph.has_edge(c2, c1):
                    edge_data = dict(self.graph.edges[c2, c1])
                    edge_data['source'] = c2
                    edge_data['target'] = c1
                    edge_key = (c2, c1)
                    if edge_key not in seen_edges:
                        relevant_edges.append(edge_data)
                        seen_edges.add(edge_key)
                
                # Find paths if no direct edge
                paths = self.find_paths(c1, c2, max_hops=max_hops)
                for path in paths[:2]:  # Top 2 paths
                    for edge in path.edges:
                        edge_key = (edge['source'], edge['target'])
                        if edge_key not in seen_edges:
                            relevant_edges.append(edge)
                            seen_edges.add(edge_key)
        
        # Add edges from each concept's neighborhood
        for concept in matched_concepts:
            subgraph = self.get_neighbors(concept, hops=1)
            for edge in subgraph.edges:
                edge_key = (edge['source'], edge['target'])
                if edge_key not in seen_edges:
                    relevant_edges.append(edge)
                    seen_edges.add(edge_key)
        
        # Sort by evidence count and limit
        relevant_edges.sort(key=lambda x: x.get('evidence_count', 0), reverse=True)
        relevant_edges = relevant_edges[:max_edges]
        
        # Collect paper evidence
        all_paper_ids = set()
        for edge in relevant_edges:
            evidence_ids = json.loads(edge.get('evidence_ids', '[]'))
            all_paper_ids.update(evidence_ids)
        
        papers = []
        for paper_id in list(all_paper_ids)[:max_papers]:
            paper_row = self.evidence_df[self.evidence_df['id'] == paper_id]
            if not paper_row.empty:
                papers.append(paper_row.iloc[0].to_dict())
        
        # Build context text
        context_parts = []
        
        context_parts.append("## Knowledge Graph Context\n")
        
        # Concepts section
        context_parts.append("### Relevant Concepts\n")
        for concept in matched_concepts:
            node = self.get_node(concept)
            if node:
                context_parts.append(
                    f"- **{node['display_name']}** ({node['type']}): {node['description']}"
                )
        context_parts.append("")
        
        # Relations section
        if relevant_edges:
            context_parts.append("### Semantic Relations\n")
            for edge in relevant_edges:
                source_node = self.get_node(edge['source'])
                target_node = self.get_node(edge['target'])
                source_name = source_node['display_name'] if source_node else edge['source']
                target_name = target_node['display_name'] if target_node else edge['target']
                
                context_parts.append(
                    f"- {source_name} **{edge['relation']}** {target_name} "
                    f"(evidence: {edge.get('evidence_count', 0)} papers)"
                )
                if edge.get('description'):
                    context_parts.append(f"  - {edge['description']}")
            context_parts.append("")
        
        # Evidence section
        if papers:
            context_parts.append("### Supporting Evidence\n")
            for paper in papers:
                context_parts.append(
                    f"- {paper.get('citation_short', 'Unknown')} ({paper.get('year', 'N/A')}): "
                    f"{paper.get('title', 'N/A')[:80]}..."
                )
        
        context_text = "\n".join(context_parts)
        
        return ContextResult(
            context_text=context_text,
            concepts=matched_concepts,
            edges=relevant_edges,
            evidence_papers=papers
        )


def test_client():
    """Test GraphKBClient functionality."""
    print("=" * 60)
    print("Testing GraphKBClient")
    print("=" * 60)
    
    client = GraphKBClient()
    client.load()
    
    print(f"\nGraph loaded: {client.graph.number_of_nodes()} nodes, {client.graph.number_of_edges()} edges")
    
    # Test concept matching
    print("\n--- Test: Concept Matching ---")
    test_terms = ['dopamine', 'PFC', 'working memory', 'Alzheimer', 'fMRI']
    for term in test_terms:
        matched = client.match_concept(term)
        print(f"  '{term}' -> {matched}")
    
    # Test get_node
    print("\n--- Test: Get Node ---")
    node = client.get_node('prefrontal_cortex')
    if node:
        print(f"  prefrontal_cortex: {node['display_name']} ({node['type']})")
        print(f"    Frequency: {node['frequency']} papers")
    
    # Test get_neighbors
    print("\n--- Test: Get Neighbors (1 hop) ---")
    subgraph = client.get_neighbors('working_memory', hops=1)
    print(f"  working_memory neighbors: {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
    for edge in subgraph.edges[:5]:
        print(f"    {edge['source']} --{edge['relation']}--> {edge['target']}")
    
    # Test find_paths
    print("\n--- Test: Find Paths ---")
    paths = client.find_paths('dopamine', 'working_memory', max_hops=2)
    print(f"  dopamine -> working_memory: {len(paths)} path(s)")
    for path in paths[:3]:
        path_str = " -> ".join(path.path)
        print(f"    {path_str} (evidence: {path.total_evidence})")
    
    # Test build_context
    print("\n--- Test: Build Context ---")
    context = client.build_context(['dopamine', 'prefrontal cortex', 'working memory'])
    print(f"  Concepts: {context.concepts}")
    print(f"  Edges: {len(context.edges)}")
    print(f"  Papers: {len(context.evidence_papers)}")
    print("\n  Context preview:")
    print(context.context_text[:500] + "...")
    
    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_client()
