"""
Graph KB Integration Module

Integrates GraphKBClient with the QA pipeline to provide
knowledge graph context for enhanced reasoning.
"""

import sys
from pathlib import Path
from typing import Optional

# Add graph module to path
GRAPH_DIR = Path(__file__).parent.parent.parent / "knowledge_base" / "graph"
if str(GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_DIR.parent))


def get_graph_kb_client():
    """
    Lazy load GraphKBClient to avoid import errors if graph not built.
    
    Returns:
        GraphKBClient instance or None if not available
    """
    try:
        from neuro_copilot.knowledge_base.graph.graph_kb_client import GraphKBClient
        
        graph_path = GRAPH_DIR / "kg.gpickle"
        if not graph_path.exists():
            print("⚠️  Graph KB not available (kg.gpickle not found)")
            return None
        
        client = GraphKBClient(graph_dir=GRAPH_DIR)
        client.load()
        return client
    except ImportError as e:
        print(f"⚠️  Graph KB import error: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Graph KB load error: {e}")
        return None


def extract_concepts_from_question(
    question: str,
    graph_client
) -> list[str]:
    """
    Extract concepts mentioned in user question.
    
    Args:
        question: User's question text
        graph_client: GraphKBClient instance
        
    Returns:
        List of matched concept names
    """
    if graph_client is None:
        return []
    
    return graph_client.match_concepts(question)


def build_graph_context(
    question: str,
    graph_client,
    max_hops: int = 2,
    max_edges: int = 10,
    max_papers: int = 5
) -> Optional[str]:
    """
    Build knowledge graph context for a question.
    
    Args:
        question: User's question
        graph_client: GraphKBClient instance
        max_hops: Maximum hops for path finding
        max_edges: Maximum edges to include
        max_papers: Maximum papers to cite
        
    Returns:
        Formatted graph context string or None
    """
    if graph_client is None:
        return None
    
    # Find concepts in question
    concepts = graph_client.match_concepts(question)
    
    if not concepts:
        return None
    
    # Build context
    context_result = graph_client.build_context(
        concepts=concepts,
        max_hops=max_hops,
        max_edges=max_edges,
        max_papers=max_papers
    )
    
    if not context_result.edges:
        return None
    
    return context_result.context_text


def enrich_kb_context(
    question: str,
    kb_context: Optional[str],
    graph_client,
    developer_mode: bool = False
) -> str:
    """
    Enrich KB context with knowledge graph information.
    
    Args:
        question: User's question
        kb_context: Original KB context from paper retrieval
        graph_client: GraphKBClient instance
        developer_mode: Print debug info
        
    Returns:
        Enhanced context string combining KB and graph context
    """
    graph_context = build_graph_context(question, graph_client)
    
    if developer_mode:
        if graph_context:
            concepts = graph_client.match_concepts(question) if graph_client else []
            print(f"   📊 Graph context: {len(concepts)} concepts matched")
        else:
            print("   📊 Graph context: None (no matching concepts)")
    
    # Combine contexts
    if graph_context and kb_context:
        return f"{graph_context}\n\n---\n\n{kb_context}"
    elif graph_context:
        return graph_context
    elif kb_context:
        return kb_context
    else:
        return ""


def get_related_concepts(
    concept: str,
    graph_client,
    relation_types: Optional[list[str]] = None
) -> list[dict]:
    """
    Get concepts related to a given concept.
    
    Args:
        concept: Concept name or alias
        graph_client: GraphKBClient instance
        relation_types: Filter by relation types (optional)
        
    Returns:
        List of related concept info dicts
    """
    if graph_client is None:
        return []
    
    subgraph = graph_client.get_neighbors(
        concept=concept,
        hops=1,
        relation_types=relation_types
    )
    
    # Build result with relation info
    results = []
    matched = graph_client.match_concept(concept)
    
    for edge in subgraph.edges:
        if edge['source'] == matched:
            results.append({
                'concept': edge['target'],
                'relation': edge['relation'],
                'direction': 'outgoing',
                'evidence_count': edge.get('evidence_count', 0)
            })
        elif edge['target'] == matched:
            results.append({
                'concept': edge['source'],
                'relation': edge['relation'],
                'direction': 'incoming',
                'evidence_count': edge.get('evidence_count', 0)
            })
    
    return results


def find_concept_paths(
    source: str,
    target: str,
    graph_client,
    max_hops: int = 2
) -> list[dict]:
    """
    Find reasoning paths between two concepts.
    
    Args:
        source: Source concept name
        target: Target concept name
        graph_client: GraphKBClient instance
        max_hops: Maximum path length
        
    Returns:
        List of path info dicts
    """
    if graph_client is None:
        return []
    
    paths = graph_client.find_paths(source, target, max_hops=max_hops)
    
    results = []
    for path in paths:
        results.append({
            'path': path.path,
            'edges': [
                {
                    'source': e['source'],
                    'target': e['target'],
                    'relation': e['relation']
                }
                for e in path.edges
            ],
            'total_evidence': path.total_evidence
        })
    
    return results


# Singleton instance
_graph_client = None


def get_cached_graph_client():
    """Get cached GraphKBClient instance."""
    global _graph_client
    if _graph_client is None:
        _graph_client = get_graph_kb_client()
    return _graph_client


def test_integration():
    """Test graph integration."""
    print("=" * 60)
    print("Testing Graph KB Integration")
    print("=" * 60)
    
    client = get_graph_kb_client()
    if client is None:
        print("❌ Graph KB not available")
        return
    
    print(f"✅ Graph loaded: {client.graph.number_of_nodes()} nodes, {client.graph.number_of_edges()} edges")
    
    # Test concept extraction
    print("\n--- Test: Extract concepts from question ---")
    question = "How does dopamine affect working memory in the prefrontal cortex?"
    concepts = extract_concepts_from_question(question, client)
    print(f"Question: {question}")
    print(f"Concepts: {concepts}")
    
    # Test graph context building
    print("\n--- Test: Build graph context ---")
    context = build_graph_context(question, client)
    if context:
        print(f"Context preview:\n{context[:500]}...")
    else:
        print("No context generated")
    
    # Test context enrichment
    print("\n--- Test: Enrich KB context ---")
    mock_kb = "## Paper 1: Working Memory Review\nContent about working memory..."
    enriched = enrich_kb_context(question, mock_kb, client, developer_mode=True)
    print(f"Enriched context length: {len(enriched)} chars")
    
    # Test related concepts
    print("\n--- Test: Get related concepts ---")
    related = get_related_concepts("working_memory", client)
    print(f"Concepts related to 'working_memory': {len(related)}")
    for r in related[:5]:
        print(f"  {r['concept']} ({r['relation']}, {r['direction']})")
    
    # Test path finding
    print("\n--- Test: Find concept paths ---")
    paths = find_concept_paths("dopamine", "working_memory", client)
    print(f"Paths from 'dopamine' to 'working_memory': {len(paths)}")
    for p in paths[:3]:
        print(f"  {' -> '.join(p['path'])} (evidence: {p['total_evidence']})")
    
    print("\n" + "=" * 60)
    print("Integration Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_integration()
