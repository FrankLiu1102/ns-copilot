"""
Context Fusion Module

Merges different knowledge sources for LLM context:
1. GraphRAG (local knowledge graph)
2. Online Search (real-time literature)
3. Local KB (SQLite paper database)
"""

from typing import Optional
from dataclasses import dataclass

from neuro_copilot.core.kb.online_search import (
    MultiSourceSearcher,
    PubMedSearcher,
    SearchResult,
)


@dataclass
class FusedContext:
    """Result of context fusion."""
    full_context: str           # Complete merged context
    graph_context: str          # GraphRAG context only
    online_context: str         # Online search context only
    local_kb_context: str       # Local KB context only
    
    # Metadata
    graph_concepts: list[str]   # Concepts matched from question
    online_results: list[SearchResult]  # Online search results
    num_sources: int            # Number of active sources
    
    def __str__(self):
        return self.full_context


class ContextFusion:
    """
    Fuses multiple knowledge sources into unified LLM context.
    
    Sources:
    1. GraphRAG - Semantic relations from local knowledge graph
    2. Online Search - Real-time literature from PubMed/etc.
    3. Local KB - Pre-indexed papers from SQLite database
    
    Usage:
        fusion = ContextFusion(
            enable_graph=True,
            enable_online=True,
            online_top_k=3
        )
        
        result = fusion.fuse(
            question="What is tau's role in Alzheimer's?",
            graph_context="## Knowledge Graph Context...",
            local_kb_context="## Relevant Papers..."
        )
    """
    
    def __init__(
        self,
        enable_graph: bool = True,
        enable_online: bool = True,
        enable_local_kb: bool = True,
        online_top_k: int = 3,
        online_searchers: list = None,
    ):
        """
        Initialize context fusion.
        
        Args:
            enable_graph: Include GraphRAG context
            enable_online: Include online search results
            enable_local_kb: Include local KB context
            online_top_k: Number of online results to fetch
            online_searchers: Custom searchers (default: PubMed)
        """
        self.enable_graph = enable_graph
        self.enable_online = enable_online
        self.enable_local_kb = enable_local_kb
        self.online_top_k = online_top_k
        
        # Initialize online searcher
        if online_searchers:
            self.online_searcher = MultiSourceSearcher(online_searchers)
        else:
            self.online_searcher = MultiSourceSearcher([PubMedSearcher()])
    
    def build_search_query(self, question: str, concepts: list[str] = None) -> str:
        """
        Build optimized search query from question and concepts.
        
        Args:
            question: User's question
            concepts: Matched concepts from GraphRAG
            
        Returns:
            Optimized search query
        """
        # If we have matched concepts, use them for a more focused search
        if concepts:
            # Use top concepts (usually most relevant)
            concept_terms = " ".join(concepts[:3])
            return concept_terms
        
        # Otherwise, extract key terms from question
        # Simple approach: remove common words
        stop_words = {
            "what", "is", "the", "how", "does", "do", "are", "in", "of", "and",
            "to", "a", "an", "for", "on", "with", "between", "role", "relationship",
            "affect", "effect", "cause", "why", "which", "where", "when"
        }
        
        words = question.lower().split()
        key_words = [w.strip("?.,!") for w in words if w.strip("?.,!") not in stop_words]
        
        # Take top 5 key words
        return " ".join(key_words[:5])
    
    def fuse(
        self,
        question: str,
        graph_context: Optional[str] = None,
        local_kb_context: Optional[str] = None,
        graph_concepts: list[str] = None,
    ) -> FusedContext:
        """
        Fuse all knowledge sources into unified context.
        
        Args:
            question: User's question
            graph_context: Pre-built GraphRAG context
            local_kb_context: Pre-built local KB context
            graph_concepts: Concepts matched from GraphRAG
            
        Returns:
            FusedContext with merged content
        """
        parts = []
        num_sources = 0
        online_results = []
        online_context = ""
        
        # 1. GraphRAG Context (semantic relations)
        if self.enable_graph and graph_context:
            parts.append(graph_context)
            num_sources += 1
        
        # 2. Online Search (real-time literature)
        if self.enable_online:
            search_query = self.build_search_query(question, graph_concepts)
            print(f"   🔍 Online search query: '{search_query}'")
            
            online_results = self.online_searcher.search(search_query, self.online_top_k)
            
            if online_results:
                online_context = self.online_searcher.format_results_as_context(online_results)
                parts.append(online_context)
                num_sources += 1
                print(f"   📚 Online search: found {len(online_results)} papers")
            else:
                print(f"   📚 Online search: no results found")
        
        # 3. Local KB Context (pre-indexed papers)
        if self.enable_local_kb and local_kb_context:
            parts.append(local_kb_context)
            num_sources += 1
        
        # Merge all parts
        if parts:
            full_context = "\n\n---\n\n".join(parts)
        else:
            full_context = ""
        
        return FusedContext(
            full_context=full_context,
            graph_context=graph_context or "",
            online_context=online_context,
            local_kb_context=local_kb_context or "",
            graph_concepts=graph_concepts or [],
            online_results=online_results,
            num_sources=num_sources,
        )


# Singleton instance for easy access
_fusion_instance = None


def get_context_fusion(
    enable_graph: bool = True,
    enable_online: bool = True,
    online_top_k: int = 3,
) -> ContextFusion:
    """Get or create context fusion instance."""
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = ContextFusion(
            enable_graph=enable_graph,
            enable_online=enable_online,
            online_top_k=online_top_k,
        )
    return _fusion_instance


def test_fusion():
    """Test context fusion."""
    print("=" * 60)
    print("Testing Context Fusion")
    print("=" * 60)
    
    fusion = ContextFusion(
        enable_graph=True,
        enable_online=True,
        online_top_k=3
    )
    
    # Mock GraphRAG context
    mock_graph_context = """## Knowledge Graph Context

### Relevant Concepts
- **Tau** (Molecule): Protein associated with neurodegeneration
- **Alzheimer** (Disease): Neurodegenerative disorder

### Semantic Relations
- Tau **ASSOCIATED_WITH** Alzheimer (evidence: 20 papers)
- Amyloid Beta **ASSOCIATED_WITH** Alzheimer (evidence: 16 papers)
"""
    
    question = "What is the relationship between tau protein and Alzheimer's disease?"
    concepts = ["tau", "alzheimer"]
    
    print(f"\nQuestion: {question}")
    print(f"Concepts: {concepts}")
    print("-" * 60)
    
    result = fusion.fuse(
        question=question,
        graph_context=mock_graph_context,
        local_kb_context=None,
        graph_concepts=concepts,
    )
    
    print(f"\n{'=' * 60}")
    print("Fused Context:")
    print("=" * 60)
    print(result.full_context[:2000] + "..." if len(result.full_context) > 2000 else result.full_context)
    
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  - Sources used: {result.num_sources}")
    print(f"  - Graph concepts: {result.graph_concepts}")
    print(f"  - Online results: {len(result.online_results)}")
    print("=" * 60)


if __name__ == "__main__":
    test_fusion()
