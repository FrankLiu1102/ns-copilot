#!/usr/bin/env python3
"""
Phase 1: Concept Node Extraction

Extract concept nodes from predefined vocabulary and paper JSONL files.
Outputs: nodes/concepts.parquet
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import hashlib

import pandas as pd


@dataclass
class ConceptNode:
    """Concept node for knowledge graph."""
    id: str                     # "concept:prefrontal_cortex"
    name: str                   # "prefrontal_cortex"
    display_name: str           # "Prefrontal Cortex"
    type: str                   # "BrainRegion"
    description: str            # "Anterior part of the frontal lobe..."
    aliases: str                # JSON string of aliases list
    frequency: int              # Number of papers mentioning this concept
    source: str                 # "vocabulary" or "discovered"


def load_concept_vocab(vocab_path: Path) -> dict:
    """Load concept vocabulary from JSON file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('concepts', {})


def load_papers(papers_dir: Path) -> list[dict]:
    """Load all papers from JSONL files."""
    papers = []
    jsonl_files = [
        papers_dir / 'working_memory' / 'wm_papers_v2.jsonl',
        papers_dir / 'parkinson' / 'parkinson_papers.jsonl',
        papers_dir / 'alzheimer' / 'alzheimer_papers.jsonl',
    ]
    
    for jsonl_path in jsonl_files:
        if jsonl_path.exists():
            print(f"Loading {jsonl_path.name}...")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        papers.append(json.loads(line))
    
    print(f"Loaded {len(papers)} papers total")
    return papers


def normalize_concept_name(name: str) -> str:
    """Normalize concept name to canonical form."""
    # Convert to lowercase, replace spaces with underscores
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', '_', normalized)
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    return normalized


def create_display_name(name: str) -> str:
    """Create human-readable display name."""
    # Replace underscores with spaces and title case
    display = name.replace('_', ' ')
    # Handle special cases
    special_cases = {
        'fmri': 'fMRI',
        'eeg': 'EEG',
        'pet': 'PET',
        'dbs': 'DBS',
        'gaba': 'GABA',
        'wm': 'Working Memory',
        'pfc': 'PFC',
        'hpc': 'Hippocampus',
    }
    if name.lower() in special_cases:
        return special_cases[name.lower()]
    return display.title()


def build_alias_index(vocab: dict) -> dict[str, str]:
    """Build index mapping aliases to canonical concept names."""
    alias_index = {}
    for concept_name, concept_data in vocab.items():
        aliases = concept_data.get('aliases', [])
        for alias in aliases:
            normalized_alias = normalize_concept_name(alias)
            alias_index[normalized_alias] = concept_name
            # Also add original form for exact matching
            alias_index[alias.lower()] = concept_name
    return alias_index


def match_concept(text: str, alias_index: dict[str, str]) -> Optional[str]:
    """Match text to a concept using alias index."""
    # Try normalized form
    normalized = normalize_concept_name(text)
    if normalized in alias_index:
        return alias_index[normalized]
    
    # Try lowercase original
    if text.lower() in alias_index:
        return alias_index[text.lower()]
    
    return None


def extract_concepts_from_papers(
    papers: list[dict],
    vocab: dict,
    alias_index: dict[str, str]
) -> tuple[dict[str, ConceptNode], dict[str, int]]:
    """
    Extract concepts from papers and count frequencies.
    
    Returns:
        - Dictionary of concept nodes
        - Dictionary of concept frequencies
    """
    concept_nodes = {}
    concept_frequencies = {}
    discovered_concepts = {}
    
    # Initialize concepts from vocabulary
    for concept_name, concept_data in vocab.items():
        node = ConceptNode(
            id=f"concept:{concept_name}",
            name=concept_name,
            display_name=create_display_name(concept_name),
            type=concept_data.get('type', 'Other'),
            description=concept_data.get('description', ''),
            aliases=json.dumps(concept_data.get('aliases', [])),
            frequency=0,
            source='vocabulary'
        )
        concept_nodes[concept_name] = node
        concept_frequencies[concept_name] = 0
    
    # Process each paper
    for paper in papers:
        paper_concepts = set()
        
        # Extract from 'concepts' field
        for concept_text in paper.get('concepts', []):
            matched = match_concept(concept_text, alias_index)
            if matched:
                paper_concepts.add(matched)
            else:
                # Track discovered concepts not in vocabulary
                normalized = normalize_concept_name(concept_text)
                if normalized and len(normalized) > 2:
                    discovered_concepts[normalized] = discovered_concepts.get(normalized, 0) + 1
        
        # Extract from 'tags' field
        for tag in paper.get('tags', []):
            matched = match_concept(tag, alias_index)
            if matched:
                paper_concepts.add(matched)
            else:
                normalized = normalize_concept_name(tag)
                if normalized and len(normalized) > 2:
                    discovered_concepts[normalized] = discovered_concepts.get(normalized, 0) + 1
        
        # Extract from 'domain' field
        for domain in paper.get('domain', []):
            matched = match_concept(domain, alias_index)
            if matched:
                paper_concepts.add(matched)
        
        # Update frequencies
        for concept_name in paper_concepts:
            if concept_name in concept_frequencies:
                concept_frequencies[concept_name] += 1
    
    # Update node frequencies
    for concept_name, freq in concept_frequencies.items():
        if concept_name in concept_nodes:
            concept_nodes[concept_name].frequency = freq
    
    # Report discovered concepts not in vocabulary
    print("\n=== Discovered concepts not in vocabulary ===")
    for name, count in sorted(discovered_concepts.items(), key=lambda x: -x[1]):
        if count >= 2:  # Only show concepts appearing in 2+ papers
            print(f"  {name}: {count} papers")
    
    return concept_nodes, concept_frequencies


def save_concepts_parquet(concept_nodes: dict[str, ConceptNode], output_path: Path):
    """Save concept nodes to Parquet file."""
    records = [asdict(node) for node in concept_nodes.values()]
    df = pd.DataFrame(records)
    
    # Sort by frequency (descending) then name
    df = df.sort_values(['frequency', 'name'], ascending=[False, True])
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} concepts to {output_path}")
    
    return df


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent
    vocab_path = base_dir / 'ontology' / 'concept_vocab.json'
    papers_dir = base_dir.parent / 'papers'
    output_path = base_dir / 'nodes' / 'concepts.parquet'
    
    print("=" * 60)
    print("Phase 1: Concept Node Extraction")
    print("=" * 60)
    
    # Load vocabulary
    print(f"\nLoading vocabulary from {vocab_path}...")
    vocab = load_concept_vocab(vocab_path)
    print(f"Loaded {len(vocab)} concepts from vocabulary")
    
    # Build alias index
    alias_index = build_alias_index(vocab)
    print(f"Built alias index with {len(alias_index)} entries")
    
    # Load papers
    print(f"\nLoading papers from {papers_dir}...")
    papers = load_papers(papers_dir)
    
    # Extract concepts
    print("\nExtracting concepts from papers...")
    concept_nodes, frequencies = extract_concepts_from_papers(papers, vocab, alias_index)
    
    # Print statistics
    print("\n=== Concept Statistics ===")
    active_concepts = [c for c, f in frequencies.items() if f > 0]
    print(f"Total concepts in vocabulary: {len(vocab)}")
    print(f"Concepts found in papers: {len(active_concepts)}")
    
    # Print top concepts by frequency
    print("\n=== Top 20 Concepts by Frequency ===")
    sorted_concepts = sorted(frequencies.items(), key=lambda x: -x[1])
    for name, freq in sorted_concepts[:20]:
        if freq > 0:
            node = concept_nodes[name]
            print(f"  {name} ({node.type}): {freq} papers")
    
    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = save_concepts_parquet(concept_nodes, output_path)
    
    # Print sample
    print("\n=== Sample Output ===")
    print(df[['id', 'name', 'type', 'frequency']].head(10).to_string())
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
