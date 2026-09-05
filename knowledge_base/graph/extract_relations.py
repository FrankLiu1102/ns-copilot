#!/usr/bin/env python3
"""
Phase 2: Relation Extraction

Extract semantic relations between concepts from paper abstracts using LLM.
Outputs: edges/semantic_edges.parquet, evidence/paper_evidence.parquet
"""

import json
import re
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
import hashlib

import pandas as pd
from openai import OpenAI

# Configuration
# Set via web UI (Configuration tab) or OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
RATE_LIMIT_DELAY = 0.5  # seconds between API calls


@dataclass
class SemanticEdge:
    """Semantic edge between two concepts."""
    id: str                      # "edge:{source}_{relation}_{target}"
    source: str                  # concept name
    target: str                  # concept name
    relation: str                # relation type
    description: str             # relation description
    weight: float                # based on evidence count
    evidence_ids: str            # JSON string of paper IDs
    evidence_count: int          # number of supporting papers


@dataclass 
class PaperEvidence:
    """Paper as evidence source."""
    id: str                      # "paper:{pmid}"
    pmid: str
    title: str
    year: int
    citation_short: str
    abstract: str
    paper_type: str
    domain: str


RELATION_EXTRACTION_PROMPT = """You are a neuroscience expert. Given a paper abstract and a list of concepts found in it, identify the semantic relationships between these concepts.

CONCEPTS FOUND IN THIS PAPER:
{concepts}

ABSTRACT:
{abstract}

VALID RELATION TYPES:
- MODULATES: Source modulates/regulates target (e.g., dopamine MODULATES prefrontal_cortex)
- LOCATED_IN: Source is located in target (e.g., dopamine LOCATED_IN basal_ganglia)
- SUPPORTS: Source supports/enables target function (e.g., prefrontal_cortex SUPPORTS working_memory)
- IMPAIRS: Source impairs/damages target (e.g., parkinson IMPAIRS motor_function)
- MEASURED_BY: Source is measured using target method (e.g., working_memory MEASURED_BY fmri)
- CORRELATES_WITH: Source correlates with target (e.g., neural_oscillations CORRELATES_WITH working_memory)
- PROJECTS_TO: Source brain region projects to target region (e.g., prefrontal_cortex PROJECTS_TO basal_ganglia)
- CAUSES: Source causes target (e.g., neurodegeneration CAUSES cognitive_decline)
- TREATS: Source treatment addresses target (e.g., deep_brain_stimulation TREATS parkinson)
- ASSOCIATED_WITH: General association (e.g., tau ASSOCIATED_WITH alzheimer)

INSTRUCTIONS:
1. Only extract relationships that are EXPLICITLY supported by the abstract
2. Do not infer relationships not mentioned in the text
3. Use the exact concept names provided (lowercase with underscores)
4. Each relationship should have a brief description (1 sentence)
5. Assign confidence 0.0-1.0 based on how explicit the evidence is

OUTPUT FORMAT (JSON array):
[
  {{
    "source": "concept_name",
    "target": "concept_name", 
    "relation": "RELATION_TYPE",
    "description": "Brief description of the relationship",
    "confidence": 0.8
  }}
]

If no clear relationships are found, return an empty array: []

OUTPUT:"""


def load_concept_vocab(vocab_path: Path) -> dict:
    """Load concept vocabulary."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('concepts', {})


def build_alias_index(vocab: dict) -> dict[str, str]:
    """Build alias to canonical name mapping."""
    alias_index = {}
    for concept_name, concept_data in vocab.items():
        aliases = concept_data.get('aliases', [])
        for alias in aliases:
            normalized = alias.lower().strip().replace(' ', '_')
            normalized = re.sub(r'[^a-z0-9_]', '', normalized)
            alias_index[normalized] = concept_name
            alias_index[alias.lower()] = concept_name
    return alias_index


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
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        papers.append(json.loads(line))
    
    return papers


def normalize_concept(text: str) -> str:
    """Normalize concept name."""
    normalized = text.lower().strip()
    normalized = re.sub(r'\s+', '_', normalized)
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    return normalized


def find_concepts_in_paper(paper: dict, alias_index: dict[str, str]) -> set[str]:
    """Find all concepts mentioned in a paper."""
    found = set()
    
    # Check concepts field
    for c in paper.get('concepts', []):
        normalized = normalize_concept(c)
        if normalized in alias_index:
            found.add(alias_index[normalized])
        elif c.lower() in alias_index:
            found.add(alias_index[c.lower()])
    
    # Check tags field
    for t in paper.get('tags', []):
        normalized = normalize_concept(t)
        if normalized in alias_index:
            found.add(alias_index[normalized])
        elif t.lower() in alias_index:
            found.add(alias_index[t.lower()])
    
    # Check domain field
    for d in paper.get('domain', []):
        normalized = normalize_concept(d)
        if normalized in alias_index:
            found.add(alias_index[normalized])
    
    return found


def extract_relations_from_abstract(
    client: OpenAI,
    abstract: str,
    concepts: list[str],
    max_retries: int = 3
) -> list[dict]:
    """Use LLM to extract relations from abstract."""
    
    prompt = RELATION_EXTRACTION_PROMPT.format(
        concepts=", ".join(concepts),
        abstract=abstract
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a neuroscience expert extracting relationships from scientific papers. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON
            try:
                # Handle both array and object responses
                result = json.loads(content)
                if isinstance(result, dict):
                    # If wrapped in object, extract relations array
                    result = result.get('relations', result.get('relationships', []))
                if isinstance(result, list):
                    return result
                return []
            except json.JSONDecodeError:
                # Try to extract JSON array from response
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return []
                
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return []


def create_edge_id(source: str, relation: str, target: str) -> str:
    """Create unique edge ID."""
    return f"edge:{source}_{relation.lower()}_{target}"


def process_papers(
    papers: list[dict],
    alias_index: dict[str, str],
    vocab: dict,
    output_dir: Path,
    resume_from: int = 0
) -> tuple[dict[str, SemanticEdge], list[PaperEvidence]]:
    """Process all papers and extract relations."""
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Edge storage: edge_id -> SemanticEdge
    edges: dict[str, SemanticEdge] = {}
    # Evidence storage
    paper_evidence: list[PaperEvidence] = []
    # Track which papers support each edge
    edge_papers: dict[str, list[str]] = {}
    edge_descriptions: dict[str, list[str]] = {}
    
    # Progress tracking
    total = len(papers)
    processed = 0
    skipped = 0
    relations_found = 0
    
    # Cache file for resuming
    cache_file = output_dir / 'extraction_cache.json'
    if cache_file.exists() and resume_from > 0:
        print(f"Loading cache from {cache_file}...")
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            edge_papers = cache.get('edge_papers', {})
            edge_descriptions = cache.get('edge_descriptions', {})
    
    print(f"\nProcessing {total} papers...")
    print("=" * 60)
    
    for i, paper in enumerate(papers):
        if i < resume_from:
            continue
            
        pmid = paper.get('external_ids', {}).get('pmid', '')
        if not pmid:
            pmid = paper.get('uid', '').replace('pubmed:', '')
        
        paper_id = f"paper:{pmid}"
        title = paper.get('paper', {}).get('title', paper.get('title', 'Unknown'))[:60]
        abstract = paper.get('abstract', '')
        
        print(f"\n[{i+1}/{total}] {title}...")
        
        # Find concepts in paper
        concepts = find_concepts_in_paper(paper, alias_index)
        
        if len(concepts) < 2:
            print(f"    Skipped: only {len(concepts)} concept(s) found")
            skipped += 1
            continue
        
        print(f"    Concepts: {', '.join(sorted(concepts))}")
        
        # Create paper evidence record
        evidence = PaperEvidence(
            id=paper_id,
            pmid=pmid,
            title=paper.get('paper', {}).get('title', paper.get('title', '')),
            year=paper.get('paper', {}).get('year', paper.get('year', 0)),
            citation_short=paper.get('citation', {}).get('short', ''),
            abstract=abstract[:500],  # Truncate for storage
            paper_type=paper.get('paper', {}).get('paper_type', ''),
            domain=','.join(paper.get('domain', []))
        )
        paper_evidence.append(evidence)
        
        # Extract relations using LLM
        if abstract:
            relations = extract_relations_from_abstract(
                client, abstract, list(concepts)
            )
            
            print(f"    Found {len(relations)} relation(s)")
            
            for rel in relations:
                source = rel.get('source', '').lower().replace(' ', '_')
                target = rel.get('target', '').lower().replace(' ', '_')
                relation_type = rel.get('relation', '').upper()
                description = rel.get('description', '')
                confidence = rel.get('confidence', 0.5)
                
                # Validate concepts exist
                if source not in vocab or target not in vocab:
                    continue
                
                # Skip low confidence
                if confidence < 0.3:
                    continue
                
                edge_id = create_edge_id(source, relation_type, target)
                
                # Track papers and descriptions for this edge
                if edge_id not in edge_papers:
                    edge_papers[edge_id] = []
                    edge_descriptions[edge_id] = []
                
                if paper_id not in edge_papers[edge_id]:
                    edge_papers[edge_id].append(paper_id)
                    edge_descriptions[edge_id].append(description)
                    relations_found += 1
                    print(f"      + {source} --{relation_type}--> {target}")
        
        processed += 1
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        # Save cache periodically
        if (i + 1) % 10 == 0:
            cache = {
                'edge_papers': edge_papers,
                'edge_descriptions': edge_descriptions,
                'last_processed': i
            }
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            print(f"    [Cache saved at {i+1}]")
    
    # Build final edges
    print("\n" + "=" * 60)
    print("Building final edges...")
    
    for edge_id, paper_ids in edge_papers.items():
        parts = edge_id.replace('edge:', '').split('_')
        # Handle relation types with underscores
        if len(parts) >= 3:
            source = parts[0]
            target = parts[-1]
            relation = '_'.join(parts[1:-1]).upper()
        else:
            continue
        
        descriptions = edge_descriptions.get(edge_id, [])
        best_description = descriptions[0] if descriptions else ''
        
        edge = SemanticEdge(
            id=edge_id,
            source=source,
            target=target,
            relation=relation,
            description=best_description,
            weight=len(paper_ids),  # Weight by evidence count
            evidence_ids=json.dumps(paper_ids),
            evidence_count=len(paper_ids)
        )
        edges[edge_id] = edge
    
    print(f"\nProcessed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Relations found: {relations_found}")
    print(f"Unique edges: {len(edges)}")
    
    return edges, paper_evidence


def save_results(
    edges: dict[str, SemanticEdge],
    paper_evidence: list[PaperEvidence],
    output_dir: Path
):
    """Save extraction results."""
    
    # Save edges
    edges_path = output_dir / 'edges' / 'semantic_edges.parquet'
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    
    edge_records = [asdict(e) for e in edges.values()]
    df_edges = pd.DataFrame(edge_records)
    df_edges = df_edges.sort_values(['evidence_count', 'source'], ascending=[False, True])
    df_edges.to_parquet(edges_path, index=False)
    print(f"\nSaved {len(df_edges)} edges to {edges_path}")
    
    # Save evidence
    evidence_path = output_dir / 'evidence' / 'paper_evidence.parquet'
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    
    evidence_records = [asdict(e) for e in paper_evidence]
    df_evidence = pd.DataFrame(evidence_records)
    df_evidence.to_parquet(evidence_path, index=False)
    print(f"Saved {len(df_evidence)} paper evidence records to {evidence_path}")
    
    # Save summary
    summary = {
        'total_edges': len(edges),
        'total_papers_with_relations': len(paper_evidence),
        'edges_by_relation': {},
        'top_edges': []
    }
    
    for edge in edges.values():
        rel = edge.relation
        summary['edges_by_relation'][rel] = summary['edges_by_relation'].get(rel, 0) + 1
    
    # Top edges by evidence count
    sorted_edges = sorted(edges.values(), key=lambda x: x.evidence_count, reverse=True)
    for edge in sorted_edges[:20]:
        summary['top_edges'].append({
            'source': edge.source,
            'relation': edge.relation,
            'target': edge.target,
            'evidence_count': edge.evidence_count,
            'description': edge.description
        })
    
    summary_path = output_dir / 'edges' / 'edge_statistics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved edge statistics to {summary_path}")
    
    return df_edges, df_evidence


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent
    vocab_path = base_dir / 'ontology' / 'concept_vocab.json'
    papers_dir = base_dir.parent / 'papers'
    
    print("=" * 60)
    print("Phase 2: Relation Extraction")
    print("=" * 60)
    
    # Load vocabulary
    print(f"\nLoading vocabulary...")
    vocab = load_concept_vocab(vocab_path)
    alias_index = build_alias_index(vocab)
    print(f"Loaded {len(vocab)} concepts")
    
    # Load papers
    print(f"\nLoading papers...")
    papers = load_papers(papers_dir)
    print(f"Loaded {len(papers)} papers")
    
    # Process papers
    edges, evidence = process_papers(
        papers, alias_index, vocab, base_dir, resume_from=0
    )
    
    # Save results
    df_edges, df_evidence = save_results(edges, evidence, base_dir)
    
    # Print sample
    print("\n" + "=" * 60)
    print("Sample Edges (Top 10 by evidence count):")
    print("=" * 60)
    print(df_edges[['source', 'relation', 'target', 'evidence_count']].head(10).to_string())
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
