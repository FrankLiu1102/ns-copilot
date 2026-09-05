"""
Knowledge Base Client - SQLite-based retrieval

Responsibilities:
- Query kb.sqlite for relevant papers
- Return structured KB entries with metadata
"""

import sqlite3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from neuro_copilot.core.config import KB_PATH, KB_TOP_K


@dataclass
class KBEntry:
    """Knowledge Base Entry"""
    uid: str
    title: str
    authors: str
    year: int
    paper_type: str
    rag_content: str
    citation_short: str
    citation_full: str
    abstract: str = ""


class KBClient:
    """Knowledge Base Client for paper retrieval"""
    
    def __init__(self, db_path: str = KB_PATH):
        self.db_path = db_path
    
    def _connect(self) -> sqlite3.Connection:
        """Connect to SQLite database"""
        return sqlite3.connect(self.db_path)
    
    def search(
        self,
        keywords: List[str],
        task_type: str = "QA",
        domain: str = "unknown",
        top_k: int = KB_TOP_K,
        prefer_paper_type: Optional[str] = None
    ) -> List[KBEntry]:
        """
        Search knowledge base for relevant papers
        
        Args:
            keywords: List of search keywords
            task_type: QA / DA / ER (affects retrieval strategy)
            domain: working_memory / parkinson / alzheimer / unknown
            top_k: Number of results to return
            prefer_paper_type: "review" or "experiment" (optional filter)
            
        Returns:
            List of KBEntry objects
        """
        if not keywords:
            keywords = ["working memory"]  # Default fallback
        
        # Use papers_v2 table (the cleaned version) if exists, else papers
        table_name = "papers_v2"
        
        conn = self._connect()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{table_name}'
        """)
        if not cursor.fetchone():
            # Fallback to original table
            table_name = "papers"
            cursor.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{table_name}'
            """)
            if not cursor.fetchone():
                print(f"⚠️  KB: No papers table found in {self.db_path}")
                conn.close()
                return []
        
        # Determine paper type preference based on task
        if prefer_paper_type is None:
            if task_type == "QA":
                prefer_paper_type = "review"
            elif task_type == "ER":
                prefer_paper_type = None  # Mix both
            elif task_type == "DA":
                # For DA, we don't retrieve KB in this phase
                conn.close()
                return []
        
        # Build query strategy:
        # When domain is specified, prioritize domain matching, use keywords for ranking
        # When domain is unknown, rely on keyword matching
        
        if domain and domain != "unknown":
            # Domain-first strategy: filter by domain, then rank by keywords
            where_clauses = ["domain = ?"]
            params = [domain]
            
            # Add paper type filter if specified
            if prefer_paper_type:
                where_clauses.append("paper_type = ?")
                params.append(prefer_paper_type)
            
            where_sql = " AND ".join(where_clauses)
            
            # Try keyword-based ranking (optional)
            # For now, just order by year and citation count
            query = f"""
                SELECT uid, title, citation_short, year, paper_type, 
                       rag_content, citation_full, abstract
                FROM {table_name}
                WHERE {where_sql}
                ORDER BY year DESC, citation_count DESC
                LIMIT ?
            """
            params.append(top_k)
        else:
            # Keyword-first strategy: match keywords across fields
            where_clauses = []
            params = []
            
            for keyword in keywords[:3]:  # Use top 3 keywords
                # Search in multiple fields
                where_clauses.append(
                    "(title LIKE ? OR abstract LIKE ? OR rag_content LIKE ?)"
                )
                kw_pattern = f"%{keyword}%"
                params.extend([kw_pattern, kw_pattern, kw_pattern])
            
            where_sql = " OR ".join(where_clauses) if where_clauses else "1=1"
            
            # Add paper type filter if specified
            if prefer_paper_type:
                where_sql = f"({where_sql}) AND paper_type = ?"
                params.append(prefer_paper_type)
            
            query = f"""
                SELECT uid, title, citation_short, year, paper_type, 
                       rag_content, citation_full, abstract
                FROM {table_name}
                WHERE {where_sql}
                ORDER BY year DESC, citation_count DESC
                LIMIT ?
            """
            params.append(top_k)
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            entries = []
            for row in rows:
                entry = KBEntry(
                    uid=row[0] or "",
                    title=row[1] or "",
                    authors=row[2] or "",  # citation_short contains authors
                    year=row[3] or 0,
                    paper_type=row[4] or "",
                    rag_content=row[5] or "",
                    citation_short=row[2] or "",  # Same as authors for now
                    citation_full=row[6] or "",
                    abstract=row[7] or ""
                )
                entries.append(entry)
            
            conn.close()
            return entries
            
        except sqlite3.Error as e:
            print(f"⚠️  KB: SQLite error: {e}")
            conn.close()
            return []
    
    def format_kb_context(self, entries: List[KBEntry]) -> str:
        """
        Format KB entries as context string for LLM
        
        Args:
            entries: List of KBEntry objects
            
        Returns:
            Formatted context string
        """
        if not entries:
            return ""
        
        context_parts = [
            "# Relevant Research Papers\n",
            "The following papers from the knowledge base may be relevant:\n"
        ]
        
        for i, entry in enumerate(entries, 1):
            context_parts.append(f"\n## Paper {i}: {entry.title}")
            context_parts.append(f"Authors: {entry.authors}")
            context_parts.append(f"Year: {entry.year}")
            context_parts.append(f"Type: {entry.paper_type}")
            context_parts.append(f"\nContent:\n{entry.rag_content[:500]}...")
            context_parts.append(f"\nCitation: {entry.citation_full}\n")
        
        return "\n".join(context_parts)


def test_kb_client():
    """Test KB client"""
    print("="*80)
    print("Testing KB Client")
    print("="*80)
    
    client = KBClient()
    
    # Test QA search
    print("\n📚 Test 1: QA search for 'working memory' + 'prefrontal'")
    results = client.search(
        keywords=["working memory", "prefrontal"],
        task_type="QA",
        top_k=3
    )
    
    print(f"   Found {len(results)} papers")
    for entry in results:
        print(f"   • {entry.title[:60]}... ({entry.year}, {entry.paper_type})")
    
    # Test ER search
    print("\n📚 Test 2: ER search for 'experiment' + 'design'")
    results = client.search(
        keywords=["experiment", "design", "working memory"],
        task_type="ER",
        top_k=3
    )
    
    print(f"   Found {len(results)} papers")
    for entry in results:
        print(f"   • {entry.title[:60]}... ({entry.year}, {entry.paper_type})")
    
    # Test context formatting
    if results:
        print("\n📄 Test 3: Format context")
        context = client.format_kb_context(results[:2])
        print(f"   Context length: {len(context)} chars")
        print(f"   Preview: {context[:200]}...")


if __name__ == "__main__":
    test_kb_client()

