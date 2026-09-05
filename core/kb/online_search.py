"""
Online Search Module

Extensible architecture for searching multiple academic databases.
Currently supports: PubMed
Future: Semantic Scholar, Google Scholar, etc.
"""

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class SearchResult:
    """Unified search result from any source."""
    pmid: str = ""                    # PubMed ID (or other unique ID)
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    journal: str = ""
    abstract: str = ""
    source: str = ""                  # "pubmed", "semantic_scholar", etc.
    url: str = ""
    relevance_score: float = 0.0      # For ranking
    
    def to_citation(self) -> str:
        """Format as citation string."""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        return f"{author_str} ({self.year}). {self.title}. {self.journal}."
    
    def to_context(self) -> str:
        """Format for LLM context."""
        lines = [
            f"**{self.title}**",
            f"Authors: {', '.join(self.authors[:3])}{'...' if len(self.authors) > 3 else ''}",
            f"Year: {self.year} | Journal: {self.journal}",
            f"Source: {self.source.upper()} | ID: {self.pmid}",
        ]
        if self.abstract:
            # Truncate long abstracts
            abstract = self.abstract[:500] + "..." if len(self.abstract) > 500 else self.abstract
            lines.append(f"Abstract: {abstract}")
        return "\n".join(lines)


class BaseSearcher(ABC):
    """Abstract base class for search engines."""
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source name (e.g., 'pubmed', 'semantic_scholar')."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Search for papers matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        pass


class PubMedSearcher(BaseSearcher):
    """
    PubMed search using NCBI E-utilities API.
    
    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    """
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, email: str = "ns-copilot@example.com", timeout: float = 10.0):
        """
        Initialize PubMed searcher.
        
        Args:
            email: Email for NCBI API (required by their policy)
            timeout: Request timeout in seconds
        """
        self.email = email
        self.timeout = timeout
    
    @property
    def source_name(self) -> str:
        return "pubmed"
    
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Search PubMed for papers.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Step 1: Search for PMIDs
            pmids = self._esearch(query, top_k)
            if not pmids:
                return []
            
            # Step 2: Fetch paper details
            results = self._efetch(pmids)
            return results
            
        except Exception as e:
            print(f"⚠️ PubMed search error: {e}")
            return []
    
    def _esearch(self, query: str, top_k: int) -> list[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(top_k),
            "retmode": "xml",
            "sort": "relevance",
            "email": self.email,
        }
        
        url = f"{self.ESEARCH_URL}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                xml_data = response.read().decode("utf-8")
            
            root = ET.fromstring(xml_data)
            pmids = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]
            return pmids
            
        except Exception as e:
            print(f"⚠️ PubMed esearch error: {e}")
            return []
    
    def _efetch(self, pmids: list[str]) -> list[SearchResult]:
        """Fetch paper details by PMIDs."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email,
        }
        
        url = f"{self.EFETCH_URL}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                xml_data = response.read().decode("utf-8")
            
            return self._parse_efetch_xml(xml_data)
            
        except Exception as e:
            print(f"⚠️ PubMed efetch error: {e}")
            return []
    
    def _parse_efetch_xml(self, xml_data: str) -> list[SearchResult]:
        """Parse PubMed efetch XML response."""
        results = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                result = SearchResult(source=self.source_name)
                
                # PMID
                pmid_elem = article.find(".//PMID")
                if pmid_elem is not None and pmid_elem.text:
                    result.pmid = pmid_elem.text
                    result.url = f"https://pubmed.ncbi.nlm.nih.gov/{result.pmid}/"
                
                # Title
                title_elem = article.find(".//ArticleTitle")
                if title_elem is not None and title_elem.text:
                    result.title = title_elem.text
                
                # Authors
                for author in article.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None and lastname.text:
                        name = lastname.text
                        if forename is not None and forename.text:
                            name = f"{forename.text} {name}"
                        result.authors.append(name)
                
                # Year
                year_elem = article.find(".//PubDate/Year")
                if year_elem is not None and year_elem.text:
                    try:
                        result.year = int(year_elem.text)
                    except ValueError:
                        pass
                
                # Journal
                journal_elem = article.find(".//Journal/Title")
                if journal_elem is not None and journal_elem.text:
                    result.journal = journal_elem.text
                
                # Abstract
                abstract_parts = []
                for abstract_text in article.findall(".//AbstractText"):
                    if abstract_text.text:
                        label = abstract_text.get("Label", "")
                        if label:
                            abstract_parts.append(f"{label}: {abstract_text.text}")
                        else:
                            abstract_parts.append(abstract_text.text)
                result.abstract = " ".join(abstract_parts)
                
                if result.title:  # Only add if we have at least a title
                    results.append(result)
            
        except ET.ParseError as e:
            print(f"⚠️ XML parse error: {e}")
        
        return results


class MultiSourceSearcher:
    """
    Aggregates results from multiple search sources.
    
    Usage:
        searcher = MultiSourceSearcher([
            PubMedSearcher(),
            SemanticScholarSearcher(),  # Future
        ])
        results = searcher.search("dopamine working memory", top_k=5)
    """
    
    def __init__(self, searchers: list[BaseSearcher] = None):
        """
        Initialize with list of searchers.
        
        Args:
            searchers: List of BaseSearcher instances. Defaults to [PubMedSearcher()].
        """
        self.searchers = searchers or [PubMedSearcher()]
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        parallel: bool = True
    ) -> list[SearchResult]:
        """
        Search all sources and merge results.
        
        Args:
            query: Search query
            top_k: Total number of results to return
            parallel: Whether to search sources in parallel
            
        Returns:
            Merged and deduplicated list of SearchResult
        """
        all_results = []
        
        if parallel and len(self.searchers) > 1:
            # Parallel search
            with ThreadPoolExecutor(max_workers=len(self.searchers)) as executor:
                futures = {
                    executor.submit(s.search, query, top_k): s.source_name
                    for s in self.searchers
                }
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        print(f"⚠️ Search error from {source}: {e}")
        else:
            # Sequential search
            for searcher in self.searchers:
                try:
                    results = searcher.search(query, top_k)
                    all_results.extend(results)
                except Exception as e:
                    print(f"⚠️ Search error from {searcher.source_name}: {e}")
        
        # Deduplicate by PMID/title
        seen = set()
        unique_results = []
        for r in all_results:
            key = r.pmid or r.title.lower()
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        # Sort by year (newest first) and return top_k
        unique_results.sort(key=lambda x: x.year, reverse=True)
        return unique_results[:top_k]
    
    def format_results_as_context(self, results: list[SearchResult]) -> str:
        """
        Format search results for LLM context.
        
        Args:
            results: List of SearchResult
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        lines = ["## Online Search Results (Real-time)\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"### [{i}] {result.to_context()}")
            lines.append("")
        
        return "\n".join(lines)


# Convenience function
def search_pubmed(query: str, top_k: int = 3) -> list[SearchResult]:
    """
    Quick search using PubMed.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        List of SearchResult
    """
    searcher = PubMedSearcher()
    return searcher.search(query, top_k)


def test_pubmed_search():
    """Test PubMed search functionality."""
    print("=" * 60)
    print("Testing PubMed Search")
    print("=" * 60)
    
    searcher = PubMedSearcher()
    
    # Test search
    query = "tau protein Alzheimer's disease"
    print(f"\nQuery: {query}")
    print("-" * 40)
    
    results = searcher.search(query, top_k=3)
    
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result.title}")
        print(f"    Authors: {', '.join(result.authors[:3])}")
        print(f"    Year: {result.year} | Journal: {result.journal}")
        print(f"    PMID: {result.pmid}")
        print(f"    Abstract: {result.abstract[:150]}...")
        print()
    
    # Test context formatting
    multi_searcher = MultiSourceSearcher()
    context = multi_searcher.format_results_as_context(results)
    print("=" * 60)
    print("Formatted Context Preview:")
    print("=" * 60)
    print(context[:1000] + "..." if len(context) > 1000 else context)
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_pubmed_search()
