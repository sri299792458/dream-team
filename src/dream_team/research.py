"""
Research module for Dream Team framework.

Integrates with Semantic Scholar API and uses LLM to analyze papers.
"""

import requests
from typing import List, Dict, Optional
import time
from dataclasses import dataclass


@dataclass
class PaperResult:
    """Semantic Scholar paper result"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    citation_count: int
    influential_citation_count: int
    url: str

    def to_paper(self) -> 'Paper':
        """Convert to Paper object"""
        from .agent import Paper
        return Paper(
            title=self.title,
            authors=self.authors,
            year=self.year,
            abstract=self.abstract,
            semantic_scholar_id=self.paper_id,
            citation_count=self.citation_count,
            relevance_score=0.0  # To be computed
        )


class SemanticScholarAPI:
    """Wrapper for Semantic Scholar API"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # Optional - higher rate limit with key
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})

        # Rate limiting: 1 request per second (per Semantic Scholar docs)
        # With key: Higher limits, but still respect 1 req/sec minimum
        self.rate_limit_delay = 1.0  # Seconds between requests

    def search(
        self,
        query: str,
        limit: int = 10,
        year_range: Optional[tuple] = None,
        fields: List[str] = None
    ) -> List[PaperResult]:
        """Search for papers"""

        if fields is None:
            fields = ["paperId", "title", "authors", "year", "abstract",
                     "citationCount", "influentialCitationCount", "url"]

        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields)
        }

        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        # Retry with exponential backoff for rate limiting
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: 5s, 10s
                    backoff_delay = 5 * (2 ** (attempt - 1))
                    print(f"   Retrying after {backoff_delay}s (attempt {attempt + 1}/{max_retries + 1})...")
                    time.sleep(backoff_delay)
                else:
                    time.sleep(self.rate_limit_delay)

                response = self.session.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()

                data = response.json()
                results = []

                for paper_data in data.get("data", []):
                    if not paper_data.get("abstract"):
                        continue  # Skip papers without abstracts

                    results.append(PaperResult(
                        paper_id=paper_data["paperId"],
                        title=paper_data["title"],
                        authors=[a.get("name", "Unknown") for a in paper_data.get("authors", [])],
                        year=paper_data.get("year", 0),
                        abstract=paper_data.get("abstract", ""),
                        citation_count=paper_data.get("citationCount", 0),
                        influential_citation_count=paper_data.get("influentialCitationCount", 0),
                        url=paper_data.get("url", "")
                    ))

                return results

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries:
                    # Rate limited - retry with backoff
                    continue
                else:
                    print(f"⚠️  Semantic Scholar API error: {e}")
                    return []
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Semantic Scholar API error: {e}")
                return []

        return []  # All retries exhausted

    def get_paper(self, paper_id: str) -> Optional[PaperResult]:
        """Get specific paper by ID"""
        fields = ["paperId", "title", "authors", "year", "abstract",
                 "citationCount", "influentialCitationCount", "url"]

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={"fields": ",".join(fields)},
                timeout=10
            )
            response.raise_for_status()

            paper_data = response.json()
            return PaperResult(
                paper_id=paper_data["paperId"],
                title=paper_data["title"],
                authors=[a.get("name", "Unknown") for a in paper_data.get("authors", [])],
                year=paper_data.get("year", 0),
                abstract=paper_data.get("abstract", ""),
                citation_count=paper_data.get("citationCount", 0),
                influential_citation_count=paper_data.get("influentialCitationCount", 0),
                url=paper_data.get("url", "")
            )
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not fetch paper {paper_id}: {e}")
            return None

    def get_citations(self, paper_id: str, limit: int = 10) -> List[PaperResult]:
        """
        Get papers that cite this paper (forward citation search).

        Useful for finding recent work building on seminal papers.
        """
        fields = ["paperId", "title", "authors", "year", "abstract",
                 "citationCount", "influentialCitationCount", "url"]

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params={
                    "fields": ",".join(fields),
                    "limit": limit
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("data", []):
                paper_data = item.get("citingPaper", {})
                if not paper_data.get("abstract"):
                    continue

                results.append(PaperResult(
                    paper_id=paper_data["paperId"],
                    title=paper_data["title"],
                    authors=[a.get("name", "Unknown") for a in paper_data.get("authors", [])],
                    year=paper_data.get("year", 0),
                    abstract=paper_data.get("abstract", ""),
                    citation_count=paper_data.get("citationCount", 0),
                    influential_citation_count=paper_data.get("influentialCitationCount", 0),
                    url=paper_data.get("url", "")
                ))

            return results

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not fetch citations for {paper_id}: {e}")
            return []

    def get_references(self, paper_id: str, limit: int = 10) -> List[PaperResult]:
        """
        Get papers that this paper cites (backward citation search).

        Useful for finding seminal works from review papers.
        """
        fields = ["paperId", "title", "authors", "year", "abstract",
                 "citationCount", "influentialCitationCount", "url"]

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params={
                    "fields": ",".join(fields),
                    "limit": limit
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("data", []):
                paper_data = item.get("citedPaper", {})
                if not paper_data.get("abstract"):
                    continue

                results.append(PaperResult(
                    paper_id=paper_data["paperId"],
                    title=paper_data["title"],
                    authors=[a.get("name", "Unknown") for a in paper_data.get("authors", [])],
                    year=paper_data.get("year", 0),
                    abstract=paper_data.get("abstract", ""),
                    citation_count=paper_data.get("citationCount", 0),
                    influential_citation_count=paper_data.get("influentialCitationCount", 0),
                    url=paper_data.get("url", "")
                ))

            return results

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not fetch references for {paper_id}: {e}")
            return []


class ResearchAssistant:
    """High-level research operations using Semantic Scholar + LLM"""

    def __init__(self, ss_api: SemanticScholarAPI, llm: 'GeminiLLM'):
        self.ss_api = ss_api
        self.llm = llm

    def research_topic(
        self,
        query: str,
        context: str = "",
        num_papers: int = 5,
        year_range: tuple = (2020, 2025)
    ) -> List['Paper']:
        """
        Research a topic: search papers + extract key insights
        """
        from .agent import Paper

        print(f"🔍 Researching: '{query}'")

        # Search Semantic Scholar
        results = self.ss_api.search(
            query=query,
            limit=num_papers * 2,  # Get more, filter down
            year_range=year_range
        )

        if not results:
            print("   No papers found")
            return []

        print(f"   Found {len(results)} papers")

        # Use LLM to analyze relevance and extract insights
        papers = []
        for i, result in enumerate(results[:num_papers]):
            print(f"   Analyzing paper {i+1}/{min(num_papers, len(results))}: {result.title[:60]}...")

            analysis_prompt = f"""You are a research assistant analyzing a scientific paper for relevance and insights.

Context: {context}

Paper Title: {result.title}
Authors: {', '.join(result.authors)}
Year: {result.year}
Abstract: {result.abstract}

Tasks:
1. Rate relevance to the context (0.0-1.0)
2. Extract 3-5 key findings that are actionable
3. List specific techniques/methods mentioned
4. Provide a brief summary of applicability

Respond in JSON format:
{{
    "relevance_score": 0.0-1.0,
    "key_findings": ["finding 1", "finding 2", ...],
    "techniques": ["technique 1", "technique 2", ...],
    "applicability": "brief summary"
}}
"""

            try:
                analysis = self.llm.generate_json(analysis_prompt, temperature=0.3)

                paper = Paper(
                    title=result.title,
                    authors=result.authors,
                    year=result.year,
                    abstract=result.abstract,
                    key_findings=analysis.get("key_findings", []),
                    techniques=analysis.get("techniques", []),
                    relevance_score=analysis.get("relevance_score", 0.0),
                    semantic_scholar_id=result.paper_id,
                    citation_count=result.citation_count
                )

                papers.append(paper)

            except Exception as e:
                print(f"   ⚠️  Error analyzing paper: {e}")
                # Fallback: create paper without LLM analysis
                paper = Paper(
                    title=result.title,
                    authors=result.authors,
                    year=result.year,
                    abstract=result.abstract,
                    semantic_scholar_id=result.paper_id,
                    citation_count=result.citation_count,
                    relevance_score=0.5
                )
                papers.append(paper)

        # Sort by relevance
        papers.sort(key=lambda p: p.relevance_score, reverse=True)

        print(f"   ✅ Analyzed {len(papers)} papers (avg relevance: {sum(p.relevance_score for p in papers)/len(papers):.2f})")

        return papers


# Global instance
_research_assistant = None


def get_research_assistant() -> ResearchAssistant:
    """Get or create global research assistant"""
    import os
    global _research_assistant
    if _research_assistant is None:
        from .llm import get_llm
        # Use API key from environment if available
        api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        ss_api = SemanticScholarAPI(api_key=api_key)
        llm = get_llm()
        _research_assistant = ResearchAssistant(ss_api, llm)
    return _research_assistant
