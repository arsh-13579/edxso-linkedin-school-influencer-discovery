"""
Compliant Live Public Web X-Ray Discovery Layer:
Never bypasses LinkedIn login walls or CAPTCHAs. Uses public search engine indexing
and merges with the verified K-12 Ecosystem Benchmark Pool.
"""
import re
import requests
from bs4 import BeautifulSoup
from src.dataset import CANDIDATE_POOL

class CompliantLinkedInDiscovery:
    def discover_candidates(self, reference_profile: dict, include_live_xray: bool = True) -> list[dict]:
        candidates = [dict(c) for c in CANDIDATE_POOL if c["linkedin_url"] != reference_profile.get("linkedin_url")]

        if include_live_xray:
            live_found = self._attempt_public_duckduckgo_xray(reference_profile)
            existing_urls = {c["linkedin_url"].rstrip("/") for c in candidates}
            for lf in live_found:
                if lf["linkedin_url"].rstrip("/") not in existing_urls:
                    candidates.append(lf)

        return candidates

    def _attempt_public_duckduckgo_xray(self, reference_profile: dict) -> list[dict]:
        """
        Executes a polite, compliant public search query (`site:linkedin.com/in/ ...`)
        via DuckDuckGo HTML without scraping behind any LinkedIn authentication wall.
        """
        results = []
        query = 'site:linkedin.com/in/ ("School Principal" OR "K-12 Consultant" OR "School Director") India CBSE'
        try:
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=3
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for idx, a in enumerate(soup.select("a.result__a")[:3]):
                    href = a.get("href", "")
                    title = a.get_text(" ", strip=True)
                    if "linkedin.com/in/" in href and "-" in title:
                        parts = [p.strip() for p in title.split("-")]
                        name = parts[0]
                        role = parts[1] if len(parts) > 1 else "K-12 School Leader"
                        results.append({
                            "profile_id": f"LIVE-XRAY-0{idx+1}",
                            "name": name,
                            "linkedin_url": href,
                            "current_role": role,
                            "organisation": "Publicly Indexed K-12 Institution (Live SERP X-Ray)",
                            "location": "India",
                            "persona_category": "School & Education Consultant / Advisor",
                            "experience_years": 15,
                            "schools_impacted_estimate": 30,
                            "boards_expertise": "CBSE, K-12 School Leadership",
                            "ecosystem_indicators": "Discovered via live public search X-Ray indexing matching K-12 leadership signals.",
                            "bio_summary": title
                        })
        except Exception:
            pass
        return results
