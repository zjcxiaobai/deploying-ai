from __future__ import annotations

import requests
from typing import List


HEADERS = {
    "User-Agent": "assignment-chat/1.0 (educational project; contact: none)"
}


def wiki_web_search(query: str, k: int = 3) -> str:
    """
    Simple web search via Wikipedia.
    Returns readable summaries.
    """
    q = (query or "").strip()
    if not q:
        return "Usage: /web <query>"

    k = max(1, min(int(k), 5))

    # 1) Search using MediaWiki API (more reliable than rest.php search)
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": k,
        "format": "json",
    }

    try:
        r = requests.get(api_url, params=params, headers=HEADERS, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"Wikipedia search failed. Error: {e}"

    hits = (data.get("query") or {}).get("search") or []
    if not hits:
        return f"No Wikipedia results for: {q}"

    # 2) Fetch summaries
    lines: List[str] = [f"Top Wikipedia matches for: **{q}**"]
    for i, h in enumerate(hits[:k], 1):
        title = h.get("title") or "Untitled"
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + requests.utils.quote(title)
        )

        try:
            s = requests.get(summary_url, headers=HEADERS, timeout=12)
            s.raise_for_status()
            sdata = s.json()
            extract = (sdata.get("extract") or "").strip()
        except Exception:
            extract = ""

        if not extract:
            extract = "(No summary available.)"
        # simplify
        snippet = extract[:300] + ("…" if len(extract) > 300 else "")
        lines.append(f"{i}. **{title}** — {snippet}")

    return "\n".join(lines)