from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

import pandas as pd
from openai import OpenAI


class SemanticStore:
    """
    Semantic search.
    with openAI.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "docs",
        data_csv: Optional[Path] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.data_csv = data_csv

        # openAI client to use embeddings
        # check the gateway first
        key = os.getenv("API_GATEWAY_KEY")
        if not key:
            raise RuntimeError("API_GATEWAY_KEY is missing")
        self.oai = OpenAI(
        base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
        api_key="any value", 
        default_headers={"x-api-key": key})


    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Embeddings
        resp = self.oai.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [d.embedding for d in resp.data]


    def ingest_from_csv(self) -> str:
        if self.collection.count() > 0:
            return "Semantic store already has documents."
        if self.data_csv is None:
            return "No CSV provided for ingestion."

        df = pd.read_csv(self.data_csv)

        ids = df["id"].astype(str).tolist()
        titles = df["title"].astype(str).tolist()
        texts = df["text"].astype(str).tolist()

        embs = self._embed(texts)
        self.collection.add(
            ids=ids,
            embeddings=embs,
            documents=texts,
            metadatas=[{"title": t} for t in titles],
        )
        return f"Ingested {len(ids)} documents into ChromaDB."

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        q_emb = self._embed([query])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(max(int(k), 1), 10),
            include=["documents", "metadatas", "distances"],
        )

        out: List[Dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(
                {
                    "id": _id,
                    "title": (meta or {}).get("title", ""),
                    "text": doc,
                    "distance": dist,
                }
            )
        return out