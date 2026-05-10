"""quick-rag: Lightweight RAG over any document corpus."""

import os, json, hashlib
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import lancedb, httpx

load_dotenv()

@dataclass
class Source:
    text: str; file: str; page: int; score: float

@dataclass
class Response:
    answer: str; sources: list[Source]


class RAG:
    SUPPORTED = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt", ".csv", ".xlsx"}
    DEFAULTS = {"ollama": "llama3.1:8b", "anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o"}
    SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context. If the context doesn't contain the answer, say so. "
        "Do NOT guess. Cite sources using [filename p.N] tags. Be concise."
    )

    def __init__(self, docs_path: str, provider: str = "ollama", model: str | None = None,
                 embed_model: str = "BAAI/bge-small-en-v1.5", rerank: bool = True,
                 persist: bool = True, db_path: str | None = None):
        self.docs_path = Path(docs_path).resolve()
        self.provider = provider
        self.model = model or self.DEFAULTS.get(provider, "llama3.1:8b")
        self.embed_model_name = embed_model
        self.persist = persist
        self.db_path = Path(db_path).resolve() if db_path else self.docs_path / ".quick_rag"
        self._ingested = False

        if not self.docs_path.exists():
            raise FileNotFoundError(f"Documents path not found: {self.docs_path}")
        if provider not in self.DEFAULTS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(self.DEFAULTS)}")

        self.embedder = SentenceTransformer(self.embed_model_name)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3") if rerank else None
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path / "lancedb"))

    def _find_docs(self) -> list[Path]:
        return [f for ext in self.SUPPORTED for f in self.docs_path.rglob(f"*{ext}") if ".quick_rag" not in str(f)]

    def _file_hash(self, path: Path) -> str:
        s = path.stat()
        return hashlib.md5(f"{path}:{s.st_size}:{s.st_mtime}".encode()).hexdigest()

    def _read_meta(self) -> dict:
        p = self.db_path / "ingested.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _write_meta(self, meta: dict):
        (self.db_path / "ingested.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def _chunk_meta(chunk) -> tuple[int, str]:
        meta = getattr(chunk, "meta", None)
        if not meta:
            return 0, ""
        page = 0
        if hasattr(meta, "doc_items") and meta.doc_items:
            prov = getattr(meta.doc_items[0], "prov", None)
            if prov and len(prov) > 0:
                page = getattr(prov[0], "page_no", 0)
        return page, (getattr(meta, "headings", None) or [""])[0]

    def ingest(self):
        files = self._find_docs()
        if not files:
            self._ingested = True
            return print(f"No supported documents found in {self.docs_path}")

        meta = self._read_meta() if self.persist else {}
        new = [f for f in files if str(f) not in meta or meta[str(f)] != self._file_hash(f)]
        if not new:
            self._ingested = True
            return print(f"All {len(files)} document(s) already ingested.")

        print(f"Ingesting {len(new)} new document(s)...")
        self._ingest_files(new)
        for f in new:
            meta[str(f)] = self._file_hash(f)
        self._write_meta(meta)
        self._ingested = True

    def _ingest_files(self, files: list[Path]):
        converter = DocumentConverter()
        chunker = HybridChunker(tokenizer=self.embed_model_name, max_tokens=512)
        all_chunks = []

        for filepath in files:
            print(f"  Parsing: {filepath.name}")
            try:
                result = converter.convert(str(filepath))
                for chunk in chunker.chunk(result.document):
                    page, heading = self._chunk_meta(chunk)
                    text = f"{heading}\n{chunk.text}" if heading else chunk.text
                    all_chunks.append({"text": text, "file": filepath.name, "page": page, "heading": heading})
            except Exception as e:
                print(f"  Warning: Failed to parse {filepath.name}: {e}")

        if not all_chunks:
            return print("  No chunks extracted.")

        print(f"  Embedding {len(all_chunks)} chunks...")
        embeddings = self.embedder.encode([c["text"] for c in all_chunks], show_progress_bar=True)
        for i, c in enumerate(all_chunks):
            c["vector"] = embeddings[i].tolist()

        try: self.db.open_table("chunks").add(all_chunks)
        except Exception: self.db.create_table("chunks", all_chunks)

        try: self.db.open_table("chunks").create_fts_index("text", replace=True)
        except Exception: print("  Warning: FTS index failed — hybrid search disabled.")
        print(f"  Done — {len(all_chunks)} chunks stored.")

    def add(self, path: str):
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        files = [p] if p.is_file() and p.suffix.lower() in self.SUPPORTED else (
            [f for ext in self.SUPPORTED for f in p.rglob(f"*{ext}")] if p.is_dir() else [])
        if not files:
            return print(f"No supported documents in {p}")
        self._ingest_files(files)
        meta = self._read_meta()
        for f in files:
            meta[str(f)] = self._file_hash(f)
        self._write_meta(meta)

    def query(self, question: str, top_k: int = 5) -> Response:
        if not self._ingested:
            self.ingest()
        try: table = self.db.open_table("chunks")
        except Exception: return Response(answer="No documents ingested yet.", sources=[])

        q_vec = self.embedder.encode(question).tolist()
        fetch_k = 20 if self.reranker else top_k

        try: results = table.search(q_vec, query_type="hybrid").text(question).limit(fetch_k).to_list()
        except Exception: results = table.search(q_vec).limit(fetch_k).to_list()

        if not results:
            return Response(answer="No relevant information found.", sources=[])

        if self.reranker and len(results) > top_k:
            scores = self.reranker.predict([(question, r["text"]) for r in results])
            for i, r in enumerate(results):
                r["_score"] = float(scores[i])
            results.sort(key=lambda r: r["_score"], reverse=True)
            results = results[:top_k]

        sources = [Source(r["text"], r["file"], r["page"], r.get("_score", r.get("_distance", 0.0))) for r in results]
        context = "\n\n---\n\n".join(f"[{r['file']} p.{r['page']}]\n{r['text']}" for r in results)
        return Response(answer=self._generate(question, context), sources=sources)

    def _generate(self, question: str, context: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        msgs = [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            url, headers = f"{host}/api/chat", {}
            body = {"model": self.model, "stream": False, "messages": msgs}
            extract = lambda r: r["message"]["content"]
        elif self.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key: raise ValueError("Set ANTHROPIC_API_KEY in your .env file.")
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": 1024, "system": self.SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}]}
            extract = lambda r: r["content"][0]["text"]
        else:
            key = os.getenv("OPENAI_API_KEY")
            if not key: raise ValueError("Set OPENAI_API_KEY in your .env file.")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}"}
            body = {"model": self.model, "messages": msgs}
            extract = lambda r: r["choices"][0]["message"]["content"]

        try:
            r = httpx.post(url, json=body, headers=headers, timeout=120.0)
            r.raise_for_status()
            return extract(r.json())
        except httpx.ConnectError:
            if self.provider == "ollama":
                raise ConnectionError(f"Can't connect to Ollama. Install: https://ollama.com then run: ollama pull {self.model}")
            raise ConnectionError(f"Can't reach {url}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404 and self.provider == "ollama":
                raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
            raise


def main():
    import argparse
    p = argparse.ArgumentParser(prog="quick-rag", description="Chat with your documents.")
    p.add_argument("command", choices=["chat", "ingest"])
    p.add_argument("docs")
    p.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-persist", action="store_true")
    a = p.parse_args()

    bot = RAG(a.docs, provider=a.provider, model=a.model, rerank=not a.no_rerank, persist=not a.no_persist)
    if a.command == "ingest":
        bot.ingest()
        return print("Done.")

    print("\nquick-rag — type 'quit' to exit.\n")
    while True:
        try: q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError): break
        if not q or q.lower() in ("quit", "exit", "q"): break
        r = bot.query(q)
        print(f"\n{r.answer}\n")
        for s in r.sources:
            print(f"  • {s.file} (p.{s.page})")
        print()

if __name__ == "__main__":
    main()