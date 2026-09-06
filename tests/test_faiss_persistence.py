"""Persistence tests for backend/faiss_store.py (Task 6).

These exercise the REAL persistence/decision logic in ``backend.faiss_store``:
manifest building, content-hash change detection, the load-or-build decision,
and safe handling of missing/corrupt state. Metadata round-tripping is checked
end to end.

Run from the PROJECT ROOT:

    python tests/test_faiss_persistence.py

Real BGE + FAISS are heavy and not always available, so by default a lightweight
deterministic embedding + an on-disk FAISS stand-in are injected. The stand-in
implements ``save_local`` / ``load_local`` by pickling the chunk Documents to
the same directory the real FAISS would, so ``source`` / ``page`` / ``row`` /
``content`` are genuinely written to and read back from disk -- the persistence
contract under test. Pass ``--real`` to instead use the project's real
embeddings and real FAISS (only works where torch/faiss/BGE are installed).

The counter on the fake embeddings is what proves "second load does not
recompute embeddings": a reload must not call ``embed_documents`` again.
"""

from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

USE_REAL = "--real" in sys.argv

# --- Inject fakes BEFORE importing the project, unless --real ----------------
# Counts embedding calls so we can prove a reload recomputes nothing.
EMBED_CALLS = {"embed_documents": 0, "embed_query": 0}


def _install_fakes() -> None:
    """Fake langchain_community.embeddings + .vectorstores with on-disk FAISS."""
    import langchain_community  # real package layer is fine; we override submodules

    # --- fake embeddings ------------------------------------------------------
    emb_mod = types.ModuleType("langchain_community.embeddings")

    class HuggingFaceBgeEmbeddings:
        def __init__(self, model_name=None, encode_kwargs=None, **kwargs):
            self.model_name = model_name
            self.encode_kwargs = encode_kwargs or {}
            self.client = object()

        def embed_documents(self, texts):
            EMBED_CALLS["embed_documents"] += 1
            return [[float(len(t)), 0.0] for t in texts]

        def embed_query(self, text):
            EMBED_CALLS["embed_query"] += 1
            return [float(len(text)), 0.0]

    emb_mod.HuggingFaceBgeEmbeddings = HuggingFaceBgeEmbeddings
    sys.modules["langchain_community.embeddings"] = emb_mod
    langchain_community.embeddings = emb_mod

    # --- fake vectorstores.FAISS with real on-disk save/load ------------------
    vs_mod = types.ModuleType("langchain_community.vectorstores")

    class FAISS:
        def __init__(self, documents):
            # keep the chunk Documents exactly (metadata included)
            self.docs = list(documents)
            self.index = types.SimpleNamespace(ntotal=len(self.docs))

        @classmethod
        def from_documents(cls, documents, embedding):
            # calling embed_documents here mirrors a real build recomputing vectors
            embedding.embed_documents([d.page_content for d in documents])
            return cls(documents)

        def save_local(self, folder_path, index_name="index"):
            folder = Path(folder_path)
            folder.mkdir(parents=True, exist_ok=True)
            # write BOTH files the real FAISS writes, so presence checks pass
            (folder / f"{index_name}.faiss").write_bytes(b"FAKEFAISS")
            with open(folder / f"{index_name}.pkl", "wb") as fh:
                pickle.dump(self.docs, fh)

        @classmethod
        def load_local(cls, folder_path, embeddings, index_name="index",
                        allow_dangerous_deserialization=False):
            folder = Path(folder_path)
            with open(folder / f"{index_name}.pkl", "rb") as fh:
                docs = pickle.load(fh)
            # NOTE: deliberately does NOT call embed_documents -> reload is free
            return cls(docs)

        def similarity_search_with_score(self, query, k=5):
            return [(d, 0.0) for d in self.docs[:k]]

    vs_mod.FAISS = FAISS
    sys.modules["langchain_community.vectorstores"] = vs_mod
    langchain_community.vectorstores = vs_mod


if not USE_REAL:
    _install_fakes()

# --- import the REAL modules under test --------------------------------------
import backend.faiss_store as fs  # noqa: E402
from backend.retriever import build_vectorstore, get_retrieved_chunks  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


# --- a small real corpus in a temp dir, with faiss_store pointed at it -------
def _point_store_at(tmp: Path) -> None:
    """Redirect faiss_store + retriever paths into an isolated temp workspace."""
    import backend.retriever as rt

    data_dir = tmp / "data"
    pdf_dir = data_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    rt.DATA_DIR = data_dir
    rt.PDF_DIR = pdf_dir
    fs.DATA_DIR = data_dir
    fs.PDF_DIR = pdf_dir
    fs.INDEX_DIR = data_dir / "faiss_index"
    fs.MANIFEST_PATH = fs.INDEX_DIR / "manifest.json"


def _write(pdf_dir: Path, name: str, text: str) -> None:
    (pdf_dir / name).write_text(text, encoding="utf-8")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="docmind_persist_"))
    try:
        _point_store_at(tmp)
        pdf_dir = fs.PDF_DIR

        _write(pdf_dir, "a.txt", "Retrieval augmented generation notes about chunking and recall.")
        _write(pdf_dir, "b.txt", "A second document covering embeddings and vector search basics.")

        print("=== 1. first run builds and saves the index + manifest ===")
        EMBED_CALLS["embed_documents"] = 0
        vs1, rebuilt1 = fs.get_or_build_vectorstore()
        check("first run reports rebuilt=True", rebuilt1 is True)
        check("index files written to disk",
              (fs.INDEX_DIR / "index.faiss").exists() and (fs.INDEX_DIR / "index.pkl").exists())
        check("manifest written to disk", fs.MANIFEST_PATH.exists())
        check("first build computed embeddings", EMBED_CALLS["embed_documents"] >= 1,
              f"{EMBED_CALLS['embed_documents']} calls")
        saved_manifest = fs._read_manifest()
        check("manifest records the embedding model",
              saved_manifest.get("embedding_model") == fs.EMBEDDING_MODEL,
              str(saved_manifest.get("embedding_model")))
        check("manifest lists both files with hashes",
              set(saved_manifest.get("files", {})) == {"a.txt", "b.txt"})

        print("\n=== 2. fresh process/session loads the saved index (no recompute) ===")
        EMBED_CALLS["embed_documents"] = 0
        vs2, rebuilt2 = fs.get_or_build_vectorstore()
        check("second run reports rebuilt=False (loaded)", rebuilt2 is False)
        check("reload recomputed NO embeddings", EMBED_CALLS["embed_documents"] == 0,
              f"{EMBED_CALLS['embed_documents']} calls")

        print("\n=== 3. persisted metadata survives the round-trip ===")
        chunks = get_retrieved_chunks(vs2, "chunking recall", k=5)
        check("loaded index returns chunks", len(chunks) > 0, f"{len(chunks)} chunks")
        required = {"rank", "score", "source", "page", "row", "content"}
        check("each chunk has the retrieval key set", all(set(c) == required for c in chunks))
        check("source metadata preserved", {c["source"] for c in chunks} <= {"a.txt", "b.txt"})
        check("txt page is 'N/A' and row is None (unchanged metadata)",
              all(c["page"] == "N/A" and c["row"] is None for c in chunks))

        print("\n=== 4. modified document triggers rebuild ===")
        _write(pdf_dir, "a.txt", "MODIFIED: retrieval augmented generation notes, now edited.")
        EMBED_CALLS["embed_documents"] = 0
        _, rebuilt_mod = fs.get_or_build_vectorstore()
        check("modified file -> rebuilt=True", rebuilt_mod is True)
        check("rebuild recomputed embeddings", EMBED_CALLS["embed_documents"] >= 1)

        print("\n=== 5. added document triggers rebuild ===")
        _write(pdf_dir, "c.txt", "A third document added to the corpus after indexing.")
        _, rebuilt_add = fs.get_or_build_vectorstore()
        check("added file -> rebuilt=True", rebuilt_add is True)
        check("manifest now lists three files",
              set(fs._read_manifest().get("files", {})) == {"a.txt", "b.txt", "c.txt"})

        print("\n=== 6. removed document triggers rebuild ===")
        (pdf_dir / "c.txt").unlink()
        _, rebuilt_rm = fs.get_or_build_vectorstore()
        check("removed file -> rebuilt=True", rebuilt_rm is True)
        check("manifest back to two files",
              set(fs._read_manifest().get("files", {})) == {"a.txt", "b.txt"})

        print("\n=== 7. unchanged corpus loads again (still no recompute) ===")
        EMBED_CALLS["embed_documents"] = 0
        _, rebuilt_same = fs.get_or_build_vectorstore()
        check("unchanged -> rebuilt=False", rebuilt_same is False)
        check("no embeddings recomputed", EMBED_CALLS["embed_documents"] == 0)

        print("\n=== 8. embedding-model mismatch triggers rebuild ===")
        current = fs.build_manifest()
        saved = fs._read_manifest()
        check("same model -> manifest matches", fs.manifest_matches(saved, current) is True)
        mismatched = dict(current, embedding_model="some-other-model")
        check("different model -> manifest does NOT match",
              fs.manifest_matches(saved, mismatched) is False)
        # drive it through the real decision path
        _, rebuilt_model = fs.get_or_build_vectorstore(embedding_model="some-other-model")
        check("model change -> rebuilt=True", rebuilt_model is True)

        print("\n=== 9. missing persistence triggers rebuild ===")
        shutil.rmtree(fs.INDEX_DIR, ignore_errors=True)
        check("index dir removed", not fs.INDEX_DIR.exists())
        EMBED_CALLS["embed_documents"] = 0
        _, rebuilt_missing = fs.get_or_build_vectorstore()
        check("missing files -> rebuilt=True", rebuilt_missing is True)
        check("rebuild after missing recomputed embeddings", EMBED_CALLS["embed_documents"] >= 1)

        print("\n=== 10. corrupted persistence triggers rebuild (no crash) ===")
        # manifest still matches, but the pickle is garbage -> load must fail safe
        (fs.INDEX_DIR / "index.pkl").write_bytes(b"not a valid pickle \x00\x01")
        check("load_vectorstore returns None on corrupt pickle",
              fs.load_vectorstore() is None)
        EMBED_CALLS["embed_documents"] = 0
        _, rebuilt_corrupt = fs.get_or_build_vectorstore()
        check("corrupt index -> rebuilt=True (no crash)", rebuilt_corrupt is True)
        check("rebuild after corruption recomputed embeddings", EMBED_CALLS["embed_documents"] >= 1)

        print("\n=== 11. corrupt manifest is handled safely ===")
        fs.MANIFEST_PATH.write_text("{ this is not valid json", encoding="utf-8")
        check("_read_manifest returns None on bad JSON", fs._read_manifest() is None)
        _, rebuilt_badmanifest = fs.get_or_build_vectorstore()
        check("bad manifest -> rebuilt=True", rebuilt_badmanifest is True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n=== summary ===")
    if failures:
        print(f"  {len(failures)} FAILURE(S): " + "; ".join(failures))
        return 1
    print("  all FAISS persistence checks passed")
    if not USE_REAL:
        print("  (ran with fake embeddings + on-disk FAISS stand-in; "
              "run with --real on a machine with BGE+faiss for the real index)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
