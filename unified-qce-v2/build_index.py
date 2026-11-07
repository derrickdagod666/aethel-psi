#!/usr/bin/env python3
"""Build the FAISS index for the Architects of Reality QCE (v2).

Corpus (in order of preference):
  1. Putt's own papers, extracted from PDF with pypdf:
       - workspace/your_files/new-theory-third-revised/new-theory-third-revised.pdf
       - workspace/user/files/uaf_paper_improved.pdf
  2. If PDF extraction yields nothing usable, the built-in fallback corpus
     (corpus_fallback.py) — summaries of the same papers, labelled as such.

Output (written next to this script):
  - syntropic.index           (FAISS IndexFlatL2 over all-mpnet-base-v2 embeddings)
  - syntropic_metadata.json   (one {"source", "text"} entry per chunk, same order)

Chunking: ~800 characters with ~150 characters of overlap, split on
paragraph/sentence boundaries where possible. Nothing is invented: every
chunk is verbatim text from the PDFs (or a labelled fallback summary).

Usage:
    python build_index.py [--force]
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.expanduser("~/workspace")

PDF_SOURCES = [
    os.path.join(WORKSPACE, "your_files/new-theory-third-revised/new-theory-third-revised.pdf"),
    os.path.join(WORKSPACE, "user/files/uaf_paper_improved.pdf"),
]

INDEX_PATH = os.path.join(HERE, "syntropic.index")
META_PATH = os.path.join(HERE, "syntropic_metadata.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_MODEL = "all-mpnet-base-v2"


def extract_pdf_text(path):
    """Return (page_texts, n_pages). Raises on failure."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page must not kill the build
            print(f"  page {i + 1}: extraction failed ({exc})")
            text = ""
        pages.append(text)
    return pages, len(reader.pages)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into ~chunk_size chunks with overlap, on sentence boundaries."""
    if len(text) <= chunk_size:
        return [{"source": source, "text": text}] if text else []
    chunks = []
    start = 0
    # Prefer splitting at sentence ends inside the window.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 > chunk_size and buf:
            chunks.append({"source": source, "text": buf.strip()})
            # Overlap: carry the tail of the finished chunk forward.
            buf = buf[-overlap:] if len(buf) > overlap else buf
            buf = (buf + " " + sent).strip()
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf:
        chunks.append({"source": source, "text": buf.strip()})
    return [c for c in chunks if len(c["text"]) >= 40]


def build_corpus():
    """Return (chunks, provenance dict)."""
    chunks = []
    provenance = {"pdfs": [], "fallback_used": False}
    for pdf in PDF_SOURCES:
        label = os.path.basename(pdf)
        if not os.path.exists(pdf):
            print(f"  MISSING: {pdf}")
            provenance["pdfs"].append({"file": label, "status": "missing"})
            continue
        try:
            pages, n_pages = extract_pdf_text(pdf)
        except Exception as exc:  # noqa: BLE001 - fall back, do not crash
            print(f"  FAILED to read {label}: {exc}")
            provenance["pdfs"].append({"file": label, "status": f"read-error: {exc}"})
            continue
        full = clean_text("\n".join(pages))
        pdf_chunks = chunk_text(full, source=label)
        print(f"  {label}: {n_pages} pages, {len(full)} chars extracted, "
              f"{len(pdf_chunks)} chunks")
        provenance["pdfs"].append(
            {"file": label, "pages": n_pages, "chars": len(full),
             "chunks": len(pdf_chunks)})
        chunks.extend(pdf_chunks)
    if not chunks:
        from corpus_fallback import FALLBACK_CHUNKS

        print("  PDF extraction yielded no usable chunks — using labelled "
              "fallback corpus (summaries, not paper text).")
        chunks = list(FALLBACK_CHUNKS)
        provenance["fallback_used"] = True
    return chunks, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the index already exists")
    args = parser.parse_args()

    if os.path.exists(INDEX_PATH) and not args.force:
        print(f"Index already exists at {INDEX_PATH} — use --force to rebuild.")
        return 0

    print("Building Syntropic Primer index from Putt's papers...")
    chunks, provenance = build_corpus()
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model {EMBED_MODEL} (downloads once, ~420 MB)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True,
                              convert_to_numpy=True).astype("float32")

    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)
    print(f"Wrote {INDEX_PATH} (dim={dim}, ntotal={index.ntotal})")
    print(f"Wrote {META_PATH} ({len(chunks)} entries)")
    print(f"Provenance: {json.dumps(provenance)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
