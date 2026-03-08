############################################################
# MODULE 1: CORE LIBRARIES
############################################################
# Removed transformers and HuggingFacePipeline — switching to Gemini via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage
import re
import unicodedata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os  # For API key
import hashlib
from langchain_community.vectorstores import FAISS


############################################################
# MODULE 2: PDF INGESTION
############################################################
Loader = DirectoryLoader(
    r"/content",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
Docs = Loader.load()

############################################################
# MODULE 2.5: DOCUMENT HASHING & VERSION CONTROL
############################################################
def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

############################################################
# MODULE 3: DOCUMENT CLEANING & NORMALIZATION
############################################################
def clean_docs(docs):
    cleaned = []
    seen_hashes = set()

    for doc in docs:
        text = unicodedata.normalize("NFKD", doc.page_content)
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()

        doc_hash = compute_sha256(text)
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)

        raw_meta = doc.metadata or {}
        source_path = raw_meta.get("source", "unknown")
        filename = os.path.basename(source_path)

        metadata = {
            "source": filename,
            "page": raw_meta.get("page", "?"),
            "doc_type": "ISO",
            "version": "2023",
            "sha256": doc_hash
        }

        cleaned.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return cleaned

Docs = clean_docs(Docs)

############################################################
# MODULE 4: TEXT CHUNKING
############################################################

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=120,          # slightly more overlap helps context continuity
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunked_docs = splitter.split_documents(Docs)

############################################################
# MODULE 5: DENSE EMBEDDINGS + FAISS
############################################################
INDEX_PATH = "faiss_index"

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if os.path.exists(INDEX_PATH):
    print("🔹 Loading existing FAISS index...")
    vector_store = FAISS.load_local(
        INDEX_PATH,
        embedder,
        allow_dangerous_deserialization=True
    )
else:
    print("🔹 Creating new FAISS index...")
    vector_store = FAISS.from_documents(
        chunked_docs,
        embedding=embedder
    )
    vector_store.save_local(INDEX_PATH)

############################################################
# MODULE 6: SPARSE RETRIEVAL (BM25)
############################################################
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([d.page_content.split() for d in chunked_docs])

############################################################
# MODULE 7: HYBRID RETRIEVAL (DYNAMIC + NORMALIZED)
############################################################
def hybrid_retrieval(query):
    k_dense, k_sparse = select_k(query)

    # ───── Dense Retrieval ─────
    dense_results = vector_store.similarity_search_with_score(
        query, k=k_dense
    )

    dense_scores = [score for _, score in dense_results]
    dense_norm = min_max_normalize(dense_scores)

    dense_map = {}
    for (doc, _), norm_score in zip(dense_results, dense_norm):
        dense_map[doc.page_content] = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "dense_score": norm_score,
            "sparse_score": 0.0
        }

    # ───── Sparse Retrieval ─────
    query_tokens = query.lower().split()
    sparse_scores = bm25.get_scores(query_tokens)

    top_sparse_idx = sorted(
        range(len(sparse_scores)),
        key=lambda i: sparse_scores[i],
        reverse=True
    )[:k_sparse]

    sparse_top_scores = [sparse_scores[i] for i in top_sparse_idx]
    sparse_norm = min_max_normalize(sparse_top_scores)

    for idx, norm_score in zip(top_sparse_idx, sparse_norm):
        doc = chunked_docs[idx]
        content = doc.page_content

        if content in dense_map:
            dense_map[content]["sparse_score"] = norm_score
        else:
            dense_map[content] = {
                "content": content,
                "metadata": doc.metadata,
                "dense_score": 0.0,
                "sparse_score": norm_score
            }

    # ───── Weighted Merge ─────
    results = []
    for item in dense_map.values():
        final_score = (
            0.6 * item["dense_score"] +
            0.4 * item["sparse_score"]
        )
        item["final_score"] = final_score
        results.append(item)

    # Sort by final score
    results.sort(key=lambda x: x["final_score"], reverse=True)

    return results

############################################################
# MODULE 7.1: QUERY ANALYSIS
############################################################
def analyze_query(query: str):
    tokens = query.strip().split()
    length = len(tokens)

    definition_keywords = {"what", "define", "definition", "meaning"}
    explanation_keywords = {"explain", "describe", "how", "why", "process"}

    lowered = query.lower()

    if any(k in lowered for k in definition_keywords):
        query_type = "definition"
    elif any(k in lowered for k in explanation_keywords):
        query_type = "explanation"
    else:
        query_type = "broad"

    return length, query_type

def select_k(query):
    length, qtype = analyze_query(query)

    if qtype == "definition":
        return 6, 4      # dense, sparse
    elif qtype == "explanation":
        return 10, 8
    else:  # broad / open-ended
        return 14, 12

############################################################
# MODULE 7.2: SCORE NORMALIZATION
############################################################
def min_max_normalize(scores):
    if not scores:
        return []

    min_s, max_s = min(scores), max(scores)
    if min_s == max_s:
        return [1.0 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]

############################################################
# MODULE 7.3: REDUNDANCY REMOVAL
############################################################
import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def deduplicate_chunks(docs, embedder, threshold=0.92):
    unique = []
    vectors = []

    for d in docs:
        vec = embedder.embed_query(d["content"])

        if all(cosine_sim(vec, v) < threshold for v in vectors):
            unique.append(d)
            vectors.append(vec)

    return unique


############################################################
# MODULE 8: CROSS-ENCODER RE-RANKING
############################################################
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cross_encoder_rerank(query, docs, top_k=4):
    if not docs:
        return []

    try:
        pairs = [(query, d["content"]) for d in docs]
        scores = cross_encoder.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [
            {
                "content": r[0]["content"],
                "metadata": r[0]["metadata"],
                "rerank_score": float(r[1])
            }
            for r in ranked[:top_k]
        ]

    except Exception as e:
        print("⚠️ Cross-encoder failed, falling back to dense scores:", e)

        # 🔁 Fallback: use already-ranked hybrid results
        return [
            {
                "content": d["content"],
                "metadata": d["metadata"],
                "rerank_score": d.get("final_score", 0.0)
            }
            for d in docs[:top_k]
        ]

############################################################
# MODULE 9: CONTEXT COMPRESSION
############################################################
def compress_chunk(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])

############################################################
# MODULE 10: CONTEXT BUILDER (improved readability for LLM)
############################################################
def build_context(docs, max_chunks=5):
    if not docs:
        return "[No relevant information found in documents]"

    parts = []
    for i, d in enumerate(docs[:max_chunks], 1):
        meta = d["metadata"]
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")

        compressed = compress_chunk(d["content"])

        header = f"Chunk {i} | Source: {source} (page {page})"
        parts.append(f"──── {header} ────\n{compressed}\n")

    return "\n".join(parts)

############################################################
# MODULE 11: STRONGER RAG PROMPT — forces explanation & interactivity
############################################################
def get_rag_prompt():
    template = """You are an excellent university-level instructor and domain expert in quality management systems (ISO 9001, etc.).

Core rules — you MUST follow ALL of them:
1. NEVER copy-paste sentences or long phrases from the context. Rewrite everything completely in your own clear, natural words.
2. Explain every important point, input, requirement or list item like you're teaching a smart colleague who is learning this for the first time.
3. When the context shows headings + bullets/numbered lists/tables → explain EACH item individually with 1–3 sentences.
4. Use simple, made-up examples when it helps clarify a concept.
5. Make your answer feel interactive: use phrases like
   • "Let's break this down step by step…"
   • "The key thing to understand here is…"
   • "Why does this matter? Because…"
   • "Think of it this way…"
   • "A practical example would be…"
6. Use markdown formatting: headings, bullets, numbered lists, short paragraphs, **bold** for emphasis.
7. End with a short "Key Takeaways" section (2–4 bullets).
8. If the context does not contain enough information → say clearly: "Based on the provided documents, I don't have sufficient detail to fully answer this.
9. Your answer MUST be at least 300 words unless the context is clearly insufficient."

Context (only use facts from here — do NOT add external knowledge):
{context}

Question: {question}

Answer (start teaching directly — no preamble like "Sure, here is…"):
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

############################################################
# MODULE 11A: SYSTEM INSTRUCTIONS (LLM BEHAVIOR)
############################################################
def get_system_message():
    return """You are an excellent university-level instructor and domain expert in quality management systems (ISO 9001, etc.).

You MUST follow ALL rules below:
1. NEVER copy-paste sentences or long phrases from the context. Rewrite everything fully in your own words.
2. Explain every important point as if teaching a smart colleague who is new to this topic.
3. If the context contains headings, bullets, lists or tables → explain EACH item individually.
4. Use simple, made-up examples where helpful.
5. Keep the tone interactive and instructional.
6. Use markdown formatting: headings, bullets, numbered lists, **bold** emphasis.
7. End with a **Key Takeaways** section (2–4 bullets).
8. If context is insufficient, explicitly say so.
9. Minimum answer length: ~300 words unless context is clearly insufficient.
"""

############################################################
# MODULE 11B: HUMAN PROMPT (TASK + CONTEXT)
############################################################
def get_human_prompt():
    template = """Context (use ONLY this information):
{context}

Question:
{question}

Your answer MUST have the following structure:

### Explanation
(Conceptual explanation in detail)

### Practical Examples
(1–2 simple made-up examples)

### Key Takeaways
- Bullet 1
- Bullet 2
- Bullet 3

Teach directly.
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


############################################################
# MODULE 12 & 13: LLM — switch to Gemini API via LangChain
############################################################
# Set your Google API key (get it from https://aistudio.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCsIUiObLaXwkGUKRVRV9iE9gzwg8WFjF8"  # Uncomment and set this

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",          # ← Updated & currently working
    temperature=0.55,
    top_p=0.9,
    max_output_tokens=10000,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

############################################################
# MODULE 14: END-TO-END PIPELINE
############################################################
def answer_query(query: str) -> dict:
    retrieved = hybrid_retrieval(query)

    # 🧹 Remove near-duplicates
    retrieved = deduplicate_chunks(retrieved, embedder)

    # 🔒 Pre-filter: only top 15 go to cross-encoder
    top_for_rerank = retrieved[:15]

    reranked_chunks = cross_encoder_rerank(query, top_for_rerank)
    context = build_context(reranked_chunks)

    system_msg = get_system_message()
    human_prompt = get_human_prompt().format(
        context=context,
        question=query
    )

    answer = llm.invoke([
        {"role": "system", "content": system_msg},
        HumanMessage(content=human_prompt)
    ])


    return {
        "question": query,
        "answer": answer.content if hasattr(answer, 'content') else answer,
        "sources": [c["metadata"] for c in reranked_chunks if "metadata" in c]
    }
