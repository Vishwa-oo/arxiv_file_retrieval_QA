import os
import re
import uuid
import requests
import fitz
import faiss
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# SETTINGS
# ============================================================

ARXIV_API = "https://export.arxiv.org/api/query"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# SHARED STATE
# ============================================================

class ResearchState(TypedDict):
    query: str
    query_type: str

    papers: list
    selected_paper: dict | None

    paper_text: str | None
    chunks: list
    vector_store_id: str | None

    summary: str | None
    final_answer: str | None

    mode: str
    qa_question: str | None
    qa_answer: str | None
    qa_history: list


# FAISS indexes are kept here while the program is running.
# The state stores only the ID of the index.
vector_stores = {}

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# LLM
# ============================================================

def call_llm(prompt, temperature=0.2):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()


# ============================================================
# QUERY UNDERSTANDING
# ============================================================

def understand_query(query):

    query = query.strip()

    if query.startswith("http") and "arxiv.org" in query:

        return {
            "query_type": "paper_url",
            "query": query
        }

    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", query):

        return {
            "query_type": "paper_id",
            "query": query
        }

    return {
        "query_type": "topic",
        "query": query
    }


# ============================================================
# ARXIV HELPERS
# ============================================================

def parse_arxiv_entries(entries):

    papers = []

    for entry in entries:

        title = entry.find("title")
        summary = entry.find("summary")
        published = entry.find("published")
        arxiv_id = entry.find("id")

        if not title or not summary or not published or not arxiv_id:
            continue

        authors = [
            author.find("name").text.strip()
            for author in entry.find_all("author")
            if author.find("name")
        ]

        categories = [
            category.get("term")
            for category in entry.find_all("category")
            if category.get("term")
        ]

        pdf_url = None

        for link in entry.find_all("link"):

            if link.get("type") == "application/pdf":

                pdf_url = link.get("href")

        papers.append({

            "title":
                title.text.strip().replace("\n", " "),

            "authors":
                authors,

            "published":
                published.text.strip(),

            "arxiv_id":
                arxiv_id.text.strip().split("/")[-1],

            "pdf_url":
                pdf_url,

            "abstract":
                summary.text.strip().replace("\n", " "),

            "categories":
                categories
        })

    return papers


def search_arxiv(topic):

    params = {

        "search_query":
            f'all:"{topic}"',

        "start":
            0,

        "max_results":
            5
    }

    response = requests.get(
        ARXIV_API,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "xml"
    )

    entries = soup.find_all("entry")

    return parse_arxiv_entries(entries)


def get_paper_by_id(paper_id):

    paper_id = paper_id.replace(
        "https://arxiv.org/abs/",
        ""
    )

    paper_id = paper_id.rstrip("/")

    params = {
        "id_list": paper_id
    }

    response = requests.get(
        ARXIV_API,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "xml"
    )

    entries = soup.find_all("entry")

    return parse_arxiv_entries(entries)


# ============================================================
# NODE 1 - QUERY UNDERSTANDING
# ============================================================

def query_understanding_node(state):

    query_info = understand_query(
        state["query"]
    )

    return {
        "query_type":
            query_info["query_type"]
    }


# ============================================================
# NODE 2 - ARXIV RETRIEVAL
# ============================================================

def retrieval_node(state):

    try:

        if state["query_type"] == "topic":

            papers = search_arxiv(
                state["query"]
            )

        else:

            paper_id = (
                state["query"]
                .rstrip("/")
                .split("/")[-1]
            )

            papers = get_paper_by_id(
                paper_id
            )

        return {
            "papers": papers
        }

    except Exception as e:

        print(
            "\nRetrieval error:",
            e
        )

        return {
            "papers": []
        }


# ============================================================
# NODE 3 - PAPER SELECTION
# ============================================================

def paper_selection_node(state):

    papers = state["papers"]

    if not papers:

        return {
            "selected_paper": None
        }

    # For a direct paper ID/URL there should normally be
    # one paper, so no ranking is necessary.
    if state["query_type"] != "topic":

        return {
            "selected_paper": papers[0]
        }

    if len(papers) == 1:

        return {
            "selected_paper": papers[0]
        }

    paper_information = ""

    for i, paper in enumerate(papers):

        paper_information += f"""

PAPER {i}

Title:
{paper["title"]}

Authors:
{", ".join(paper["authors"])}

Abstract:
{paper["abstract"]}

"""

    prompt = f"""
You are helping select an academic research paper.

User research topic:
{state["query"]}

Candidate papers:
{paper_information}

Choose the single paper that is most relevant
to the user's research topic.

Return ONLY the paper number.
"""

    try:

        answer = call_llm(
            prompt,
            temperature=0.0
        )

        match = re.search(
            r"\d+",
            answer
        )

        if match:

            index = int(
                match.group()
            )

        else:

            index = 0

        index = max(
            0,
            min(
                index,
                len(papers) - 1
            )
        )

    except Exception as e:

        print(
            "\nLLM selection failed. "
            "Using the first paper.",
            e
        )

        index = 0

    return {
        "selected_paper":
            papers[index]
    }


# ============================================================
# NODE 4 - PDF ACQUISITION AND PARSING
# ============================================================

def pdf_acquisition_node(state):

    paper = state["selected_paper"]

    if not paper:

        return {
            "paper_text": None
        }

    pdf_url = paper.get(
        "pdf_url"
    )

    if not pdf_url:

        return {
            "paper_text": None
        }

    try:

        response = requests.get(
            pdf_url,
            timeout=60
        )

        response.raise_for_status()

        with open(
            "paper.pdf",
            "wb"
        ) as file:

            file.write(
                response.content
            )

        doc = fitz.open(
            "paper.pdf"
        )

        pages = []

        for page in doc:

            page_text = page.get_text()

            if page_text:

                pages.append(
                    page_text
                )

        doc.close()

        text = "\n".join(
            pages
        ).strip()

        if len(text) < 500:

            print(
                "\nWarning: very little text "
                "was extracted from the PDF."
            )

        return {
            "paper_text": text
        }

    except Exception as e:

        print(
            "\nPDF acquisition error:",
            e
        )

        return {
            "paper_text": None
        }


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=1500,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end == len(text):

            break

        start = end - overlap

    return chunks


# ============================================================
# NODE 5 - CHUNK AND EMBED
# ============================================================

def chunk_and_embed_node(state):

    text = state["paper_text"]

    if not text:

        return {
            "chunks": [],
            "vector_store_id": None
        }

    chunks = create_chunks(
        text
    )

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    store_id = str(
        uuid.uuid4()
    )

    vector_stores[
        store_id
    ] = index

    print(
        f"\nCreated {len(chunks)} text chunks."
    )

    return {

        "chunks":
            chunks,

        "vector_store_id":
            store_id
    }


# ============================================================
# VECTOR SEARCH HELPER
# ============================================================

def retrieve_chunks(
    chunks,
    index,
    question,
    k=4
):

    if not chunks or index is None:

        return []

    question_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        .astype("float32")
    )

    k = min(
        k,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        k
    )

    retrieved = []

    for i in indices[0]:

        if 0 <= i < len(chunks):

            retrieved.append(
                chunks[i]
            )

    return retrieved


# ============================================================
# NODE 6 - SUMMARIZATION
# ============================================================

def summarization_node(state):

    paper = state["selected_paper"]
    chunks = state["chunks"]
    store_id = state["vector_store_id"]

    index = vector_stores.get(
        store_id
    )

    if not paper or not chunks or index is None:

        return {
            "summary":
                "Unable to create the briefing because "
                "paper content was not available."
        }

    briefing_questions = [

        "What is the main research problem and motivation?",

        "What methodology, model, dataset, or experimental approach does the paper use?",

        "What are the main results and important claims?",

        "What limitations, weaknesses, or future work does the paper mention?",

        "What is the conclusion and why does the paper matter?"
    ]

    selected_chunks = []

    for question in briefing_questions:

        results = retrieve_chunks(
            chunks,
            index,
            question,
            k=3
        )

        selected_chunks.extend(
            results
        )

    # Remove duplicate chunks while keeping order.
    selected_chunks = list(
        dict.fromkeys(
            selected_chunks
        )
    )

    context = "\n\n".join(

        f"RETRIEVED SECTION {i + 1}:\n{chunk}"

        for i, chunk in enumerate(
            selected_chunks
        )
    )

    prompt = f"""
You are an academic research assistant.

Create a structured executive briefing for this paper.

Paper title:
{paper["title"]}

Use ONLY the retrieved paper content below.

Your answer must contain:

1. Why this paper matters
2. Problem statement
3. Method / approach
4. Key results / claims
5. Limitations
6. Suggested follow-up questions

Rules:

- Do not invent facts.
- Do not use outside knowledge.
- If a requested detail is not supported by the
  retrieved content, say that it is not clearly stated.
- Keep the language clear and suitable for a researcher
  who wants a quick understanding of the paper.

Retrieved paper content:

{context}
"""

    try:

        summary = call_llm(
            prompt,
            temperature=0.1
        )

    except Exception as e:

        summary = (
            "LLM summarization failed: "
            + str(e)
        )

    return {
        "summary":
            summary
    }


# ============================================================
# NODE 7 - SYNTHESIS
# ============================================================

def synthesis_node(state):

    paper = state["selected_paper"]
    summary = state["summary"]

    if not paper:

        return {
            "final_answer":
                "No suitable paper was found."
        }

    final_answer = f"""
# Executive Briefing

## Title

{paper["title"]}

## Authors

{", ".join(paper["authors"])}

## arXiv ID

{paper["arxiv_id"]}

## Published

{paper["published"]}

## Link

https://arxiv.org/abs/{paper["arxiv_id"]}

## Briefing

{summary}

## QA Mode

The paper has been parsed, chunked and indexed in a
local FAISS vector store.

You can now ask follow-up questions about the paper.
"""

    return {
        "final_answer":
            final_answer
    }


# ============================================================
# NODE 8 - QA / RAG
# ============================================================

def qa_node(state):

    question = state["qa_question"]

    chunks = state["chunks"]

    store_id = state["vector_store_id"]

    index = vector_stores.get(
        store_id
    )

    if not question:

        return {
            "qa_answer":
                "Please enter a question."
        }

    if not chunks or index is None:

        return {
            "qa_answer":
                "The paper content is not available for QA."
        }

    retrieved = retrieve_chunks(
        chunks,
        index,
        question,
        k=5
    )

    if not retrieved:

        return {
            "qa_answer":
                "I could not retrieve relevant paper content."
        }

    context = "\n\n".join(

        f"RETRIEVED CHUNK {i + 1}:\n{chunk}"

        for i, chunk in enumerate(
            retrieved
        )
    )

    prompt = f"""
You are a grounded academic QA assistant.

Answer the user's question using ONLY the retrieved
chunks from the research paper.

Question:
{question}

Retrieved paper chunks:

{context}

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not supported by these chunks,
   say exactly:

"I could not find that information in the retrieved
paper content."

4. Keep the answer concise and clear.
"""

    try:

        answer = call_llm(
            prompt,
            temperature=0.0
        )

    except Exception as e:

        answer = (
            "QA failed: "
            + str(e)
        )

    history = list(
        state.get(
            "qa_history",
            []
        )
    )

    history.append({

        "question":
            question,

        "answer":
            answer
    })

    return {

        "qa_answer":
            answer,

        "qa_history":
            history
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def start_router(state):

    if state.get(
        "mode"
    ) == "qa":

        return "qa"

    return "research"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    builder = StateGraph(
        ResearchState
    )

    builder.add_node(
        "query_understanding",
        query_understanding_node
    )

    builder.add_node(
        "retrieval",
        retrieval_node
    )

    builder.add_node(
        "paper_selection",
        paper_selection_node
    )

    builder.add_node(
        "pdf_acquisition",
        pdf_acquisition_node
    )

    builder.add_node(
        "chunk_and_embed",
        chunk_and_embed_node
    )

    builder.add_node(
        "summarization",
        summarization_node
    )

    builder.add_node(
        "synthesis",
        synthesis_node
    )

    builder.add_node(
        "qa",
        qa_node
    )

    # START decides whether this is a new research request
    # or a follow-up QA request.
    builder.add_conditional_edges(

        START,

        start_router,

        {
            "research":
                "query_understanding",

            "qa":
                "qa"
        }
    )

    builder.add_edge(
        "query_understanding",
        "retrieval"
    )

    builder.add_edge(
        "retrieval",
        "paper_selection"
    )

    builder.add_edge(
        "paper_selection",
        "pdf_acquisition"
    )

    builder.add_edge(
        "pdf_acquisition",
        "chunk_and_embed"
    )

    builder.add_edge(
        "chunk_and_embed",
        "summarization"
    )

    builder.add_edge(
        "summarization",
        "synthesis"
    )

    builder.add_edge(
        "synthesis",
        END
    )

    builder.add_edge(
        "qa",
        END
    )

    memory = MemorySaver()

    return builder.compile(
        checkpointer=memory
    )


# ============================================================
# RUN THE AGENT
# ============================================================

def run_research_assistant():

    graph = build_graph()

    query = input(
        "\nEnter a research topic, arXiv ID, "
        "or arXiv URL: "
    ).strip()

    initial_state = {

        "query":
            query,

        "query_type":
            "",

        "papers":
            [],

        "selected_paper":
            None,

        "paper_text":
            None,

        "chunks":
            [],

        "vector_store_id":
            None,

        "summary":
            None,

        "final_answer":
            None,

        "mode":
            "research",

        "qa_question":
            None,

        "qa_answer":
            None,

        "qa_history":
            []
    }

    config = {

        "configurable": {

            "thread_id":
                "research-session-1"
        }
    }

    result = graph.invoke(
        initial_state,
        config=config
    )

    print("\n" + "=" * 80)
    print("RETRIEVED PAPERS")
    print("=" * 80)

    if result["papers"]:

        df = pd.DataFrame(
            result["papers"]
        )

        print(
            df[
                [
                    "title",
                    "authors",
                    "published",
                    "arxiv_id",
                    "pdf_url"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No papers found."
        )

        return

    print("\n" + "=" * 80)
    print("SELECTED PAPER")
    print("=" * 80)

    if result["selected_paper"]:

        print(
            result[
                "selected_paper"
            ]["title"]
        )

    else:

        print(
            "No paper selected."
        )

        return

    print("\n" + "=" * 80)
    print("EXECUTIVE BRIEFING")
    print("=" * 80)

    print(
        result["final_answer"]
    )

    # --------------------------------------------------------
    # QA LOOP
    # --------------------------------------------------------

    while True:

        question = input(
            "\nAsk a question about the paper "
            "(or type 'exit'): "
        ).strip()

        if question.lower() == "exit":

            break

        qa_state = {

            **result,

            "mode":
                "qa",

            "qa_question":
                question
        }

        result = graph.invoke(
            qa_state,
            config=config
        )

        print("\n" + "-" * 80)
        print("GROUNDED ANSWER")
        print("-" * 80)

        print(
            result["qa_answer"]
        )


if __name__ == "__main__":

    run_research_assistant()
