# Autonomous arXiv Paper Digest & QA Agent

A local research assistant built with Python and LangGraph.

The system accepts either:
- a natural-language research topic
- an arXiv paper ID
- an arXiv paper URL

It retrieves the paper from the official arXiv API, downloads and parses the PDF, chunks the text, creates local embeddings, stores them in a local FAISS vector index, generates an executive briefing with a local LLM, and supports grounded follow-up questions using RAG.

## 1. Architecture

```text
                         USER INPUT
                             |
                             v
                  +----------------------+
                  | Query Understanding  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Conditional Router   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |    arXiv Retrieval   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   Paper Selection    |
                  |      (LLM)           |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   Fetch & Parse PDF  |
                  |      PyMuPDF         |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   Chunk & Embed      |
                  | SentenceTransformer  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   Local FAISS Index  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |    Summarization     |
                  |       (LLM)          |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Executive Briefing  |
                  +----------+-----------+
                             |
                             v
                       QA / RAG LOOP
                             |
                             v
                  Question -> FAISS -> LLM
                             |
                             v
                    Grounded Answer
```

### State

The LangGraph state contains:

```text
query
query_type
papers
selected_paper
paper_text
chunks
vector_store_id
summary
final_answer
mode
qa_question
qa_answer
qa_history
```

The state is passed between nodes. `MemorySaver` provides in-process LangGraph checkpointing for the research session.

## 2. Main nodes

1. **Query Understanding**
   - Detects topic search, paper ID, or paper URL.

2. **arXiv Retrieval**
   - Uses the official arXiv Atom API.
   - Retrieves title, authors, abstract, PDF link, categories, date and ID.

3. **Paper Selection**
   - For topic searches, candidate abstracts are sent to the local LLM.
   - For a direct paper ID/URL, the returned paper is selected directly.
   - If LLM selection fails, the first candidate is used as a graceful fallback.

4. **Fetch & Parse**
   - Downloads the PDF.
   - Uses PyMuPDF to extract text.
   - Handles missing/broken extraction with a warning and empty state.

5. **Chunk & Embed**
   - Splits extracted text into overlapping chunks.
   - Uses `all-MiniLM-L6-v2` locally for embeddings.
   - Stores vectors in a local FAISS index.

6. **Summarization**
   - Retrieves chunks related to the research problem, method, results, limitations and conclusion.
   - Sends only retrieved paper content to the local LLM.
   - Produces a structured executive briefing.

7. **QA / RAG**
   - Embeds the user's question.
   - Retrieves the most relevant paper chunks from FAISS.
   - Gives only those chunks to the LLM.
   - Instructs the LLM not to use outside knowledge.
   - If the information is not supported by the retrieved content, it returns a grounded "not found" response.

## 3. Technology choices

- **Python** — main language
- **LangGraph** — stateful workflow orchestration
- **arXiv API** — paper retrieval
- **BeautifulSoup** — parsing the arXiv Atom/XML response
- **PyMuPDF** — PDF parsing
- **Sentence Transformers** — local embeddings
- **FAISS** — local vector search
- **Ollama + Qwen 2.5 3B** — local open-weight LLM
- **Pandas** — displaying retrieved paper metadata

No paid API key is required.

## 4. Setup

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama

Install Ollama from:

https://ollama.com/

Then download the model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before starting the Python program.

The application expects the Ollama API at:

```text
http://localhost:11434/api/generate
```

The first use of the Sentence Transformer model will also download:

```text
all-MiniLM-L6-v2
```

## 5. Run

```bash
python arxiv_research_agent.py
```

The program asks for:

```text
Enter a research topic, arXiv ID, or arXiv URL:
```

Examples:

```text
recent work on KV-cache compression for LLMs
```

or:

```text
2306.04338
```

or:

```text
https://arxiv.org/abs/2306.04338
```

After the briefing is generated, the program enters QA mode:

```text
Ask a question about the paper (or type 'exit'):
```

## 6. Example run

For the final submission, run the program once and paste the actual output here.

Recommended test:

```text
2306.04338
```

Then ask 2–3 questions such as:

```text
What is the main problem addressed by this paper?
```

```text
What methodology does the paper use?
```

```text
What limitations are discussed?
```

Also test an out-of-paper question to demonstrate grounding, for example:

```text
What is the author's favorite programming language?
```

The expected behavior is that the system should not invent an answer when the retrieved paper content does not support it.

## 7. Failure handling

The system handles several realistic failure cases:

### No arXiv result

If retrieval returns no papers:

```text
No papers found.
```

### PDF failure

The PDF node catches download or parsing errors and returns an empty paper text state instead of crashing the whole workflow.

### Poor PDF extraction

If very little text is extracted, the program prints a warning.

### LLM selection failure

If the local LLM cannot select a candidate paper, the system falls back to the first candidate.

### Missing QA context

If no vector index or chunks are available, QA returns a clear message instead of fabricating an answer.

## 8. Design Decisions & Tradeoffs

I chose LangGraph because the assessment specifically focuses on explicit stateful agent design. The workflow is represented as nodes, edges, conditional routing and shared state rather than as one large prompt.

I chose a local Ollama model instead of a paid hosted API. This avoids requiring a paid API key and makes the project reproducible with open/local tools. The tradeoff is that local model quality and speed depend on the user's hardware.

I chose FAISS because it is simple, local and sufficient for a single-paper QA workflow. A production system could use a persistent vector database.

The embedding model is also local. This keeps the retrieval pipeline independent of a paid embedding API.

The current FAISS index is held in memory and referenced by an ID in LangGraph state. This keeps the graph state simple, but the vector index itself is not persistent across a full process restart.

The paper selection step uses an LLM for topic searches, while direct paper-ID/URL requests do not need ranking.

For summarization, the system retrieves chunks relevant to the problem, methodology, results, limitations and conclusion instead of sending the entire PDF to the LLM. This reduces the amount of context passed to the model and helps keep the briefing grounded.

## 9. Known limitations

- PDF extraction can be imperfect for scanned or unusual PDFs.
- Figures and tables are not interpreted as images.
- The local LLM may be slower or less capable on low-resource hardware.
- The FAISS index is in memory for the current process.
- The arXiv topic search is intentionally small and retrieves five candidates.
- The system is designed for one active research session rather than multi-user deployment.

## 10. What I would do next with more time

- Persist FAISS indexes and metadata to disk or a local vector database.
- Improve section-aware PDF parsing.
- Add better ranking across multiple candidate papers.
- Store paper-specific chunk metadata such as page number and section.
- Return source chunk/page references in QA answers.
- Add a small evaluation set for retrieval and groundedness.
- Add support for comparing several selected papers.

## 11. Assessment mapping

| Assessment requirement | Implementation |
|---|---|
| Natural-language topic | Query understanding + arXiv search |
| arXiv ID / URL | Direct arXiv lookup |
| Stateful graph | LangGraph + shared `ResearchState` |
| Query understanding | `query_understanding_node` |
| arXiv retrieval | `retrieval_node` |
| Selection/ranking | `paper_selection_node` |
| Fetch & parse | `pdf_acquisition_node` |
| Chunk & embed | `chunk_and_embed_node` |
| Vector DB | Local FAISS |
| Executive briefing | `summarization_node` + `synthesis_node` |
| QA loop | `qa_node` |
| Grounded QA | FAISS retrieval + context-only prompt |
| Failure handling | Retrieval/PDF/LLM/QA fallbacks |
| Local/free tooling | Ollama + local embedding model + FAISS |

## 12. Submission checklist

- [ ] Push the project to GitHub or create a zip.
- [ ] Include `arxiv_research_agent.py`.
- [ ] Include `requirements.txt`.
- [ ] Include this README.
- [ ] Run one real example and paste the output into the README.
- [ ] Include 2–3 real QA exchanges.
- [ ] Record the 4-minute reflection video.
- [ ] Submit the repository/zip and README.

