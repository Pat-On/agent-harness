# agent-harness

An agent harness built from scratch: a ReAct loop over an OpenAI-compatible model, with
filesystem and git tools, containerised `bash`, durable memory in `AGENTS.md`, web search,
and RAG over a local knowledge corpus.

## Requirements

- **Python 3.11–3.14.** `chromadb` depends on `onnxruntime`, which publishes no wheels for
  3.15 yet, so `pip install` fails with `ResolutionImpossible` on newer interpreters.
- **Docker**, running — the `bash` tool executes inside a container.
- An OpenAI API key, and a Tavily key for web search.

## Setup

If your default `python3` is outside the supported range, create the venv with an explicit
interpreter (`python3.12 -m venv ...`) rather than the bare command.

```bash
python3 -m venv .venv-host
source .venv-host/bin/activate
python -V                      # confirm the version is in range
pip install -r requirements.txt
```

Activate the venv in every new shell before running anything — `which python` should point
inside it. Without it, `pip` installs against the system interpreter and may refuse
outright (`externally-managed-environment` on Debian-based distros).

## Keys

`.env` at the project root — see `.env.example` for the variable names:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Sandbox image

```bash
docker build -f harness/sandbox/Dockerfile -t agent-harness:01 .
```

The tag must match `SANDBOX_IMAGE` in `harness/config.py`.

## Knowledge index (RAG)

`knowledge/` holds the corpus — `.md` and `.txt` files, read recursively. Indexing is
manual and nothing detects a stale index, so re-run it after every corpus change:

```bash
python -m harness.rag_index
```

It reports the files found, the chunk count, the embedding model, and where the index was
written. The index lives in `.rag_index/` (ChromaDB; derived, gitignored, regenerable in
seconds). Sessions query it through the `recall` tool.

## Run

```bash
python -m harness.agent
```

## Configuration

Operator-tunable values live in `harness/config.py`: model id, step budget, tool timeouts,
the `bash` allow/deny lists, sandbox image and paths, web-search result count, and the
`RAG_*` knobs (corpus path, index path, embedding model, chunk size and overlap, default
result count).
