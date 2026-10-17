# agent-harness

## Setup

**Use Python 3.12 explicitly.** Bare `python3` is brew's 3.15 on this machine, and
`onnxruntime` (a hard dependency of `chromadb`) publishes no cp315 wheel — `pip install`
then fails with `ResolutionImpossible` after backtracking through every chromadb release.

```bash
deactivate
rm -rf .venv-host
/usr/bin/python3.12 -m venv .venv-host
source .venv-host/bin/activate
python -V                      # must print 3.12.x
pip install -r requirements.txt
echo ".venv-host/" >> .gitignore
```

`.venv` is the sandbox-side venv (used from inside the container tooling), `.venv-host` is
the one that runs the harness:

```bash
source .venv-host/bin/activate
```

## Keys

`.env` at the project root (names documented in `.env.example`):

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Sandbox image

Docker must be running before the harness starts.

```bash
docker build -f harness/sandbox/Dockerfile -t agent-harness:01 .
```

The tag must match `SANDBOX_IMAGE` in `harness/config.py`.

## Knowledge index (RAG)

`knowledge/` holds the corpus (`.md` and `.txt` only, recursive). Re-run the indexer after
every corpus change — indexing is manual, and nothing detects a stale index:

```bash
python -m harness.rag_index
echo ".rag_index/" >> .gitignore
```

Expect: `Found 9 files` → `Chunked into 9 chunks` → `Embedding via text-embedding-3-small`
→ `Index built`. The index lives in `.rag_index/` (ChromaDB, derived, regenerable).

Query it from a session with the `recall` tool, e.g. *"what's our on-call escalation policy
for P0 incidents?"*.

## Run

```bash
python -m harness.agent
```
