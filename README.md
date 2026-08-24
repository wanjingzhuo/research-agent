# Research Agent

A minimal ReAct-loop research agent built from scratch to understand how agentic
loops work: model decides the next step, a Python loop executes it, the result
is appended to state, and the model decides again — until it has enough to
answer, or the step limit is reached.

This is a self-built companion project, separate from the instructor's
`research-scout` (NTU PACE SCTP DSAI Module 6). Same underlying pattern, built
independently cell by cell to understand every moving part before following
the instructor's five-prompt version.

## What it does

Give it a research question. It will:

1. Search the web (DuckDuckGo, via `ddgs`)
2. Read pages it finds relevant (BeautifulSoup, capped at 5000 characters)
3. Decide whether it has enough — it is **not allowed to finish** until it has
   successfully read at least 3 distinct pages
4. Write a report where every finding is tagged with the URL it came from, or
   `[no source]` if nothing supports it
5. List which sources were actually read vs. only found in search results but
   never opened

## Why it's built this way

A few design choices came out of debugging real runs, not just following a
spec:

- **HTTP errors and empty pages are treated as failed reads.** A page that
  returns a 403, a 406, or 200-with-no-readable-text does not count toward
  the "3 distinct pages" requirement — only pages that actually yielded
  content do.
- **The report's "Sources Read" list is cross-checked against the real run
  state**, not just trusted. The model is asked to report which URLs it
  read, and `run_evals()` verifies that claim against what actually
  happened, flagging any mismatch.
- **The current date is injected into the prompt.** Without it, the model
  defaults to whatever "today" it assumed from training and can generate
  searches for the wrong year (verified: without this, a query for "this
  year's PSLE syllabus changes" searched for 2024 instead of 2026).
- **429 and 5xx errors are retried** (respecting `Retry-After` when the
  server provides it); other API failures are reported with the actual
  status code and response body, not a generic message.
- **A single failed step does not kill the run.** Failures (a bad API call,
  malformed JSON from the model, an HTTP error) are logged into the agent's
  own state and the loop continues.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (never commit this):

```
API_BASE_URL=https://opencode.ai/zen/v1
API_KEY=your-own-key-here
MODEL=hy3-free
```

## Usage

### Command line

```bash
python research_agent.py "What is the gold price today, and what moved it this past month?"
python research_agent.py "your question" --eval
```

`--eval` runs 5 automated checks after the agent finishes (used search, read
more than one distinct valid source, finished within the step limit,
produced a non-trivial report, and the sources-read list matches what
actually happened) and prints a score.

Every run is saved to a timestamped `run_YYYYMMDD_HHMMSS.json` containing the
full state, the report, and (if `--eval` was used) the eval results.

### Web interface

```bash
streamlit run app.py
```

Opens a local page where you can type a question, adjust the max step count
with a slider, and watch each SEARCH / READ / FINISH step appear live as the
agent runs. The step log auto-collapses once the run finishes, leaving the
final report visible. Runs through the UI are saved the same way as CLI runs.

## Project structure

```
research_agent.py   # core agent: tools, loop, eval, save
app.py               # Streamlit UI
requirements.txt
.env                 # not committed
```

## What this project is not

It has no memory across runs, no access to private data, and no ability to
take actions beyond searching and reading public web pages. It cannot judge
whether a source is trustworthy — that judgment stays with whoever reads the
report.
