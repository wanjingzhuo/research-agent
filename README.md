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
2. Read pages it finds relevant — HTML (BeautifulSoup) or PDF (pypdf), capped
   at 5000 characters
3. Decide whether it has enough — it is **not allowed to finish** until it has
   successfully read at least 3 distinct pages
4. Write a report where every finding is tagged with the URL it came from, or
   `[no source]` if nothing supports it
5. List which sources were actually read vs. only found in search results but
   never opened

## Why it's built this way

A few design choices came out of debugging real runs, not just following a
spec. Some were discovered independently here; others were borrowed back
after comparing this project against the instructor's spec-driven
`research-scout` implementation, which had already solved a few of the same
problems more thoroughly.

Found and fixed here first:

- **HTTP errors, empty pages, and PDFs are all handled explicitly.** A page
  that returns a 403, a 406, or 200-with-no-readable-text does not count
  toward the "3 distinct pages" requirement — only pages that actually
  yielded content do. PDF links (detected by Content-Type or a `.pdf`
  extension) are parsed with `pypdf` instead of being fed to an HTML parser,
  which previously returned raw compressed binary as garbage text.
- **A source only counts if it was actually cited.** A page can be
  successfully read but never end up supporting any finding (e.g. its content
  turned out to be navigation menus or unrelated). Such pages are excluded
  from "Sources Read" and from the "distinct valid source" eval check, and
  are reported separately as read-but-not-quoted for debugging.
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
- **The model cannot search the exact same query twice.** A repeated query
  wastes a step and a real search request; it's rejected and logged so the
  model is prompted to rephrase instead.

Borrowed from `research-scout` after comparing the two implementations:

- **Every decision includes a `reason` field**, so the model's stated
  rationale is visible in the run log at each step, not just the action it
  took.
- **The model is told its exact step budget up front** ("you have at most N
  steps total") and instructed to plan accordingly, rather than discovering
  the limit only when it runs out.
- **The "Sources Read" / "Also found" lists are built entirely from `state`**,
  not written by the model. The model is told not to write its own source
  list — the program constructs it after the run from what actually
  happened, so it can't drift from reality the way a model-authored list can.
- **The model cannot READ the same URL twice** — repeating one is refused
  and logged, so a step isn't wasted re-fetching something already read
  (successfully or not).
- **FINISH is refused if the report is empty**, not just if too few pages
  were read — an empty report used to be accepted as long as the page count
  was met.

## Known limitations

- `read_webpage` extracts visible text only — it does not preserve hyperlinks
  found within a page. If a page links to a more detailed source (e.g. a
  syllabus PDF linked from a summary page), the agent has no way to discover
  that link unless it also turns up separately in search results.
- No mechanism yet distinguishes "this question needs live information" from
  "this is common knowledge" — the 3-page-minimum rule applies uniformly, so
  trivial questions still trigger a full search-and-read cycle.

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
MODEL=your-model-name-here
```

Available model names can change over time — check `GET <API_BASE_URL>/models`
for the current list rather than assuming an example model name still works.

## Usage

### Command line

```bash
python research_agent.py "What is the gold price today, and what moved it this past month?"
python research_agent.py "your question" --eval
```

`--eval` runs automated checks after the agent finishes (search was used,
more than one distinct *cited* source, finished within the step limit,
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
research_agent.py     # core agent: tools, loop, eval, save
app.py                 # Streamlit UI
requirements.txt
.env                   # not committed
scripts/               # helper scripts (e.g. regression test runner)
dev-notebooks/         # notebook history from the fake-model prototype
                        # through the working agent — kept for reference,
                        # not used at runtime
```

## What this project is not

It has no memory across runs, no access to private data, and no ability to
take actions beyond searching and reading public web pages. It cannot judge
whether a source is trustworthy — that judgment stays with whoever reads the
report.
