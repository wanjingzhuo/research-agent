#!/usr/bin/env python
# coding: utf-8
"""
research_agent.py

A small research agent that can search the web, read pages, and produce a
sourced report. Ported from agent.ipynb, keeping the notebook's final logic
for every function unchanged.

Usage:
    python research_agent.py "What is the gold price today, and what moved it this past month?"
    python research_agent.py "..." --eval
"""

import os
import json
import time
import argparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv


# =============================================================================
# 1. Load .env configuration
# =============================================================================

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

assert API_BASE_URL, "API_BASE_URL missing in .env"
assert API_KEY, "API_KEY missing in .env"
assert MODEL, "MODEL missing in .env"
# print(f"model={MODEL}, base_url={API_BASE_URL}")


# =============================================================================
# 2. Tools: search_web / read_webpage
# =============================================================================

def search_web(query):
    print(f"  [search_web] searching: {query}")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]


def read_webpage(url):
    print(f"  [read_webpage] reading: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if not resp.ok:
            return f"[failed to read: HTTP {resp.status_code} {resp.reason}]"
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        if not text.strip():
            return "[failed to read: page returned no readable text]"
        return text[:5000]
    except Exception as e:
        return f"[failed to read: {e}]"


# =============================================================================
# 3. TOOL_DESCRIPTIONS
# =============================================================================

TOOL_DESCRIPTIONS = """
You are a research agent. You have three actions:

SEARCH: search the web. Reply {"action": "SEARCH", "query": "..."}
READ: read one web page. Reply {"action": "READ", "url": "..."}
FINISH: you have enough info. Reply {"action": "FINISH", "report": "..."}

Rules:
- You must READ at least 3 different pages before you are allowed to FINISH.
- Prefer news articles and analysis over live price-ticker pages or shopping/dealer
  sites, because they usually explain WHY something changed, not just WHAT the
  number is right now.

When you FINISH, your report must have this structure:

1. A list of findings. Each finding must end with the URL it came from, in
   square brackets, like this: [https://example.com/page]. If a finding is
   not supported by any specific source you read, end it with [no source]
   instead. Only cite a URL in a finding if you actually used the READ action
   on that page — do not cite a URL you only saw in search results.

2. Two source lists at the end, exactly in this format:

SOURCES READ:
- <url>
- <url>

LINKS FOUND BUT NOT READ:
- <url>
- <url>

Reply with ONLY a JSON object, nothing else. No markdown, no explanation.
"""


# =============================================================================
# 4. ask_model (with 429 retry)
# =============================================================================

def ask_model(goal, state, max_retries=3):
    today = datetime.today().strftime("%Y-%m-%d")
    messages = [
        {"role": "system", "content": TOOL_DESCRIPTIONS + f"\n\nToday's date is {today}."},
        {"role": "user", "content": f"Goal: {goal}\n\nState so far:\n{json.dumps(state, indent=2)}"}
    ]

    last_error = None

    for attempt in range(max_retries):
        resp = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "User-Agent": "research-agent/0.1"
            },
            json={"model": MODEL, "messages": messages}
        )

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"  [ask_model] rate limited (429), waiting {wait}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            print(f"  [ask_model] server error ({resp.status_code}), waiting 5s before retry {attempt + 1}/{max_retries}")
            time.sleep(5)
            continue

        if not resp.ok:
            raise RuntimeError(
                f"API call failed: {resp.status_code} {resp.reason}\n"
                f"Response body: {resp.text[:500]}"
            )

        raw = resp.json()["choices"][0]["message"]["content"]
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)

    raise RuntimeError(f"API call failed after {max_retries} retries. Last error: {last_error}")


# =============================================================================
# 5. run_agent (with forced 3-page gate, failure tolerance)
# =============================================================================

MAX_STEPS = 10

def run_agent(goal, on_step=None, max_steps=None):
    """
    Run the agent loop until it FINISHES or hits max_steps.

    If on_step is provided, it is called as on_step(step_number, decision, result)
    after every step (SEARCH, READ, FINISH, or ERROR). `decision` is the action
    dict for that step (or an {"action": "ERROR", "detail": ...} dict for error
    steps); `result` is the corresponding tool/report output, or None for errors.
    on_step is purely a side channel for observers (e.g. a UI) — when it is not
    provided, behavior is unchanged from before it existed.

    max_steps caps how many loop iterations are attempted before giving up; if
    omitted, it defaults to the module-level MAX_STEPS constant, so callers
    that don't pass it (e.g. the CLI) keep their existing behavior unchanged.
    This is unrelated to the separate rule below that requires at least 3
    distinct pages to be read before a FINISH is accepted.
    """
    if max_steps is None:
        max_steps = MAX_STEPS

    state = []
    report = None

    def emit(step, decision, result):
        if on_step is not None:
            on_step(step, decision, result)

    for step in range(1, max_steps + 1):
        try:
            decision = ask_model(goal, state)
        except (RuntimeError, json.JSONDecodeError) as e:
            print(f"STEP {step}: ask_model failed — {e}")
            error_entry = {"action": "ERROR", "detail": str(e)}
            state.append(error_entry)
            emit(step, error_entry, None)
            continue

        print(f"STEP {step}: {decision.get('action', 'UNKNOWN')}")

        if decision.get("action") == "SEARCH":
            result = search_web(decision["query"])
            state.append({"action": "SEARCH", "query": decision["query"], "result": result})
            emit(step, decision, result)

        elif decision.get("action") == "READ":
            result = read_webpage(decision["url"])
            state.append({"action": "READ", "url": decision["url"], "result": result})
            emit(step, decision, result)

        elif decision.get("action") == "FINISH":
            valid_reads = {
                s["url"] for s in state
                if s["action"] == "READ" and not s["result"].startswith("[failed to read:")
            }
            if len(valid_reads) < 3:
                print(f"  [blocked] FINISH rejected — only {len(valid_reads)} distinct page(s) read, need 3")
                error_entry = {
                    "action": "ERROR",
                    "detail": f"FINISH rejected: only {len(valid_reads)} distinct pages read, minimum is 3"
                }
                state.append(error_entry)
                emit(step, error_entry, None)
                continue
            report = decision.get("report", "")
            emit(step, decision, report)
            break

        else:
            print(f"  [warning] unrecognized action: {decision}")
            error_entry = {"action": "ERROR", "detail": f"unrecognized decision: {decision}"}
            state.append(error_entry)
            emit(step, error_entry, None)

    if report is None:
        report = "Step limit reached before finishing."

    print("\n--- FINAL REPORT ---")
    print(report)

    return state, report


# =============================================================================
# 6. run_evals (with SOURCES READ verification)
# =============================================================================

def run_evals(state, report):
    results = []

    # 1. Used the SEARCH tool
    used_search = any(s["action"] == "SEARCH" for s in state)
    results.append(("Used the search tool", used_search))

    # 2. Read more than one distinct valid source
    valid_reads = {
        s["url"] for s in state
        if s["action"] == "READ" and not s["result"].startswith("[failed to read:")
    }
    read_more_than_one = len(valid_reads) > 1
    results.append(("Read more than one distinct valid source", read_more_than_one))

    # 3. Finished within the step limit (didn't trigger the fallback text)
    finished_in_time = report != "Step limit reached before finishing."
    results.append(("Finished within the step limit", finished_in_time))

    # 4. Produced actual content (not empty or unusually short)
    produced_report = len(report.strip()) > 50
    results.append(("Produced a non-trivial report", produced_report))

    # 5. SOURCES READ list matches the actual valid_reads
    if "SOURCES READ:" in report:
        try:
            sources_section = report.split("SOURCES READ:")[1].split("LINKS FOUND BUT NOT READ:")[0]
            claimed_reads = {
                line.strip().lstrip("- ").strip()
                for line in sources_section.strip().split("\n") if line.strip()
            }
        except IndexError:
            claimed_reads = set()
    else:
        claimed_reads = set()

    sources_match = claimed_reads == valid_reads
    results.append(("SOURCES READ list matches actual valid reads", sources_match))

    # Print results
    print("\n--- EVAL RESULTS ---")
    passed = 0
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"{status}  {name}")

    print(f"\nScore: {passed} of {len(results)}")

    missing = valid_reads - claimed_reads
    extra = claimed_reads - valid_reads
    if missing:
        print(f"  [detail] actually read but missing from SOURCES READ: {missing}")
    if extra:
        print(f"  [detail] claimed in SOURCES READ but never actually read: {extra}")

    eval_summary = {
        "checks": [{"name": name, "passed": ok} for name, ok in results],
        "score": f"{passed} of {len(results)}",
        "sources_mismatch": {
            "missing_from_list": sorted(missing),
            "falsely_claimed": sorted(extra)
        }
    }
    return eval_summary


def save_run(goal, state, report, eval_summary=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {"goal": goal, "state": state, "report": report}
    if eval_summary is not None:
        data["eval"] = eval_summary
    with open(f"run_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [saved] run_{timestamp}.json")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Research agent: search, read, and report on a topic.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="What is the gold price today, and what moved it this past month?",
        help="The research goal/question to answer.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run run_evals() on the result and include the eval summary in the saved run.",
    )
    args = parser.parse_args()

    state, report = run_agent(args.goal)

    eval_summary = None
    if args.eval:
        eval_summary = run_evals(state, report)

    save_run(args.goal, state, report, eval_summary)


if __name__ == "__main__":
    main()
