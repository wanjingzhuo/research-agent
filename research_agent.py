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
import re
import time
import argparse
from datetime import datetime
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from pypdf import PdfReader


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
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]
    except Exception as e:
        return [{"title": "[search failed]", "url": "", "snippet": f"[search failed: {e}]"}]


def read_webpage(url):
    print(f"  [read_webpage] reading: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if not resp.ok:
            return f"[failed to read: HTTP {resp.status_code} {resp.reason}]"

        content_type = resp.headers.get("Content-Type", "")
        is_pdf = "application/pdf" in content_type.lower() or url.lower().endswith(".pdf")

        if is_pdf:
            reader = PdfReader(BytesIO(resp.content))
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        else:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

        if not text.strip():
            return "[failed to read: page returned no readable text]"
        return text[:5000]
    except Exception as e:
        return f"[failed to read: {e}]"


# =============================================================================
# 3. build_tool_descriptions (system prompt)
# =============================================================================

def build_tool_descriptions(step_limit):
    """The system prompt sent on every agent step. step_limit is passed in
    (rather than hardcoded) so the number the model is told always matches
    max_steps for this particular run — which can vary (e.g. via the UI's
    "Max steps" slider) rather than being a single fixed constant.
    """
    return f"""
You are a research agent. You have three actions:

SEARCH: search the web. Reply {{"reason": "one short sentence", "action": "SEARCH", "query": "..."}}
READ: read one web page. Reply {{"reason": "one short sentence", "action": "READ", "url": "..."}}
FINISH: you have enough info. Reply {{"reason": "one short sentence", "action": "FINISH", "report": "..."}}

Rules:
- You must READ at least 3 different pages before you are allowed to FINISH.
- Prefer primary sources, news, and in-depth analysis over pages that only
show a number or a product listing without explanation.
- You have at most {step_limit} steps total in this run, including this one.
  If you have not FINISHed by step {step_limit}, the run ends without a
  report — plan your searches and reads so you can finish within that budget.
- You cannot READ the same URL twice: asking again is refused and wastes a
  step, so pick a different page.
- You cannot SEARCH the exact same query twice: repeating it is refused and
  wastes a step, so rephrase or narrow your query if the first search didn't
  find what you needed.
- FINISH is refused if the report is empty, or if you have not yet read at
  least 3 different pages successfully.
- "report" must be a single plain-text string (with newlines inside it as
  needed) — never a nested object or list. Everything you want to say goes
  inside that one string.

When you FINISH, your report must have this structure:

A list of findings, one per line, each starting with "- ". Do not run
multiple findings together in one paragraph. Each finding must end with the
URL it came from, in square brackets, like this: [https://example.com/page].
If a finding is not supported by any specific source you read, end it with
[no source] instead. Only cite a URL in a finding if you actually used the
READ action on that page — do not cite a URL you only saw in search results.

Do not write your own source lists — the program adds "SOURCES READ" and
"LINKS FOUND BUT NOT READ" lists automatically from the pages you actually
read.

Reply with ONLY a JSON object, nothing else. No markdown, no explanation.
"""


# =============================================================================
# 4. ask_model (with 429 retry)
# =============================================================================

def ask_model(goal, state, max_steps, max_retries=3):
    today = datetime.today().strftime("%Y-%m-%d")
    messages = [
        {"role": "system", "content": build_tool_descriptions(max_steps) + f"\n\nToday's date is {today}."},
        {"role": "user", "content": f"Goal: {goal}\n\nState so far:\n{json.dumps(state, indent=2)}"}
    ]

    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "User-Agent": "research-agent/0.1"
                },
                json={"model": MODEL, "messages": messages},
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"  [ask_model] request failed ({type(e).__name__}: {e}), waiting 5s before retry {attempt + 1}/{max_retries}")
            time.sleep(5)
            continue

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

        try:
            raw = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"API response did not contain choices[0].message.content ({type(e).__name__}: {e})\n"
                f"Response body: {resp.text[:500]}"
            )
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)

    raise RuntimeError(f"API call failed after {max_retries} retries. Last error: {last_error}")


# =============================================================================
# 5. format_source_lists
# =============================================================================

# Matches a URL cited in findings text, e.g. "... rose 2% [https://example.com/page]".
CITED_URL_RE = re.compile(r"\[(https?://[^\]\s]+)\]")


def extract_cited_urls(text):
    """URLs the findings text actually cites (the "[https://...]" markers),
    as opposed to pages that were merely fetched. Used to tell a page that
    was read and genuinely used from one that was read but turned out to be
    navigation/noise the model never drew on.
    """
    return set(CITED_URL_RE.findall(text))


def format_source_lists(state, cited_urls):
    """Build the "SOURCES READ" / "LINKS FOUND BUT NOT READ" lists straight
    from state — the ground truth of what actually happened — instead of
    trusting the model's own report text for them. Neither list contains
    repeats.

    SOURCES READ only includes pages that were both successfully read AND
    actually cited in the findings (present in cited_urls) — a page that
    loaded fine but turned out to be nav/boilerplate noise the model never
    quoted doesn't belong there. Such a page is also left out of LINKS FOUND
    BUT NOT READ: it was read, so "not read" would be wrong too. It simply
    doesn't appear in either list (run_agent prints a debug line about it
    instead).

    A page that failed to load is excluded from both lists for the same
    reason as before: it was attempted, so it isn't "also found" either.
    """
    pages_read = []
    seen_read = set()
    attempted_urls = set()

    for entry in state:
        if entry.get("action") != "READ":
            continue
        url = entry.get("url", "")
        if not url:
            continue
        attempted_urls.add(url)
        result = entry.get("result", "")
        if (
            not result.startswith("[failed to read:")
            and url in cited_urls
            and url not in seen_read
        ):
            pages_read.append(url)
            seen_read.add(url)

    also_found = []
    seen_found = set()
    for entry in state:
        if entry.get("action") != "SEARCH":
            continue
        for r in entry.get("result", []):
            url = r.get("url", "")
            if url and url not in attempted_urls and url not in seen_found:
                also_found.append(url)
                seen_found.add(url)

    lines = ["SOURCES READ:"]
    lines += [f"- {u}" for u in pages_read]
    lines.append("")
    lines.append("LINKS FOUND BUT NOT READ:")
    lines += [f"- {u}" for u in also_found]
    return "\n".join(lines)


# =============================================================================
# 6. run_agent (with forced 3-page gate, failure tolerance)
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
            decision = ask_model(goal, state, max_steps)
        except (RuntimeError, json.JSONDecodeError) as e:
            print(f"STEP {step}: ask_model failed — {e}")
            error_entry = {"action": "ERROR", "detail": str(e)}
            state.append(error_entry)
            emit(step, error_entry, None)
            continue

        reason = decision.get("reason", "")
        print(f"STEP {step}: {decision.get('action', 'UNKNOWN')} — reason: {reason}")

        if decision.get("action") == "SEARCH":
            query = decision["query"]
            already_searched = {s["query"] for s in state if s.get("action") == "SEARCH"}
            if query in already_searched:
                print(f"  [blocked] SEARCH rejected — {query!r} was already searched this run")
                error_entry = {
                    "action": "ERROR",
                    "detail": f"SEARCH rejected: query {query!r} was already searched earlier in this run; try a different query"
                }
                state.append(error_entry)
                emit(step, error_entry, None)
                continue
            result = search_web(query)
            state.append({"action": "SEARCH", "reason": reason, "query": query, "result": result})
            emit(step, decision, result)

        elif decision.get("action") == "READ":
            url = decision["url"]
            already_attempted = {s["url"] for s in state if s.get("action") == "READ"}
            if url in already_attempted:
                print(f"  [blocked] READ rejected — {url!r} was already read this run")
                error_entry = {
                    "action": "ERROR",
                    "detail": f"READ rejected: {url!r} was already read earlier in this run; choose a different URL"
                }
                state.append(error_entry)
                emit(step, error_entry, None)
                continue
            result = read_webpage(url)
            state.append({"action": "READ", "reason": reason, "url": url, "result": result})
            emit(step, decision, result)

        elif decision.get("action") == "FINISH":
            report_field = decision.get("report", "")
            if not isinstance(report_field, str):
                print(f"  [blocked] FINISH rejected — report was {type(report_field).__name__}, not plain text")
                error_entry = {
                    "action": "ERROR",
                    "detail": (
                        f'FINISH rejected: "report" must be a single plain-text string, '
                        f"not a {type(report_field).__name__}. Write your findings as "
                        'one string (e.g. "- finding one [url]\\n- finding two [url]"), '
                        "not a nested object or list."
                    )
                }
                state.append(error_entry)
                emit(step, error_entry, None)
                continue

            findings_text = report_field.strip()
            if not findings_text:
                print("  [blocked] FINISH rejected — report was empty")
                error_entry = {
                    "action": "ERROR",
                    "detail": "FINISH rejected: the report was empty"
                }
                state.append(error_entry)
                emit(step, error_entry, None)
                continue

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

            cited_urls = extract_cited_urls(findings_text)
            read_but_not_quoted = sorted(valid_reads - cited_urls)
            if read_but_not_quoted:
                print(f"  [detail] read but not quoted in findings: {read_but_not_quoted}")

            state.append({"action": "FINISH", "reason": reason})
            report = f"{findings_text}\n\n{format_source_lists(state, cited_urls)}"
            emit(step, decision, report)
            break

        else:
            print(f"  [warning] unrecognized action: {decision}")
            error_entry = {
                "action": "ERROR",
                "detail": (
                    f"unrecognized decision: {decision}. Your reply must be a JSON "
                    'object with a top-level "action" key set to exactly "SEARCH", '
                    '"READ", or "FINISH" — plus "reason" and the one matching field: '
                    '"query" for SEARCH, "url" for READ, "report" for FINISH. No '
                    "other shape is accepted."
                )
            }
            state.append(error_entry)
            emit(step, error_entry, None)

    if report is None:
        report = "Step limit reached before finishing."

    print("\n--- FINAL REPORT ---")
    print(report)

    return state, report


# =============================================================================
# 7. run_evals (with SOURCES READ verification)
# =============================================================================

def run_evals(state, report):
    results = []
    # Check 5 below (SOURCES READ list matches actual valid reads) is now a
    # transitional double-check rather than load-bearing: format_source_lists
    # builds SOURCES READ straight from state, so this should always pass by
    # construction. Kept as a safety net while that change beds in — once
    # we're confident, this check can retire.

    # 1. Used the SEARCH tool
    used_search = any(s["action"] == "SEARCH" for s in state)
    results.append(("Used the search tool", used_search))

    # valid_reads: successfully READ this run, regardless of whether the
    # findings text ever cites them. This is the ground truth run_agent's
    # FINISH gate uses too ("read >= 3 distinct pages") — that gate runs
    # before the report (and thus citations) exist, so it can only ever
    # judge by "was it read", not "was it cited". Keep using valid_reads
    # wherever a check is describing that same read-time fact.
    valid_reads = {
        s["url"] for s in state
        if s["action"] == "READ" and not s["result"].startswith("[failed to read:")
    }

    # cited_urls / cited_valid_reads: unlike the FINISH gate, run_evals runs
    # after the report exists, so it can ask the stricter question — of the
    # pages actually read, which ones does the report's findings text cite.
    # A page that loaded fine but was never quoted (e.g. nav/boilerplate
    # noise) isn't a source that backs the report.
    findings_portion = report.split("SOURCES READ:")[0] if "SOURCES READ:" in report else report
    cited_urls = extract_cited_urls(findings_portion)
    cited_valid_reads = valid_reads & cited_urls

    # 2. Consulted more than one distinct source that the report actually
    # cites (cited_valid_reads) — not merely successfully READ (valid_reads).
    read_more_than_one = len(cited_valid_reads) > 1
    results.append(("Read more than one distinct valid source", read_more_than_one))

    # 3. Finished within the step limit (didn't trigger the fallback text)
    finished_in_time = report != "Step limit reached before finishing."
    results.append(("Finished within the step limit", finished_in_time))

    # 4. Produced actual content (not empty or unusually short)
    produced_report = len(report.strip()) > 50
    results.append(("Produced a non-trivial report", produced_report))

    # 5. SOURCES READ list matches the valid_reads that were actually cited
    # in the findings. SOURCES READ intentionally excludes a page that was
    # read successfully but never quoted (see format_source_lists), so the
    # ground truth to compare against is cited_valid_reads, not all of
    # valid_reads.
    expected_sources = cited_valid_reads

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

    sources_match = claimed_reads == expected_sources
    results.append(("SOURCES READ list matches actual valid reads", sources_match))

    # Read successfully but never cited in the findings — not a PASS/FAIL
    # check, just information for manual review (see sources_mismatch below).
    read_but_not_quoted = valid_reads - cited_urls

    # Print results
    print("\n--- EVAL RESULTS ---")
    passed = 0
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"{status}  {name}")

    print(f"\nScore: {passed} of {len(results)}")

    missing = expected_sources - claimed_reads
    extra = claimed_reads - expected_sources
    if missing:
        print(f"  [detail] actually read but missing from SOURCES READ: {missing}")
    if extra:
        print(f"  [detail] claimed in SOURCES READ but never actually read: {extra}")
    if read_but_not_quoted:
        print(f"  [detail] read but never quoted in the report: {read_but_not_quoted}")

    eval_summary = {
        "checks": [{"name": name, "passed": ok} for name, ok in results],
        "score": f"{passed} of {len(results)}",
        "sources_mismatch": {
            "missing_from_list": sorted(missing),
            "falsely_claimed": sorted(extra),
            "read_but_not_quoted": sorted(read_but_not_quoted)
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
