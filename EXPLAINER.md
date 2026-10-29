# Agent overload in OpenPoke: what I built, how I tested it, what is left

This is a take-home for General Magic. It is a fork of the open-source OpenPoke project. All numbers below come from files in `eval/results/`. You can rerun every offline check without any key.

## 1. The problem

The article names one limit: the whole list of execution agents goes into every prompt of the interaction agent. After heavy use, the list grows to hundreds of names and the model loses focus. I call this agent overload. When I read the code, I found that the same pattern shows up in eight more places:

- An execution agent reads its whole log on every reuse, and the log never shrinks.
- Nothing limits how many agents run at once. Twenty due reminders start twenty agents together.
- A flood of important emails starts one interaction run per email.
- Chat messages can overlap, so two runs write to the same history.
- A reminder can repeat every second.
- A model call that fails once is lost. Nothing counts cost.
- If the classifier fails, the email watcher still marks the mail as seen. The mail is lost.
- When the manager loop hits its step limit, the user gets silence.

On Windows the app did not start, because `fcntl` exists only on Unix. Also, a crash during a roster save can empty the roster file. I fixed both.

## 2. What I built

Governor (limits on work):
- One shared cap on parallel agents (default 5) and a cap of 4 agents per message.
- One interaction run at a time per chat, with a queue of 10. A longer queue gets an HTTP 429 reply.
- A reminder floor of 300 seconds, 20 reminders per agent, and 100 in total.
- Retries with backoff for 429 and 5xx errors, token counting, and a `GET /api/v1/meta/load` view.
- When the manager loop gives up, the user gets a plain-language reply.

Context diet (smaller prompts):
- An execution agent now sees only the newest 20 log entries (12,000 characters at most) plus a header line.
- Above 200 agents, the interaction prompt shows the 8 best matches for the new message, the 3 most recently used agents, and a names-only tail. Below 200 agents the prompt is identical to the original.
- A `find_agents` tool searches the full roster. The code refuses to create a new agent in a large roster until `find_agents` ran in that turn. This rule exists because a live test showed that a prompt hint alone did not stop the model from making duplicates.

Email digest:
- One interaction run per poll. Up to 3 urgent emails go out alone and the rest go in one digest.
- If the classifier fails, the email stays unseen for up to 3 attempts.

## 3. How I tested it

Everything runs offline first, with a fake model and a fake Gmail, and the live model only checks the result at the end.

- `pytest`: 232 tests. I wrote the tests before the code.
- `eval/baseline.py` measures the original and the new code on the same scenarios.
- `eval/regression_eval.py` runs 7 regression checks and a table of known gaps. The new code passes 7 of 7.
- `eval/routing_eval.py` uses 150 made-up agents and 133 labelled questions, split into a tuning set and a test set. It checks whether the right agent is still visible.
- `eval/live_eval.py` runs the real interaction agent with the real model (Haiku 5.5) on the original code and the new code. It has a dollar cap and runs as a dry run unless you add `--run`.
- `eval/real_gmail_check.py` reads a real inbox with a guard that blocks every call except fetch and profile. Tests prove that send, draft, reply, forward, delete and label calls fail before any network call.

## 4. Results

Offline, original against new:

| Measure | Original | New |
|---|---|---|
| Execution prompt, 200 log entries (characters) | 79,381 | 11,666 |
| Peak parallel agents, 20 due reminders | 20 | 5 |
| Peak parallel agents, 8 started by one message | 8 | 4 |
| Model calls for 10 / 50 important emails | 20 / 100 | 11 / 51 |
| Overlapping chat runs, 5 fast messages | 5 | 1 |
| Emails lost when the classifier fails on 30% of 50 emails | 15 | 0 |

Live model, 76 test questions, 2 repeats each:

| Roster | Version | Correct | Duplicates | Input tokens per question |
|---|---|---|---|---|
| 300 | Original | 99% | 0% | 21,083 |
| 300 | New | 98% | 0% | 17,487 (17% fewer) |
| 600 | Original | 99% | 0% | 33,785 |
| 600 | New | 97% | 1% | 19,238 (43% fewer) |

Real inbox, 40 emails, messy agent names, 300 agents (40 questions, 2 repeats):

| Version | Correct | Duplicates | Input tokens per question |
|---|---|---|---|
| Original | 93% | 6% | 26,841 |
| New | 95% | 3% | 21,832 |

The real classifier marked 6 of the 40 emails as important. One poll causes 6 interaction runs in the original and 3 in the new code. I did the real inbox read once, it was read-only, and I disconnected Gmail afterward.

### What the live test taught me

My first version of the roster fix was worse than the original. At 150 agents it scored 65% on reworded questions and made duplicates 4% of the time. I raised the threshold to 200 agents, added the names tail, and then added the find-before-create rule in code. The final numbers above come from that corrected version. The offline evaluator alone did not catch this. Only the real model did.

## 4b. Where each number comes from

Every table above comes from a file in `eval/results/`. You rebuild a number by running the script in the second column. The "runs" column counts each time a question is asked, so a question asked twice counts twice.

| Claim | Result file | Script | Model | Runs |
|---|---|---|---|---|
| Prompt sizes, calls for email floods, peak parallel agents, overlapping chats | `phase0-original.json` to `phase9-cleanup.json`, `baseline.json` | `eval/baseline.py` | fake | none |
| 7 of 7 old behaviors unchanged, and the table of gaps | `regression_eval.json` | `eval/regression_eval.py` | fake | none |
| Is the right agent visible in the prompt, and how big the prompt gets from 10 to 1,000 agents | `routing_eval.json` | `eval/routing_eval.py` | none | 133 questions |
| Accuracy at 150 agents (the first, flawed version) | `live_eval_150.json` | `eval/live_eval.py` | real, Haiku 5.5 | 76 questions x 2 repeats |
| Accuracy at 300 and 600 agents, before the extra rule | `live_eval_300_600.json` | `eval/live_eval.py` | real | 152 per size per version |
| Accuracy at 300 and 600 agents, final version | `live_eval_300_600_gate.json` | `eval/live_eval.py` | real | 152 per size |
| Speed at 300 agents | `live_eval_latency_300.json` | `eval/live_eval.py` | real | 36 questions x 2 repeats, per version |
| Real inbox: importance count and messy-name test | not committed, kept private | `eval/real_gmail_check.py` and `eval/live_eval.py --real-file` | real | 40 emails, 40 questions x 2 |

The tests themselves are in `tests/`. There are 233 tests in 23 files, and they use a fake model and a fake Gmail. Run them with `python -m pytest -q`.

### The 76 questions

The file `eval/datasets/routing_queries.json` holds 133 questions about 150 made-up agents in `eval/datasets/routing_agents.json`. A coding agent that I directed wrote them. They do not come from real users. Each question has an expected answer: an agent name, or "none, make a new one". I fixed the split by position before I scored anything. 57 questions are for tuning and 76 are for the final report. The 76 test questions fall into six groups:

| Group | Count | Example | Right answer |
|---|---|---|---|
| direct | 14 | "Triage the new bug reports from today" | the bug-triage agent |
| paraphrase (reworded) | 13 | "Set up my weekly chat with my counsellor" | the agent for counseling sessions, though no word matches |
| disambiguation | 18 | "Draft a reminder for Acme's unpaid invoice" | the Acme invoice agent, not the Globex one |
| follow-up | 10 | "do that again" | the agent used just before |
| no match | 12 | "Research electric bikes for commuting" | create a new agent |
| off-topic | 9 | "tell me a joke" | no agent |

### What counts as correct

For a question with an expected agent, the run is correct if the manager's accepted choice is that agent. A clone of that agent also counts, because the roster padding adds look-alikes with a number suffix. For a no-match or off-topic question, the run is correct if the manager creates a new agent or calls none. A duplicate means the manager created a new agent although the expected one existed. A wrong reuse means it picked a different existing agent.

### Do these tests measure the right thing?

They measure routing: does the manager pick the right worker without making a duplicate, and what does that cost in tokens, calls and seconds? That is the thing the roster change touches, so the test fits the change. They do not measure whether a worker then does its job well. I stubbed that part out because I did not change it. They also use made-up agents with tidy names, single-turn questions, one model, and a small sample. The real-inbox run partly covers the messy-name gap, but only at 40 questions.

## 5. Honest limits

- I tested one model (Haiku 5.5) on made-up agents with single-turn questions. A weaker model can behave differently.
- In a large roster, the new version makes 0.2 to 0.35 more model calls per question. The model searches before it creates an agent.
- The difference in accuracy between the two versions is one or two points. That is inside the noise of 76 questions.
- The new version is a little slower end to end. In a live run at 300 agents (36 questions asked twice, 72 runs per version), a question took 4.36 seconds on average on the original and 4.79 seconds on the new version. The new prompts are 16% shorter, but the extra search call costs more time than the shorter prompt saves. The gap is smaller than the spread inside each version, so treat it as a hint and not as proof. Nearly all of the slowdown is in questions that need a new agent.
- Choosing agents for the bounded view takes about 160 ms per message at 300 agents, 580 ms at 1,000, and 1.9 seconds at 3,000, measured on Windows. Reading file details costs most of that time. Tokenizing and scoring cost a few milliseconds. A cache of log contents saves about 70%, but it adds stale-data risk, so I did not build it. A single directory scan is the better fix if rosters reach 1,000 agents.
- The reminder check samples only the first 6 occurrences. Retry counts reset on restart. The trigger count check is not atomic.
- The digest names at most 20 emails.
- The real inbox is small and quiet, and the questions are templates.

### Ideas I measured and chose not to build

- Merging near-duplicate agents. A merge must move logs, rewrite reminder rows, and keep old names working across three stores that share no transaction. A word-overlap rule can only separate real duplicates from agents that are meant to be separate (one invoice agent per client) when it compares names, and the margin is thin. With the find-before-create rule on, the live tests showed 0 duplicates out of 152 at 300 agents and 1 out of 152 at 600. A wrong merge sends a request into the wrong history, which is worse than a duplicate.
- A cache for agent selection. See the latency limit above.

### Two ideas I built and tested after the main work

Prompt caching (kept). The model client now asks OpenRouter to cache the unchanging start of each prompt (the system prompt and tool list) for Anthropic models. Setting `OPENPOKE_PROMPT_CACHE=0` turns it off. If a provider rejects the request, the code retries once without it. In a live run of 72 questions at 300 agents, the provider served 9,755 of 17,608 input tokens per question from the cache (55%). In a second run right after, it served 15,526 of 17,557 (88%), because the cache was already warm. A cached token costs a tenth of a normal one, so this is expected to cut input cost roughly in half. I did not measure dollars per version, so treat that as an estimate. The run also kept accuracy at 100% on those 72 questions.

Server-side auto-search (built, measured, removed). The idea: when the roster scorer says nothing fits, the server runs the agent search itself, so the model skips the extra call that makes new-topic questions slow. I built it with tests, and two live runs showed no change. New-topic questions still took 3.3 calls and about 6.4 seconds, against 1.9 calls and 4.2 seconds on the original. I found two reasons.
1. The "nothing fits" signal is rare. Only 1 of 12 new-topic questions triggered it, because common words such as "find" and "plan" match some agent name.
2. The extra search call is not wasted work. When the model searches, it picks its own keywords, and it often tries synonyms. That is how it finds a reworded match that the scorer missed. A server-side search on the user's own words cannot do that. This is the same reason the first version of the roster fix, which had no forced search, made duplicates.
I reverted the change (commit `c0460eb`) and kept the two result files, `eval/results/live_eval_latency_300_v2.json` and `_v3.json`, as the evidence. The slower speed on new-topic questions is the price of avoiding duplicates.

## 6. Other gaps I found

I fixed these in code:
- The Windows start-up failure and the roster corruption.
- The dead model name and the Composio calls that changed.
- Log file names that collided.
- Lost emails after a classifier failure.

I flag these without code:
- There is no login, CORS is open, and one route deletes data.
- Email text goes into the prompt, so a hostile email can try to steer the agent (prompt injection). The rule "ask before sending" exists only in the prompt, not in code.
- Gmail calls block the event loop.
- The UI hides agent status, and a 30-second poll hides slowness.
- No cap exists on model calls per message beyond the 8-step loop limit.

## 7. How to rerun

```
pip install -r server/requirements.txt pytest pytest-asyncio
python -m pytest -q
python eval/baseline.py
python eval/regression_eval.py
python eval/routing_eval.py
python eval/live_eval.py            # dry run, prints a cost estimate
```

Settings for every limit are in `.env.example`.
