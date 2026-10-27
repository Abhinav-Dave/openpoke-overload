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

## 5. Honest limits

- I tested one model (Haiku 5.5) on made-up agents with single-turn questions. A weaker model can behave differently.
- In a large roster, the new version makes 0.2 to 0.35 more model calls per question. The model searches before it creates an agent.
- The difference in accuracy between the two versions is one or two points. That is inside the noise of 76 questions.
- Choosing agents for the bounded view re-reads every log on each message. It takes about 155 ms at 300 agents and 511 ms at 1,000. A cache keyed on log file time fixes this. I did not build it.
- The reminder check samples only the first 6 occurrences. Retry counts reset on restart. The trigger count check is not atomic.
- The digest names at most 20 emails.
- The real inbox is small and quiet, and the questions are templates.

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
