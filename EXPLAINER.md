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
- Above 1,000 agents, `find_agents` also ranks agents by meaning with an embedding model. It merges that list with the word-search list by reciprocal rank fusion: each agent gets 1 / (60 + its place) from each list, and the sums set the order. If the embedding call fails, `find_agents` returns exactly the old word-search result. `OPENPOKE_EMBED=0` does the same. The bounded view in the prompt still uses word search only.

Email digest:
- One interaction run per poll. Up to 3 urgent emails go out alone and the rest go in one digest.
- If the classifier fails, the email stays unseen for up to 3 attempts.

## 3. How I tested it

Everything runs offline first, with a fake model and a fake Gmail, and the live model only checks the result at the end.

- `pytest`: 320 tests. I wrote the tests before the code.
- `eval/baseline.py` measures the original and the new code on the same scenarios.
- `eval/regression_eval.py` runs 7 regression checks and a table of known gaps. The new code passes 7 of 7.
- `eval/routing_eval.py` uses 150 made-up agents and 133 labelled questions, split into a tuning set and a test set. It checks whether the right agent is still visible.
- `eval/find_eval.py` calls the real `find_agents` with word search alone and with embeddings, on 30 new reworded questions plus the older ones. It calls the embedding model only, never a chat model.
- `eval/live_eval.py` runs the real interaction agent with the real model (Haiku 5.5) on the original code and the new code. It has a dollar cap and runs as a dry run unless you add `--run`.
- `eval/real_gmail_check.py` reads a real inbox with a guard that blocks every call except fetch and profile. Tests prove that send, draft, reply, forward, delete and label calls fail before any network call.

## 4. Results

Offline, original against new:

| Measure | Original | New | Change | What it means |
|---|---|---|---|---|
| Execution prompt, 200 log entries (characters) | 79,381 | 11,666 | -85% | Good: an agent with a long history no longer sends all of it on every call, so cost stops growing with age. |
| Peak parallel agents, 20 due reminders | 20 | 5 | -75% | Good: a burst of reminders can no longer start 20 model-backed agents at once and hit rate limits. |
| Peak parallel agents, 8 started by one message | 8 | 4 | -50% | Good: one message can start only 4 agents, and the rest wait for a free slot. |
| Model calls for 10 / 50 important emails | 20 / 100 | 11 / 51 | -45% / -49% | Good: a flood of email costs about half as many model calls. |
| Overlapping chat runs, 5 fast messages | 5 | 1 | -80% | Good: replies no longer collide and write over each other. |
| Emails lost when the classifier fails on 30% of 50 emails | 15 | 0 | -100% | Good: a failed classifier call now retries the email instead of marking it seen and losing it. |

Live model, 76 test questions, 2 repeats each:

| Roster | Measure | Original | New | Change | What it means |
|---|---|---|---|---|---|
| 300 | Correct | 99% | 98% | -1 point | Neutral: the same accuracy within the noise of the test. |
| 300 | Duplicates | 0% | 0% | none | Good: the bounded view does not make the manager create copies of agents it can reuse. |
| 300 | Input tokens per question | 21,083 | 17,487 | -17% | Good: each question costs about a sixth less to read. |
| 300 | Seconds per question (separate run, 36 questions asked twice) | 4.36 | 4.79 | +10% | Slightly bad: the model sometimes searches the roster before it answers, which costs an extra call. This is the honest tradeoff for the smaller prompt. |
| 600 | Correct | 99% | 97% | -2 points | Slightly bad: the price of showing the model fewer names. It is small and I report it. |
| 600 | Duplicates | 0% | 1% | +1 point | Slightly bad: 1 question in 100 created a copy of an agent that existed. |
| 600 | Input tokens per question | 33,785 | 19,238 | -43% | Good: this is the main win, because the original prompt keeps growing with the roster and the new one stays flat. |

Real inbox, 40 emails, messy agent names, 300 agents (40 questions, 2 repeats):

| Measure | Original | New | Change | What it means |
|---|---|---|---|---|
| Correct | 93% | 95% | +2 points | Good: with realistic messy names the new version routes at least as well as the original. |
| Duplicates | 6% | 3% | -3 points | Good: half as many unneeded new agents, because the manager searches before it creates. |
| Input tokens per question | 26,841 | 21,832 | -19% | Good: cheaper on a real roster as well as the made-up one. |
| Interaction runs for one poll with 6 important emails | 6 | 3 | -50% | Good: the digest batches emails, so the user is not interrupted six times. |

The real classifier marked 6 of the 40 emails as important. I did this first real inbox read read-only, and I disconnected Gmail afterward.

### A bigger real-inbox test: 350 emails, 292 messy agents

The 40-email test above is small, so I repeated it with a much larger inbox. I read 350 emails from my own inbox (the last 90 days) with the same read-only guard. The guard allows only fetch and profile calls. The real classifier marked 33 of the 350 emails as important, and 13 of those as urgent. I built one agent per email, named the way a rushed user names things, such as the sender plus the subject. That gave 292 agents, padded to a roster of 300. There are 146 questions, such as "Did [sender] get back to me yet?". Each one comes from one known email, so the right agent is known without hand labelling. I ran each question twice on each version with the real model (Haiku 5.5). The run cost about $2.0, and the email read and classification cost $0.40 more.

| Measure | Original | New | Change | What it means |
|---|---|---|---|---|
| Correct | 70.2% | 95.5% | +25 points | Good: with realistic messy names the original often fails to find the right agent, and the new version finds it almost every time. |
| Duplicates (a new agent although one existed) | 26.8% | 0.7% | -26 points | Good: this is the main source of the gain. The new version must search before it creates an agent, and the original did not. |
| Wrong agent reused | 4.1% | 3.4% | -0.7 points | Neutral: about the same on both versions. |
| Input tokens per question | 34,737 | 29,431 | -15% | Good: each question costs about a seventh less to read. |
| Model calls per question | 2.38 | 2.66 | +12% | Slightly bad: the search step adds a call in about 7% of runs and a longer loop in some others. |
| Seconds per question (mean) | 5.3 | 5.5 | +4% | Slightly bad: a small cost for the extra search. |
| Interaction runs for one poll with 33 important emails | 33 | 4 | -88% | Good: 3 urgent emails go out alone and the rest go in one digest. |

The 12 questions with no matching agent were answered correctly by both versions. The raw results are kept private, because they contain real sender names. The scripts that made them are in the repo: `eval/real_gmail_check.py` (read-only fetch, classify, build) and `eval/live_eval.py --real-file`.

Read this result with care:

- The questions are templates built from sender names, so word search finds them more easily than a real user's wording will.
- Both repeats use the same 146 questions, so the sample is 146 questions and not 292.
- It is one inbox from one person, and one run per version.
- It is larger than the made-up roster result because this roster is almost all messy names. In the 40-email test, most of the 300 agents were tidy made-up ones, and the gap was 93% against 95%.

### The same overload scenarios with the real model

The table above uses a scripted stand-in for the model, so it shows what the code does when the model behaves in a fixed way. I also ran the same scenarios with the real model (Haiku 5.5, fake Gmail), one run per version. The script is `python eval/baseline.py --live`, and the results are in `live_baseline_orig.json` and `live_baseline_new.json`. The two runs cost $0.47 and $0.28 and used 3.9 million and 2.0 million input tokens.

| Measure | Original | New | Change | What it means |
|---|---|---|---|---|
| Interaction runs for 10 / 50 important emails | 12 / 43 | 1 / 1 | -92% / -98% | Good: one digest message replaces one manager run per email. |
| Model calls for 10 / 50 important emails | 101 / 201 | 12 / 51 | -88% / -75% | Good: this is where the cost saving comes from. |
| Seconds for 10 / 50 important emails | 107 / 322 | 32 / 149 | -70% / -54% | Good: the flood finishes faster because there are fewer runs to wait for. |
| Peak parallel agents, 20 due reminders | 20 | 5 | -75% | Good: the real model confirms the cap holds. |
| Peak overlapping runs, 5 chat messages at once | 5 | 1 | -80% | Good: chats now run one at a time. |
| Interaction prompt at 300 agents (tokens) | 5,943 | 4,687 | -21% | Good: a smaller prompt on every message. |
| Execution prompt, 200 log entries (tokens) | 19,845 | 2,916 | -85% | Good: the log tail cap works with the real model too. |

Two cautions. First, each number is one run, and the real model varies from run to run, so read the direction and not the exact digits. Second, the "8 agents in one message" test did not work with the real model. In both versions the model started none of the eight agents when I asked, so I do not use that row. The scripted version still covers it. In the full live run of the new code, the chat test read 0 because state from the earlier test leaked in. I ran it again alone (`--only burst_chat`) and it read 1, and the result file says so.

### What the live test taught me

My first version of the roster fix was worse than the original. At 150 agents it scored 65% on reworded questions and made duplicates 4% of the time. I raised the threshold to 200 agents, added the names tail, and then added the find-before-create rule in code. The final numbers above come from that corrected version. The offline evaluator alone did not catch this. Only the real model did.

### Embedding search in find_agents

Word search (BM25) ranks agents by the words they share with the keywords. It misses a request in other words, such as "chat with my counsellor" for the agent that books therapy sessions. Above 1,000 agents, `find_agents` now also ranks by meaning and merges the two lists. I measured this offline on the test split: 31 reworded questions (13 older and 18 new), 32 direct and look-alike questions, and 21 questions with no matching agent. Each question goes to `find_agents` as written. The script is `eval/find_eval.py`, and the results are in `find_eval.json`.

| Roster | Measure | BM25 | Hybrid | Change | What it means |
|---|---|---|---|---|---|
| 1,500 | Reworded questions with the right agent in the top 5 (31) | 19% | 81% | +61 points | Good: this is the gain the change is for. Word search finds 6 of 31, and the blend finds 25. |
| 1,500 | Reworded questions with the right agent first (31) | 19% | 42% | +23 points | Good, but partial: the right agent is often in the five but not first. The model sees all five. |
| 1,500 | Older reworded questions, top 5 (13) | 15% | 92% | +77 points | Good: the gain holds on questions written before this change. |
| 1,500 | New reworded questions, top 5 (18) | 22% | 72% | +50 points | Good: the gain also holds on the 18 new test questions. |
| 1,500 | Direct and look-alike questions with the right agent first (32) | 100% | 100% | none | Good: when the words match, the blend does not push the right agent down. |
| 1,500 | No-match questions that return no agent (21) | 29% | 29% | none | Neutral: the same 6 of 21 come back empty. Word search already returns some agent for the other 15, and the model must judge them. |
| 1,500 | Milliseconds per `find_agents` call | 544 | 1,598 | +194% | Bad: each search now waits about one second for the embedding call. |
| 150, blend forced on | Reworded questions with the right agent in the top 5 (31) | 39% | 94% | +55 points | Good: the gain does not come only from the clones in the 1,500-agent roster. |
| 150, blend forced on | Direct and look-alike questions with the right agent first (32) | 100% | 100% | none | Good: the same as at 1,500 agents. |

How I chose the settings. I wrote the rules in the docstring of `eval/find_eval.py` before the first real run.
- I tuned on the dev split at 1,500 agents. Two values of MIN_SIM tied: 0.25 and 0.40 both put the right agent in the top 5 for 82% of reworded questions. At 0.25, the empty rate for no-match questions fell from 29% to 18%. That is more than the 10-point limit, so the rule chose 0.40.
- Direct hit@1 stayed at 100% at every value. Because of that, I did not need the fallback rule that keeps the word-search top hit first.
- I ran the test split once. Reworded hit@5 gained 19 of 31 (the rule needs at least 3). Direct hit@1 lost 0 of 32 (the rule allows 1). Empty no-match answers fell by 0 of 21 (the rule allows 2). So `OPENPOKE_EMBED` stays on by default, with `OPENPOKE_EMBED_MIN_SIM=0.4`.
- The whole offline eval made 304 embedding calls with 56,818 tokens, for $0.0011. No hybrid row fell back to word search.

Read this result with care:
- A coding agent wrote all the questions, including the 30 new ones. They do not come from real users.
- I tuned MIN_SIM on the dev split. The sample is small: 31 reworded test questions.
- At 1,500 agents, most agents are numbered clones that copy the text of an original. A clone of the right agent counts as a hit, and the top five often holds several copies of one agent. This makes hit@5 at 1,500 agents a coarse measure. The 150-agent rows have no clones.
- Offline, each question goes to `find_agents` as written. In a live run, the model picks the keywords.

The live check does not show the offline gain. I ran the new code at 1,500 agents with the real model (Haiku 5.5), once with `OPENPOKE_EMBED=0` and once with `OPENPOKE_EMBED=1`. It used the 76 test questions plus the 18 new test questions, 2 repeats each, so 188 runs per setting. I left out the 12 new dev questions, because I tuned MIN_SIM on them. The cache was warm before the first question, and no search fell back to word search.

| Roster | Measure | BM25 | Hybrid | Change | What it means |
|---|---|---|---|---|---|
| 1,500 | Correct, all questions (188 runs) | 96.3% | 94.1% | -2.2 points | Slightly bad, but inside the noise: 7 misses against 11. |
| 1,500 | Correct, reworded questions (62 runs) | 91.9% | 87.1% | -4.8 points | Slightly bad: no live gain on the questions the change is for. |
| 1,500 | Correct, direct and look-alike questions (64 runs) | 100% | 98.4% | -1.6 points | Neutral: the one miss came in a run with no search, where both settings run the same code. |
| 1,500 | Duplicates (146 runs with an expected agent) | 0% | 0.7% | +0.7 points | Slightly bad: 1 run made a new yoga agent after a search. |
| 1,500 | Runs where the model called `find_agents` | 19.1% | 16.5% | -2.6 points | Neutral: the model searched in fewer than 1 run in 5. |
| 1,500 | Correct in runs with a search | 100% (36 runs) | 93.5% (31 runs) | -6.5 points | Slightly bad: 2 misses against 0. One of them is the same wrong agent that the word-search setting also chose without a search. |
| 1,500 | Input tokens per question | 19,962 | 19,589 | -2% | Neutral: the same prompt in both settings. |
| 1,500 | Seconds per question | 6.21 | 6.54 | +5% | Slightly bad: a search now also waits for the embedding call. |

Why the offline gain does not show up live:
- The model calls `find_agents` in fewer than 1 run in 5. In the other runs it picks from the bounded view and the names tail, which embeddings do not change. 16 of the 18 misses came from those runs, and there both settings run the same code, so those misses are model noise.
- When the model searches, it picks its own keywords and often tries synonyms. Word search then finds the agent more often than it does with the user's own words. With embeddings off, all 36 runs with a search were correct.

The two runs cost $0.42 and $0.41 by token count. The account-spend reading from OpenRouter was the same for both runs, so I do not trust it.

My reading: with the user's own words as keywords, the change helps `find_agents` a lot, and it does not hurt the questions whose words match. But this live test shows no gain for the user, and it adds about one second to each search. I kept `OPENPOKE_EMBED=1` as the default, because that was the rule I set before the runs, and the rule uses the offline test. The live sample is small: at most 36 runs with a search per setting. `OPENPOKE_EMBED=0` gives exactly the old behaviour. If you want the conservative choice, turn it off until a larger live test shows a gain.

### Word search against meaning search: how each works, and the tradeoffs

How the word search works today. It lives in `server/services/execution/roster_view.py` and uses only the standard library. For each agent, it builds one text. The text holds the agent name three times, so a name hit counts more than a stray word. It also holds the first request (cut to 100 characters) and the last 3 requests (300 characters each). It splits the user message into words and drops filler words such as "the", "send" and "find". It then scores every agent with BM25, a classic formula that rewards rare words and penalizes long texts. Above 200 agents, the prompt shows the 8 top scorers, the 3 most recently used agents, and a names-only tail. If the best score is below 2.0, the code says "no agent fits", and the find-before-create rule makes the model search before it creates anything.

How the meaning search works. Above 1,000 agents, `find_agents` also sends the keywords and the agent texts to an embedding model. The model turns each text into a list of 1,536 numbers (a vector), and texts with a similar meaning get close vectors. The code ranks agents by cosine similarity (how close two vectors point) and drops scores below 0.4. It then merges that list with the word-search list by reciprocal rank fusion, which uses only the rank positions. Vectors are cached on disk, and a changed agent log makes the code embed that agent again.

| | Word search (BM25) | Meaning search (embeddings) |
|---|---|---|
| Finds a reworded request ("chat with my counsellor" for a therapy agent) | Weak: 6 of 31 reworded test questions in the top 5 | Strong: 25 of 31 in the top 5 |
| Finds the right agent among look-alikes (Acme invoice against Globex invoice) | 32 of 32 first | 32 of 32 first, kept by the merge |
| Says "nothing fits" | Gives a clear signal, but only 6 of 21 no-match questions return nothing | The same 6 of 21, because a nearest neighbor always exists |
| Time per search at 1,500 agents | 544 ms | 1,598 ms |
| Cost | Zero | About $0.001 for the whole offline eval |
| Network and privacy | None, all local | The agent request text goes to the embedding provider |
| Works offline and in tests | Yes | Only with a fake embedder in tests |
| Failure mode | Misses a paraphrase | A timeout blocks the server for up to 5 seconds, then falls back to word search for 60 seconds |
| Disk | None | 12.4 MB of vectors at 1,500 agents |
| Easy to explain | Yes: you can see which words matched | No: a similarity score has no visible reason |
| Live result with the real model | 96.3% correct at 1,500 agents | 94.1%, no gain and inside the noise |

Pros of word search, as the default. It is free, fast, private, deterministic and easy to debug. It never hurts a question whose words match, and it kept duplicates at 0% in the live tests. The model also repairs its weak spot: when the model searches, it picks its own synonyms, so the paraphrase gap is mostly closed before the code sees the query.

Cons of word search. It cannot match two texts that share no word. The offline test shows this clearly, with 19% of reworded questions found in the top 5. A user who words a request in a new way, and a model that does not search, can still miss the right agent.

Pros of meaning search. It closes that gap offline, from 19% to 81% in the top 5. It does not push down the right agent for direct questions. It also fails safely, because any error returns the exact word-search result.

Cons of meaning search. It adds a network call, about one second per search, and a 12 MB cache. It sends the user's request text to a third party. It cannot say "no match" any better than word search. It blocks the event loop while it waits. It starts cold, because the first searches embed only 128 new agents each. In the live test it gave no gain. The model rarely calls `find_agents`. When the model does call it, the model already tries synonyms.

Why I used both, and why only in `find_agents`. Word search stays the default path because it is cheap enough to run on every message. Meaning search runs only inside `find_agents`, which is a rare call and has no strict time limit. I did not put it in the bounded view for three reasons. First, a network call would sit in front of every message. Second, the "nothing fits" rule depends on a BM25 score that merged ranks do not have. Third, the find-before-create rule already forces a search before a new agent exists, so the search fix covers the duplicate risk.

My recommendation. For a roster under 1,000 agents, word search is the right tool and meaning search never runs. Above that size, the offline result is strong and the live result is flat. The evidence supports keeping the feature built and tested. It does not prove that users gain from it. Set `OPENPOKE_EMBED=0` for the conservative choice until a larger live test with real user wording shows a gain.

## 4b. Where each number comes from

Every table above comes from a file in `eval/results/`. You rebuild a number by running the script in the second column. The "runs" column counts each time a question is asked, so a question asked twice counts twice.

| Claim | Result file | Script | Model | Runs |
|---|---|---|---|---|
| Prompt sizes, calls for email floods, peak parallel agents, overlapping chats | `phase0-original.json` to `phase9-cleanup.json`, `baseline.json` | `eval/baseline.py` | fake | none |
| The same scenarios with the real model | `live_baseline_orig.json`, `live_baseline_new.json` | `eval/baseline.py --live` | real, Haiku 5.5 | 1 run per version |
| 7 of 7 old behaviors unchanged, and the table of gaps | `regression_eval.json` | `eval/regression_eval.py` | fake | none |
| Is the right agent visible in the prompt, and how big the prompt gets from 10 to 1,000 agents | `routing_eval.json` | `eval/routing_eval.py` | none | 133 questions |
| Accuracy at 150 agents (the first, flawed version) | `live_eval_150.json` | `eval/live_eval.py` | real, Haiku 5.5 | 76 questions x 2 repeats |
| Accuracy at 300 and 600 agents, before the extra rule | `live_eval_300_600.json` | `eval/live_eval.py` | real | 152 per size per version |
| Accuracy at 300 and 600 agents, final version | `live_eval_300_600_gate.json` | `eval/live_eval.py` | real | 152 per size |
| Speed at 300 agents | `live_eval_latency_300.json` | `eval/live_eval.py` | real | 36 questions x 2 repeats, per version |
| Real inbox: importance count and messy-name test | not committed, kept private | `eval/real_gmail_check.py` and `eval/live_eval.py --real-file` | real | 40 emails, 40 questions x 2 |
| Larger real inbox: importance count and messy-name test | not committed, kept private | `eval/real_gmail_check.py` and `eval/live_eval.py --real-file` | real | 350 emails, 146 questions x 2 |
| `find_agents` with and without embeddings, offline | `find_eval.json` | `eval/find_eval.py` | embedding model only (`openai/text-embedding-3-small`) | 84 test and 61 dev questions per mode, plus 7 dev sweeps |
| Accuracy at 1,500 agents with embeddings off and on | `live_eval_1500_embed0.json`, `live_eval_1500_embed1.json` | `eval/live_eval.py --extra-queries` | real, Haiku 5.5 | 94 questions x 2 repeats per setting |

One caution about `routing_eval.json`. The committed file comes from the first version of the evaluator. A fresh run of `eval/routing_eval.py` on today's code gives different numbers. For example, visible recall on the test split is 1.0, not 0.855. The new embedding code does not cause this: a fresh run before and after the change gives the same numbers.

The tests themselves are in `tests/`. There are 320 tests in 26 files, and they use a fake model, a fake Gmail and a fake embedder. No test reaches the network. Run them with `python -m pytest -q`.

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

A second file, `eval/datasets/routing_paraphrase_queries.json`, holds 30 more reworded questions for the embedding test. It is a separate file. Because of that, a rerun of the older checks gives the older numbers. The rules for the 30 questions:
- A coding agent wrote them, the same way as the older questions.
- No content word of a question is a word of its agent's name.
- No question uses an agent that one of the 23 older reworded questions uses.
- The split is by position, the same as before: 12 dev and 18 test.

I froze the file before the first hybrid run and did not edit it after. Its sha256, with LF line endings as git stores it, is `6c718c0578e3a824b151401cc199a2cf80394793a44c731d292d2495fb8c043f`. With the 13 older reworded test questions, the test split has 31 reworded questions.

### What counts as correct

For a question with an expected agent, the run is correct if the manager's accepted choice is that agent. A clone of that agent also counts, because the roster padding adds look-alikes with a number suffix. For a no-match or off-topic question, the run is correct if the manager creates a new agent or calls none. A duplicate means the manager created a new agent although the expected one existed. A wrong reuse means it picked a different existing agent.

### Do these tests measure the right thing?

They measure routing: does the manager pick the right worker without making a duplicate, and what does that cost in tokens, calls and seconds? That is the thing the roster change touches, so the test fits the change. They do not measure whether a worker then does its job well. I stubbed that part out because I did not change it. They also use made-up agents with tidy names, single-turn questions, one model, and a small sample. The real-inbox run partly covers the messy-name gap, but only at 40 questions.

The embedding test has one more gap. The offline check gives `find_agents` the user's own words. In a live run the model picks its own keywords, and it often tries synonyms, so word search does better live than offline. The live run is the fairer measure of what a user gets.

## 5. Honest limits

- I tested one model (Haiku 5.5) on made-up agents with single-turn questions. A weaker model can behave differently.
- In a large roster, the new version makes 0.2 to 0.35 more model calls per question. The model searches before it creates an agent.
- The difference in accuracy between the two versions is one or two points. That is inside the noise of 76 questions.
- The new version is a little slower end to end. In a live run at 300 agents (36 questions asked twice, 72 runs per version), a question took 4.36 seconds on average on the original and 4.79 seconds on the new version. The new prompts are 16% shorter, but the extra search call costs more time than the shorter prompt saves. The gap is smaller than the spread inside each version, so treat it as a hint and not as proof. Nearly all of the slowdown is in questions that need a new agent.
- Choosing agents for the bounded view takes about 160 ms per message at 300 agents, 580 ms at 1,000, and 1.9 seconds at 3,000, measured on Windows. Reading file details costs most of that time. Tokenizing and scoring cost a few milliseconds. A cache of log contents saves about 70%, but it adds stale-data risk, so I did not build it. A single directory scan is the better fix if rosters reach 1,000 agents.
- The reminder check samples only the first 6 occurrences. Retry counts reset on restart. The trigger count check is not atomic.
- The digest names at most 20 emails.
- The real-inbox tests use one inbox from one person, and the questions are templates built from sender names.

Limits of the embedding search in `find_agents`:
- Agent text leaves the machine. With embeddings on, the `find_agents` keywords and the request text of every agent go to the embedding provider (OpenRouter, which passes them to OpenAI). The prompts already go to OpenRouter, but the bounded view sends only a few agents per message. Embedding search sends every agent over time. `OPENPOKE_EMBED=0` stops this. The server logs one line about it at the first call.
- A cold cache fills slowly. Each call embeds at most 128 new agents. A roster of 1,500 agents therefore needs 12 searches to give every agent a vector. Until then, the agents without a vector get word search only. In the eval, warming 1,500 agents took 13 calls and 35 seconds.
- The cache is one JSON file per model. When the text of an agent changes, the code rewrites the whole file. At 1,500 agents the file is 12.4 MB.
- The call blocks the event loop for up to 5 seconds (`TIMEOUT_S`), like the Gmail calls. After a failure, embedding search stays off for 60 seconds.
- Each search is slower. Above 1,000 agents, a `find_agents` call took 1,598 ms on average with embeddings and 544 ms without, offline.
- 13 of the 30 new questions share no word with their agent's name or requests. The other 17 share at least one request word, so word search has some chance on them.

### Ideas I measured and chose not to build

- Embeddings in the bounded view. With it, every message waits for one embedding call. One short input took 0.75 to 1.6 seconds in my tests. I ran `eval/routing_eval.py` again on today's code. At 1,000 agents, the bounded view already shows the right agent with its description for 80% of test questions. It shows the right name for 87%. The model calls `find_agents` for only some messages, so there the wait happens less often.

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
python eval/baseline.py --live --max-usd 1.5   # real model, needs OPENROUTER_API_KEY in the environment
python eval/regression_eval.py
python eval/routing_eval.py
python eval/live_eval.py            # dry run, prints a cost estimate
python eval/find_eval.py            # embedding model only, needs OPENROUTER_API_KEY, about 14 minutes and $0.002
OPENPOKE_EMBED=1 python eval/live_eval.py --trees new --sizes 1500 --extra-queries eval/datasets/routing_paraphrase_queries.json --run --max-usd 1.75 --out eval/results/live_eval_1500_embed1.json
```

For the last line, run it again with `OPENPOKE_EMBED=0` and `_embed0.json` to get the word-search side.

Settings for every limit are in `.env.example`.
