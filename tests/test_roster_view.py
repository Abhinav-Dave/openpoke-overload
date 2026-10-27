import os
import random

import pytest

from server.services.execution import get_agent_roster, get_execution_agent_logs
from server.services.execution import roster_view
from server.services.execution.roster_view import select_agents, tokenize

AGENTS = [
    ("invoice-followups-acme", "Chase Acme Corp about the unpaid March invoice and draft a polite reminder"),
    ("invoice-followups-globex", "Follow up with Globex on overdue invoice 4471 and ask for payment date"),
    ("flight-booking-tokyo", "Find flights to Tokyo in November and watch for price drops"),
    ("flight-booking-berlin", "Search flights to Berlin for the conference week"),
    ("newsletter-digest", "Summarize my newsletters every morning into five bullet points"),
    ("dentist-appointment", "Schedule a dentist cleaning for next month and add to calendar"),
    ("landlord-repairs", "Email the landlord about the broken kitchen faucet and track replies"),
    ("job-applications-stripe", "Track my Stripe job application and watch for recruiter emails"),
    ("job-applications-notion", "Track my Notion application and draft thank-you note after interview"),
    ("team-standup-notes", "Collect standup updates from teammates and send a daily summary"),
    ("tax-documents", "Gather tax documents from email: W2, 1099 and receipts"),
    ("gym-membership-cancel", "Cancel my gym membership and confirm the refund"),
    ("mom-birthday-gift", "Plan a birthday gift for mom, find ideas under 50 dollars"),
    ("conference-speaker-outreach", "Email speakers about the spring conference schedule"),
    ("package-tracking-amazon", "Track my Amazon package delivery and alert on delays"),
    ("mortgage-refinance", "Compare mortgage refinance offers from three banks"),
    ("car-insurance-renewal", "Get quotes for car insurance renewal before the policy expires"),
    ("school-pickup-schedule", "Coordinate school pickup rotation with other parents"),
    ("podcast-guest-booking", "Invite guests for the podcast and schedule recording slots"),
    ("receipts-expense-report", "Collect receipts and build the monthly expense report"),
    ("wedding-rsvp-tracker", "Track wedding RSVP replies and count the guests"),
    ("apartment-hunting-brooklyn", "Watch listing emails for two bedroom apartments in Brooklyn"),
    ("vendor-contract-review", "Review the vendor contract renewal terms and flag auto-renew clauses"),
    ("github-pr-reviews", "Summarize GitHub pull request review requests assigned to me"),
    ("customer-support-escalations", "Triage support escalations from enterprise customers"),
    ("recipe-meal-plan", "Plan weekly meals and compile a grocery list from recipes"),
    ("visa-application-canada", "Track Canada visa application status and required documents"),
    ("board-meeting-prep", "Prepare the board meeting agenda and collect slide decks"),
    ("book-club", "Pick next book club read and email the members"),
    ("energy-bill-utilities", "Review electricity bill and compare utility providers"),
    ("hotel-booking-tokyo", "Book a hotel in Tokyo near Shibuya for the November trip"),
    ("press-inquiries", "Respond to journalist inquiries and log press mentions"),
]
NAMES = [n for n, _ in AGENTS]

QUERIES = [
    ("did Acme pay the invoice yet", "invoice-followups-acme"),
    ("send Globex a reminder about the invoice", "invoice-followups-globex"),
    ("any cheaper flights to Tokyo", "flight-booking-tokyo"),
    ("Berlin flight options", "flight-booking-berlin"),
    ("summarise my newsletters", "newsletter-digest"),
    ("move the dentist appointment", "dentist-appointment"),
    ("the faucet is still leaking, ping the landlord", "landlord-repairs"),
    ("status of my Stripe application", "job-applications-stripe"),
    ("write a thank you note to Notion after the interview", "job-applications-notion"),
    ("what did the team say in standup today", "team-standup-notes"),
    ("where are my 1099 and W2 forms", "tax-documents"),
    ("did the gym refund go through", "gym-membership-cancel"),
    ("gift ideas for mom", "mom-birthday-gift"),
    ("is my Amazon package delayed", "package-tracking-amazon"),
    ("refinance rates from the banks", "mortgage-refinance"),
    ("renew car insurance quotes", "car-insurance-renewal"),
    ("how many guests replied to the wedding", "wedding-rsvp-tracker"),
    ("any new Brooklyn listings", "apartment-hunting-brooklyn"),
    ("book the hotel near Shibuya", "hotel-booking-tokyo"),
    ("check canada visa status", "visa-application-canada"),
    ("remind me about that thing with the contract", "vendor-contract-review"),
]
OFF_TOPIC = ["what is the weather like in Paris", "tell me a joke about cats", "play some jazz music"]


@pytest.fixture
def seeded():
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    base = 1_700_000_000
    for i, (name, request) in enumerate(AGENTS):
        roster.add_agent(name)
        logs.record_request(name, request)
        logs.record_agent_response(name, "Done, nothing else to report.")
        os.utime(logs._log_path(name), (base + i, base + i))  # later in the list = more recent
    return NAMES


def names(shown):
    return [a.name for a in shown]


def test_tokenize_lowercases_splits_and_drops_one_letter_tokens():
    assert tokenize("Invoice-Followups_ACME, a W2! x 4471") == ["invoice", "followups", "acme", "w2", "4471"]
    assert tokenize("") == []


def test_ranks_expected_agent_first_and_in_top_k(seeded):
    top1 = topk = 0
    for query, expected in QUERIES:
        shown, _, no_match = select_agents(query, seeded, k=3, hot_n=0)
        top1 += names(shown)[:1] == [expected]
        topk += expected in names(shown)
        assert not no_match, query
    n = len(QUERIES)
    print(f"\nlabelled set: {len(NAMES)} agents, {n} queries, top1={top1}/{n}, top3={topk}/{n}")
    assert top1 >= 0.85 * n
    assert topk == n


def test_near_duplicate_names_are_told_apart(seeded):
    shown, _, _ = select_agents("globex invoice", seeded, k=2, hot_n=0)
    assert names(shown)[0] == "invoice-followups-globex"
    shown, _, _ = select_agents("acme invoice", seeded, k=2, hot_n=0)
    assert names(shown)[0] == "invoice-followups-acme"


@pytest.mark.parametrize("query", OFF_TOPIC)
def test_off_topic_query_is_no_match(seeded, query):
    shown, hidden, no_match = select_agents(query, seeded)
    assert no_match
    assert len(shown) + hidden == len(NAMES)
    assert len(shown) <= roster_view.HOT_N  # only recently used agents are offered


def test_hot_agents_always_included_and_deduplicated(seeded):
    # last three in the list have the newest mtime; query matches one of them as well
    shown, hidden, _ = select_agents("press journalist inquiries", seeded, k=2, hot_n=3)
    got = names(shown)
    assert got[0] == "press-inquiries"
    assert set(NAMES[-3:]) <= set(got)
    assert len(got) == len(set(got))
    assert len(got) + hidden == len(NAMES)


def test_small_roster_returns_everything_names_only(seeded):
    few = NAMES[:11]
    shown, hidden, no_match = select_agents("anything at all", few, k=8, hot_n=3)
    assert names(shown) == few and hidden == 0 and not no_match
    assert all(a.description == "" for a in shown)


def test_empty_roster():
    assert select_agents("hello", []) == ([], 0, False)


def test_exact_name_outside_top_k_is_hidden_but_reachable_by_name(seeded):
    # Documented behaviour: hidden agents are not listed, the prompt says they can still be
    # reused by exact name, and send_message_to_agent looks agents up by name, not by the view.
    shown, hidden, _ = select_agents("gift ideas for mom", seeded, k=2, hot_n=0)
    assert "book-club" not in names(shown) and hidden == len(NAMES) - len(shown)
    assert "book-club" in get_agent_roster().get_agents()


def test_include_forces_named_agents_into_view(seeded):
    shown, _, _ = select_agents("unrelated words", seeded, k=2, hot_n=0, include=["book-club"])
    assert "book-club" in names(shown)


def test_description_is_first_request_truncated(seeded):
    logs = get_execution_agent_logs()
    logs.record_request("dentist-appointment", "A later request that must not become the description")
    shown, _, _ = select_agents("newsletters", seeded, k=1, hot_n=0)
    assert shown[0].description.startswith("Summarize my newsletters")
    get_agent_roster().add_agent("long-one")
    logs.record_request("long-one", "word " * 100)
    shown, _, _ = select_agents("zzz", seeded + ["long-one"], k=1, hot_n=0, include=["long-one"])
    assert len(shown[-1].description) <= roster_view.DESC_CHARS + 3


def test_recent_requests_are_searchable(seeded):
    get_execution_agent_logs().record_request("book-club", "Also arrange the lasagna potluck dinner")
    shown, _, no_match = select_agents("lasagna potluck", seeded, k=1, hot_n=0)
    assert names(shown) == ["book-club"] and not no_match


def test_accuracy_check_discriminates_a_random_scorer(seeded, monkeypatch):
    """Break-on-purpose guard: swapping in a random scorer must drop below the accuracy bar."""
    rng = random.Random(0)
    monkeypatch.setattr(roster_view, "_scores", lambda q, docs: [rng.random() + 1 for _ in docs])
    top1 = sum(names(select_agents(q, seeded, k=3, hot_n=0)[0])[:1] == [e] for q, e in QUERIES)
    assert top1 < 0.85 * len(QUERIES)


@__import__("pytest").fixture(autouse=True)
def _bound_at_eleven(monkeypatch):
    # these tests exercise the bounded view on small rosters; production activates it above 200 agents
    from server.services.execution import roster_view as _rv

    monkeypatch.setattr(_rv, "BOUND_ABOVE", 11)
