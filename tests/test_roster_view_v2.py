import os

import pytest

from server.agents.interaction_agent.agent import prepare_message_with_history
from server.services.execution import get_agent_roster, get_execution_agent_logs, roster_view
from server.services.execution.roster_view import select_agents

AGENTS = [
    ("invoice-followups-acme", "Chase Acme Corp about the unpaid March invoice and draft a polite reminder"),
    ("flight-booking-tokyo", "Find flights to Tokyo in November and watch for price drops"),
    ("newsletter-digest", "Summarize my newsletters every morning into five bullet points"),
    ("dentist-appointment", "Schedule a dentist cleaning for next month and add to calendar"),
    ("landlord-repairs", "Email the landlord about the broken kitchen faucet and track replies"),
    ("job-applications-stripe", "Track my Stripe job application and watch for recruiter emails"),
    ("tax-documents", "Gather tax documents from email: W2, 1099 and receipts"),
    ("gym-membership-cancel", "Cancel my gym membership and confirm the refund"),
    ("mom-birthday-gift", "Plan a birthday gift for mom, find ideas under 50 dollars"),
    ("package-tracking-amazon", "Track my Amazon package delivery and alert on delays"),
    ("mortgage-refinance", "Compare mortgage refinance offers from three banks"),
    ("car-insurance-renewal", "Get quotes for car insurance renewal before the policy expires"),
    ("wedding-rsvp-tracker", "Track wedding RSVP replies and count the guests"),
    ("visa-application-canada", "Track Canada visa application status and required documents"),
]
NAMES = [n for n, _ in AGENTS]


@pytest.fixture
def seeded():
    roster, logs = get_agent_roster(), get_execution_agent_logs()
    base = 1_700_000_000
    for i, (name, request) in enumerate(AGENTS):
        roster.add_agent(name)
        logs.record_request(name, request)
        os.utime(logs._log_path(name), (base + i, base + i))
    return NAMES


def block(text):
    content = prepare_message_with_history(text, "", message_type="user")[0]["content"]
    return content.split("<active_agents>\n", 1)[1].split("\n</active_agents>", 1)[0]


@pytest.mark.parametrize("vague", ["send it", "do that again", "yes", "ok", "", "go on please"])
def test_vague_messages_never_get_the_no_match_line(seeded, vague):
    shown, _, no_match = select_agents(vague, seeded)
    assert not no_match
    assert "No listed agent fits" not in block(vague)
    # hot agents stay listed
    assert {n for n in NAMES[-roster_view.HOT_N:]} <= {a.name for a in shown}


def test_clearly_new_topic_still_gets_no_match(seeded):
    assert select_agents("what is the weather like in Paris", seeded)[2]


def test_one_content_token_is_vague_two_is_not(seeded):
    assert not select_agents("zebra", seeded)[2]
    assert select_agents("zebra giraffe", seeded)[2]


def test_coverage_rule_flags_partial_match_with_good_score(seeded):
    # "tokyo" alone scores high on flight-booking-tokyo, but 1 of 4 query words is covered.
    q = "tokyo zebra giraffe walrus"
    assert not select_agents(q, seeded, min_coverage=0.0)[2]
    assert select_agents(q, seeded, min_coverage=0.5)[2]


def test_coverage_rule_keeps_a_well_covered_query(seeded):
    q = "tokyo flights november"
    assert not select_agents(q, seeded, min_coverage=0.9)[2]


def test_coverage_default_is_a_fraction():
    assert 0.0 <= roster_view.MIN_COVERAGE <= 1.0
    assert roster_view.MIN_CONTENT_TOKENS == 2


@__import__("pytest").fixture(autouse=True)
def _bound_at_eleven(monkeypatch):
    # these tests exercise the bounded view on small rosters; production activates it above 200 agents
    from server.services.execution import roster_view as _rv

    monkeypatch.setattr(_rv, "BOUND_ABOVE", 11)
