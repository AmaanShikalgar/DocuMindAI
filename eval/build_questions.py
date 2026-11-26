"""Source of truth for eval/questions.jsonl.

Questions are written here with their evidence quotes; this script finds the page each quote is
on (so page numbers can't drift) and refuses to write the file if a quote isn't found verbatim.

    python eval/build_questions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

from documind.pdf import parse_pdf  # noqa: E402
from run_eval import locate  # noqa: E402

NIST, PAPER, IRS = "nist-csf-2.0.pdf", "lost-in-the-middle.pdf", "irs-p463-2025.pdf"

# (doc, question, reference answer, [verbatim evidence quotes]); no quotes = unanswerable
Q = [
    # ---- NIST Cybersecurity Framework 2.0 ----
    (NIST, "What are the CSF Functions?", "There are six: Govern, Identify, Protect, Detect, Respond, and Recover.",
     ["There are six CSF Functions: Govern, Identify, Protect, Detect, Respond, and Recover."]),
    (NIST, "What are the four CSF Tiers called?", "Partial (Tier 1), Risk Informed (Tier 2), Repeatable (Tier 3), and Adaptive (Tier 4).",
     ["There are four Tiers: Partial (Tier 1), Risk Informed (Tier 2), Repeatable (Tier 3), and Adaptive (Tier 4)."]),
    (NIST, "What is a Community Profile?", "A baseline of CSF outcomes created and published to address shared interests and goals among a number of organizations.",
     ["A Community Profile is a baseline of CSF outcomes that is created and published to address shared interests and goals among a number of organizations."]),
    (NIST, "What does a Current Profile describe?", "The Core outcomes an organization is currently achieving (or attempting to achieve) and how or to what extent each is being achieved.",
     ["A Current Profile specifies the Core outcomes that an organization is currently achieving (or attempting to achieve)"]),
    (NIST, "What kinds of changes should a Target Profile anticipate?", "Anticipated changes to the organization's cybersecurity posture, such as new requirements, new technology adoption, and threat intelligence trends.",
     ["A Target Profile considers anticipated changes to the organization’s cybersecurity posture, such as new requirements, new technology adoption, and threat intelligence trends."]),
    (NIST, "What are Quick-Start Guides?", "Brief documents on specific CSF-related topics, often tailored to specific audiences.",
     ["Quick-Start Guides (QSGs) are brief documents on specific CSF-related topics and are often tailored to specific audiences."]),
    (NIST, "What do Informative References point to?", "Sources of guidance on each outcome from existing global standards, guidelines, frameworks, regulations, policies, etc.",
     ["Informative References that point to sources of guidance on each outcome from existing global standards, guidelines, frameworks, regulations, policies"]),
    (NIST, "What is the outcome of the Cybersecurity Supply Chain Risk Management category (GV.SC)?",
     "Cyber supply chain risk management processes are identified, established, managed, monitored, and improved by organizational stakeholders.",
     ["Cyber supply chain risk management processes are identified, established, managed, monitored, and improved"]),
    (NIST, "What does subcategory PR.AA-02 say?", "Identities are proofed and bound to credentials based on the context of interactions.",
     ["Identities are proofed and bound to credentials based on the context of interactions"]),
    (NIST, "Which subcategory covers monitoring external service providers?", "DE.CM-06: external service provider activities and services are monitored to find potentially adverse events.",
     ["DE.CM-06: External service provider activities and services are monitored to find potentially adverse events"]),
    (NIST, "What does RC.RP-02 require for recovery actions?", "Recovery actions are selected, scoped, prioritized, and performed.",
     ["Recovery actions are selected, scoped, prioritized, and performed"]),
    (NIST, "How is cybersecurity risk strategy applied at Tier 1 for governance?", "It is managed in an ad hoc manner.",
     ["Application of the organizational cybersecurity risk strategy is managed in an ad hoc manner."]),
    (NIST, "What is the first step in creating and using an Organizational Profile?", "Scope the Organizational Profile.",
     ["Scope the Organizational Profile."]),
    (NIST, "What is the purpose of the DETECT function?", "Possible cybersecurity attacks and compromises are found and analyzed.",
     ["Possible cybersecurity attacks and compromises are found and analyzed"]),
    # ---- Lost in the Middle (Liu et al., TACL 2024) ----
    (PAPER, "Where in the input context do language models use relevant information best?",
     "At the very beginning (primacy bias) or the end (recency bias) of the input context; performance is a U-shaped curve.",
     ["language model performance is highest when relevant information occurs at the very beginning (primacy bias) or end of its input context (recency bias)"]),
    (PAPER, "What was GPT-3.5-Turbo's closed-book accuracy on multi-document QA?", "56.1%.",
     ["when predicting without any documents (i.e., the closed-book setting; 56.1%)"]),
    (PAPER, "How were the distractor documents collected?", "With a retrieval system: Contriever fine-tuned on MS-MARCO, keeping documents that don't contain the answer.",
     ["we use a retrieval system (Contriever, fine-tuned on MS-MARCO"]),
    (PAPER, "Which dataset is the multi-document QA task built from?", "NaturalQuestions-Open.",
     ["We instantiate this task with data from NaturalQuestions-Open"]),
    (PAPER, "What are the keys and values in the synthetic key-value retrieval task?", "128-bit UUIDs (unique, randomly generated).",
     ["All keys and values are 128-bit UUIDs."]),
    (PAPER, "How many key-value pairs were used in the key-value retrieval experiments?", "75, 140, and 300 key-value pairs, with 500 examples each.",
     ["We experiment with input contexts containing 75, 140, and 300 key-value pairs (500 examples each)."]),
    (PAPER, "How much does using 50 retrieved documents instead of 20 help?", "Only marginally: about 1.5% for GPT-3.5-Turbo and about 1% for Claude-1.3.",
     ["only marginally improves performance (∼1.5% for GPT-3.5-Turbo and ∼1% for claude-1.3)"]),
    (PAPER, "When are encoder-decoder models robust to the position of relevant information?", "Only when evaluated on sequences within their training-time sequence length.",
     ["but only when evaluated on sequences within its training-time sequence length"]),
    (PAPER, "What effect does query-aware contextualization have on the key-value retrieval task?", "It dramatically improves it: all models reach near-perfect performance at 75, 140 and 300 pairs.",
     ["all models achieve near-perfect performance on the 75, 140, and 300 key-value pair settings"]),
    (PAPER, "Does query-aware contextualization fix the position problem in multi-document QA?", "No, it does not substantially improve robustness to the position of relevant information in multi-document QA.",
     ["does not substantially improve robustness of language models to changing the position of relevant information in multi-document QA"]),
    (PAPER, "How do 7B Llama-2 models behave compared with the 13B and 70B models?", "The 7B models are solely recency-biased, while the 13B and 70B models show a U-shaped curve.",
     ["the 7B Llama-2 models are solely recency-biased, while the 13B and 70B models exhibit a U-shaped performance curve"]),
    (PAPER, "Which effect from psychology do the authors relate their findings to?", "The serial-position effect.",
     ["The serial-position effect plays a role in understanding how humans develop short- and long-term memory."]),
    (PAPER, "Which open models were evaluated?", "MPT-30B-Instruct and LongChat-13B (16K).",
     ["MPT-30B-Instruct, LongChat-13B (16K)"]),
    (PAPER, "In the open-domain QA case study, how does reader performance compare with retriever recall as more documents are added?",
     "Reader performance saturates long before retriever performance does, so readers don't use the extra context effectively.",
     ["reader model performance saturates long before retriever performance saturates"]),
    # ---- IRS Publication 463 (2025) ----
    (IRS, "What is the 2025 standard mileage rate for business use of a car?", "70 cents per mile.",
     ["For 2025, the standard mileage rate for the cost of operating your car for business use is 70 cents"]),
    (IRS, "What is the most I can deduct for business gifts to one person in a year?", "$25 per person per tax year.",
     ["You can deduct no more than $25 for business gifts you give directly or indirectly to each person during your tax year."]),
    (IRS, "What share of unreimbursed business meal costs can generally be deducted?", "50%.",
     ["you can generally deduct only 50% of the unreimbursed cost of your meals"]),
    (IRS, "What meal deduction percentage applies to workers under DOT hours of service limits?", "80% of business-related meal expenses.",
     ["you can deduct 80% of your business-related meal expenses"]),
    (IRS, "How long can a work assignment last and still count as temporary?", "It must realistically be expected to last (and actually last) 1 year or less.",
     ["realistically expected to last (and does in fact last) for 1 year or less"]),
    (IRS, "How long do I have to return excess reimbursements under an accountable plan?", "Within 120 days after the expense was paid or incurred.",
     ["You return any excess reimbursement within 120 days after the expense was paid or incurred."]),
    (IRS, "For foreign travel, how is \"one week\" defined?", "7 consecutive days.",
     ["One week means 7 consecutive days."]),
    (IRS, "When is a trip abroad lasting more than a week still treated as entirely for business?", "When less than 25% of the total time outside the U.S. was spent on nonbusiness activities.",
     ["You spent less than 25% of the total time you were outside the United States on nonbusiness activities."]),
    (IRS, "What is the daily deduction limit for business travel by cruise ship?", "Twice the highest federal per diem rate allowable at the time of travel.",
     ["The limit is twice the highest federal per diem rate allowable at the time of your travel."]),
    (IRS, "What is my tax home?", "Generally your regular place of business or post of duty, regardless of where your family home is.",
     ["your tax home is your regular place of business or post of duty, regardless of where you maintain your family home"]),
    (IRS, "If I have more than one regular place of business, which one is my tax home?", "Your main place of business.",
     ["If you have more than one regular place of business, your tax home is your main place of business."]),
    (IRS, "Who can use Form 2106?", "Only Armed Forces reservists, qualified performing artists, fee-basis state or local government officials, and employees with impairment-related work expenses.",
     ["Form 2106 is only used by Armed Forces reservists, qualified performing artists, fee-basis state or local government officials, and employees with impairment-related work expenses"]),
    (IRS, "Which items under $4 don't count toward the gift limit?", "Items costing $4 or less that have your name clearly and permanently imprinted and are one of many identical items you widely distribute, like pens.",
     ["Has your name clearly and permanently imprinted on the gift"]),
    (IRS, "Is a job I expect to last under a year always treated as temporary?",
     "Generally yes: an assignment expected to last 1 year or less is temporary unless facts and circumstances indicate otherwise.",
     ["If you expect an assignment or job to last for 1 year or less, it is temporary unless there are facts and circumstances that indicate otherwise."]),
    # ---- unanswerable: the right behaviour is to abstain ----
    (NIST, "How much does NIST charge for CSF 2.0 certification?", "", []),
    (NIST, "Which countries require CSF 2.0 by law?", "", []),
    (PAPER, "How did Gemini perform on the multi-document QA task?", "", []),
    (PAPER, "How many GPU hours did the experiments take?", "", []),
    (IRS, "What is the per diem lodging rate for Honolulu in 2025?", "", []),
    (IRS, "What is the standard mileage rate for 2030?", "", []),
]


def main() -> None:
    pages = {}
    for doc in {d for d, *_ in Q}:
        pages[doc] = [p.text for p in parse_pdf((HERE / "docs" / doc).read_bytes()).pages]
    rows, bad = [], 0
    for i, (doc, question, answer, quotes) in enumerate(Q, 1):
        ev = []
        for quote in quotes:
            found = [n for n, t in enumerate(pages[doc], 1) if locate(t, quote)]
            if not found:
                print(f"q{i:02d}: NOT FOUND in {doc}: {quote[:80]!r}")
                bad += 1
                continue
            # a passage repeated on several pages (summary tables, appendices) counts wherever it's retrieved
            ev += [{"page": n, "quote": quote} for n in found]
        row = {"id": f"q{i:02d}", "doc": doc, "question": question}
        if quotes:
            row.update(answer=answer, evidence=ev)
        else:
            row.update(answer="", answerable=False)
        rows.append(row)
    if bad:
        sys.exit(f"{bad} quotes not found; questions.jsonl not written")
    (HERE / "questions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    print(f"wrote {len(rows)} questions ({sum(bool(q[3]) for q in Q)} answerable)")


if __name__ == "__main__":
    main()
