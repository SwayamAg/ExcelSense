"""The routing query suite.

Nine categories, as the specification requires: direct questions, paraphrases,
executive language, why questions, how questions, ranking questions, follow-ups,
ambiguous questions, and unsupported or out-of-domain questions.

Entries are (question, expected_problem_id). None means the question must not be
routed at all - it belongs in the refusal or ambiguity path.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# 1. Direct questions - the plain form of each analysis
# --------------------------------------------------------------------------
DIRECT = [
    ("How many policies are in the portfolio?", "PS-001"),
    ("What is the total APE of the portfolio?", "PS-003"),
    ("What is the total sum assured exposure?", "PS-004"),
    ("What is the overall lapse rate?", "PS-035"),
    ("How many claims have been made?", "PS-052"),
    ("What is the distribution of policies across policy statuses?", "PS-002"),
    ("What is the split between traditional and ULIP policies?", "PS-006"),
    ("How many distinct agents, branches and clients are there?", "PS-010"),
    ("What proportion of the portfolio is still actively paying premium?", "PS-011"),
    ("What is the total and average renewal premium due?", "PS-027"),
    ("What proportion of claims have been settled?", "PS-054"),
    ("What is the age profile of our customers?", "PS-077"),
    ("What is the gender split of our customers?", "PS-078"),
    ("Do we have any overseas policyholders?", "PS-087"),
    ("What data quality issues should I know about?", "PS-096"),
    ("Do the five datasets reconcile with each other?", "PS-095"),
    ("What questions can this data not answer?", "PS-098"),
    ("How much renewal premium is still outstanding?", "PS-048"),
    ("What are the leading causes of death in our claims?", "PS-053"),
    ("How long does it take to intimate and settle a claim?", "PS-056"),
]

# --------------------------------------------------------------------------
# 2. Paraphrases - the same analysis asked differently
# --------------------------------------------------------------------------
PARAPHRASE = [
    ("How big is the book?", "PS-001"),
    ("Total number of policies please", "PS-001"),
    ("What share of policies have lapsed?", "PS-035"),
    ("How much premium does the book represent?", "PS-003"),
    ("How much cover have we written altogether?", "PS-004"),
    ("Break the book down by status", "PS-002"),
    ("How many unit linked policies do we have?", "PS-006"),
    ("Which plan sells the most policies?", "PS-007"),
    ("Which plans bring in the most premium?", "PS-008"),
    ("What do our customers typically pay?", "PS-005"),
    ("How old are our policyholders?", "PS-077"),
    ("What is our renewal collection rate?", "PS-044"),
    ("How late do customers pay their renewals?", "PS-046"),
    ("What are people dying from?", "PS-053"),
    ("Which cities are our biggest markets?", "PS-086"),
    ("Which states write the most business?", "PS-084"),
    ("How many policies went in force each month?", "PS-032"),
    ("Are we writing bigger policies than before?", "PS-091"),
    ("Which customers hold more than one policy?", "PS-082"),
    ("How spread out is our branch network?", "PS-090"),
]

# --------------------------------------------------------------------------
# 3. Executive language
# --------------------------------------------------------------------------
EXECUTIVE = [
    ("Give me a portfolio health snapshot", "PS-009"),
    ("Brief me on the whole book", "PS-101"),
    ("Give me a complete business review across every domain", "PS-101"),
    ("Where is the business losing the most value?", "PS-099"),
    ("What are the three things I should act on this quarter?", "PS-102"),
    ("Where should we focus retention effort for the biggest impact?", "PS-100"),
    ("Where should we focus agent remediation effort?", "PS-076"),
    ("Build a value segmentation of the customer base", "PS-083"),
    ("Give me a full performance scorecard for every product", "PS-020"),
    ("Build a composite quality score for every agent", "PS-073"),
]

# --------------------------------------------------------------------------
# 4. Why questions - diagnostic
# --------------------------------------------------------------------------
WHY = [
    ("Why are policies lapsing?", "PS-040"),
    ("What is driving lapses across the portfolio?", "PS-040"),
    ("Why does the worst-performing product have such a high lapse rate?", "PS-014"),
    ("Why is renewal collection weakest in the 37th month bucket?", "PS-050"),
    ("Why does the top agent outperform their peers?", "PS-068"),
    ("Which claims took longest to settle and why?", "PS-057"),
]

# --------------------------------------------------------------------------
# 5. How questions - comparison and mechanism
# --------------------------------------------------------------------------
HOW = [
    ("How do traditional and ULIP policies compare across key metrics?", "PS-016"),
    ("How do surrendered policies differ from those still in force?", "PS-043"),
    ("How does persistency differ across the 13th, 25th and 37th month buckets?", "PS-045"),
    ("How do active and inactive agents differ?", "PS-071"),
    ("How does renewal premium compare to original APE?", "PS-028"),
    ("How does claim amount compare to the policy sum assured?", "PS-060"),
    ("How does performance vary across branches?", "PS-089"),
    ("How concentrated is the portfolio across products?", "PS-019"),
    ("How has the product mix shifted over time?", "PS-092"),
    ("How long does it take for a policy to go from login to in-force?", "PS-029"),
]

# --------------------------------------------------------------------------
# 6. Ranking questions
# --------------------------------------------------------------------------
RANKING = [
    ("Which product has the highest lapse rate?", "PS-013"),
    ("Which products have the best customer retention?", "PS-017"),
    ("Who are the top-producing agents?", "PS-066"),
    ("Which agents have the worst lapse rates?", "PS-039"),
    ("Which agents have the best first-year persistency?", "PS-067"),
    ("Which agents have the best renewal collection?", "PS-049"),
    ("Which agents raise the most conduct concerns?", "PS-072"),
    ("Which are the highest-value policies in the book?", "PS-023"),
    ("Which are the largest claims?", "PS-061"),
    ("Which branches write the most business?", "PS-034"),
    ("Which occupations generate the most premium?", "PS-080"),
    ("Which states have the worst persistency?", "PS-085"),
    ("Which products carry the highest claim incidence?", "PS-059"),
    ("Which products are slowest to issue?", "PS-030"),
    ("Which agents are statistical outliers on quality metrics?", "PS-075"),
]

# --------------------------------------------------------------------------
# 7. Entity-scoped questions, including follow-up phrasing
# --------------------------------------------------------------------------
ENTITY_SCOPED = [
    ("Tell me about policy 11000000015", "PS-031"),
    ("Show policy details for AppID 15011000000", "PS-031"),
    ("Give me the full record for policy 19102929551", "PS-031"),
    ("How does agent 977337 compare against their peers?", "PS-069"),
]

# --------------------------------------------------------------------------
# 8. Ambiguous - genuinely answerable more than one way, must not be guessed
# --------------------------------------------------------------------------
AMBIGUOUS_QUERIES = [
    "performance",
    "how are we doing",
    "numbers",
    "show me the data",
    "analysis please",
]

# --------------------------------------------------------------------------
# 9. Unsupported and out-of-domain - must be refused, with the missing concept named
# --------------------------------------------------------------------------
UNSUPPORTED = [
    ("How do smokers compare to non-smokers on premiums?", "smoker"),
    ("What is the average BMI of our policyholders?", "bmi"),
    ("Show me commission paid to each agent", "commission"),
    ("Which product is most profitable?", "profit"),
    ("What is our expense ratio?", "expense"),
    ("What is the NAV of the ULIP fund?", "nav"),
    ("How have fund values performed?", "fund value"),
    ("Which riders are sold most often?", "rider"),
    ("Who are the nominees on our largest policies?", "nominee"),
    ("How many policies reached maturity?", "maturity"),
    ("Compare performance by zone", "zone"),
    ("Which sales territory performs best?", "territory"),
    ("Which agents missed their target?", "target"),
    ("Does agent tenure affect persistency?", "agent tenure"),
    ("What surrender value did customers receive?", "surrender value"),
    ("What reason did customers give for surrendering?", "surrender reason"),
    ("Why did each policy lapse? Show the lapse reason", "lapse reason"),
    ("Show me the premium payment history for each policy", "payment history"),
    ("Compare channels on persistency", "compare channels"),
    ("What is the marital status of our customers?", "marital status"),
    ("How many customer service calls did we receive?", "customer service"),
    ("Show underwriting risk scores", "risk score"),
]

OUT_OF_DOMAIN = [
    "What is the weather today?",
    "purple monkey dishwasher",
    "Tell me a joke",
    "What is the capital of France?",
    "How do I reset my password?",
]

# Identifiers that do not exist in the data.
INVALID_ENTITIES = [
    "Tell me about policy 00000000000",
    "Show me policy 99999999999",
]


IN_DOMAIN = (DIRECT + PARAPHRASE + EXECUTIVE + WHY + HOW + RANKING + ENTITY_SCOPED)

CATEGORIES = {
    "direct": DIRECT,
    "paraphrase": PARAPHRASE,
    "executive": EXECUTIVE,
    "why": WHY,
    "how": HOW,
    "ranking": RANKING,
    "entity_scoped": ENTITY_SCOPED,
}
