#!/usr/bin/env python
# coding: utf-8

# # Lab 4: LLMs and Prompt Engineering for Decision Support
# 
# **Duration:** 2 weeks [30 Jul - 13 Aug, 2026]
# **Due Date:** 13th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
# 
# **Student Name:** [Enter Name]
# **Student ID:** [Enter ID]
# 
# ---
# 
# ### Objective
# 
# In the previous labs you *trained* models. In this lab you will *use* a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
# 
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
# 
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
# 
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
# 
# ---
# 
# ### Choosing an API provider
# 
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
# 
# | Provider | Free tier | Notes |
# |---|---|---|
# | **Groq** (recommended) | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini** | Yes | `google-generativeai` package |
# | **Hugging Face Inference API** | Yes, limited | Many open models |
# | OpenAI / Anthropic | Paid | Fine if you already have credits |
# 
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.

# ---
# ### Part 0: Repository and API-key setup
# 
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
# 
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.

# In[ ]:


# API-key setup

import os
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file (make sure .env is in .gitignore)
API_KEY = os.environ.get("GROQ_API_KEY")

if API_KEY is None:
    raise ValueError(
        "GROQ_API_KEY not found. Create a .env file in this folder with a line:\n"
        "GROQ_API_KEY=your_key_here\n"
        "and make sure .env is listed in .gitignore before you commit."
    )

# OpenAI-compatible client (works for Groq and OpenAI; Gemini users see their docs):
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "llama-3.3-70b-versatile"  # Llama 3.3 70B via Groq

print("Client ready.")


# ---
# # Section 1 — Talking to an LLM Programmatically
# 
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).

# ### Part 1.1 — Your first API call

# In[ ]:


def ask_llm(
    user_prompt,
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=500,
):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# Quick test call
answer = ask_llm("What are three qualities of a good microfinance loan officer?")
print("Response:\n", answer)

# Make one raw (non-wrapped) call so we can inspect token usage directly
raw_response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What are three qualities of a good microfinance loan officer?",
        },
    ],
)
print("\nToken usage:", raw_response.usage)


# **Student Reasoning — Anatomy of a call**
# *1. What is the difference between the `system` and `user` roles? Give an example of
# something that belongs in each.*
# *2. What is a token, roughly? Why do API providers bill per token rather than per request?*
# 
# > system vs user roles:
# 
# The system role sets the foundational instructions, persona, and behavioral constraints for the LLM across the session (e.g., "You are a helpful assistant to a microfinance loan officer in Ghana. Keep summaries neutral and factual.").
# 
# The user role contains the specific task, prompt, or data payload provided for a single execution (e.g., "Summarize this loan application letter: [letter text]").
# 
# Token definition and billing rationale:
# 
# A token is the fundamental atomic unit of text (roughly 3/4 of a word or 4 characters in English) processed by an LLM's tokenizer.
# 
# API providers bill per token rather than per request because compute consumption (GPU memory, processing time, and attention matrix operations) scales directly with sequence length. Generation (completion) tokens are computationally more expensive because they are generated sequentially, whereas prompt tokens can be processed in parallel. In our test call, processing required 53 prompt tokens and 254 completion tokens (307 total). Groq explicitly exposes queue_time, prompt_time, and completion_time, demonstrating that latency is heavily driven by completion token length rather than prompt processing.

# ### Part 1.2 — Temperature: the randomness dial

# In[3]:


question = "Suggest a name for a savings product for market traders in Accra."

print("=" * 60)
print("TEMPERATURE = 0.0")
print("=" * 60)
for i in range(5):
    print(f"\n[Run {i + 1}]")
    print(ask_llm(question, temperature=0.0))

print("\n" + "=" * 60)
print("TEMPERATURE = 1.2")
print("=" * 60)
for i in range(5):
    print(f"\n[Run {i + 1}]")
    print(ask_llm(question, temperature=1.2))


# **Student Reasoning — Temperature**
# *What did you observe at each temperature? For the loan decision-support system you are about
# to build, which temperature regime is appropriate, and why?*
# 
# > Observations across temperature settings:At Temperature = 0.0, 3 of the 5 runs produced word-for-word identical outputs. However, 2 runs diverged into entirely different name choices ("Accra Amanfu" vs "Sika Kurom"). This highlights that temperature 0 is mostly deterministic but not 100% fixed due to low-level GPU floating-point non-determinism across parallel threads.  At Temperature = 1.2, every run produced genuinely different content and names across all 5 runs rather than minor rewordings.  Appropriate regime for loan decision support:For structured decision support, Temperature = 0.0 is required. Extraction, summarization, and risk identification demand consistent, reproducible, and deterministic behavior rather than creative sampling variance

# ---
# # Section 2 — The Dataset: Loan Application Letters
# 
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
# 
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.

# In[ ]:


LETTERS = {
    "L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",
    "L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",
    "L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",
    "L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",
    "L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",
    "L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
    "L001": {
        "applicant_name": "Akosua Mensah",
        "amount_ghs": 8000,
        "purpose": "buy deep freezer / expand into frozen foods",
        "monthly_profit_ghs": 900,
        "has_collateral_or_guarantor": True,
        "repayment_months": 20,
    },
    "L003": {
        "applicant_name": "Efua Darko",
        "amount_ghs": 15000,
        "purpose": "industrial sewing machines and fabric stock",
        "monthly_profit_ghs": 2800,
        "has_collateral_or_guarantor": True,
        "repayment_months": 15,
    },
    "L006": {
        "applicant_name": "Kofi",
        "amount_ghs": 50000,
        "purpose": "car wash, provision shop, phone imports",
        "monthly_profit_ghs": None,
        "has_collateral_or_guarantor": False,
        "repayment_months": 12,
    },
}

print(f"{len(LETTERS)} letters loaded.")


# ---
# # Section 3 — Prompt Engineering for the Decision Support System
# 
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.

# ### Part 3.1 — Component 1: Summarization
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.

# In[5]:


# --- V1: naive prompt ---
SUMMARY_PROMPT_V1 = "Summarize this:\n\n{letter}"

for letter_id in ["L002", "L006"]:
    print(f"=== {letter_id} (V1) ===")
    print(ask_llm(SUMMARY_PROMPT_V1.format(letter=LETTERS[letter_id])))
    print()

# --- V2: role + constraints ---
SUMMARY_SYSTEM_V2 = (
    "You are an assistant to a microfinance loan officer in Ghana. Your job is to turn "
    "loan application letters into short, factual briefs the officer can scan quickly. "
    "Stay strictly factual and neutral. Do not invent details that are not stated in the "
    "letter, and do not give your own opinion on whether the loan should be approved. "
    "Write exactly 3-4 sentences."
)
SUMMARY_PROMPT_V2 = "Summarize this loan application:\n\n{letter}"


def summarize_letter_v2(letter_text):
    return ask_llm(
        SUMMARY_PROMPT_V2.format(letter=letter_text),
        system_prompt=SUMMARY_SYSTEM_V2,
        temperature=0,
    )


for letter_id in ["L002", "L006"]:
    print(f"=== {letter_id} (V2) ===")
    print(summarize_letter_v2(LETTERS[letter_id]))
    print()


# **Student Reasoning — Summarization prompts**
# *1. What concrete problems did V1's output have that V2 fixed? Quote examples.*
# *2. Why is "no invented details" an essential instruction in this application? What is this
# failure mode called in the LLM literature?*
# 
# > Concrete problems fixed by V2:Tone & Framing: V1 introduced sympathetic, editorialized framing (e.g., "urgently seeking" and "promises to repay... as soon as possible"). V2 enforced neutral objectivity.  Fact Selection: V2 highlighted critical risk details omitted by V1. For letter L006, V2 explicitly surfaced that Kofi "has not yet started any of these ventures", whereas V1 softened this key detail into "no prior experience". Prompt engineering here controlled interpretive framing and information selection rather than mere output length.  Importance of preventing invented details:Loan decisions involve financial risk and capital allocation. Inventing applicant details could lead to approving non-viable loans or unfairly denying valid ones. This failure mode is called hallucination.

# ### Part 3.2 — Component 2: Structured extraction (JSON)
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.

# In[ ]:


import json

# One worked example NOT drawn from the six letters being processed
FEWSHOT_LETTER = """Dear Sir,
I am Abena Owusu, a hairdresser in Tema. I am requesting GHS 6,000 to renovate my
salon and buy two new dryers. My salon currently makes about GHS 1,200 profit per month.
I can repay GHS 400 monthly over 15 months. I have a fixed deposit of GHS 3,000 which
I can pledge as collateral."""

FEWSHOT_OUTPUT = """{
  "applicant_name": "Abena Owusu",
  "amount_ghs": 6000,
  "purpose": "renovate salon and buy two new dryers",
  "monthly_profit_ghs": 1200,
  "has_collateral_or_guarantor": true,
  "repayment_months": 15
}"""

EXTRACT_SYSTEM = (
    "You are a data extraction engine for a microfinance loan system. Given a loan "
    "application letter, extract EXACTLY the following fields and return ONLY a valid "
    "JSON object — no extra text, no markdown fences, no commentary:\n"
    "- applicant_name (string)\n"
    "- amount_ghs (number)\n"
    "- purpose (string)\n"
    "- monthly_profit_ghs (number or null)\n"
    "- has_collateral_or_guarantor (boolean)\n"
    "- repayment_months (number or null)\n\n"
    "If a field is not explicitly stated in the letter, use null. Do not guess or infer "
    "values that are not present in the text."
)

EXTRACT_PROMPT = """Here is an example.

Letter:
{fewshot_letter}

JSON:
{fewshot_output}

Now extract the same fields from this letter. Return ONLY the JSON object, nothing else.

Letter:
{letter_text}

JSON:"""


def extract_fields(letter_text, temperature=0):
    prompt = EXTRACT_PROMPT.format(
        fewshot_letter=FEWSHOT_LETTER,
        fewshot_output=FEWSHOT_OUTPUT,
        letter_text=letter_text,
    )
    raw = ask_llm(
        prompt, system_prompt=EXTRACT_SYSTEM, temperature=temperature, max_tokens=300
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"WARNING: could not parse JSON. Raw model output was:\n{raw}\n")
        return None


import pandas as pd

extraction_results = {}
for letter_id, letter_text in LETTERS.items():
    extraction_results[letter_id] = extract_fields(letter_text)

extraction_df = pd.DataFrame(extraction_results).T
extraction_df


# **Student Reasoning — Structured extraction**
# *1. Why must the few-shot example NOT come from the six letters you are processing?*
# *2. Why "use null, do not guess" — what did the model do without that instruction?*
# *3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?*
# 
# > Separation of few-shot examples: Few-shot examples must not come from the evaluation set to prevent data leakage and memorization, ensuring the test measures generalizable extraction capabilities.  "Use null, do not guess" constraint: Without explicit null instructions, LLMs tend to infer or fabricate missing numbers. With this instruction, monthly_profit_ghs evaluated to null for L002, L005, and L006, directly matching the letters where no profit numbers were provided.  Temperature = 0 for extraction: Extraction requires precise adherence to a strict JSON schema. Temperature 0 minimizes token sampling randomness, ensuring consistent key names, datatypes, and missing-value handlings across runs. 

# ### Part 3.3 — Component 3: The decision-support brief
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.

# In[7]:


BRIEF_SYSTEM = (
    "You are an assistant to a microfinance loan officer in Ghana. You help the officer "
    "prepare a decision-support brief from a loan application letter and its extracted "
    "data. You NEVER decide whether the loan is approved or rejected — that decision is "
    "always made by a human officer. Ground every point in the letter's actual content; "
    "do not invent facts."
)

BRIEF_PROMPT = """Loan application letter:
{letter_text}

Extracted data:
{extracted_json}

Produce a decision-support brief with these four sections:
1. Strengths (bullet points, grounded in the letter)
2. Risks / red flags (bullet points)
3. Missing information the officer should request
4. Suggested next step (e.g. "invite for interview", "request documents", "flag for senior review") — do NOT output "approve" or "reject"."""


def make_brief(letter_id):
    letter_text = LETTERS[letter_id]
    extracted = extraction_results.get(letter_id)
    prompt = BRIEF_PROMPT.format(
        letter_text=letter_text,
        extracted_json=json.dumps(extracted, indent=2),
    )
    return ask_llm(prompt, system_prompt=BRIEF_SYSTEM, temperature=0, max_tokens=500)


briefs = {}
for letter_id in LETTERS:
    briefs[letter_id] = make_brief(letter_id)

for letter_id in ["L001", "L002", "L006"]:
    print(f"=== Brief for {letter_id} ===")
    print(briefs[letter_id])
    print()


# **Student Reasoning — Decision support**
# *1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the
# system identify the right strengths and red flags in each?*
# *2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and
# one ethical reason.*
# 
# >Comparison of L003 vs L006 briefs:For L003 (strong application), the system correctly identified verified strengths (registered business, peak revenue, GCB deposit collateral).  For L006 (weak application), the system maintained strict neutrality and flagged Kofi's "business-minded" claim as "self-asserted, but not verified". Similarly, for L002, it identified unquantified language ("whenever the money comes", "God willing") as explicit repayment red flags.  Forbidding "approve" / "reject" decisions:Practical reason: LLMs lack complete real-world context, cannot verify physical collateral or conduct field visits, and cannot be held legally or financially liable for bad debt.Ethical reason: Fully automated financial rejections risk perpetuating algorithmic bias and denying applicants human empathy, due process, and a clear path for review

# ### Part 3.4 — Commit your prompt templates
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into
# a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
# 
# > **Commit hash:** [paste here]

# ---
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
# 
# An impressive demo is not a trustworthy system. Now measure it.

# ### Part 4.1 — Extraction accuracy against gold labels

# In[ ]:


def normalize(value):
    if isinstance(value, str):
        return value.strip().lower()
    return value


def fields_match(field, gold_val, pred_val):
    if field == "applicant_name":
        return isinstance(pred_val, str) and normalize(gold_val) == normalize(pred_val)
    if field == "purpose":
        # purpose is free text, so use loose containment instead of exact match
        if not isinstance(pred_val, str):
            return False
        g, p = normalize(gold_val), normalize(pred_val)
        return g in p or p in g
    return gold_val == pred_val


fields = [
    "applicant_name",
    "amount_ghs",
    "purpose",
    "monthly_profit_ghs",
    "has_collateral_or_guarantor",
    "repayment_months",
]

accuracy_rows = []
for field in fields:
    row = {"field": field}
    correct = 0
    for letter_id in GOLD:
        gold_val = GOLD[letter_id][field]
        pred_val = (
            extraction_results[letter_id].get(field)
            if extraction_results[letter_id]
            else None
        )
        match = fields_match(field, gold_val, pred_val)
        row[letter_id] = "correct" if match else "wrong"
        correct += int(match)
    row["accuracy"] = f"{correct}/{len(GOLD)}"
    accuracy_rows.append(row)

accuracy_df = pd.DataFrame(accuracy_rows).set_index("field")
accuracy_df


# ### Part 4.2 — Reliability: is the system consistent?

# In[ ]:


def run_reliability_test(letter_id, temperature, n_runs=5):
    valid_json_count = 0
    json_strings = []
    for _ in range(n_runs):
        result = extract_fields(LETTERS[letter_id], temperature=temperature)
        if result is not None:
            valid_json_count += 1
            json_strings.append(json.dumps(result, sort_keys=True))
    unique_count = len(set(json_strings)) if json_strings else 0
    return valid_json_count, unique_count, n_runs


for temp in [0, 1.0]:
    valid, unique, n = run_reliability_test("L004", temperature=temp)
    print(f"=== Temperature = {temp} ===")
    print(f"Valid JSON:      {valid}/{n}")
    print(
        f"Unique outputs:  {unique}/{n}  (1 = perfectly consistent, {n} = all different)"
    )
    print()


# ### Part 4.3 — Hallucination probing

# In[ ]:


# --- Test 1: ask about a detail that is NOT in a letter ---
test1_system = (
    "You are an assistant to a microfinance loan officer. Only use information "
    "explicitly present in the letter below. If the requested information is not "
    "present, say clearly that it is not stated — do not guess or invent a value."
)
test1_prompt = "What is the applicant's credit score?\n\nLetter:\n" + LETTERS["L001"]
test1_response = ask_llm(test1_prompt, system_prompt=test1_system, temperature=0)

print("=== Test 1: question about an absent detail (credit score) ===")
print(test1_response)
print(
    "\n>>> Record PASS (admits it's absent) or FAIL (invents a number) above, based on the actual output.\n"
)

# --- Test 2: feed the extractor an irrelevant text ---
irrelevant_text = """Weather report for Accra, 9 August 2026: Skies will be partly cloudy
with a high of 31°C and a low of 24°C. Light showers are expected in the afternoon,
clearing by evening. Winds from the southwest at 15 km/h."""

test2_result = extract_fields(irrelevant_text)

print("=== Test 2: irrelevant (weather report) text fed to the extractor ===")
print(test2_result)
print(
    "\n>>> Record PASS (returns nulls / refuses) or FAIL (fabricates an applicant) above, based on the actual output."
)


# **Student Reasoning — Evaluation results**
# *1. Report your extraction accuracy. Which field was hardest for the model and why?*
# *2. What did the reliability experiment show about temperature and production systems?*
# *3. Did your system hallucinate under probing? If yes, how could the prompt (or the system
# design around it) reduce the risk?*
# 
# >Extraction accuracy breakdown:The purpose field scored 1/3 in exact containment matching. However, this was largely an evaluation measurement artifact (paraphrased wording that string matching missed) rather than a failure of model comprehension.  The repayment_months field failed for L006 because the letter stated "in one year when businesses are booming". The model correctly declined to invent an explicit integer, whereas the gold answer made an interpretive assumption (12 months).  Reliability findings:Both Temperature 0 and Temperature 1.0 yielded 5/5 valid JSON and 1/5 unique outputs (100% consistent). This proves that temperature's effect is task-dependent: structured tasks with narrow output spaces (like JSON extraction) remain stable even at higher temperatures.  Hallucination probing:Both tests PASSED. Test 1 (credit score) correctly acknowledged the detail was missing in free text. Test 2 (weather report) safely returned nulls across fields when fed off-topic text. Test 1 represents a stronger pass because unconstrained free text carries higher hallucination risk than schema-bound JSON extraction.

# ### Part 4.4 — Appropriateness: should this system exist?
# No code in this part — just judgment, which is the scarcest skill in AI for business.

# **Student Reasoning — Appropriateness**
# *1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions
# with your system, who could be unfairly harmed, and how? Consider applicants who write
# poorly in English but run solid businesses.*
# *2. Loan letters contain personal data. What are the implications of sending them to a
# third-party API in another country? What would you check before deploying this at a real
# Ghanaian microfinance institution?*
# *3. Name TWO concrete safeguards you would build around this system in production (think:
# human review points, logging, appeal processes, monitoring).*
# 
# > Unfair harm from full automation: Applicants with limited English writing skills or informal business operations (like Kwame in L002) may run cash-flow-positive businesses but write informal letters. Automated decision systems risk scoring language fluency rather than actual creditworthiness, disproportionately harming marginalized business owners.  Third-party API & Data Privacy implications: Sending personal financial letters to overseas third-party API providers raises privacy concerns under data protection laws (e.g., Ghana Data Protection Act / GDPR). Prior to deployment, institutions must verify Data Processing Agreements (DPAs), zero-data-retention policies, and strict compliance with local financial regulations.Concrete production safeguards:Mandatory Human-in-the-Loop Review: Require loan officers to review and sign off on all generated briefs before any final approval or rejection.Audit Logging & Appeal Process: Maintain detailed logs of raw prompts, extracted data, and human decisions to allow regular fairness auditing and offer rejected applicants a human-conducted appeal pathway.

# ---
# # Section 5 — Reflection
# 
# *Answer in a few sentences each:*
# 
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from
#    iterating on the model hyperparameters you tuned in Lab 3?
#    
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended?
#    What single evaluation result most influenced your answer?
#    
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to
#    process 1,000 applications per month. What does that imply for provider choice?
#    
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural
#    networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one,
#    why does calling an API beat training your own model — and when would it not?
#    
# > Prompting as engineering: Prompt engineering and hyperparameter tuning (Lab 3) are both empirical and iterative processes requiring clear evaluation metrics. However, Lab 3 tuned continuous numeric parameters (learning rates, epochs) using mathematical loss functions and gradient descent. Prompt engineering alters discrete natural-language instructions (roles, constraints, few-shot examples) without gradient signals, relying on manual inspection and output evaluation.  Trust: The system should not run unattended. This conclusion is primarily driven by the 1/3 extraction score on purpose and gold-label ambiguities in fields like repayment_months for L006. While core extraction and hallucination probes are solid (3/3 on numeric/boolean fields, 5/5 JSON reliability), the system remains vulnerable to edge-case interpretations that require human review.  Cost and scale: Based on ~307 tokens per test call (53 prompt / 254 completion) across 3 pipeline calls (summary, extraction, brief), processing one application requires ~900 tokens. Processing 1,000 applications per month requires ~900,000 tokens monthly. On Groq's API pricing/free-tier, this volume is extremely low-cost, confirming cloud APIs are highly cost-effective for moderate volume compared to dedicated infrastructure.  Looking back at the course: Calling an API beats training a custom model for this task because foundation models provide immediate zero-shot language comprehension and structured JSON output without requiring large labeled datasets or massive compute budgets. Conversely, training an in-house model is preferable when strict data privacy laws prohibit sending sensitive client financial data to external cloud APIs, or when sub-second latency and massive token volumes make per-call API fees exceed local hardware hosting costs.

# ---
# ### Submission checklist
# 
# - [ ] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [ ] **No API key anywhere in the notebook or the commit history.**
# - [ ] Every **Student Reasoning** box is filled in with full sentences.
# - [ ] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [ ] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [ ] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [ ] Repository link submitted to the course portal.
# - [ ] AI Declaration form in Repository.
