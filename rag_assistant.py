import os
import re
import json
import psycopg2
from sentence_transformers import SentenceTransformer
from google import genai


# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================

DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "research_paper_assistant"
DB_USER = "postgres"
DB_PASSWORD = "postgres"


# ==========================================
# 2. GEMINI CONFIGURATION
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY missing. Set the API key in your terminal."
    )

client = genai.Client(api_key=api_key)

# Keep the model name that currently works in your environment.
GEMINI_MODEL = "gemini-3.5-flash-lite"


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Embedding model loaded!")


# ==========================================
# 4. HELPER: FORMAT RETRIEVED SOURCES
# ==========================================

def build_source_context(results, max_chars=2500):
    """Return formatted source context and a Source ID -> row mapping."""
    source_parts = []
    source_map = {}

    for rank, row in enumerate(results, start=1):
        chunk_id, paper_id, page_number, chunk_text, distance = row
        source_id = f"S{rank}"
        source_map[source_id] = row

        source_parts.append(
            f"""
Source ID: {source_id}
Paper ID: {paper_id}
Page: {page_number}
Chunk ID: {chunk_id}

Evidence text:
{str(chunk_text)[:max_chars]}
"""
        )

    return "\n\n".join(source_parts), source_map


def normalize_text(text):
    """Normalize whitespace for exact quote checks."""
    return re.sub(r"\s+", " ", str(text)).strip()


# ==========================================
# 5. RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve_chunks(question, top_k=5):
    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    embedding_string = (
        "[" + ",".join(map(str, query_embedding.tolist())) + "]"
    )

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id,
                    paper_id,
                    page_number,
                    chunk_text,
                    embedding <=> %s::vector AS distance
                FROM paper_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding_string, embedding_string, top_k)
            )
            return cursor.fetchall()
    finally:
        conn.close()


# ==========================================
# 6. GENERATE ANSWER USING GEMINI
# ==========================================

def generate_answer(question, results):
    if not results:
        return "No relevant research-paper chunks were found."

    context, _ = build_source_context(results)

    prompt = f"""
You are an academic research-paper assistant.
Answer the user's question using ONLY the research-paper context below.

Rules:
1. Do not invent facts, numbers, findings, or citations.
2. If the provided excerpts do not establish a claim, say so explicitly.
3. Do not say the entire paper lacks evidence unless the entire paper
   has actually been checked. Say "the retrieved excerpts do not establish..."
4. Explain in simple, clear English.
5. Cite each factual statement about the paper using a paper ID and page.
6. Use one page per citation in exactly this form:
   (Paper PAPER_ID, p. PAGE_NUMBER)
7. Do not combine pages into formats such as "pp. 1-2, 7".
8. Do not include unrelated paper facts that do not help answer the question.
9. Separate known findings from unavailable or uncertain information.

Research-paper context:
{context}

User question:
{question}

Write a concise answer that directly addresses the user's question.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text or "Gemini returned an empty response."


# ==========================================
# 7. CITATION VALIDATION
# ==========================================

def expand_page_numbers(page_text):
    """Expand '1-2, 7' or '1–2; 7' into individual page numbers."""
    pages = []

    for part in re.split(r"\s*[,;]\s*", page_text.strip()):
        part = part.strip()
        if not part:
            continue

        match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)

        if match:
            start_page = int(match.group(1))
            end_page = int(match.group(2))

            if end_page >= start_page and end_page - start_page <= 100:
                pages.extend(str(page) for page in range(start_page, end_page + 1))
            else:
                pages.extend([str(start_page), str(end_page)])
        elif part.isdigit():
            pages.append(part)

    return pages


def validate_citations(answer, results):
    """
    Check whether recognized paper ID/page pairs occur in retrieved chunks.
    This checks citation existence, not whether the source supports the claim.
    """
    valid_citations = {
        (str(row[1]).strip(), str(row[2]).strip())
        for row in results
    }

    # Captures both single pages (p. 2) and page lists/ranges (pp. 1-2, 7).
    pattern = re.compile(
        r"\bPaper\s+([^,\s)]+)\s*,?\s*(pp?\.?|pages?)\s*"
        r"(\d+(?:\s*[-–—]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-–—]\s*\d+)?)*)",
        flags=re.IGNORECASE
    )

    found_citations = []
    invalid_citations = []

    for match in pattern.finditer(answer):
        paper_id = match.group(1).strip()
        page_text = match.group(3)

        for page_number in expand_page_numbers(page_text):
            citation = (paper_id, page_number)
            found_citations.append(citation)

            if citation not in valid_citations:
                invalid_citations.append(
                    f"Paper {paper_id}, p. {page_number}"
                )

    return found_citations, list(dict.fromkeys(invalid_citations))


# ==========================================
# 8. DETECT META-STATEMENTS ABOUT MISSING EVIDENCE
# ==========================================

def is_context_limitation_statement(claim_text):
    """
    Returns True for statements about what the retrieved context does not
    contain/establish. These are not claims about what the paper itself found.
    """
    text = normalize_text(claim_text).lower()

    missing_evidence_patterns = [
        r"\b(provided|retrieved|available)\s+(context|evidence|text|excerpts?)\b.*"
        r"\b(no mention|does not mention|doesn't mention|does not contain|"
        r"contains no|does not establish|doesn't establish|lack|lacks|"
        r"insufficient|not enough|cannot establish|cannot determine)\b",
        r"\b(no mention|not mentioned|not found|does not establish|"
        r"does not contain|doesn't contain|lacks sufficient evidence|"
        r"insufficient evidence|not enough evidence|cannot determine|"
        r"not known whether)\b.*\b(context|evidence|text|excerpts?|chunks?)\b",
        r"\b(the|this|these)\s+(context|excerpts?|chunks?|retrieved sources?)\b.*"
        r"\b(do not|does not|don't|doesn't|cannot|can't)\b.*"
        r"\b(establish|mention|contain|show|provide|prove)\b",
        r"\b(available|retrieved|provided)\s+context\b.*\b(not|no|doesn't|does not)\b",
        r"\bthe claim\b.*\b(cannot be established|is not established)\b"
    ]

    return any(re.search(pattern, text, flags=re.IGNORECASE)
               for pattern in missing_evidence_patterns)


# ==========================================
# 9. ORIGINAL QUESTION ASSESSMENT (SEPARATE API CALL)
# ==========================================

def assess_original_question(question, results):
    """
    Assess only a testable proposition in the user's original question.
    The generated answer is deliberately NOT sent to this call.
    """
    evidence_context, source_map = build_source_context(results)

    prompt = f"""
You are an academic evidence assessor. Assess the user's ORIGINAL QUESTION
against only the retrieved excerpts.

USER QUESTION:
{question}

RETRIEVED EXCERPTS:
{evidence_context}

First decide whether the question asks to verify one concrete factual
proposition (for example: "Does the proposed method achieve 99% accuracy?").
If the question is open-ended (for example: "What is the method?" or
"List the limitations"), set proposition to null and verdict to "NOT_APPLICABLE".

If there is a concrete proposition, assess whether the excerpts establish it:
- SUPPORTED: the excerpt text directly supports the proposition.
- CONTRADICTED: the excerpt text directly contradicts the proposition.
- NOT_ESTABLISHED: the retrieved excerpts do not establish the proposition.
  This does not prove that the proposition is false.
- UNCERTAIN: the excerpts are ambiguous.

Do not use the generated answer; assess the question directly.
Do not say the entire paper lacks evidence, because only the retrieved
excerpts are provided.
Use source IDs only when they directly contain relevant evidence. For
NOT_ESTABLISHED, an empty source ID list is allowed.
Return JSON only, no Markdown fences, exactly this structure:
{{
  "proposition": null,
  "verdict": "NOT_APPLICABLE",
  "evidence_source_ids": [],
  "explanation": "A concise assessment limited to the retrieved excerpts."
}}
Allowed verdicts: SUPPORTED, CONTRADICTED, NOT_ESTABLISHED, UNCERTAIN,
NOT_APPLICABLE.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                 flags=re.IGNORECASE).strip()
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Question assessor returned invalid JSON.")

    allowed = {
        "SUPPORTED", "CONTRADICTED", "NOT_ESTABLISHED",
        "UNCERTAIN", "NOT_APPLICABLE"
    }

    verdict = data.get("verdict")
    if verdict not in allowed:
        raise ValueError(f"Invalid original-question verdict: {verdict}")

    source_ids = data.get("evidence_source_ids", [])
    if not isinstance(source_ids, list):
        raise ValueError("Question assessor source IDs must be a list.")

    if any(source_id not in source_map for source_id in source_ids):
        raise ValueError("Question assessor returned an unknown source ID.")

    # Deterministic guardrail for percentage claims:
    # if the user asks about an exact percentage and that percentage
    # does not occur in any retrieved excerpt, those excerpts cannot
    # establish that exact percentage, regardless of the model's verdict.
    question_percentages = re.findall(
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*%",
        question
    )
    combined_source_text = " ".join(
        str(row[3]) for row in results if row[3] is not None
    )

    for percentage in question_percentages:
        percent_pattern = (
            rf"(?<![\d.]){re.escape(percentage)}"
            rf"(?:\.0+)?\s*(?:%|percent)(?![A-Za-z])"
        )
        if not re.search(percent_pattern, combined_source_text, flags=re.IGNORECASE):
            data["verdict"] = "NOT_ESTABLISHED"
            data["evidence_source_ids"] = []
            data["explanation"] = (
                f"The requested figure {percentage}% does not appear in the "
                "retrieved excerpts. Therefore, these excerpts do not establish "
                "the exact percentage. This does not prove the entire paper "
                "never reports it or that the claim is false."
            )
            break

    # Guardrail: no evidence IDs means it cannot be called SUPPORTED.
    if data.get("verdict") == "SUPPORTED" and not data.get("evidence_source_ids"):
        data["verdict"] = "UNCERTAIN"
        data["explanation"] = (
            "The assessor did not identify a valid retrieved source "
            "that supports this proposition."
        )

    return data


# ==========================================
# 10. ANSWER CLAIM VERIFICATION (SEPARATE API CALL)
# ==========================================

def verify_answer_claims(question, answer, results):
    """
    Extract and verify claims stated in the answer only.
    Original-question assessment occurs in a different API call.
    Supported claims must include an exact quote from a retrieved chunk.
    """
    evidence_context, source_map = build_source_context(results)

    prompt = f"""
You are an academic fact-checker. Verify substantive factual claims in
the generated answer against the retrieved excerpts.

ORIGINAL USER QUESTION:
{question}

GENERATED ANSWER:
{answer}

RETRIEVED EXCERPTS:
{evidence_context}

Extract only substantive claims about the research paper, its algorithm,
experiments, findings, or technical approach that directly help answer the
ORIGINAL USER QUESTION above. Omit unrelated paper facts even if they are true.

EXCLUDE statements whose only meaning is:
- the retrieved context does not mention a requested metric;
- the available excerpts do not establish something;
- there is insufficient information in the provided context;
- it is unknown whether a proposition is true because the context does
  not say so.

Those are statements about the limits of the current retrieval, not findings
about the paper. Do not turn them into answer claims.

For each remaining factual claim, return:
- claim: one concise factual claim from the answer
- verdict:
  SUPPORTED = exact retrieved text supports it
  CONTRADICTED = retrieved text directly contradicts it
  NOT_ESTABLISHED = retrieved text does not establish it
  UNCERTAIN = ambiguous/incomplete evidence
- evidence_source_ids: only IDs of directly relevant retrieved excerpts
- evidence_quotes: a list of source_id and a short EXACT verbatim quote
  copied from that source. Quotes must match the source text exactly except
  for whitespace differences. If there is no quote, use [].
- explanation: concise reason

Never invent a quote or source ID. Do not provide a SUPPORTED verdict unless
at least one exact quote from a source directly supports the claim.
If the answer contains only context-limitation statements, return an empty
claims list.

Return only valid JSON, no Markdown fences, exactly this structure:
{{
  "claims": [
    {{
      "claim": "A factual claim from the answer",
      "verdict": "SUPPORTED",
      "evidence_source_ids": ["S1"],
      "evidence_quotes": [
        {{"source_id": "S1", "quote": "exact verbatim text from the source"}}
      ],
      "explanation": "Why the quoted text supports or fails to support the claim."
    }}
  ]
}}
Allowed verdicts: SUPPORTED, CONTRADICTED, NOT_ESTABLISHED, UNCERTAIN.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                 flags=re.IGNORECASE).strip()
    data = json.loads(raw)

    if not isinstance(data, dict) or not isinstance(data.get("claims", []), list):
        raise ValueError("Answer-claim verifier returned invalid JSON.")

    allowed = {"SUPPORTED", "CONTRADICTED", "NOT_ESTABLISHED", "UNCERTAIN"}
    claims = []
    quote_checks = []

    for claim in data.get("claims", []):
        if not isinstance(claim, dict):
            continue

        claim_text = str(claim.get("claim", "")).strip()
        if not claim_text or is_context_limitation_statement(claim_text):
            continue

        verdict = claim.get("verdict")
        if verdict not in allowed:
            verdict = "UNCERTAIN"

        source_ids = claim.get("evidence_source_ids", [])
        if not isinstance(source_ids, list):
            source_ids = []

        source_ids = list(dict.fromkeys(
            source_id for source_id in source_ids
            if source_id in source_map
        ))

        # Verify that each alleged verbatim quote really appears in its source.
        verified_quotes = []
        for quote_item in claim.get("evidence_quotes", []):
            if not isinstance(quote_item, dict):
                continue

            source_id = quote_item.get("source_id")
            quote = str(quote_item.get("quote", "")).strip()

            if source_id not in source_map or not quote:
                continue

            source_text = normalize_text(source_map[source_id][3])
            normalized_quote = normalize_text(quote)

            if normalized_quote and normalized_quote in source_text:
                verified_quotes.append({
                    "source_id": source_id,
                    "quote": quote
                })

        # A SUPPORT claim is accepted only when an exact source quote
        # can be verified in the source chunk and that source ID is valid.
        if verdict == "SUPPORTED":
            supporting_quote_ids = {
                item["source_id"] for item in verified_quotes
            }
            if not supporting_quote_ids:
                verdict = "UNCERTAIN"
                explanation = (
                    "Downgraded to UNCERTAIN: the claimed supporting quote "
                    "could not be matched exactly to a retrieved source chunk."
                )
            else:
                explanation = str(claim.get("explanation", "")).strip()
        else:
            explanation = str(claim.get("explanation", "")).strip()

        claims.append({
            "claim": claim_text,
            "verdict": verdict,
            "evidence_source_ids": source_ids,
            "verified_quotes": verified_quotes,
            "explanation": explanation or "No explanation was provided."
        })

    return {"claims": claims, "quote_checks": quote_checks}


# ==========================================
# 11. VERIFICATION REPORT
# ==========================================

def display_verification_report(question_assessment, claim_report):
    print("\n========== ORIGINAL QUESTION ASSESSMENT ==========\n")

    if question_assessment is None:
        print("Assessment could not be completed.")
    elif question_assessment.get("verdict") == "NOT_APPLICABLE":
        print("This is an open-ended information question; no single proposition to score.")
        print("Explanation:", question_assessment.get("explanation", ""))
    else:
        print("Proposition:", question_assessment.get("proposition") or "Not identified")
        print("Verdict:", question_assessment.get("verdict", "UNCERTAIN"))
        print(
            "Evidence source IDs:",
            ", ".join(question_assessment.get("evidence_source_ids", [])) or "None"
        )
        print("Explanation:", question_assessment.get("explanation", ""))

    print("\n========== SUBSTANTIVE ANSWER CLAIM VERIFICATION ==========\n")

    if claim_report is None:
        print("Answer-claim verification could not be completed.")
        return

    claims = claim_report.get("claims", [])
    if not claims:
        print(
            "No substantive research-paper claims were extracted. "
            "Statements only about missing retrieved evidence are not scored here."
        )

    allowed = ["SUPPORTED", "CONTRADICTED", "NOT_ESTABLISHED", "UNCERTAIN"]
    counts = {verdict: 0 for verdict in allowed}

    for number, claim in enumerate(claims, start=1):
        verdict = claim.get("verdict", "UNCERTAIN")
        if verdict not in counts:
            verdict = "UNCERTAIN"
        counts[verdict] += 1

        print(f"Claim #{number}")
        print("Claim:", claim["claim"])
        print("Verdict:", verdict)
        print(
            "Evidence source IDs:",
            ", ".join(claim.get("evidence_source_ids", [])) or "None"
        )
        print("Evidence assessment:", claim.get("explanation", ""))

        for quote in claim.get("verified_quotes", []):
            print(
                f"Verified quote from {quote['source_id']}: "
                f'"{quote["quote"]}"'
            )
        print("-" * 70)

    print("\n========== ANSWER CLAIM SUMMARY ==========")
    print("Substantive claims assessed:", len(claims))
    for verdict in allowed:
        print(f"{verdict.replace('_', ' ').title()}:", counts[verdict])

    print(
        "\nImportant: A NOT_ESTABLISHED result means the retrieved excerpts "
        "do not establish the proposition; it does not prove the proposition false."
    )


# ==========================================
# 12. MAIN PROGRAM
# ==========================================

def main():
    print("\n==========================================")
    print(" AI Research Paper Intelligence Assistant")
    print("==========================================")

    question = input("\nEnter your question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    try:
        print("\nSearching research papers...")
        results = retrieve_chunks(question)

        if not results:
            print("No chunks found in the database.")
            return

        print(f"Retrieved {len(results)} relevant chunks.")

        print("\nGenerating answer with Gemini...")
        answer = generate_answer(question, results)

        found_citations, invalid_citations = validate_citations(answer, results)

        print("\n========== AI ANSWER ==========\n")
        print(answer)

        print("\n========== CITATION VALIDATION ==========\n")
        if not found_citations:
            print("WARNING: No recognized citations were found.")
        elif invalid_citations:
            print("WARNING: These recognized citations do not match retrieved paper/page pairs:")
            for citation in invalid_citations:
                print(" -", citation)
        else:
            print("Citation validation passed.")
            print(f"Checked {len(found_citations)} page citation(s).")
            print(
                "All recognized paper ID/page combinations match retrieved sources. "
                "This checks citation existence, not factual support."
            )

        # Separate API call 1: assess only the original question.
        print("\nAssessing the original question independently...")
        try:
            question_assessment = assess_original_question(question, results)
        except Exception as error:
            question_assessment = None
            print("Original-question assessment error:", error)

        # Separate API call 2: assess only substantive claims in the answer.
        print("\nVerifying substantive answer claims and source quotes...")
        try:
            claim_report = verify_answer_claims(question, answer, results)
        except Exception as error:
            claim_report = None
            print("Answer-claim verification error:", error)

        display_verification_report(question_assessment, claim_report)

        print("\n========== RETRIEVED SOURCES ==========\n")
        for rank, row in enumerate(results, start=1):
            chunk_id, paper_id, page_number, chunk_text, distance = row
            print(f"Source #{rank}")
            print(f"Paper ID: {paper_id}")
            print(f"Page: {page_number}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Distance: {distance:.4f}")
            print("-" * 60)

    except Exception as error:
        print("\nAn error occurred:")
        print(error)


# ==========================================
# 13. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
