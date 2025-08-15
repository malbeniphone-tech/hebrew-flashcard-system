"""
Flashcard System
================

This module implements a simple Hebrew‑aware flashcard and quiz generator.

Given a text or a path to a ``.docx``, ``.pdf`` or ``.md`` file, the
system extracts the text, identifies the most significant sentences
using a TF‑IDF based ranking, and then generates a set of flashcards
summarising the key ideas.  It also produces a multiple‑choice quiz to
help the user test their knowledge.  After the quiz is completed the
system can analyse incorrect answers and generate additional
flashcards focusing on topics that require more attention.

Key features
------------

* **File support** – The system can read plain text strings, Word
  documents (`.docx`), PDFs and Markdown files.  Word documents are
  parsed directly from their XML representation.  PDFs are read via
  a lightweight HTTP service available at ``http://localhost:8451``.
  Markdown files are treated as plain text after stripping simple
  formatting.

* **Sentence ranking** – Sentences are scored using a TF‑IDF model
  built with ``scikit‑learn``.  The top ranked sentences are assumed
  to carry the most important information.  This heuristic mirrors
  common extractive summarisation techniques【280062428660715†L46-L67】.

* **Flashcard generation** – For each selected sentence the system
  identifies a salient word (the token with the highest TF‑IDF
  weight) and replaces it with a blank (``______``) to form a
  question.  The missing word becomes the answer.  Although naive,
  this approach yields short, fill‑in‑the‑blank flashcards that
  encourage active recall.

* **Quiz creation** – A multiple‑choice quiz of configurable length
  (default 10 questions) is assembled from the flashcards.  Each
  question presents the blanked sentence and four candidate answers:
  the correct word plus three distractors sampled from the answers
  of other flashcards.

* **Adaptive reinforcement** – After the user answers the quiz the
  ``evaluate_quiz`` function reports the score and identifies which
  cards were answered incorrectly.  These topics can then be passed
  back into ``generate_additional_flashcards`` to produce new
  questions that emphasise the concepts requiring further study.

Limitations
-----------

This implementation avoids heavy external dependencies because the
network is restricted.  It therefore does not use pre‑trained
transformer models for summarisation or question generation.  The
TF‑IDF based approach is language‑agnostic and works reasonably well
for Hebrew text, but it cannot perform abstractive summarisation or
deep semantic analysis.  The user may wish to integrate more
advanced models such as mT5 or BART for higher quality output when
network access is available【70391832399734†L250-L267】.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence, Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Optional transformers integration
try:
    # We lazily import transformers so that systems without the dependency can
    # still run the TF‑IDF based logic.  If transformers is unavailable or
    # the required models cannot be downloaded (e.g., due to lack of network
    # connectivity), the summariser variables below will remain None and the
    # fallback extractive summarisation will be used instead.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    _SUMMARISER_MODEL_NAME = "imvladikon/het5_small_summarization"
    _summariser = None  # type: Optional[callable]
    # For question generation we attempt to use a Hebrew instruction‑tuned
    # model.  We defer loading the model until requested to avoid
    # unnecessary overhead when not needed.  Note: these models are
    # extremely large (billions of parameters) and may not run in
    # constrained environments.  They are included here to illustrate how
    # one might integrate an open‑source Hebrew Q&A generator when
    # resources permit.
    _QA_MODEL_NAME = "ronigold/dictalm2.0-instruct-fine-tuned-alpaca-gpt4-hebrew"
    _qa_pipeline = None  # type: Optional[callable]
except Exception:
    # If transformers isn't installed, we simply skip summarisation
    _summariser = None  # type: ignore


def _get_summariser() -> Optional[callable]:
    """Initialise and cache the Hebrew summarisation pipeline.

    Returns a callable that takes a string and returns a summary.  If
    ``transformers`` is not available or the model cannot be loaded,
    this function returns ``None``.  The pipeline is created lazily on
    first use to avoid unnecessary overhead.
    """
    global _summariser
    if _summariser is not None:
        return _summariser
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(_SUMMARISER_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(_SUMMARISER_MODEL_NAME)
        _summariser = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
        )
        return _summariser
    except Exception:
        # Model loading failed (e.g., no internet).  Leave summariser as None.
        _summariser = None
        return None

# ----------------------------------------------------------------------------
# Optional: Hebrew question generation using DictaLM‑based models
# ----------------------------------------------------------------------------

def _get_qa_generator() -> Optional[callable]:
    """Initialise and cache a question generation pipeline for Hebrew.

    This function lazily loads a large transformer model fine‑tuned for
    generating question–answer pairs in Hebrew.  If the model cannot be
    loaded (due to missing dependencies, lack of network connectivity or
    insufficient resources) the function returns ``None``.  The pipeline
    uses the text‑generation task and expects prompts instructing the model
    to produce a series of questions and answers separated by newlines.

    Returns
    -------
    Optional[callable]
        A callable text generation pipeline, or ``None`` if loading failed.
    """
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline
    try:
        # We avoid importing heavy modules at the top‑level to allow the
        # library to be absent.  Import here to catch failures gracefully.
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
        model = AutoModelForCausalLM.from_pretrained(_QA_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(_QA_MODEL_NAME)
        _qa_pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
        )
        return _qa_pipeline
    except Exception:
        _qa_pipeline = None
        return None


def generate_question_answer_pairs(text: str, num_pairs: int = 5) -> List[Tuple[str, str]]:
    """Generate question–answer pairs from ``text`` using a Hebrew LLM.

    This function attempts to use a large instruction‑tuned model to
    produce natural question–answer pairs from the input text.  It
    constructs a prompt asking the model to generate ``num_pairs`` pairs
    separated by newlines.  The output is parsed heuristically: each
    question is expected to end with a question mark and the following
    sentence is taken as its answer.  If the model cannot be loaded or
    the generation fails, an empty list is returned.

    Parameters
    ----------
    text : str
        The context from which to extract questions and answers.
    num_pairs : int, optional
        The number of pairs to request from the model.  Defaults to 5.

    Returns
    -------
    List[Tuple[str, str]]
        A list of (question, answer) tuples.  The list may be empty if
        generation fails.
    """
    qa_gen = _get_qa_generator()
    if qa_gen is None:
        return []
    # Compose an instruction in Hebrew.  We ask the model to produce
    # numbered Q&A pairs to aid parsing.  The numbers help delineate
    # boundaries but are removed in the final output.
    prompt = (
        f"כתוב {num_pairs} שאלות ותשובות בקשר לקטע הבא בעברית. "
        f"הצג כל שאלה בשורה חדשה ולאחריה את התשובה באותה שורה בצורה ברורה.\n\n"
        f"קטע:\n{text}\n"
        f"שאלות ותשובות:"
    )
    try:
        # We generate up to 1024 new tokens to accommodate several pairs.
        outputs = qa_gen(prompt, max_new_tokens=1024, do_sample=False)
        if not outputs:
            return []
        generated = outputs[0]["generated_text"]
        # Find the portion after the prompt
        # We look for the delimiter "שאלות ותשובות:" which we included
        idx = generated.find("שאלות ותשובות:")
        if idx >= 0:
            generated = generated[idx + len("שאלות ותשובות:"):].strip()
        lines = [ln.strip() for ln in generated.split('\n') if ln.strip()]
        pairs: List[Tuple[str, str]] = []
        for line in lines:
            # Attempt to split question and answer by the last question mark
            qm_pos = line.rfind('?')
            if qm_pos > 0:
                question = line[: qm_pos + 1].strip()
                answer = line[qm_pos + 1:].strip()
                # Remove leading numbering (e.g., "1.") if present
                question = re.sub(r'^\d+\.\s*', '', question)
                answer = re.sub(r'^[-\d\.\s]+', '', answer)
                pairs.append((question, answer))
            if len(pairs) >= num_pairs:
                break
        return pairs
    except Exception:
        return []


def summarize_text(text: str, max_length: int = 100) -> Optional[str]:
    """Return a concise summary of ``text`` using the Hebrew summariser.

    If the summariser is unavailable or an error occurs, ``None`` is
    returned so that callers can fall back to extractive techniques.

    Parameters
    ----------
    text : str
        The input text to summarise.
    max_length : int
        The maximum number of tokens to generate in the summary.  The
        default of 100 tokens generally produces a few sentences.
    Returns
    -------
    Optional[str]
        A summary string, or ``None`` if summarisation failed.
    """
    summariser = _get_summariser()
    if summariser is None:
        return None
    try:
        summary_outputs = summariser(
            text,
            max_length=max_length,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        if summary_outputs:
            return summary_outputs[0]["summary_text"].strip()
    except Exception:
        # Summarisation failed unexpectedly
        return None
    return None


# -----------------------------------------------------------------------------
# Text extraction utilities
# -----------------------------------------------------------------------------

def _parse_docx(path: str) -> str:
    """Extract raw text from a .docx file.

    A .docx file is a zip archive containing several XML parts.  The
    paragraphs live in ``word/document.xml``.  We read that file,
    strip XML tags and return the concatenated paragraph text.  This
    approach avoids the need for the ``python-docx`` library which
    cannot be installed in the restricted environment.
    """
    text = []
    with zipfile.ZipFile(path) as docx_zip:
        with docx_zip.open('word/document.xml') as document_xml:
            xml = document_xml.read().decode('utf-8')
    # Extract text between <w:t> tags
    texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', xml)
    # If there are paragraph breaks, we attempt to split them
    # Roughly count paragraphs by counting closing paragraph tags
    paragraphs = []
    para_breaks = [m.start() for m in re.finditer(r'</w:p>', xml)]
    if para_breaks:
        # Create a simple segmentation: divide runs evenly among paragraphs
        runs_per_para = max(1, len(texts) // len(para_breaks))
        for i in range(0, len(texts), runs_per_para):
            paragraphs.append(' '.join(texts[i:i + runs_per_para]))
    else:
        paragraphs = [' '.join(texts)]
    return '\n'.join(paragraphs)


def _parse_pdf(path: str) -> str:
    """Extract text from a PDF using the local pdf reader service.

    The execution environment exposes a pdf reader at
    ``http://localhost:8451`` which can parse PDF files and return
    their text as JSON.  We send a GET request to the service with a
    ``file://`` URL pointing to the absolute file path.  The service
    responds with a structure containing the extracted text for each
    page.  We concatenate the pages into a single string.
    """
    abs_path = os.path.abspath(path)
    uri = f'file://{abs_path}'
    url = f'http://localhost:8451/{uri}'
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get('pages', [])
    text_pages = []
    for page in pages:
        page_text = page.get('text', '')
        text_pages.append(page_text.strip())
    return '\n'.join(text_pages)


def _parse_md(path: str) -> str:
    """Read a markdown file and strip simple formatting.

    We remove common markdown syntax such as headings (``#``),
    emphasis (``*`` or ``_``) and links.  This is a minimal
    implementation; more sophisticated markdown parsing could be
    implemented using a library when network access is available.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'^[#]+\s*', '', text, flags=re.MULTILINE)
    text = text.replace('*', '').replace('_', '')
    text = re.sub(r'\[[^\]]*\]\([^\)]*\)', '\1', text)
    return text


def extract_text(input_data: str) -> str:
    """Return plain text from input data.

    ``input_data`` may be either a raw text string or a file path.  If
    the string corresponds to an existing file on disk it will be
    parsed according to its extension.  Otherwise the string is
    returned as‑is.

    Supported file types:

    * ``.docx`` – parsed with the internal XML parser.
    * ``.pdf`` – parsed via the pdf reader service.
    * ``.md`` – stripped of markdown syntax.
    * Other files – read as plain UTF‑8 text.
    """
    if os.path.exists(input_data):
        ext = os.path.splitext(input_data)[1].lower()
        if ext == '.docx':
            text = _parse_docx(input_data)
        elif ext == '.pdf':
            text = _parse_pdf(input_data)
        elif ext == '.md':
            text = _parse_md(input_data)
        else:
            with open(input_data, 'r', encoding='utf-8') as f:
                text = f.read()
        return _clean_text(text)
    else:
        return _clean_text(input_data)


# -----------------------------------------------------------------------------
# Text cleaning helper
# -----------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Normalise and clean extracted text.

    This helper collapses consecutive whitespace into single spaces and
    removes non‑printable control characters.  It preserves sentence
    punctuation so that downstream sentence splitting continues to work.
    """
    if not text:
        return ''
    # Replace newlines and tabs with spaces
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove non‑printable characters
    text = ''.join(ch for ch in text if ch.isprintable())
    return text.strip()


# -----------------------------------------------------------------------------
# Text processing and flashcard generation
# -----------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Naively split text into sentences.

    We look for sentence terminating punctuation marks common in
    Hebrew and other languages: period (``.``), exclamation mark
    (``!``), question mark (``?``) and the Hebrew sof pasuq (``׃``).
    The function returns a list of trimmed sentences.
    """
    if not text:
        return []
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<=[\.\!?׃])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _choose_keyword(sentence: str, vectorizer: TfidfVectorizer, tfidf_matrix, sentence_index: int) -> str:
    """
    Pick the most salient single‑word token in a sentence based on TF‑IDF weights.

    This helper examines the TF‑IDF weights produced by ``vectorizer`` for
    the given sentence and returns the highest scoring token that meets
    a few criteria:

    * The token must have a positive TF‑IDF weight.
    * It must not be purely numeric.
    * It must be at least 3 characters long.
    * It must not contain whitespace (i.e. avoid multi‑word bigrams).

    Using only single words avoids unnatural multi‑word blanks such as
    ``"הם חיות"`` which can be difficult to guess.  If no suitable token
    is found then we fall back to the first word of the sentence.
    """
    # Extract TF‑IDF scores for the sentence
    row = tfidf_matrix[sentence_index].toarray().flatten()
    tokens = vectorizer.get_feature_names_out()
    # Pair tokens with their scores and sort descending
    token_scores = [(tok, row[i]) for i, tok in enumerate(tokens)]
    token_scores.sort(key=lambda x: x[1], reverse=True)
    for tok, score in token_scores:
        # Skip zero‑weight tokens
        if score <= 0:
            continue
        # Skip pure numbers
        if tok.isdigit():
            continue
        # Skip very short tokens
        if len(tok) < 3:
            continue
        # Ignore tokens containing whitespace (avoid bigrams)
        if any(ch.isspace() for ch in tok):
            continue
        return tok
    # Fallback: use the first word of the sentence
    words = sentence.split()
    return words[0] if words else ''


def _blank_sentence(sentence: str, keyword: str) -> str:
    """Replace all occurrences of ``keyword`` in ``sentence`` with blanks.

    The replacement is case‑insensitive and uses underscores to mark
    the missing word.  If the keyword does not appear in the sentence
    (e.g., due to case differences) the sentence is returned
    unchanged.
    """
    pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
    blank = '_' * max(len(keyword), 4)
    new_sentence = pattern.sub(blank, sentence)
    return new_sentence


@dataclass
class Flashcard:
    """Represents a single flashcard with a question and an answer."""
    question: str
    answer: str
    context: str = field(default='')


def generate_flashcards(text: str, num_cards: int = 20) -> List[Flashcard]:
    """Generate a list of flashcards from the input text.

    This function first attempts to identify explicit sections or headings
    in the text based on delimiters such as a colon (``:``), hyphen (``-``)
    or en dash (``–``).  When such a heading is found, the portion
    preceding the delimiter is taken as a *concept* and the following
    lines until the next heading constitute the *definition*.  Each
    concept–definition pair becomes a flashcard where the front asks
    ``מהי …?`` or ``מהם …?`` depending on whether the concept appears to
    be singular or plural.  If no suitable headings are discovered the
    function falls back to extractive summarisation: it ranks sentences
    by TF–IDF, selects the top ``num_cards`` sentences and forms
    questions of the form ``מהו X?`` where ``X`` is a salient keyword
    extracted from the sentence.

    Parameters
    ----------
    text : str
        The source text from which to generate flashcards.
    num_cards : int, optional
        Maximum number of flashcards to return.  Default is 20.

    Returns
    -------
    List[Flashcard]
        A list of flashcards derived from the text.
    """
    # Attempt to extract concept–definition sections
    sections: List[Tuple[str, str]] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    current_heading: Optional[str] = None
    current_content: List[str] = []

    def flush_section() -> None:
        """Finalize the current section by appending it to the sections list."""
        nonlocal current_heading, current_content
        if current_heading:
            content = " ".join(current_content).strip()
            if content:
                sections.append((current_heading, content))
        current_heading = None
        current_content = []

    for ln in lines:
        # Look for a heading delimiter. We only treat a colon, " – " (en dash with spaces)
        # or " - " (hyphen with spaces) as section separators.  This avoids
        # splitting words such as "חד-פעמי".
        delim = None
        for d in (':', ' – ', ' - '):
            if d in ln:
                delim = d
                break
        if delim:
            parts = ln.split(delim, 1)
            heading = parts[0].strip()
            remainder = parts[1].strip()
            flush_section()
            current_heading = heading
            if remainder:
                current_content.append(remainder)
        else:
            current_content.append(ln)
    flush_section()

    flashcards: List[Flashcard] = []
    
    def _summarise_answer(ans: str) -> str:
        """Return a concise version of ``ans`` if a summariser is available.

        When the answer text is long (more than ~40 words) we attempt to
        shorten it using the ``summarize_text`` function defined above.  This
        uses a Hebrew transformer model to extract the main points of the
        answer.  If summarisation is unavailable or fails, the original
        answer is returned unchanged.  The maximum length is scaled to
        roughly one third of the original word count to preserve salient
        details without overwhelming the learner.

        Parameters
        ----------
        ans : str
            The original answer string.

        Returns
        -------
        str
            A potentially shortened version of the answer.
        """
        words = ans.split()
        if len(words) < 40:
            return ans
        # Compute a max_length proportional to the input size, capped at 120
        max_len = max(40, min(len(words) // 3, 120))
        summary = summarize_text(ans, max_length=max_len)
        if summary and len(summary.split()) < len(words):
            # Remove trailing punctuation from summary for consistency
            return summary.strip()
        return ans
    # Build flashcards from structured sections if any were found
    #
    # In addition to simple concept–definition pairs, attempt to derive
    # more specific question/answer pairs from the text.  We look for
    # sentences that state for whom a policy is intended ("מיועד/ת ל…"),
    # bullet‑style enumerations of different plans (lines beginning
    # with "כלבים" or "חתולים"), and comparative phrases describing
    # advantages versus competitors ("מול X, הפניקס …").  These
    # heuristics help produce flashcards closer to the user’s
    # examples, such as asking to which animals a policy applies or
    # summarising multiple insurance tracks.
    if sections:
        for heading, content in sections:
            # Skip enumerated lines that describe specific age categories (e.g.
            # "כלבים וחתולים עד גיל 8") since they will be summarised
            # separately as part of the plan enumeration flashcard.  Including
            # them here leads to unnatural questions such as "מהי כלבים וחתולים עד גיל 8".
            if re.match(r'^(כלבים|חתולים)', heading):
                continue
            # Choose interrogative based on simple plural heuristic
            interrogative = "מהי"
            if heading.endswith("ים") or heading.endswith("ות") or heading.endswith("אות"):
                interrogative = "מהם"
            question = f"{interrogative} {heading}?"
            answer = _summarise_answer(content)
            flashcards.append(Flashcard(question=question, answer=answer, context=content))
        # Derive additional Q/A pairs from within the section content
        # Split the entire text into sentences for pattern matching
        sentences = _split_sentences(text)
        extra_cards: List[Flashcard] = []
        # 1. Questions about intended audience (מיועדת ל…)
        for sent in sentences:
            # Look for phrases like "פוליסת הפניקס מיועדת לכלבים וחתולים…"
            m = re.search(r'([^\.\n\?]+?)\s+מיועד[ת] ל\s*([^\.\n]+)', sent)
            if m:
                subject = m.group(1).strip()
                targets = m.group(2).strip()
                # Formulate question.  If the subject contains the word
                # "פוליסת" then ask "לאיזה בעלי חיים מיועדת …?", otherwise
                # ask a generic "למי מיועדת …?".  Strip leading determiners
                # such as "ה" for better readability.
                subj_clean = subject.strip()
                # Remove leading "ה" if present to avoid unnatural phrasing
                subj_clean = re.sub(r'^ה', '', subj_clean)
                # If the subject mentions a policy, attempt to extract just the
                # policy name (e.g. "פוליסת הפניקס") rather than the entire
                # heading.  This avoids overly long questions such as
                # "פוליסת הפניקס – עיקרי התנאים והכיסויים פוליסת הפניקס".
                if 'פוליס' in subj_clean:
                    mpol = re.search(r'(פוליס[^\s]*\s+\S+)', subj_clean)
                    if mpol:
                        subj_clean = mpol.group(1).strip()
                # Formulate specific or generic question depending on subject
                if 'פוליס' in subj_clean:
                    q = f"לאיזה בעלי חיים מיועדת {subj_clean}?"
                else:
                    q = f"למי מיועדת {subj_clean}?"
                answer = _summarise_answer(targets.strip())
                extra_cards.append(Flashcard(question=q, answer=answer, context=sent))
        # 2. Summarise enumerated plans beginning with "כלבים" or "חתולים"
        enumerated: List[List[str]] = []
        current_enum: List[str] = []
        for ln in lines:
            if re.match(r'^(כלבים|חתולים)', ln):
                current_enum.append(ln)
            else:
                if current_enum:
                    enumerated.append(current_enum)
                    current_enum = []
        if current_enum:
            enumerated.append(current_enum)
        for enum in enumerated:
            # Join the enumeration lines into a single answer string
            ans = " ".join(enum)
            # Only create this card if there are at least two items (otherwise
            # it’s just a single plan and will already be covered by the
            # concept‑definition card)
            if len(enum) >= 2:
                q = "מה הם המסלולים וסכומי הביטוח שלהם בהפניקס?"
                answer = _summarise_answer(ans)
                extra_cards.append(Flashcard(question=q, answer=answer, context=ans))
        # 3. Comparative advantages versus competitors
        for sent in sentences:
            m = re.search(r'מול\s+([^,]+),\s*הפניקס\s+([^\.]+)', sent)
            if m:
                competitor = m.group(1).strip()
                advantage = m.group(2).strip()
                q = f"מה יתרונותיה של הפניקס מול {competitor}?"
                answer = _summarise_answer(advantage)
                extra_cards.append(Flashcard(question=q, answer=answer, context=sent))
        # Merge extra cards into flashcards while preserving order and
        # limiting to the requested number.  We append extra_cards after
        # the primary section cards so that high‑level concepts appear
        # first.  Duplicate questions are filtered.
        seen_questions: set[str] = set()
        final_cards: List[Flashcard] = []
        for card in flashcards + extra_cards:
            if card.question not in seen_questions:
                final_cards.append(card)
                seen_questions.add(card.question)
            if len(final_cards) >= num_cards:
                break
        # ------------------------------------------------------------------
        # Scenario cards: derive "אם לקוח מתלבט" questions from comparative advantages
        #
        # For each card that asks about advantages versus a competitor ("מה יתרונותיה של X מול Y")
        # generate an additional scenario card that frames the same answer as a customer dilemma.
        # These cards help practise handling objections and explaining benefits in a sales context.
        scenario_cards: List[Flashcard] = []
        for card in list(final_cards):
            m = re.match(r"מה יתרונותיה של\s+(.*?)\s+מול\s+(.*)\?", card.question)
            if m:
                subject = m.group(1).strip()
                competitor = m.group(2).strip()
                scenario_q = f"אם לקוח מתלבט בין {subject} ל{competitor}, כיצד תציג את היתרונות?"
                scenario_cards.append(Flashcard(question=scenario_q, answer=card.answer, context=card.context))
        for sc in scenario_cards:
            if len(final_cards) >= num_cards:
                break
            if sc.question not in seen_questions:
                final_cards.append(sc)
                seen_questions.add(sc.question)
        return final_cards

    # Fallback: extractive summarisation
    sentences = _split_sentences(text)
    if not sentences:
        return []
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1).A1
    ranked_indices = np.argsort(scores)[::-1]
    selected_indices = ranked_indices[: num_cards]
    for idx in selected_indices:
        sentence = sentences[idx]
        keyword = _choose_keyword(sentence, vectorizer, tfidf_matrix, idx)
        question = f"מהו {keyword}?"
        answer = sentence.strip()
        flashcards.append(Flashcard(question=question, answer=answer, context=sentence))
    return flashcards


@dataclass
class MCQQuestion:
    """Represents a multiple choice question with options."""
    prompt: str
    options: List[str]
    correct_index: int


def generate_mcq(flashcards: Sequence[Flashcard], num_questions: int = 10) -> List[MCQQuestion]:
    """Create a multiple choice quiz from a list of flashcards.

    For each question we use the flashcard's question as the prompt and
    prepare four answer choices: the correct answer and three
    distractors drawn at random from the answers of other flashcards.
    If there are fewer than four unique answers available we will
    duplicate distractors as necessary.  The returned list is
    truncated or padded to ``num_questions`` questions.
    """
    if not flashcards:
        return []
    # Helper to shorten long definitions for use in MCQ options.  We
    # truncate to a maximum number of words so that answer choices
    # remain concise.  This avoids entire paragraphs appearing as
    # options, which can overwhelm the learner.  See docs
    #  【280062428660715†L46-L67】 for discussion of extractive summarisation.
    def _shorten_definition(defn: str, max_words: int = 25) -> str:
        """Return a shortened version of the definition limited to
        ``max_words`` words.  If the definition is shorter than the
        limit it is returned unchanged.  An ellipsis is appended to
        indicate truncation."""
        words = defn.split()
        if len(words) <= max_words:
            return defn.strip()
        return " ".join(words[:max_words]).strip() + " …"

    # Pre‑compute shortened versions of all flashcard answers for use
    # when building multiple‑choice options.  We use the original
    # definitions as keys to preserve mapping back to full answers.
    all_answers = [fc.answer for fc in flashcards]
    shortened_lookup = {ans: _shorten_definition(ans) for ans in all_answers}
    questions: List[MCQQuestion] = []
    indices = list(range(len(flashcards)))
    random.shuffle(indices)
    for idx in indices[:num_questions]:
        fc = flashcards[idx]
        correct = fc.answer
        distractors: List[str] = []
        other_answers = [ans for ans in all_answers if ans != correct]
        # Select up to three distractor definitions from other flashcards
        if other_answers:
            distractors = random.sample(other_answers, k=min(3, len(other_answers)))
        # If there are fewer than three distractors, pad with the correct answer to avoid index errors
        while len(distractors) < 3:
            distractors.append(correct)
        # Build options list using shortened versions for readability
        options_full = distractors + [correct]
        # Shuffle options to randomise order
        random.shuffle(options_full)
        # Determine index of correct answer
        correct_index = options_full.index(correct)
        # Replace options with shortened strings
        options = [shortened_lookup.get(opt, opt) for opt in options_full]
        # Build a clearer prompt: ask for the definition of the concept rather than the concept itself.
        # Extract the keyword from the original question, removing pronouns such as
        # "מהו", "מהי" or "מהם" at the start and any trailing question mark.  For example
        # "מהם יתרונות הפניקס?" -> "יתרונות הפניקס".
        keyword = re.sub(r'^(מהו|מהי|מהם)\s+', '', fc.question)
        keyword = re.sub(r'\?$', '', keyword).strip()
        prompt = f"מהי ההגדרה של {keyword}?"
        questions.append(MCQQuestion(prompt=prompt, options=options, correct_index=correct_index))
    return questions


def evaluate_quiz(questions: Sequence[MCQQuestion], user_answers: Sequence[int]) -> Tuple[int, List[int]]:
    """Evaluate user answers to the MCQ quiz.

    Returns a tuple containing the total score (number of correct
    answers) and a list of indices indicating which questions were
    answered incorrectly.
    """
    score = 0
    wrong_indices: List[int] = []
    for i, (q, ans) in enumerate(zip(questions, user_answers)):
        if ans == q.correct_index:
            score += 1
        else:
            wrong_indices.append(i)
    return score, wrong_indices


def generate_additional_flashcards(flashcards: Sequence[Flashcard], wrong_indices: Sequence[int], count: int = 5) -> List[Flashcard]:
    """Generate additional flashcards for incorrectly answered topics.

    We reuse the original context sentences of the incorrectly answered
    cards, but create new questions by selecting a different keyword if
    possible.  This encourages the learner to engage with the same
    information from another angle.  The number of new cards can be
    controlled with ``count``.
    """
    additional: List[Flashcard] = []
    # Extract contexts for incorrectly answered flashcards
    contexts = [flashcards[i].context for i in wrong_indices]
    if not contexts:
        return additional
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(contexts)
    for i, ctx in enumerate(contexts[:count]):
        # Extract the original keyword from the question (handles 'מהו', 'מהי', 'מהם')
        orig_question = flashcards[wrong_indices[i]].question
        original_keyword = re.sub(r'^(מהו|מהי|מהם)\s+', '', orig_question)
        original_keyword = re.sub(r'\?$', '', original_keyword).strip()
        row = tfidf_matrix[i].toarray().flatten()
        tokens = vectorizer.get_feature_names_out()
        token_scores = [(tok, row[j]) for j, tok in enumerate(tokens)]
        token_scores.sort(key=lambda x: x[1], reverse=True)
        new_keyword = original_keyword
        for tok, score in token_scores:
            if score > 0 and tok != original_keyword and len(tok) >= 3:
                new_keyword = tok
                break
        # Formulate new flashcard: ask about the new keyword, answer is the full sentence
        question = f"מהו {new_keyword}?"
        additional.append(Flashcard(question=question, answer=ctx.strip(), context=ctx))
    return additional


def simulate_quiz_session(text: str, flashcard_count: int = 20, quiz_length: int = 10) -> None:
    """Simulate an interactive session in the console.

    This helper function demonstrates the usage of the API.  It
    generates flashcards and a quiz from the provided text, asks the
    user to answer each question via the terminal, evaluates the
    responses and then shows additional flashcards for reinforcement.
    """
    flashcards = generate_flashcards(text, num_cards=flashcard_count)
    mcq = generate_mcq(flashcards, num_questions=quiz_length)
    print(f"נוצרו {len(flashcards)} כרטיסיות למידה ו-{len(mcq)} שאלות אמריקאיות.")
    user_answers: List[int] = []
    for idx, question in enumerate(mcq):
        print(f"\nשאלה {idx + 1}: {question.prompt}")
        for opt_idx, opt in enumerate(question.options):
            print(f"  {opt_idx + 1}. {opt}")
        while True:
            try:
                choice = int(input("בחר מספר תשובה: ")) - 1
                if 0 <= choice < len(question.options):
                    user_answers.append(choice)
                    break
            except ValueError:
                pass
            print("בחירה לא חוקית. נסה שוב.")
    score, wrong = evaluate_quiz(mcq, user_answers)
    print(f"\nהציון שלך: {score}/{len(mcq)}")
    if wrong:
        print("טעויות התגלו בשאלות הבאות: ", [i + 1 for i in wrong])
        additional = generate_additional_flashcards(flashcards, wrong)
        if additional:
            print("\nכרטיסיות נוספות לחיזוק:")
            for fc in additional:
                print(f"שאלה: {fc.question}\nתשובה: {fc.answer}\n")
    else:
        print("כל הכבוד! ענית נכון על כל השאלות.")


def load_and_run(path_or_text: str, flashcard_count: int = 20, quiz_length: int = 10) -> None:
    """Convenience wrapper to extract text and run a quiz session.

    Pass a file path or raw text to ``path_or_text``.  The function
    will parse the file if necessary, generate flashcards and quiz
    questions, and launch the console interaction.  This is a useful
    entry point when running the module as a script.
    """
    text = extract_text(path_or_text)
    simulate_quiz_session(text, flashcard_count=flashcard_count, quiz_length=quiz_length)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hebrew Flashcard and Quiz Generator')
    parser.add_argument('input', help='Path to a .docx, .pdf, .md file or raw text')
    parser.add_argument('--cards', type=int, default=20, help='Number of flashcards to generate')
    parser.add_argument('--questions', type=int, default=10, help='Number of MCQ questions')
    args = parser.parse_args()
    load_and_run(args.input, flashcard_count=args.cards, quiz_length=args.questions)