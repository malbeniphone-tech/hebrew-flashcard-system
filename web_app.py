"""
A lightweight HTTP server providing a web interface to the flashcard system.

The server is implemented with Python's built‑in ``http.server`` and
``cgi`` modules to avoid external dependencies.  It offers three
endpoints:

* ``GET /`` – Presents a form where users can paste text, upload a
  file, or supply a remote URL.  Users can also set the number of
  flashcards and quiz questions.  Submission posts to ``/generate``.

* ``POST /generate`` – Processes the input, generates flashcards and
  a quiz, stores them in an in‑memory session and returns an HTML
  page showing the flashcards and a quiz form.  The page contains a
  hidden field with the session identifier and posts to ``/quiz``.

* ``POST /quiz`` – Evaluates the submitted quiz answers, reports the
  score, shows which questions were missed and displays additional
  flashcards to reinforce weak areas.

Note: This server is intended for development and demonstration
purposes.  It does not implement authentication or persistent
storage.  In a production environment you should deploy the
application using a proper WSGI framework and host.
"""

import cgi
import html
import io
import os
import random
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import requests
import re

from flashcard_system import (
    extract_text,
    generate_flashcards,
    generate_mcq,
    evaluate_quiz,
    generate_additional_flashcards,
    Flashcard,
    MCQQuestion,
)


class SessionData:
    """Stores flashcards and quiz data for a user session."""

    def __init__(self, flashcards, mcqs):
        self.flashcards: list[Flashcard] = flashcards
        self.mcqs: list[MCQQuestion] = mcqs


class FlashcardRequestHandler(BaseHTTPRequestHandler):
    # In‑memory session store; maps session IDs to SessionData
    sessions: dict[str, SessionData] = {}

    def _send_html(self, content: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def do_GET(self):
        if self.path == '/':
            self._send_html(self.render_home())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/generate':
            self.handle_generate()
        elif self.path == '/quiz':
            self.handle_quiz()
        else:
            self.send_error(404)

    # ------------------------------------------------------------------
    # Page rendering
    # ------------------------------------------------------------------

    def render_home(self) -> str:
        """Return the HTML for the home page with the input form."""
        return f"""
<!DOCTYPE html>
<html lang="he">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>יצירת כרטיסיות ובחינה</title>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }}
    .container {{ max-width: 800px; margin: 20px auto; padding: 20px; background: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h1 {{ text-align: center; }}
    label {{ display: block; margin-top: 10px; font-weight: bold; }}
    input[type="text"], textarea {{ width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
    input[type="number"] {{ width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; }}
    input[type="file"] {{ margin-top: 4px; }}
    button {{ margin-top: 15px; background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
    button:hover {{ background-color: #0056b3; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>יצירת כרטיסיות ובחינה</h1>
    <form action="/generate" method="post" enctype="multipart/form-data">
      <label for="text">מלל חופשי:</label>
      <textarea id="text" name="text" rows="6" placeholder="כתוב כאן טקסט בעברית"></textarea>

      <label for="url">או הזן קישור למסמך:</label>
      <input type="text" id="url" name="url" placeholder="https://example.com/file.txt">

      <label for="file">או העלה קובץ:</label>
      <input type="file" id="file" name="file">

      <label for="cards">מספר כרטיסיות (ברירת מחדל 20):</label>
      <input type="number" id="cards" name="cards" value="20" min="1" max="100">

      <label for="questions">מספר שאלות במבחן (ברירת מחדל 10):</label>
      <input type="number" id="questions" name="questions" value="10" min="1" max="50">

      <button type="submit">צור</button>
    </form>
  </div>
</body>
</html>
        """

    def render_quiz(self, session_id: str, flashcards: list[Flashcard], mcqs: list[MCQQuestion]) -> str:
        """Return HTML for the quiz page with interactive flashcards and questions."""
        # Build interactive flashcards: each card flips on click to reveal answer and context
        cards_html_parts = []
        for fc in flashcards:
            # Determine the keyword from the question (formats: מהו/מהי/מהם <keyword>?) for highlighting
            keyword = re.sub(r'^(מהו|מהי|מהם)\s+', '', fc.question)
            keyword = re.sub(r'\?$', '', keyword).strip()
            # Highlight the keyword in the context for the back side
            try:
                pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
                highlighted_context = pattern.sub(lambda m: f"<strong>{m.group(0)}</strong>", fc.context, count=1)
            except Exception:
                highlighted_context = html.escape(fc.context)
            front = f"<strong>מושג:</strong> {html.escape(keyword)}"
            back = f"<strong>הגדרה:</strong> {html.escape(fc.answer)}<br><small>{highlighted_context}</small>"
            card_html = f"""
        <div class="flashcard" onclick="flipCard(this)">
          <div class="flashcard-inner">
            <div class="flashcard-front">{front}</div>
            <div class="flashcard-back">{back}</div>
          </div>
        </div>
            """
            cards_html_parts.append(card_html)
        flashcards_html = '\n'.join(cards_html_parts)
        # Only render flashcards; no CSV export is generated. The interactive cards themselves are intended for practice.
        # Build the quiz form
        question_forms = []
        for q_idx, q in enumerate(mcqs):
            opts_html = ''.join(
                f"<label><input type=\"radio\" name=\"q{q_idx}\" value=\"{i}\" required> {html.escape(opt)}</label><br>"
                for i, opt in enumerate(q.options)
            )
            question_forms.append(f"<div><p><strong>שאלה {q_idx + 1}:</strong> {html.escape(q.prompt)}</p>{opts_html}</div>")
        quiz_html = ''.join(question_forms)
        return f"""
<!DOCTYPE html>
<html lang="he">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>מבחן כרטיסיות</title>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }}
    .container {{ max-width: 900px; margin: 20px auto; padding: 20px; background: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h2 {{ text-align: center; }}
    .flashcard-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-bottom: 20px; }}
    .flashcard {{ width: 200px; height: 140px; perspective: 1000px; cursor: pointer; }}
    .flashcard-inner {{ position: relative; width: 100%; height: 100%; transition: transform 0.6s; transform-style: preserve-3d; }}
    .flashcard.is-flipped .flashcard-inner {{ transform: rotateY(180deg); }}
    .flashcard-front, .flashcard-back {{ position: absolute; width: 100%; height: 100%; backface-visibility: hidden; border: 1px solid #ccc; border-radius: 8px; padding: 10px; box-sizing: border-box; display: flex; justify-content: center; align-items: center; text-align: center; }}
    .flashcard-back {{ transform: rotateY(180deg); background-color: #f8f8f8; }}
    button {{ margin-top: 15px; background-color: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
    button:hover {{ background-color: #218838; }}
  </style>
  <script>
    function flipCard(el) {{
      el.classList.toggle('is-flipped');
    }}
  </script>
</head>
<body>
  <div class="container">
    <h2>כרטיסיות שנוצרו</h2>
    <div class="flashcard-container">
      {flashcards_html}
    </div>
    <h2>מבחן</h2>
    <form action="/quiz" method="post">
      <input type="hidden" name="session_id" value="{session_id}">
      {quiz_html}
      <button type="submit">שלח תשובות</button>
    </form>
  </div>
</body>
</html>
        """

    def render_results(self, score: int, total: int, wrong_indices: list[int], additional: list[Flashcard]) -> str:
        """Return HTML for the results page with optional reinforcement flashcards."""
        wrong_str = ', '.join(str(i + 1) for i in wrong_indices) if wrong_indices else 'לא היו טעויות'
        # Build interactive additional flashcards if any
        additional_html = ''
        if additional:
            cards_parts = []
            for fc in additional:
                # Extract keyword from the question of the additional card
                keyword = re.sub(r'^(מהו|מהי|מהם)\s+', '', fc.question)
                keyword = re.sub(r'\?$', '', keyword).strip()
                # Highlight the keyword in the context
                try:
                    pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
                    highlighted = pattern.sub(lambda m: f"<strong>{m.group(0)}</strong>", fc.context, count=1)
                except Exception:
                    highlighted = html.escape(fc.context)
                front = f"<strong>מושג:</strong> {html.escape(keyword)}"
                back = f"<strong>הגדרה:</strong> {html.escape(fc.answer)}<br><small>{highlighted}</small>"
                cards_parts.append(f"""
            <div class="flashcard" onclick="flipCard(this)">
              <div class="flashcard-inner">
                <div class="flashcard-front">{front}</div>
                <div class="flashcard-back">{back}</div>
              </div>
            </div>
                """)
            cards_html = '\n'.join(cards_parts)
            additional_html = f"""
    <h2>כרטיסיות לחיזוק</h2>
    <div class="flashcard-container">
      {cards_html}
    </div>
            """
        return f"""
<!DOCTYPE html>
<html lang="he">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>תוצאות מבחן</title>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; }}
    .container {{ max-width: 900px; margin: 20px auto; padding: 20px; background: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h2 {{ text-align: center; }}
    a {{ color: #007bff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .flashcard-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-bottom: 20px; }}
    .flashcard {{ width: 200px; height: 140px; perspective: 1000px; cursor: pointer; }}
    .flashcard-inner {{ position: relative; width: 100%; height: 100%; transition: transform 0.6s; transform-style: preserve-3d; }}
    .flashcard.is-flipped .flashcard-inner {{ transform: rotateY(180deg); }}
    .flashcard-front, .flashcard-back {{ position: absolute; width: 100%; height: 100%; backface-visibility: hidden; border: 1px solid #ccc; border-radius: 8px; padding: 10px; box-sizing: border-box; display: flex; justify-content: center; align-items: center; text-align: center; }}
    .flashcard-back {{ transform: rotateY(180deg); background-color: #f8f8f8; }}
  </style>
  <script>
    function flipCard(el) {{
      el.classList.toggle('is-flipped');
    }}
  </script>
</head>
<body>
  <div class="container">
    <h2>תוצאות</h2>
    <p>הציון שלך: {score}/{total}</p>
    <p>שאלות בהן טעית: {wrong_str}</p>
    {additional_html}
    <p><a href="/">חזרה לעמוד הראשי</a></p>
  </div>
</body>
</html>
        """

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def handle_generate(self):
        """Handle POST to /generate: parse input and build quiz."""
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
            'REQUEST_METHOD': 'POST',
            'CONTENT_TYPE': self.headers['Content-Type'],
        })
        text_input = form.getvalue('text') or ''
        url_input = form.getvalue('url') or ''
        file_item = form['file'] if 'file' in form else None
        num_cards = int(form.getvalue('cards') or 20)
        num_questions = int(form.getvalue('questions') or 10)
        # Determine source of text
        raw_text = ''
        if file_item is not None and file_item.filename:
            file_data = file_item.file.read()
            # Save uploaded file to a temporary location
            tmp_path = f'/tmp/upload_{secrets.token_hex(8)}'
            with open(tmp_path, 'wb') as tmp_file:
                tmp_file.write(file_data)
            raw_text = extract_text(tmp_path)
            os.remove(tmp_path)
        elif url_input:
            try:
                resp = requests.get(url_input)
                resp.raise_for_status()
                # If the content type is text or JSON we treat as text; otherwise we save and extract
                content_type = resp.headers.get('Content-Type', '')
                if 'text' in content_type or 'json' in content_type:
                    raw_text = resp.text
                else:
                    # Save binary content
                    tmp_path = f'/tmp/remote_{secrets.token_hex(8)}'
                    with open(tmp_path, 'wb') as tmp_file:
                        tmp_file.write(resp.content)
                    raw_text = extract_text(tmp_path)
                    os.remove(tmp_path)
            except Exception as e:
                raw_text = ''
        elif text_input:
            raw_text = text_input
        else:
            # No input provided
            self._send_html('<p>שגיאה: לא סופק מלל או קובץ.</p><p><a href="/">חזור</a></p>', status=400)
            return
        # Generate flashcards and quiz
        flashcards = generate_flashcards(raw_text, num_cards)
        mcqs = generate_mcq(flashcards, num_questions)
        # Store session
        session_id = secrets.token_hex(8)
        FlashcardRequestHandler.sessions[session_id] = SessionData(flashcards, mcqs)
        self._send_html(self.render_quiz(session_id, flashcards, mcqs))

    def handle_quiz(self):
        """Handle POST to /quiz: evaluate quiz and show results."""
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length).decode('utf-8')
        params = parse_qs(body)
        session_id = params.get('session_id', [''])[0]
        session = FlashcardRequestHandler.sessions.get(session_id)
        if not session:
            self._send_html('<p>שגיאה: מושב לא קיים. אנא התחל מחדש.</p><p><a href="/">חזור</a></p>', status=400)
            return
        user_answers: list[int] = []
        # Extract answers: keys like q0, q1...
        for idx in range(len(session.mcqs)):
            ans_list = params.get(f'q{idx}', [])
            if ans_list:
                try:
                    user_answers.append(int(ans_list[0]))
                except ValueError:
                    user_answers.append(-1)
            else:
                user_answers.append(-1)
        score, wrong = evaluate_quiz(session.mcqs, user_answers)
        additional = generate_additional_flashcards(session.flashcards, wrong)
        # Remove session to free memory
        del FlashcardRequestHandler.sessions[session_id]
        self._send_html(self.render_results(score, len(session.mcqs), wrong, additional))


def run_server(host: str = '0.0.0.0', port: int = 8000):
    """Start the HTTP server."""
    server = HTTPServer((host, port), FlashcardRequestHandler)
    print(f"Server running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.shutdown()


if __name__ == '__main__':
    run_server()