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
    make_short_question,
    make_bullets,
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
        # Serve the home page or static assets
        if self.path == '/':
            self._send_html(self.render_home())
        elif self.path.startswith('/static/'):
            # Serve static files from the 'static' directory relative to this script
            rel_path = self.path.lstrip('/')
            fs_path = os.path.join(os.path.dirname(__file__), rel_path)
            if os.path.isfile(fs_path):
                try:
                    with open(fs_path, 'rb') as f:
                        data = f.read()
                    # Determine a simple content type based on extension
                    if fs_path.endswith('.css'):
                        ctype = 'text/css'
                    elif fs_path.endswith('.js'):
                        ctype = 'application/javascript'
                    else:
                        ctype = 'application/octet-stream'
                    self.send_response(200)
                    self.send_header("Content-type", ctype)
                    self.end_headers()
                    self.wfile.write(data)
                except Exception:
                    self.send_error(500, "Error reading static file")
            else:
                self.send_error(404)
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
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>יצירת כרטיסיות ובחינה</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <main class="container">
    <h1>יצירת כרטיסיות ובחינה</h1>
    <div class="panel">
      <form action="/generate" method="post" enctype="multipart/form-data" class="row">
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

        <button class="primary" type="submit">צור</button>
      </form>
    </div>
  </main>
</body>
</html>
        """

    def render_quiz(self, session_id: str, flashcards: list[Flashcard], mcqs: list[MCQQuestion]) -> str:
        """Return HTML for the quiz page with interactive flashcards and questions."""
        # Build cards using concise question and bullet answer lists
        cards_html_parts = []
        for fc in flashcards:
            q_short = make_short_question(fc.question)
            bullets = make_bullets(fc.answer)
            bullets_html = ''.join(f"• {html.escape(p)}<br>" for p in bullets)
            cards_html_parts.append(
                f"<div class=\"flashcard\"><div class=\"q\">{html.escape(q_short)}</div><div class=\"a\">{bullets_html}</div></div>"
            )
        flashcards_html = ''.join(cards_html_parts)
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
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>כרטיסיות ומבחן</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <main class="container">
    <h1>כרטיסיות</h1>
    <section class="flashgrid panel">
      {flashcards_html}
    </section>
    <h1>מבחן</h1>
    <form action="/quiz" method="post" class="panel">
      <input type="hidden" name="session_id" value="{session_id}">
      {quiz_html}
      <button class="primary" type="submit">שלח תשובות</button>
    </form>
  </main>
  <script src="/static/app.js"></script>
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
                q_short = make_short_question(fc.question)
                bullets = make_bullets(fc.answer)
                bullets_html = ''.join(f"• {html.escape(p)}<br>" for p in bullets)
                cards_parts.append(
                    f"<div class=\"flashcard\"><div class=\"q\">{html.escape(q_short)}</div><div class=\"a\">{bullets_html}</div></div>"
                )
            cards_html = ''.join(cards_parts)
            additional_html = f"""
    <h2>כרטיסיות לחיזוק</h2>
    <section class="flashgrid panel">
      {cards_html}
    </section>
            """
        return f"""
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>תוצאות מבחן</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <main class="container">
    <h1>תוצאות</h1>
    <div class="panel">
      <p>הציון שלך: {score}/{total}</p>
      <p>שאלות בהן טעית: {wrong_str}</p>
      {additional_html}
      <p><a href="/">חזרה לעמוד הראשי</a></p>
    </div>
  </main>
  <script src="/static/app.js"></script>
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