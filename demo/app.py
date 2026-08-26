"""
app.py

Thin Flask application entry point. All scoring logic lives in
api/routes.py — this file just wires up the app and serves a minimal
HTML page for the live demo.
"""

import os
import sys

from flask import Flask, render_template

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.routes import api  # noqa: E402

app = Flask(__name__, template_folder="templates")
app.register_blueprint(api, url_prefix="/api")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
