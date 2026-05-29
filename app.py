import sys
import io
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from src.routes import register_routes

# Fix UnicodeEncodeError trên Windows terminal (cp1252 không encode được emoji)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

app = Flask(__name__)
CORS(app)

register_routes(app)

if __name__ == "__main__":
    print("Leaf Search Server running at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)