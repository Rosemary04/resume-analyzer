from flask import Flask, request, render_template
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDF
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)
key = os.getenv("AZURE_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# PDF to text
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Azure AI processing
def analyze_resume(text):
    sentiment = client.analyze_sentiment([text])[0]
    key_phrases = client.extract_key_phrases([text])[0]
    return sentiment, key_phrases

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["resume"]
        if file and file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
            sentiment, key_phrases = analyze_resume(text)
            return render_template("result.html",
                                   sentiment=sentiment.sentiment,
                                   scores=sentiment.confidence_scores,
                                   phrases=key_phrases.key_phrases)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
