from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import spacy

app = Flask(__name__, template_folder="templates")  # ✅ Ensures Flask finds templates

# Load NLP Model
nlp = spacy.load("en_core_web_md")

# Load Resume Data
with open("resume.txt", "r") as file:
    my_resume = file.read()

def extract_job_description(url):
    """
    Web Scraper to Extract Job Description from LinkedIn
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extracting Job Description (Modify selector based on LinkedIn structure)
    job_desc = soup.find("div", {"class": "show-more-less-html__markup"})
    
    if job_desc:
        return job_desc.get_text()
    return None

def compute_match(job_desc):
    """
    NLP-Based Job Match Scoring with AI/ML filtering
    """
    job_doc = nlp(job_desc)
    resume_doc = nlp(my_resume)

    # Extract important words (skills, job roles, tools) using Named Entity Recognition (NER)
    job_keywords = [ent.text.lower() for ent in job_doc.ents if ent.label_ in {"ORG", "PRODUCT", "GPE", "TECH"}]
    resume_keywords = [ent.text.lower() for ent in resume_doc.ents if ent.label_ in {"ORG", "PRODUCT", "GPE", "TECH"}]

    # Predefined AI/ML keywords to boost AI-related jobs
    ai_keywords = {
        "machine learning", "deep learning", "artificial intelligence", "data science", 
        "python", "tensorflow", "pytorch", "nlp", "neural networks", "mlops", "data engineering",
        "computer vision", "transformers", "predictive modeling"
    }

    # Add weight to AI/ML skills by duplicating them
    job_keywords += [word for word in job_keywords if word in ai_keywords] * 5
    resume_keywords += [word for word in resume_keywords if word in ai_keywords] * 5

    # Convert lists back to text for NLP comparison
    job_text = " ".join(job_keywords)
    resume_text = " ".join(resume_keywords)

    job_cleaned = nlp(job_text)
    resume_cleaned = nlp(resume_text)

    # Compute Similarity Score
    match_score = resume_cleaned.similarity(job_cleaned) * 100  # Convert to percentage
    return round(match_score, 2)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    job_url = data.get("job_url")

    job_desc = extract_job_description(job_url)
    if not job_desc:
        return jsonify({"error": "Failed to retrieve job description"}), 400

    match_score = compute_match(job_desc)
    return jsonify({"match_score": match_score})

if __name__ == '__main__':
    app.run(debug=True)
