# Resume-to-Job-Description Matcher with Gap Analysis

## Demo Link
- Add your Hugging Face Spaces URL here after deployment: `https://huggingface.co/spaces/<username>/<space-name>`

## Introduction
This NLP app compares a candidate resume against a target job description and returns:
- A match score.
- Extracted skills from both documents.
- Skill gaps (skills the job requests but the resume does not clearly show).
- Actionable suggestions to improve resume alignment.

The system includes three matching approaches to satisfy ARI 525 experimental requirements:
- TF-IDF + cosine similarity (baseline).
- Sentence-transformers bi-encoder.
- Fine-tuned cross-encoder.

## Usage
### 1) Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2) Run the app
```bash
python app/gradio_app.py
```
Open the printed Gradio URL in your browser.

### 3) Provide inputs
- Paste resume and JD text directly, or upload `.txt` / `.pdf` files.
- Choose `tfidf`, `bi_encoder`, or `cross_encoder`.
- Click **Analyze Match**.

### 4) Example
Resume input (short):
```text
Data analyst with Python, SQL, Tableau, and AWS. Built dashboards and automated reports.
```

JD input (short):
```text
Need data analyst with SQL, Python, Tableau, cloud exposure, and strong communication.
```

Expected behavior:
- Higher score on TF-IDF and bi-encoder.
- Gap list may highlight communication if weakly signaled.
- Suggestions include adding quantified communication outcomes in experience bullets.

## Documentation
### System flow
1. Text preprocessing: parse PDF/TXT, clean noise, normalize text, detect sections.
2. Skill extraction: hybrid dictionary + alias normalization + regex certification detection.
3. Matching: one of three approaches selected in UI.
4. Gap analysis: identify missing JD skills and prioritize by salience.
5. Recommendation generation: produce concrete bullet-level resume improvements.

### Models and frameworks
- `scikit-learn` for TF-IDF + cosine baseline.
- `sentence-transformers/all-MiniLM-L6-v2` for bi-encoder semantic matching.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for pairwise scoring and fine-tuning.
- `gradio` for hosted interface.
- `PyPDF2` for PDF parsing.

### Data
- Resume sources: Kaggle resume datasets and/or public anonymized examples.
- Job descriptions: public postings (LinkedIn/Indeed/Career pages) collected with policy compliance.
- Skill taxonomy: O*NET/ESCO-inspired skill lists with local normalization aliases.
- Evaluation set: target ~50 labeled resume-JD pairs with low/medium/high match quality.

### ARI 525 experiments
Run comparative experiments:
```bash
python scripts/run_experiments.py
```

Fine-tune cross-encoder:
```bash
python scripts/train_cross_encoder.py --data data/evaluation/labeled_pairs.json --epochs 2 --batch-size 8 --output models/cross_encoder
```

Report in user guide:
- Ranking metrics: NDCG@k, MRR.
- Classification metrics: Precision/Recall/F1 (thresholded labels).
- Inference metrics: average latency and throughput.
- Human evaluation summary and labeling protocol.

## Contributions
- Built an end-to-end modular NLP pipeline instead of a single pretrained-model wrapper.
- Implemented and compared 3 matching approaches under a shared evaluation setup.
- Added gap analysis and recommendation logic tied to extracted skill evidence.
- Delivered an interactive Gradio app to compare approaches side-by-side.

## Limitations
- Skill extraction currently depends on dictionary coverage and lexical normalization.
- Cross-encoder fine-tuning quality depends on labeled dataset size and consistency.
- Results are guidance signals, not a hiring decision tool.
- Domain shift (non-standard resume style or niche roles) may reduce reliability.

## Project Structure
```text
NLP_HW4/
  app/
    gradio_app.py
    config.yaml
  data/
    evaluation/labeled_pairs.json
    raw/resumes/
    raw/job_descriptions/
  scripts/
    run_experiments.py
    train_cross_encoder.py
  src/
    preprocessing.py
    skill_extraction.py
    gap_analysis.py
    evaluation.py
    pipeline.py
    matching/
      tfidf_matcher.py
      biencoder_matcher.py
      crossencoder_matcher.py
  results/metrics/
  requirements.txt
  README.md
```

## Final Submission Checklist
- [ ] Hosted app is live and link is added above.
- [ ] User guide includes all required sections.
- [ ] Experiment table compares all three approaches.
- [ ] Demo link + user guide posted in Discord #NLP-Apps before presentation.
