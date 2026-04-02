# vConstruct Buildathon'26 — Construction Drawing AI Pipeline

> Hackathon project built for vConstruct Buildathon'26. An end-to-end AI pipeline that classifies construction drawings by engineering discipline and automatically redacts sensitive information from title blocks.

---

## Problem Statement

Construction drawing packages contain PDFs from multiple disciplines mixed together, and each drawing's title block carries sensitive information — engineer names, company details, contact numbers, addresses — that needs to be removed before sharing externally.

Doing this manually across hundreds of drawings is slow and error-prone.

---

## What This Project Does

### Input
- A folder of labeled training PDFs organized by discipline (Architectural, Electrical, Fire Protection, Mechanical, Plumbing, Structural)
- A folder of unlabeled PDFs to be classified and redacted

### Processing

**Step 1 — Text Extraction**
PyMuPDF extracts raw text from every page of each PDF for both training and inference.

**Step 2 — Classification**
A TF-IDF + Logistic Regression model is trained on the labeled PDFs. It learns vocabulary patterns specific to each discipline. At inference time, a hybrid scoring approach combines the ML model's confidence with domain keyword matching to improve accuracy on low-confidence predictions.

**Step 3 — Redaction (4 Layers)**

- Layer 1 — Regex patterns scan the title block region for emails, phone numbers, websites, and company names
- Layer 2 — Column-split detection finds label-value pairs (e.g. "CLIENT: XYZ Architects") and redacts the value side
- Layer 3 — Detects honorifics, name initials, and all-caps company names
- Layer 4 — Redacts images (logos, stamps, signatures) that overlap with already-flagged text regions

All redactions are applied as black filled boxes. The main drawing content (walls, dimensions, symbols, room labels) is never touched.

### Output
- Redacted PDFs saved to `output/redacted_pdfs/` with `REDACTED_` prefix
- `output/classification_results.csv` — file name, predicted discipline, confidence score
- `output/classification_results_detailed.csv` — includes redaction count per file

---

## Project Structure

```
├── demo.py                          # Main pipeline (classification + redaction)
├── dataset/
│   ├── Architectural/               # Training PDFs
│   ├── Electrical/
│   ├── Fire Protection/
│   ├── Mechanical/
│   ├── Plumbing/
│   ├── Structural/
│   └── Data to be Classified and Redacted/   # Test PDFs
└── output/
    ├── redacted_pdfs/
    └── classification_results.csv
```

---

## How to Run

```bash
pip install -r requirements.txt
python demo.py
```

---

## Tech Stack

- Python
- PyMuPDF (fitz) — PDF parsing and redaction
- scikit-learn — TF-IDF vectorization + Logistic Regression
- pandas / numpy — data handling and output

---

*Built at vConstruct Buildathon'26 by Vaibhav Sonar*
