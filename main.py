import os
import re
import fitz
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DATASET_DIR = "dataset"
TEST_DIR = os.path.join(DATASET_DIR, "Data to be Classified and Redacted")

OUTPUT_DIR = "output"
REDACTED_DIR = os.path.join(OUTPUT_DIR, "redacted_pdfs")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "classification_results.csv")


CLASSES = [
    "Architectural",
    "Electrical",
    "Fire Protection",
    "Mechanical",
    "Plumbing",
    "Structural",
]


TITLE_BLOCK_Y_FRACTION = 0.65
TITLE_BLOCK_X_FRACTION = 0.70


DRAWING_VOCAB = {
    "plan","section","detail","details","notes","revision","elevation",
    "foundation","kitchen","bedroom","bathroom","garage","living",
    "room","door","window","pipe","valve","dimension","scale",
    "north","south","east","west","floor","ceiling","wall",
    "beam","column","duct","schedule","diagram","view","views",
    "tank","water","domestic","fire","fighting","overflow","outlet",
    "drain","section","typical","thickness","insert"
}


RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"\+?\d[\d\s\-]{7,15}")
RE_WEBSITE = re.compile(r"(https?://|www\.)", re.I)
RE_ADDRESS = re.compile(r"\d{1,5} .* (street|road|colony|nagar|sector)", re.I)

RE_COMPANY = re.compile(r"(architects?|consultants?|engineers?|pvt\.?\s*ltd)", re.I)
RE_HONORIFIC = re.compile(r"(mr\.?|mrs\.?|ms\.?|dr\.?)\s+[A-Z][A-Za-z]+")

RE_NAME_INITIAL = re.compile(r"\b[A-Z][a-z]{2,15}\s+[A-Z]\.\b")

RE_ALLCAPS_COMPANY = re.compile(r"^[A-Z]{4,}(?:\s+[A-Z]{4,})*$")


class WordToken:

    def __init__(self, x0, y0, x1, y1, text):
        self.rect = fitz.Rect(x0, y0, x1, y1)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.text = text

    @property
    def y_centre(self):
        return (self.y0 + self.y1) / 2

    @property
    def x_centre(self):
        return (self.x0 + self.x1) / 2


def extract_text_from_pdf(path):

    text = []
    doc = fitz.open(path)

    for page in doc:
        text.append(page.get_text("text"))

    doc.close()

    return "\n".join(text).lower()


def load_training_data():

    texts = []
    labels = []

    for cls in CLASSES:

        folder = os.path.join(DATASET_DIR, cls)

        if not os.path.isdir(folder):
            continue

        for pdf in Path(folder).glob("*.pdf"):

            text = extract_text_from_pdf(str(pdf))

            texts.append(text)
            labels.append(cls)

    return texts, labels


def build_classifier(texts, labels):

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=1000))
    ])

    clf.fit(texts, labels)

    return clf


def predict(clf, text, filename):

    probs = clf.predict_proba([text])[0]
    idx = np.argmax(probs)

    return clf.classes_[idx], float(probs[idx])


def get_title_block_words(page):

    ph = page.rect.height
    pw = page.rect.width

    y_thresh = ph * TITLE_BLOCK_Y_FRACTION
    x_thresh = pw * TITLE_BLOCK_X_FRACTION

    tokens = []

    for w in page.get_text("words"):

        tok = WordToken(w[0], w[1], w[2], w[3], w[4])

        if tok.y_centre > y_thresh or tok.x_centre > x_thresh:
            tokens.append(tok)

    return tokens


def layer1_regex(page):

    ph = page.rect.height
    pw = page.rect.width

    y_thresh = ph * TITLE_BLOCK_Y_FRACTION
    x_thresh = pw * TITLE_BLOCK_X_FRACTION

    rects = []

    for w in page.get_text("words"):

        tok = WordToken(w[0], w[1], w[2], w[3], w[4])

        if not (tok.y_centre > y_thresh or tok.x_centre > x_thresh):
            continue

        txt = tok.text.strip()

        if txt.lower() in DRAWING_VOCAB:
            continue

        if len(txt) <= 4:
            continue

        if RE_EMAIL.search(txt):
            rects.append(tok.rect)

        elif RE_PHONE.search(txt):
            rects.append(tok.rect)

        elif RE_WEBSITE.search(txt):
            rects.append(tok.rect)

        elif RE_ADDRESS.search(txt):
            rects.append(tok.rect)

        elif RE_COMPANY.search(txt):
            rects.append(tok.rect)

        elif RE_HONORIFIC.search(txt):
            rects.append(tok.rect)

    return rects


def collect_multiline_values(label_tok, tokens):

    rects = []
    base_y = label_tok.y_centre

    for tok in tokens:

        if tok.x0 <= label_tok.x1:
            continue

        if abs(tok.y_centre - base_y) > 28:
            continue

        if tok.text.lower() in DRAWING_VOCAB:
            continue

        rects.append(tok.rect)

    return rects


def layer2_column_split(tokens, page_width):

    rects = []

    labels = [
        "project","client","drawn","checked","approved",
        "architect","company","address","title"
    ]

    for tok in tokens:

        if tok.text.lower() in labels:

            rects.extend(
                collect_multiline_values(tok, tokens)
            )

    return rects


def layer3_names(tokens, page_height):

    rects = []

    for tok in tokens:

        txt = tok.text.strip()

        if txt.lower() in DRAWING_VOCAB:
            continue

        if len(txt) <= 4:
            continue

        if tok.y_centre < page_height * TITLE_BLOCK_Y_FRACTION:
            continue

        if RE_NAME_INITIAL.match(txt):
            rects.append(tok.rect)

        elif RE_ALLCAPS_COMPANY.match(txt) and txt.lower() not in DRAWING_VOCAB:
            rects.append(tok.rect)

    return rects


def layer4_images(page, text_rects):

    rects = []

    for img in page.get_image_info(xrefs=True):

        bbox = img.get("bbox")

        if not bbox:
            continue

        rect = fitz.Rect(bbox)

        for anchor in text_rects:

            if rect.intersects(anchor):

                rects.append(rect)
                break

    return rects


def redact_pdf(input_path, output_path):

    doc = fitz.open(input_path)

    total = 0

    for page in doc:

        pw = page.rect.width
        ph = page.rect.height

        tb_tokens = get_title_block_words(page)

        rects1 = layer1_regex(page)
        rects2 = layer2_column_split(tb_tokens, pw)
        rects3 = layer3_names(tb_tokens, ph)

        text_rects = rects1 + rects2 + rects3

        rects4 = layer4_images(page, text_rects)

        all_rects = text_rects + rects4

        for rect in all_rects:

            page.add_redact_annot(rect, fill=(0,0,0))
            total += 1

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

    doc.save(output_path)
    doc.close()

    return total


def run_pipeline():

    os.makedirs(REDACTED_DIR, exist_ok=True)

    print("Loading training data")

    texts, labels = load_training_data()

    if len(texts) == 0:
        print("No training data found")
        return

    clf = build_classifier(texts, labels)

    results = []

    for pdf in Path(TEST_DIR).glob("*.pdf"):

        fname = pdf.name

        text = extract_text_from_pdf(str(pdf))

        pred, conf = predict(clf, text, fname)

        out_path = os.path.join(
            REDACTED_DIR,
            "REDACTED_" + fname
        )

        try:
            n = redact_pdf(str(pdf), out_path)
        except Exception as e:
            print("Redaction error:", e)
            n = 0

        print(fname, pred, conf, "redactions:", n)

        results.append({
            "file_name": fname,
            "predicted_class": pred,
            "confidence_score": conf
        })

    df = pd.DataFrame(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(CSV_OUTPUT, index=False)

    print("Pipeline complete")


if __name__ == "__main__":
    run_pipeline()