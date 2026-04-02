"""
Verification & Diagnosis Script
Run this after construction_pipeline.py to audit results.
"""

import os
import re
import fitz
import pandas as pd
from pathlib import Path

CSV_PATH = "output/classification_results.csv"
REDACTED_DIR = "output/redacted_pdfs"
TEST_DIR = "dataset/Data to be Classified and Redacted"

# Filename prefix/keyword hints for quick sanity check
FILENAME_HINTS = {
    "Architectural": ["arch", "a-rinker", "layout", "elevation", "floor", "plan", "demo", "section", "a1", "a2", "a3"],
    "Structural":    ["struct", "s-rinker", "footing", "foundation", "ftg", "s1", "s2"],
    "Mechanical":    ["mech", "m-rinker", "hvac", "m1", "m2", "m-"],
    "Plumbing":      ["plumb", "p-rinker", "toilet", "riser", "p1", "p14", "p-"],
    "Electrical":    ["elec", "e-rinker", "electrical", "voltage", "e1", "e4"],
    "Fire Protection": ["fire", "fa", "fp", "alarm", "sprinkler"],
}


def guess_class_from_filename(filename: str) -> str | None:
    """Guess the likely class purely from filename patterns."""
    fn = filename.lower()
    for cls, hints in FILENAME_HINTS.items():
        if any(h in fn for h in hints):
            return cls
    return None


def check_text_extractable(pdf_path: str) -> tuple[bool, int]:
    """Return (has_text, char_count) for a PDF."""
    try:
        doc = fitz.open(pdf_path)
        total_chars = sum(len(page.get_text("text").strip()) for page in doc)
        doc.close()
        return total_chars > 50, total_chars
    except:
        return False, 0


def verify_redaction(original_path: str, redacted_path: str) -> dict:
    """Compare original vs redacted to confirm black boxes were applied."""
    result = {"redacted_file_exists": False, "size_change_kb": 0, "pages_match": False}
    if not os.path.exists(redacted_path):
        return result
    result["redacted_file_exists"] = True
    orig_size = os.path.getsize(original_path) / 1024
    red_size = os.path.getsize(redacted_path) / 1024
    result["size_change_kb"] = round(red_size - orig_size, 1)

    try:
        orig_doc = fitz.open(original_path)
        red_doc = fitz.open(redacted_path)
        result["pages_match"] = len(orig_doc) == len(red_doc)
        orig_doc.close()
        red_doc.close()
    except:
        pass
    return result


def run_verification():
    print("=" * 70)
    print("VERIFICATION REPORT")
    print("=" * 70)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"\nTotal files classified: {total}")

    # ── Confidence distribution ──
    print("\n── Confidence Score Distribution ──")
    bins = [(0.0, 0.3, "LOW (<0.30)"), (0.3, 0.6, "MEDIUM (0.30–0.60)"), (0.6, 1.01, "HIGH (>0.60)")]
    for lo, hi, label in bins:
        count = ((df["confidence_score"] >= lo) & (df["confidence_score"] < hi)).sum()
        print(f"  {label}: {count} files")

    # ── Class distribution ──
    print("\n── Predicted Class Distribution ──")
    print(df["predicted_class"].value_counts().to_string())

    # ── Filename vs Prediction mismatch check ──
    print("\n── Filename Hint vs Prediction Check ──")
    mismatches = []
    no_hint = []
    for _, row in df.iterrows():
        fn = row["file_name"]
        pred = row["predicted_class"]
        conf = row["confidence_score"]
        hint = guess_class_from_filename(fn)
        if hint is None:
            no_hint.append(fn)
        elif hint != pred:
            mismatches.append({
                "file": fn,
                "predicted": pred,
                "filename_suggests": hint,
                "confidence": conf,
            })

    if mismatches:
        print(f"\n  ⚠️  {len(mismatches)} possible misclassification(s):")
        for m in mismatches:
            print(f"    ✗ {m['file']}")
            print(f"      Predicted : {m['predicted']} (conf: {m['confidence']})")
            print(f"      Suggested : {m['filename_suggests']}")
    else:
        print("  ✅ No filename-based mismatches detected.")

    if no_hint:
        print(f"\n  ℹ️  {len(no_hint)} files have no filename hint (ambiguous):")
        for f in no_hint:
            print(f"    - {f}")

    # ── Low-confidence files (image-based PDFs) ──
    low_conf = df[df["confidence_score"] < 0.3]
    if not low_conf.empty:
        print(f"\n── Low-Confidence Files (likely image-based, no extractable text) ──")
        for _, row in low_conf.iterrows():
            pdf_path = os.path.join(TEST_DIR, row["file_name"])
            has_text, char_count = check_text_extractable(pdf_path)
            status = f"{char_count} chars extracted" if has_text else "⚠️  NO TEXT (image scan)"
            print(f"  {row['file_name']}")
            print(f"    Predicted: {row['predicted_class']} | Conf: {row['confidence_score']} | {status}")

    # ── Redaction verification ──
    print("\n── Redaction Verification ──")
    missing_redacted = []
    page_mismatches = []
    for _, row in df.iterrows():
        fn = row["file_name"]
        orig = os.path.join(TEST_DIR, fn)
        red = os.path.join(REDACTED_DIR, f"REDACTED_{fn}")
        info = verify_redaction(orig, red)
        if not info["redacted_file_exists"]:
            missing_redacted.append(fn)
        elif not info["pages_match"]:
            page_mismatches.append(fn)

    if not missing_redacted:
        print(f"  ✅ All {total} redacted PDFs exist in output folder.")
    else:
        print(f"  ⚠️  {len(missing_redacted)} redacted files MISSING:")
        for f in missing_redacted:
            print(f"    - {f}")

    if page_mismatches:
        print(f"  ⚠️  {len(page_mismatches)} files have page count mismatch after redaction:")
        for f in page_mismatches:
            print(f"    - {f}")
    else:
        print("  ✅ All redacted PDFs have the correct page count.")

    # ── Final summary ──
    print("\n── Overall Health ──")
    high_conf = (df["confidence_score"] >= 0.6).sum()
    print(f"  High-confidence predictions : {high_conf}/{total} ({100*high_conf//total}%)")
    print(f"  Possible mismatches         : {len(mismatches)}")
    print(f"  Missing redacted PDFs       : {len(missing_redacted)}")

    # Save verification report
    report = df.copy()
    report["filename_hint"] = report["file_name"].apply(guess_class_from_filename)
    report["hint_matches"] = report.apply(
        lambda r: "✅" if r["filename_hint"] == r["predicted_class"]
        else ("❓ no hint" if r["filename_hint"] is None else "⚠️ mismatch"),
        axis=1
    )
    report_path = "output/verification_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\n  Full report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()