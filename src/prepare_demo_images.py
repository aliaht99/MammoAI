"""
Demo Image Preparation — MammoDoctor Showcase
=============================================
Extracts 7 representative DICOM cases from CBIS-DDSM that cover every possible
prediction type the app can show. Converts to PNG and creates a cheat-sheet
with exact clinical form values to enter for each image.

Output:
    demo_images/
        1_CLEAR_MALIGNANT_mass_spiculated.png
        2_MALIGNANT_calc_pleomorphic.png
        3_SUSPICIOUS_mass_illdef.png
        4_PROBABLY_BENIGN_mass_lobulated.png
        5_CLEAR_BENIGN_mass_circumscribed.png
        6_BENIGN_BIOPSY_NEEDED.png
        7_BENIGN_NO_CALLBACK.png
        DEMO_CHEATSHEET.txt

Usage:
    cd /Users/alihamza/Desktop/AICD
    python src/prepare_demo_images.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

try:
    import pydicom
    DICOM_OK = True
except ImportError:
    DICOM_OK = False
    print("ERROR: pydicom not installed — run: pip install pydicom")
    exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/alihamza/Desktop/AICD/manifest-ZkhPvrLo5216730872708713142")
OUT_DIR  = Path("/Users/alihamza/Desktop/AICD/demo_images")
OUT_DIR.mkdir(exist_ok=True)

# Load CSVs
def load_csv(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()
    return df

print("Loading CSVs...")
mass_test  = load_csv(DATA_DIR / "mass_case_description_test_set.csv")
mass_train = load_csv(DATA_DIR / "mass_case_description_train_set.csv")
calc_test  = load_csv(DATA_DIR / "calc_case_description_test_set.csv")
calc_train = load_csv(DATA_DIR / "calc_case_description_train_set.csv")
mass_all   = pd.concat([mass_test, mass_train], ignore_index=True)
calc_all   = pd.concat([calc_test, calc_train], ignore_index=True)

# ── Helper: DICOM → PNG ────────────────────────────────────────────────────
def dcm_to_png(dcm_relative_path: str, output_path: Path,
               size: int = 600, clahe: bool = True) -> bool:
    full = DATA_DIR / dcm_relative_path.strip().strip('"')
    if not full.exists():
        # try without the trailing newline / quote issues
        candidates = list(DATA_DIR.glob(f"**/{full.name}"))
        if not candidates:
            return False
        full = candidates[0]
    try:
        ds  = pydicom.dcmread(str(full))
        arr = ds.pixel_array.astype(float)
        arr = ((arr - arr.min()) / (arr.ptp() + 1e-8) * 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("L")
        pil = pil.resize((size, size), Image.LANCZOS)
        # CLAHE-like: enhance contrast
        pil = ImageEnhance.Contrast(pil).enhance(1.8)
        pil.save(output_path)
        return True
    except Exception as e:
        print(f"  ERROR loading {full.name}: {e}")
        return False

# ── Case Selection ─────────────────────────────────────────────────────────
# 7 scenarios covering every prediction type + clinical form settings
scenarios = [
    # (label, search_df, filter_conditions, output_name, cheatsheet_entry)
    {
        "id": 1,
        "name": "1_CLEAR_MALIGNANT_mass_spiculated",
        "title": "SCENARIO 1 — Clear Malignant (Mass, Spiculated)",
        "df": mass_all,
        "filters": {
            "pathology": "MALIGNANT",
            "mass_margins": "SPICULATED",
            "assessment": 5,
        },
        "clinical_form": {
            "Abnormality Type":  "Mass",
            "BI-RADS Assessment": "5 — Highly Suggestive",
            "Subtlety":           "5 (obvious)",
            "Breast Side":        "LEFT",
            "View":               "CC",
            "Breast Density":     "3 — Heterogeneous",
            "Mass Shape":         "IRREGULAR",
            "Mass Margins":       "SPICULATED",
        },
        "expected": "🔴 MALIGNANT — Very High Risk (>80%). Biopsy strongly recommended.",
        "birads_map": "BI-RADS 5 — Highly Suggestive of Malignancy",
    },
    {
        "id": 2,
        "name": "2_MALIGNANT_calc_pleomorphic",
        "title": "SCENARIO 2 — Malignant Calcifications (Pleomorphic)",
        "df": calc_all,
        "filters": {
            "pathology": "MALIGNANT",
            "calc_type": "PLEOMORPHIC",
            "assessment": 5,
        },
        "clinical_form": {
            "Abnormality Type":   "Calcification",
            "BI-RADS Assessment": "5 — Highly Suggestive",
            "Subtlety":           "4",
            "Breast Side":        "RIGHT",
            "View":               "MLO",
            "Breast Density":     "2 — Scattered",
            "Calc Type":          "PLEOMORPHIC",
            "Calc Distribution":  "CLUSTERED",
        },
        "expected": "🔴 MALIGNANT — High Risk (60–80%).",
        "birads_map": "BI-RADS 4C/5 — High Suspicion",
    },
    {
        "id": 3,
        "name": "3_SUSPICIOUS_mass_illdef",
        "title": "SCENARIO 3 — Suspicious (BI-RADS 4, Ill-defined margins)",
        "df": mass_all,
        "filters": {
            "pathology": "MALIGNANT",
            "mass_margins": "ILL_DEFINED",
            "assessment": 4,
        },
        "clinical_form": {
            "Abnormality Type":   "Mass",
            "BI-RADS Assessment": "4 — Suspicious",
            "Subtlety":           "3",
            "Breast Side":        "LEFT",
            "View":               "MLO",
            "Breast Density":     "3 — Heterogeneous",
            "Mass Shape":         "IRREGULAR",
            "Mass Margins":       "ILL_DEFINED",
        },
        "expected": "🟡 SUSPICIOUS — Medium-High Risk (50–70%). Biopsy/follow-up advised.",
        "birads_map": "BI-RADS 4B/4C — Moderate-High Suspicion",
    },
    {
        "id": 4,
        "name": "4_PROBABLY_BENIGN_mass_lobulated",
        "title": "SCENARIO 4 — Probably Benign (BI-RADS 3, Lobulated)",
        "df": mass_all,
        "filters": {
            "pathology": "BENIGN",
            "mass_shape": "LOBULATED",
            "assessment": 3,
        },
        "clinical_form": {
            "Abnormality Type":   "Mass",
            "BI-RADS Assessment": "3 — Probably Benign",
            "Subtlety":           "2",
            "Breast Side":        "RIGHT",
            "View":               "CC",
            "Breast Density":     "2 — Scattered",
            "Mass Shape":         "LOBULATED",
            "Mass Margins":       "CIRCUMSCRIBED",
        },
        "expected": "🔵 PROBABLY BENIGN — Medium Risk (25–45%). 6-month follow-up.",
        "birads_map": "BI-RADS 3/4A — Probably Benign",
    },
    {
        "id": 5,
        "name": "5_CLEAR_BENIGN_mass_circumscribed",
        "title": "SCENARIO 5 — Clear Benign (Circumscribed, BI-RADS 2)",
        "df": mass_all,
        "filters": {
            "pathology": "BENIGN_WITHOUT_CALLBACK",
            "mass_margins": "CIRCUMSCRIBED",
            "assessment": 2,
        },
        "clinical_form": {
            "Abnormality Type":   "Mass",
            "BI-RADS Assessment": "2 — Benign",
            "Subtlety":           "1 (barely visible)",
            "Breast Side":        "LEFT",
            "View":               "CC",
            "Breast Density":     "1 — Almost entirely fat",
            "Mass Shape":         "ROUND",
            "Mass Margins":       "CIRCUMSCRIBED",
        },
        "expected": "🟢 BENIGN — Low Risk (<15%). Routine screening.",
        "birads_map": "BI-RADS 1/2 — Benign / Negative",
    },
    {
        "id": 6,
        "name": "6_BENIGN_biopsy_needed",
        "title": "SCENARIO 6 — Benign but Biopsy Was Required (Sub-class demo)",
        "df": mass_all,
        "filters": {
            "pathology": "BENIGN",
            "assessment": 4,
            "mass_margins": "MICROLOBULATED",
        },
        "clinical_form": {
            "Abnormality Type":   "Mass",
            "BI-RADS Assessment": "4 — Suspicious",
            "Subtlety":           "3",
            "Breast Side":        "RIGHT",
            "View":               "MLO",
            "Breast Density":     "3 — Heterogeneous",
            "Mass Shape":         "IRREGULAR",
            "Mass Margins":       "MICROLOBULATED",
        },
        "expected": "⚠️ BENIGN (pathology confirmed) but biopsy WAS needed.\n"
                    "Sub-class model: Biopsy Likely (>50%). Shows BENIGN sub-class working.",
        "birads_map": "BI-RADS 4A/4B — Biopsy required",
    },
    {
        "id": 7,
        "name": "7_BENIGN_NO_CALLBACK",
        "title": "SCENARIO 7 — Benign, No Recall Needed (BENIGN_WITHOUT_CALLBACK)",
        "df": calc_all,
        "filters": {
            "pathology": "BENIGN_WITHOUT_CALLBACK",
            "calc_type": "PUNCTATE",
            "assessment": 2,
        },
        "clinical_form": {
            "Abnormality Type":   "Calcification",
            "BI-RADS Assessment": "2 — Benign",
            "Subtlety":           "1",
            "Breast Side":        "LEFT",
            "View":               "CC",
            "Breast Density":     "2 — Scattered",
            "Calc Type":          "PUNCTATE",
            "Calc Distribution":  "DIFFUSELY_SCATTERED",
        },
        "expected": "🟢 BENIGN — Very Low Risk (<10%). No recall/biopsy.\n"
                    "Sub-class model: Biopsy Unlikely (<20%). Shows BENIGN_WITHOUT_CALLBACK.",
        "birads_map": "BI-RADS 1/2 — No follow-up needed",
    },
]

# ── Extract Images ─────────────────────────────────────────────────────────
print("\nExtracting representative DICOM cases...\n")
cheatsheet_lines = []
cheatsheet_lines.append("=" * 70)
cheatsheet_lines.append("  MAMMODCOTOR — DEMO CHEAT SHEET")
cheatsheet_lines.append("  7 test cases covering every prediction type")
cheatsheet_lines.append("=" * 70)
cheatsheet_lines.append("")

extracted = []
for sc in scenarios:
    df = sc["df"].copy()
    # Apply filters
    for col, val in sc["filters"].items():
        col_clean = col.replace(" ", "_")
        if col_clean in df.columns:
            if isinstance(val, str):
                df = df[df[col_clean].str.upper().str.startswith(str(val).upper())]
            else:
                df = df[df[col_clean] == val]

    row = df.iloc[0] if len(df) > 0 else None

    out_path = OUT_DIR / f"{sc['name']}.png"

    if row is not None and "image_file_path" in row.index:
        dcm_path = str(row["image_file_path"]).strip()
        ok = dcm_to_png(dcm_path, out_path)
        if ok:
            print(f"  ✅ Scenario {sc['id']}: {sc['title']}")
            print(f"     Patient: {row.get('patient_id','?')} | "
                  f"Assessment: {row.get('assessment','?')} | "
                  f"Pathology: {row.get('pathology','?')}")
            extracted.append(sc)
        else:
            # Try backup — find any file for this patient
            folder_pattern = dcm_path.split("/")[0]
            matches = list(DATA_DIR.glob(f"{folder_pattern}/**/*.dcm"))
            if matches:
                ok2 = dcm_to_png(str(matches[0].relative_to(DATA_DIR)), out_path)
                if ok2:
                    print(f"  ✅ Scenario {sc['id']}: {sc['title']} (backup path)")
                    extracted.append(sc)
                    row_info = f"Patient: {row.get('patient_id','?')}"
                else:
                    print(f"  ❌ Scenario {sc['id']}: DICOM not found — skipped")
            else:
                print(f"  ❌ Scenario {sc['id']}: No DICOMs found — skipped")
    else:
        print(f"  ⚠️ Scenario {sc['id']}: No matching CSV row for filters {sc['filters']}")

    # Cheatsheet entry (even if image not found)
    cheatsheet_lines.append(f"{'─'*70}")
    cheatsheet_lines.append(f"📁 IMAGE FILE:  demo_images/{sc['name']}.png")
    cheatsheet_lines.append(f"🏷️  SCENARIO:   {sc['title']}")
    cheatsheet_lines.append("")
    cheatsheet_lines.append("  CLINICAL FORM VALUES (Tab: Clinical Form → Run Full Analysis):")
    for k, v in sc["clinical_form"].items():
        cheatsheet_lines.append(f"    {k:<25} → {v}")
    cheatsheet_lines.append("")
    cheatsheet_lines.append(f"  EXPECTED RESULT:")
    cheatsheet_lines.append(f"    {sc['expected']}")
    cheatsheet_lines.append(f"    Ensemble BI-RADS: {sc['birads_map']}")
    cheatsheet_lines.append("")

cheatsheet_lines.append("=" * 70)
cheatsheet_lines.append("")
cheatsheet_lines.append("HOW TO USE IN THE APP:")
cheatsheet_lines.append("  1. Launch:  cd /Users/alihamza/Desktop/AICD")
cheatsheet_lines.append("              /opt/anaconda3/envs/aicd/bin/streamlit run mammo_doctor.py")
cheatsheet_lines.append("  2. Upload the PNG from demo_images/ in the sidebar")
cheatsheet_lines.append("  3. Go to 🔬 AI Analysis tab — CNN result appears automatically")
cheatsheet_lines.append("  4. Go to 📋 Clinical Form tab — enter values from this sheet")
cheatsheet_lines.append("  5. Click Run Full Analysis → see Stage 1 + Sub-class + SHAP")
cheatsheet_lines.append("  6. Go to 📄 Report tab → Download PDF report")
cheatsheet_lines.append("")
cheatsheet_lines.append("DATASETS USED TO TRAIN ALL MODELS:")
cheatsheet_lines.append("  ┌─────────────────────────────────────────────────────────────┐")
cheatsheet_lines.append("  │ CBIS-DDSM  │ 3,568 cases │ US, 1990s film   │ TCIA         │")
cheatsheet_lines.append("  │ VinDr-Mammo│20,000 images│ Vietnam, digital  │ PhysioNet    │")
cheatsheet_lines.append("  │ Combined   │23,568 cases │ Multi-continental │ Public only  │")
cheatsheet_lines.append("  └─────────────────────────────────────────────────────────────┘")
cheatsheet_lines.append("")
cheatsheet_lines.append("=" * 70)

cheatsheet_text = "\n".join(cheatsheet_lines)
cheat_path = OUT_DIR / "DEMO_CHEATSHEET.txt"
with open(cheat_path, "w") as f:
    f.write(cheatsheet_text)

print(f"\n{'='*60}")
print(f"  Extracted {len(extracted)}/7 scenarios")
print(f"  Images saved to: {OUT_DIR}")
print(f"  Cheat sheet: {cheat_path}")
print(f"{'='*60}\n")
print(cheatsheet_text)
