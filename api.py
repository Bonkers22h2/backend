from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field
import joblib
import pandas as pd
import json
import os
import sqlite3
from pathlib import Path

from sqlalchemy.orm import Session

from db import AuditLog, get_db, init_db
from security import decrypt_aes_gcm, encrypt_aes_gcm, sign_payload, verify_signature

app = FastAPI()


# --- SQLite persistence (stores each /predict result) ---
DB_PATH = Path(os.getenv("SQLITE_PATH", "")) if os.getenv("SQLITE_PATH") else Path(__file__).with_name("results.db")


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                risk_features_json TEXT NOT NULL,
                career_features_json TEXT NOT NULL,
                translated_risk_features_json TEXT NOT NULL,
                translated_career_features_json TEXT NOT NULL,
                risk_pred INTEGER,
                track_pred TEXT,
                recommended_track TEXT NOT NULL,
                health_status TEXT NOT NULL,
                actionable_advice TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _insert_prediction(
    *,
    risk_features: dict,
    career_features: dict,
    translated_risk_features: dict,
    translated_career_features: dict,
    risk_pred: int,
    track_pred: str,
    recommended_track: str,
    health_status: str,
    actionable_advice: str,
) -> int:
    with _get_db_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                risk_features_json,
                career_features_json,
                translated_risk_features_json,
                translated_career_features_json,
                risk_pred,
                track_pred,
                recommended_track,
                health_status,
                actionable_advice
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                json.dumps(risk_features, ensure_ascii=False),
                json.dumps(career_features, ensure_ascii=False),
                json.dumps(translated_risk_features, ensure_ascii=False),
                json.dumps(translated_career_features, ensure_ascii=False),
                int(risk_pred),
                str(track_pred),
                str(recommended_track),
                str(health_status),
                str(actionable_advice),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

# Allow Next.js (port 3000) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    _init_db()
    init_db()


def _extract_student_no(payload: dict) -> str | None:
    """Best-effort student number extraction from incoming payload."""
    candidates = [
        "student_no",
        "studentNo",
        "student_number",
        "studentNumber",
        "Student No",
        "Student Number",
    ]

    # Check common nesting patterns
    for container_key in (None, "risk_features", "career_features"):
        container = payload if container_key is None else payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in candidates:
            value = container.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return None

# 1. Load All Models & Mappers
risk_model = joblib.load('risk_assessment_svm_model.pkl')
risk_scaler = joblib.load('risk_assessment_scaler.pkl')
risk_columns = joblib.load('risk_feature_columns.pkl')

track_model = joblib.load('track_recommendation_dt.pkl')
track_scaler = joblib.load('track_recommendation_scaler.pkl')
track_columns = joblib.load('track_feature_columns.pkl')
track_encoder = joblib.load('track_label_encoder.pkl')

# 2. Define the Incoming Data Structure
class StudentData(BaseModel):
    student_number: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "studentNumber",
            "student_number",
            "studentNo",
            "student_no",
        ),
    )
    risk_features: dict
    career_features: dict


def translate_gwa_to_20_scale(ph_gwa):
    """
    Translate Philippine GWA (1.00 best, 3.00 passing, >3.00 failing)
    to the 0-20 scale expected by the SVM model.
    """
    if ph_gwa > 3.00:
        return 0.0

    translated = 20.0 - ((ph_gwa - 1.00) * 5.0)
    return max(0.0, min(20.0, translated))


def translate_gwa_to_4_scale(ph_gwa):
    """
    Translate Philippine GWA (1.00 best, 3.00 passing, >3.00 failing)
    to the 0-4 scale expected by the track model.
    """
    if ph_gwa > 3.00:
        return 0.0

    translated = 4.0 - ((ph_gwa - 1.00) * 1.5)
    return max(0.0, min(4.0, translated))


def apply_gwa_translation(risk_features, career_features):
    """
    Apply Philippine GWA translation to fields expected by each model.
    """
    translated_risk = dict(risk_features)
    translated_career = dict(career_features)

    risk_grade_keys = [
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Previous qualification (grade)",
        "Admission grade",
    ]

    for key in risk_grade_keys:
        if key in translated_risk:
            try:
                translated_risk[key] = translate_gwa_to_20_scale(float(translated_risk[key]))
            except (ValueError, TypeError):
                pass

    if "GPA" in translated_career:
        try:
            translated_career["GPA"] = translate_gwa_to_4_scale(float(translated_career["GPA"]))
        except (ValueError, TypeError):
            pass

    return translated_risk, translated_career

# 3. The 15-Rule Expert Inference Engine
def expert_system_advising(risk_level, predicted_career, risk_raw, career_raw):
    """
    Combines ML outputs with 15 heuristic rules to provide prescriptive advice.
    """
    track_mapping = {
        'Data & AI': 'Data Science & Artificial Intelligence',
        'Software Engineer': 'Software Engineering',
        'Software Development': 'Software Engineering',
        'Security & Infrastructure': 'Cybersecurity & Network Administration',
        'Database & Systems': 'Information Systems Management',
        'Graphics Programmer': 'Game Development / Computer Graphics',
        'Specialized Roles': 'General IT / Elective Mix'
    }

    recommended_track = track_mapping.get(predicted_career, 'General IT / Elective Mix')

    # --- START HARD OVERRIDES ---
    # Normalize possible original PH GWA or translated scales into model scales.
    def _to_20_scale(value):
        try:
            numeric = float(value)
        except (ValueError, TypeError):
            return None

        if 0.0 <= numeric <= 5.0:
            return translate_gwa_to_20_scale(numeric)
        return max(0.0, min(20.0, numeric))

    def _to_4_scale(value):
        try:
            numeric = float(value)
        except (ValueError, TypeError):
            return None

        if 0.0 <= numeric <= 5.0:
            return translate_gwa_to_4_scale(numeric)
        return max(0.0, min(4.0, numeric))

    # 1. Risk Override: force high risk when either sem grade is below passing.
    sem1_grade = _to_20_scale(risk_raw.get("Curricular units 1st sem (grade)"))
    sem2_grade = _to_20_scale(risk_raw.get("Curricular units 2nd sem (grade)"))
    if (sem1_grade is not None and sem1_grade < 10.0) or (sem2_grade is not None and sem2_grade < 10.0):
        risk_level = 1

    # 2. Track Override: prioritize selected domain for strong GPA profiles.
    target_domain = career_raw.get("Interested Domain")
    gpa_value = _to_4_scale(career_raw.get("GPA"))
    if target_domain and gpa_value is not None and gpa_value >= 2.5:
        recommended_track = track_mapping.get(target_domain, recommended_track)
    # --- END HARD OVERRIDES ---

    advice_list = []

    # --- Category 4: External Factor Heuristics (High Priority) ---
    if risk_raw.get("Tuition fees up to date") == 0:
        advice_list.append("Financial Risk: Your academic risk is primarily driven by financial stress. Please visit the Student Affairs office to discuss installment plans.")

    if risk_raw.get("Scholarship holder") == 1 and risk_level == 1:
        advice_list.append("Scholarship at Risk: Current performance may lead to loss of financial aid. Priority advising needed to maintain required GPA.")

    if risk_raw.get("Unemployment rate", 0) > 12 and risk_raw.get("GDP", 0) < 0:
        advice_list.append("Market Awareness: Economic factors indicate a tough job market. Focus on Software Engineering for maximum freelance flexibility.")

    # --- Category 3: Skill Gap Heuristics ---
    if recommended_track == 'Data Science & Artificial Intelligence' and career_raw.get("Python") == "Weak":
        advice_list.append("Technical Gap: Your logic fits AI, but Python syntax will hold you back. Complete a Python intensive before the semester starts.")

    if recommended_track == 'Software Engineering' and career_raw.get("Java") == "Weak":
        advice_list.append("Language Barrier: Software Engineering tracks rely heavily on Java/C#. Recommend Object-Oriented Programming (OOP) drills.")

    if recommended_track == 'Cybersecurity & Network Administration' and career_raw.get("SQL") == "Weak":
        advice_list.append("Security Risk: Database security is a pillar of this track. Improve SQL skills to understand injection vulnerabilities.")

    # --- Category 2: Intervention Path (High Risk: 1) ---
    if risk_level == 1:
        health_status = "🔴 High Risk: Early Intervention Triggered."

        if risk_raw.get("Curricular units 1st sem (approved)", 5) < 3:
            advice_list.append("Critical Academic Load: Passing few core units indicates foundational struggle. Mandatory Peer Tutoring is required.")

        if recommended_track == 'Data Science & Artificial Intelligence':
            advice_list.append("Caution: AI electives have high failure rates for at-risk students. Mandatory Math/Logic refresher required.")
        elif recommended_track == 'Software Engineering':
            advice_list.append("Intervention Required: Heavy coding loads may lead to burnout. Reduce semester load by 3-6 units.")
        elif recommended_track == 'Cybersecurity & Network Administration':
            advice_list.append("Bridge Course Needed: Enroll in a remedial Networking workshop before taking Ethical Hacking electives.")

    # --- Category 1: Success Path (Low Risk: 0) ---
    else:
        health_status = "🟢 Low Risk: Academic Standing is Solid."

        if risk_raw.get("Age at enrollment", 20) > 25:
            advice_list.append("Non-Traditional Success: You show high resilience. Leverage your experience for leadership roles in group projects.")

        if recommended_track == 'Data Science & Artificial Intelligence':
            advice_list.append("High Alignment: Standing is strong enough for math-heavy AI. Focus on Advanced Statistics and Calculus.")
        elif recommended_track == 'Software Engineering':
            advice_list.append("Ready for Industry: Well-positioned for full-stack dev. Begin building a GitHub portfolio for internships.")
        elif recommended_track == 'Cybersecurity & Network Administration':
            advice_list.append("Specialization Match: Profile supports technical infrastructure rigor. Prepare for CompTIA Security+ certification.")
        elif recommended_track == 'Information Systems Management':
            advice_list.append("Strategic Fit:  Balanced profile. Focus on Database Management and ERP electives.")

    # Fallback if no specific rules triggered
    if not advice_list:
        advice_list.append("Standard Pathway: You are cleared to proceed with the standard prerequisite sequencing for your track.")

    return recommended_track, health_status, " ".join(advice_list)

# 4. The API Endpoint
@app.post("/predict")
def make_prediction(data: StudentData, db: Session = Depends(get_db)):
    try:
        raw_payload = data.model_dump()
        student_no = _extract_student_no(raw_payload)

        encrypted_student_no = encrypt_aes_gcm(student_no) if student_no else None
        encrypted_input = encrypt_aes_gcm(json.dumps(raw_payload, ensure_ascii=False))

        translated_risk_features, translated_career_features = apply_gwa_translation(
            data.risk_features,
            data.career_features,
        )

        # A. Process Risk (SVM)
        df_risk = pd.DataFrame([translated_risk_features]).reindex(columns=risk_columns, fill_value=0)
        risk_input = risk_scaler.transform(df_risk)
        risk_pred = int(risk_model.predict(risk_input)[0])

        # B. Process Track (Decision Tree)
        df_career = pd.DataFrame([translated_career_features])
        df_encoded = pd.get_dummies(df_career).reindex(columns=track_columns, fill_value=0)
        track_input = track_scaler.transform(df_encoded)
        track_pred_num = track_model.predict(track_input)[0]
        track_text = track_encoder.inverse_transform([track_pred_num])[0]

        # C. Run Hybrid Expert Inference
        final_track, final_health, final_advice = expert_system_advising(
            risk_pred,
            track_text,
            translated_risk_features,
            translated_career_features
        )

        audit_error = None
        try:
            signature = sign_payload({"recommendation": final_track})
            db.add(
                AuditLog(
                    encrypted_student_no=encrypted_student_no,
                    encrypted_input=encrypted_input,
                    recommendation=final_track,
                    health_status=final_health,
                    actionable_advice=final_advice,
                    signature=signature,
                )
            )
            db.commit()
        except Exception as db_exc:
            db.rollback()
            audit_error = str(db_exc)

        try:
            _insert_prediction(
                risk_features=data.risk_features,
                career_features=data.career_features,
                translated_risk_features=translated_risk_features,
                translated_career_features=translated_career_features,
                risk_pred=risk_pred,
                track_pred=track_text,
                recommended_track=final_track,
                health_status=final_health,
                actionable_advice=final_advice,
            )
        except Exception:
            pass

        if audit_error is not None:
            pass

        return {
            "recommended_track": final_track,
            "health_status": final_health,
            "actionable_advice": final_advice,
        }
    except Exception as e:
        return {"error": str(e)}


class VerifyRequest(BaseModel):
    recommendation: str
    signature: str


@app.post("/verify")
def verify_recommendation(payload: VerifyRequest):
    return {"valid": bool(verify_signature({"recommendation": payload.recommendation}, payload.signature))}


@app.get("/api/search")
def search_audit_log(student_no: str, db: Session = Depends(get_db)):
    wanted = str(student_no).strip()
    if not wanted:
        raise HTTPException(status_code=400, detail="student_no is required")

    rows = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).all()
    matches = []
    for row in rows:
        try:
            decrypted_student = decrypt_aes_gcm(row.encrypted_student_no) if row.encrypted_student_no else None
        except Exception:
            continue

        if decrypted_student != wanted:
            continue

        try:
            decrypted_input_text = decrypt_aes_gcm(row.encrypted_input)
            decrypted_input_json = json.loads(decrypted_input_text)
        except Exception:
            decrypted_input_json = None

        matches.append(
            {
                "input": decrypted_input_json,
                "recommendation": row.recommendation,
                "health_status": row.health_status,
                "actionable_advice": row.actionable_advice,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            }
        )

    if not matches:
        raise HTTPException(status_code=404, detail="Not found")

    return {"results": matches}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
