from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Allow Next.js (port 3000) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    risk_features: dict
    career_features: dict

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
def make_prediction(data: StudentData):
    try:
        # A. Process Risk (SVM)
        df_risk = pd.DataFrame([data.risk_features]).reindex(columns=risk_columns, fill_value=0)
        risk_input = risk_scaler.transform(df_risk)
        risk_pred = int(risk_model.predict(risk_input)[0])

        # B. Process Track (Decision Tree)
        df_career = pd.DataFrame([data.career_features])
        df_encoded = pd.get_dummies(df_career).reindex(columns=track_columns, fill_value=0)
        track_input = track_scaler.transform(df_encoded)
        track_pred_num = track_model.predict(track_input)[0]
        track_text = track_encoder.inverse_transform([track_pred_num])[0]

        # C. Run Hybrid Expert Inference
        final_track, final_health, final_advice = expert_system_advising(
            risk_pred,
            track_text,
            data.risk_features,
            data.career_features
        )

        return {
            "recommended_track": final_track,
            "health_status": final_health,
            "actionable_advice": final_advice
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
