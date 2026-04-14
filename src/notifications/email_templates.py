from typing import Any, Dict

def _fmt_dt(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value).replace("T", " ").split(".")[0]

def build_prep_email(patient_record: Dict[str, Any]) -> Dict[str, str]:
    patient_name = patient_record.get("patient_name", "Patient")
    patient_email = patient_record.get("patient_email", "mayochatbot1@gmail.com")

    colonoscopy_dt = _fmt_dt(patient_record.get("colonoscopy_datetime"))
    prep_start = _fmt_dt(patient_record.get("bowel_prep_start_datetime"))
    prep_end = _fmt_dt(patient_record.get("bowel_prep_end_datetime"))

    prep_agent = patient_record.get("prep_agent", "your prescribed bowel prep")
    diet_protocol = patient_record.get("diet_protocol", "follow your prep instructions")
    current_medications = patient_record.get("current_medications", "your medications on file")

    subject = f"Colonoscopy prep reminder for {colonoscopy_dt}"

    body = f"""Hi {patient_name},

This is a reminder for your upcoming colonoscopy.

Your appointment
- Colonoscopy time: {colonoscopy_dt}
- Prep start time: {prep_start}
- Prep end time: {prep_end}

Your prep instructions
- Prep agent: {prep_agent}
- Diet instructions: {diet_protocol}

Medication reminder
- Medications on file: {current_medications}
- Follow the instructions given by your care team.

Please contact your care team if you have severe vomiting, severe abdominal pain, dizziness, trouble finishing the prep, or urgent medication questions.

Thank you,
Mayo Chatbot Team
"""

    return {
        "to": patient_email,
        "subject": subject,
        "body": body,
    }