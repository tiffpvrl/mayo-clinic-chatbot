from typing import Any, Dict
from datetime import datetime

def _fmt_dt(value):
    if value is None:
        return "N/A"

    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y at %-I:%M %p")
    except Exception:
        return str(value)

def _get_schedule_times(patient_record: Dict[str, Any]) -> Dict[str, str]:
    colonoscopy_dt = _fmt_dt(
        patient_record.get("demo_colonoscopy_datetime")
        or patient_record.get("colonoscopy_datetime")
    )
    prep_start = _fmt_dt(
        patient_record.get("demo_bowel_prep_start_datetime")
        or patient_record.get("bowel_prep_start_datetime")
    )
    prep_end = _fmt_dt(
        patient_record.get("demo_bowel_prep_end_datetime")
        or patient_record.get("bowel_prep_end_datetime")
    )

    return {
        "colonoscopy_dt": colonoscopy_dt,
        "prep_start": prep_start,
        "prep_end": prep_end,
    }


def build_prep_email(patient_record: Dict[str, Any], email_type: str = "confirmation") -> Dict[str, str]:
    patient_name = patient_record.get("patient_name", "Patient")
    patient_email = patient_record.get("patient_email", "mayochatbot1@gmail.com")

    times = _get_schedule_times(patient_record)
    colonoscopy_dt = times["colonoscopy_dt"]
    prep_start = times["prep_start"]
    prep_end = times["prep_end"]

    prep_agent = patient_record.get("prep_agent", "your prescribed bowel prep")
    diet_protocol = patient_record.get("diet_protocol", "follow your prep instructions")
    current_medications = patient_record.get("current_medications", "your medications on file")

    if email_type == "confirmation":
        subject = f"Your colonoscopy appointment details for {colonoscopy_dt}"
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

    elif email_type == "prep_24h":
        subject = "Reminder: Your colonoscopy prep starts tomorrow"
        body = f"""Hi {patient_name},

Your colonoscopy preparation begins tomorrow.

Prep details
- Prep start time: {prep_start}
- Colonoscopy time: {colonoscopy_dt}

Please remember
- Prep agent: {prep_agent}
- Diet instructions: {diet_protocol}
- Review any medication instructions from your care team

This step is important for a successful procedure.

Thank you,
Mayo Chatbot Team
"""

    elif email_type == "prep_start":
        subject = "Your bowel prep starts now"
        body = f"""Hi {patient_name},

It is time to begin your bowel preparation.

Prep details
- Prep start time: {prep_start}
- Prep end time: {prep_end}
- Prep agent: {prep_agent}
- Diet instructions: {diet_protocol}

Medication reminder
- Medications on file: {current_medications}
- Follow the instructions given by your care team.

Please contact your care team if you have severe vomiting, severe abdominal pain, dizziness, trouble finishing the prep, or urgent medication questions.

Thank you,
Mayo Chatbot Team
"""

    elif email_type == "procedure_day":
        subject = "Reminder: Your colonoscopy is today"
        body = f"""Hi {patient_name},

This is a reminder that your colonoscopy is today.

Your appointment
- Colonoscopy time: {colonoscopy_dt}
- Prep end time: {prep_end}

Please remember
- Follow the instructions from your care team about eating, drinking, and medications
- Arrange transportation if sedation is planned

Thank you,
Mayo Chatbot Team
"""

    else:
        raise ValueError(f"Unsupported email_type: {email_type}")

    return {
        "to": patient_email,
        "subject": subject,
        "body": body,
    }