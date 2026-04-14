import smtplib
from email.mime.text import MIMEText

from src.config import SMTP_EMAIL, SMTP_PASSWORD, SMTP_SERVER, SMTP_PORT


def send_email(to_email: str, subject: str, body: str) -> None:
    if not SMTP_EMAIL:
        raise ValueError("Missing SMTP_EMAIL.")
    if not SMTP_PASSWORD:
        raise ValueError("Missing SMTP_PASSWORD.")
    if not to_email:
        raise ValueError("Missing recipient email.")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)