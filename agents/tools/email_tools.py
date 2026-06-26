"""Email notification tool for LangGraph agents."""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Send an email notification to a recipient.

    Use for reports, approvals, supplier communications, and non-urgent alerts.
    Do not send emails for routine internal operations.

    Args:
        recipient_email: Valid email address of the recipient.
        subject: Concise subject line (under 80 characters).
        body: Plain text email body. Be clear and actionable.

    Returns:
        Confirmation string or error message.
    """
    import django.core.mail as mail
    from django.conf import settings
    try:
        mail.send_mail(
            subject=subject,
            message=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient_email],
            fail_silently=False,
        )
        return f"Email sent to {recipient_email}: '{subject}'"
    except Exception as exc:
        return f"Failed to send email to {recipient_email}: {exc}"
