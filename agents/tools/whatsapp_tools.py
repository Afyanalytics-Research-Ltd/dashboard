"""WhatsApp notification tool for LangGraph agents.

Uses Twilio's WhatsApp API when TWILIO_* environment variables are configured.
Falls back to logging when credentials are absent (useful in development).
"""
from __future__ import annotations

import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger("agents")


@tool
def send_whatsapp_message(phone_number: str, message: str) -> str:
    """Send a WhatsApp message to a phone number.

    Use for urgent operational alerts that require immediate attention —
    stockout warnings, critical equipment failures, emergency escalations.
    Phone numbers must be in E.164 format (e.g. +254700000000).

    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_FROM
    environment variables. Without them the message is logged only.

    Args:
        phone_number: Recipient phone number in E.164 format.
        message: Message text (max 1600 characters).

    Returns:
        Confirmation string or error message.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    whatsapp_from = os.getenv("TWILIO_WHATSAPP_FROM", "").strip()

    if account_sid and auth_token and whatsapp_from:
        try:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            msg = client.messages.create(
                body=message[:1600],
                from_=whatsapp_from,
                to=f"whatsapp:{phone_number}",
            )
            return f"WhatsApp sent (SID: {msg.sid}) to {phone_number}"
        except Exception as exc:
            logger.error("WhatsApp send failed: %s", exc)
            return f"Failed to send WhatsApp to {phone_number}: {exc}"

    # No Twilio credentials — log and return stub confirmation
    logger.warning(
        "WhatsApp stub (no Twilio config) → %s: %.100s",
        phone_number,
        message,
    )
    return (
        f"[WhatsApp stub — configure TWILIO_* env vars to enable] "
        f"Would send to {phone_number}: {message[:120]}..."
    )
