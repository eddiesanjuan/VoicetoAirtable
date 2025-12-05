"""
Quick POC validation tests for Voice-to-Airtable.
Run with: python -m pytest tests/test_poc.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import app, classify_intent, extract_lead_fields

client = TestClient(app)


# =============================================================================
# FEAT-001: Voice Transcription Processing
# =============================================================================

def test_health_check():
    """Verify server is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_webhook_receives_transcription():
    """Verify webhook accepts transcription payload."""
    payload = {
        "transcription": "New lead from John Smith about custom cabinets"
    }
    response = client.post("/webhook/wispr", json=payload)
    assert response.status_code == 200
    assert "status" in response.json()


# =============================================================================
# FEAT-002: Intent Classification
# =============================================================================

@pytest.mark.asyncio
async def test_create_lead_intent():
    """Verify create_lead intent is classified correctly."""
    transcription = "New lead from John Smith, he called about custom millwork for his beach house"
    result = await classify_intent(transcription)
    assert result.intent == "create_lead"
    assert result.confidence > 0.5


@pytest.mark.asyncio
async def test_query_intent():
    """Verify query_lead intent is classified correctly."""
    transcription = "What's the status on the Johnson project?"
    result = await classify_intent(transcription)
    assert result.intent == "query_lead"


@pytest.mark.asyncio
async def test_unknown_intent():
    """Verify unknown intent for irrelevant transcriptions."""
    transcription = "The weather is nice today"
    result = await classify_intent(transcription)
    assert result.intent == "unknown"


# =============================================================================
# FEAT-003: Lead Field Extraction
# =============================================================================

@pytest.mark.asyncio
async def test_full_field_extraction():
    """Verify all fields are extracted from complete transcription."""
    transcription = """Got a call from Sarah Johnson at 555-123-4567,
    she's interested in custom doors for her property at 123 Seaside Drive in Destin.
    Referral from the Hendersons."""

    result = await extract_lead_fields(transcription)

    assert result.customer_name == "Sarah Johnson"
    assert "555-123-4567" in (result.contact_phone or "")
    assert "Seaside" in (result.property_address or "")
    assert result.lead_source == "Referral"
    assert "custom doors" in (result.initial_notes or "").lower() or "custom doors" in transcription.lower()


@pytest.mark.asyncio
async def test_partial_field_extraction():
    """Verify extraction handles missing fields gracefully."""
    transcription = "New lead, Bob called about cabinets"
    result = await extract_lead_fields(transcription)

    assert result.customer_name == "Bob" or "Bob" in (result.initial_notes or "")
    assert result.contact_phone is None  # Not provided
    assert result.property_address is None  # Not provided


# =============================================================================
# FEAT-005: End-to-End Flow
# =============================================================================

def test_end_to_end_create_lead():
    """Test complete flow from webhook to lead creation intent."""
    payload = {
        "transcription": """Just got off the phone with Mike Thompson, 850-555-9876,
        he wants a quote for custom millwork at his new construction on 456 Gulf Shore Blvd
        in Panama City Beach. He found us on Google."""
    }

    response = client.post("/webhook/wispr", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["intent"] == "create_lead"
    assert "extracted_fields" in data
    assert data["extracted_fields"]["customer_name"] is not None


# =============================================================================
# Test Utilities
# =============================================================================

def test_classify_endpoint():
    """Test the standalone classify endpoint."""
    payload = {"transcription": "I need to create a new lead for the Martinez family"}
    response = client.post("/test/classify", json=payload)
    assert response.status_code == 200
    assert response.json()["intent"] == "create_lead"


def test_extract_endpoint():
    """Test the standalone extract endpoint."""
    payload = {"transcription": "Lead from Tom Wilson, phone 555-0123, for 789 Beach Rd"}
    response = client.post("/test/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_name"] is not None or data["raw_transcription"] is not None
