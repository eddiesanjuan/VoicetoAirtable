"""
Voice-to-Airtable: EF San Juan Internal Tool
Converts voice recordings into CRM Lead records.
- OpenAI Whisper for transcription
- Claude for intent classification and field extraction
- Airtable API for record creation

Supports:
- Direct audio upload (for Airtable Interface Extension)
- Text webhook (for Wispr integration)
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic
import httpx
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG if os.getenv("DEBUG") == "true" else logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Voice-to-Airtable",
    description="EF San Juan internal tool for voice-driven lead creation",
    version="0.2.0"
)

# Add CORS for Airtable Interface Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://airtable.com", "https://*.airtable.com", "*"],  # Allow Airtable and dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API clients lazily (avoid crash if env vars not set at import time)
_anthropic_client = None
_openai_client = None

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# Airtable configuration
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
CRM_BASE_ID = os.getenv("EF_SANJUAN_CRM_BASE_ID")
LEADS_TABLE_ID = os.getenv("LEADS_TABLE_ID")

# =============================================================================
# Pydantic Models
# =============================================================================

class WisprWebhook(BaseModel):
    """Incoming webhook payload from Wispr"""
    transcription: str
    timestamp: Optional[str] = None
    user_id: Optional[str] = None
    audio_duration: Optional[float] = None


class IntentResult(BaseModel):
    """Result of intent classification"""
    intent: str  # create_lead, update_lead, query_lead, unknown
    confidence: float
    message: Optional[str] = None


class ExtractedLead(BaseModel):
    """Extracted lead fields from transcription"""
    customer_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    property_address: Optional[str] = None
    lead_source: Optional[str] = None
    job_segment: Optional[str] = None
    priority: Optional[str] = None
    initial_notes: Optional[str] = None
    raw_transcription: str


class CreateLeadResponse(BaseModel):
    """Response from lead creation"""
    status: str
    record_id: Optional[str] = None
    lead_name: Optional[str] = None
    fields_populated: list[str] = []
    message: str
    airtable_url: Optional[str] = None


# =============================================================================
# Claude Prompts
# =============================================================================

INTENT_CLASSIFICATION_PROMPT = """You are an AI assistant for EF San Juan, a custom millwork company.
Analyze this voice transcription and classify the user's intent.

Transcription: "{transcription}"

Classify as ONE of these intents:
- create_lead: User wants to log a new potential customer/lead
- update_lead: User wants to update an existing lead's information
- query_lead: User is asking about the status of an existing lead
- unknown: The transcription is not related to lead management

Respond in JSON format:
{{
  "intent": "create_lead|update_lead|query_lead|unknown",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

FIELD_EXTRACTION_PROMPT = """You are an AI assistant for EF San Juan, a custom millwork company.
Extract lead information from this voice transcription for creating a CRM record.

Transcription: "{transcription}"

Extract these fields if present (leave null if not mentioned):
- customer_name: Full name of the potential customer
- contact_phone: Phone number (format as XXX-XXX-XXXX if possible)
- contact_email: Email address
- property_address: Property/project address (include city if mentioned)
- lead_source: How they heard about us. Map to: Referral, Website, Walk-in, Phone Call, Repeat Customer, Other
- job_segment: Type of project. Map to:
  - RR = Residential Remodel/Renovation
  - RN = Residential New Construction
  - CR = Commercial Remodel
  - CN = Commercial New Construction
- priority: If urgency mentioned. Map to: Low, Medium, High, Critical
- initial_notes: Any other relevant details (what they want, referral source name, etc.)

Respond in JSON format only:
{{
  "customer_name": "string or null",
  "contact_phone": "string or null",
  "contact_email": "string or null",
  "property_address": "string or null",
  "lead_source": "string or null",
  "job_segment": "string or null",
  "priority": "string or null",
  "initial_notes": "string or null"
}}"""


# =============================================================================
# Core Functions
# =============================================================================

async def transcribe_audio(audio_file: UploadFile) -> str:
    """Use OpenAI Whisper to transcribe audio file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe with Whisper
        with open(tmp_path, "rb") as audio:
            transcript = get_openai_client().audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="text"
            )

        # Cleanup
        os.unlink(tmp_path)

        logger.info(f"Transcribed audio: {transcript[:100]}...")
        return transcript

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


async def classify_intent(transcription: str) -> IntentResult:
    """Use Claude to classify the intent of a transcription."""
    try:
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": INTENT_CLASSIFICATION_PROMPT.format(transcription=transcription)
            }]
        )

        # Parse JSON response
        result_text = response.content[0].text
        # Handle potential markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        return IntentResult(
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.5),
            message=result.get("reasoning")
        )
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        return IntentResult(intent="unknown", confidence=0.0, message=str(e))


async def extract_lead_fields(transcription: str) -> ExtractedLead:
    """Use Claude to extract lead fields from transcription."""
    try:
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": FIELD_EXTRACTION_PROMPT.format(transcription=transcription)
            }]
        )

        # Parse JSON response
        result_text = response.content[0].text
        # Handle potential markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        return ExtractedLead(
            customer_name=result.get("customer_name"),
            contact_phone=result.get("contact_phone"),
            contact_email=result.get("contact_email"),
            property_address=result.get("property_address"),
            lead_source=result.get("lead_source"),
            job_segment=result.get("job_segment"),
            priority=result.get("priority"),
            initial_notes=result.get("initial_notes"),
            raw_transcription=transcription
        )
    except Exception as e:
        logger.error(f"Field extraction error: {e}")
        return ExtractedLead(raw_transcription=transcription)


async def create_airtable_lead(lead: ExtractedLead) -> CreateLeadResponse:
    """Create a new Lead record in EF San Juan CRM via Airtable API."""

    if not all([AIRTABLE_API_KEY, CRM_BASE_ID, LEADS_TABLE_ID]):
        return CreateLeadResponse(
            status="error",
            message="Airtable configuration missing. Check environment variables.",
            fields_populated=[]
        )

    # Build Airtable fields
    fields = {}
    fields_populated = []

    # Generate Lead Name
    lead_name_parts = []
    if lead.customer_name:
        fields["Customer Name"] = lead.customer_name
        fields_populated.append("Customer Name")
        lead_name_parts.append(lead.customer_name)

    if lead.property_address:
        fields["Property Address"] = lead.property_address
        fields_populated.append("Property Address")
        lead_name_parts.append(lead.property_address)

    # Note: Lead Name is a formula field in Airtable, auto-generated from other fields

    if lead.contact_phone:
        fields["Contact Phone"] = lead.contact_phone
        fields_populated.append("Contact Phone")

    if lead.contact_email:
        fields["Contact Email"] = lead.contact_email
        fields_populated.append("Contact Email")

    if lead.lead_source:
        fields["Lead Source"] = lead.lead_source
        fields_populated.append("Lead Source")
    else:
        fields["Lead Source"] = "Phone Call"  # Default
        fields_populated.append("Lead Source (default)")

    if lead.job_segment:
        fields["Job Segment"] = lead.job_segment
        fields_populated.append("Job Segment")

    if lead.priority:
        fields["Priority"] = lead.priority
        fields_populated.append("Priority")

    # Always set status to New (valid options: New, Contacted, Qualified, Converted to Opportunity, Lost)
    fields["Status"] = "New"
    fields_populated.append("Status")

    # Combine initial notes with raw transcription
    notes_parts = []
    if lead.initial_notes:
        notes_parts.append(lead.initial_notes)
    notes_parts.append(f"\n\n---\nVoice transcription ({datetime.now().isoformat()}):\n{lead.raw_transcription}")
    fields["Initial Notes"] = "\n".join(notes_parts)
    fields_populated.append("Initial Notes")

    # Make Airtable API request
    url = f"https://api.airtable.com/v0/{CRM_BASE_ID}/{LEADS_TABLE_ID}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"fields": fields}

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                record_id = data.get("id")
                airtable_url = f"https://airtable.com/{CRM_BASE_ID}/{LEADS_TABLE_ID}/{record_id}"

                return CreateLeadResponse(
                    status="created",
                    record_id=record_id,
                    lead_name=fields.get("Lead Name", "Unknown"),
                    fields_populated=fields_populated,
                    message=f"Successfully created lead for {lead.customer_name or 'unknown customer'}",
                    airtable_url=airtable_url
                )
            else:
                error_detail = response.text
                logger.error(f"Airtable error: {response.status_code} - {error_detail}")
                return CreateLeadResponse(
                    status="error",
                    message=f"Airtable API error: {response.status_code}",
                    fields_populated=fields_populated
                )

    except Exception as e:
        logger.error(f"Airtable request error: {e}")
        return CreateLeadResponse(
            status="error",
            message=str(e),
            fields_populated=fields_populated
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Voice-to-Airtable",
        "status": "running",
        "version": "0.2.0",
        "target": "EF San Juan CRM",
        "recorder": "http://localhost:8000/recorder"
    }


@app.get("/recorder")
async def serve_recorder():
    """Serve the web recorder UI."""
    import pathlib
    recorder_path = pathlib.Path(__file__).parent.parent / "web-recorder.html"
    return FileResponse(recorder_path, media_type="text/html")


@app.post("/webhook/wispr")
async def wispr_webhook(payload: WisprWebhook):
    """
    Receive voice transcription from Wispr and create Lead record.

    FEAT-001: Voice Transcription Processing
    FEAT-005: End-to-End Voice-to-Lead Flow
    """
    logger.info(f"Received transcription: {payload.transcription[:100]}...")

    # FEAT-002: Intent Classification
    intent_result = await classify_intent(payload.transcription)
    logger.info(f"Intent: {intent_result.intent} (confidence: {intent_result.confidence})")

    if intent_result.intent != "create_lead":
        return {
            "status": "skipped",
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "message": f"Intent '{intent_result.intent}' not yet implemented. Only 'create_lead' is supported in MVP."
        }

    # FEAT-003: Lead Field Extraction
    extracted = await extract_lead_fields(payload.transcription)
    logger.info(f"Extracted: customer={extracted.customer_name}, phone={extracted.contact_phone}")

    # FEAT-004: Airtable Lead Creation
    result = await create_airtable_lead(extracted)

    # FEAT-006: Confirmation Response
    return {
        "status": result.status,
        "intent": intent_result.intent,
        "intent_confidence": intent_result.confidence,
        "record_id": result.record_id,
        "lead_name": result.lead_name,
        "fields_populated": result.fields_populated,
        "message": result.message,
        "airtable_url": result.airtable_url,
        "extracted_fields": {
            "customer_name": extracted.customer_name,
            "contact_phone": extracted.contact_phone,
            "contact_email": extracted.contact_email,
            "property_address": extracted.property_address,
            "lead_source": extracted.lead_source,
            "job_segment": extracted.job_segment,
            "priority": extracted.priority
        }
    }


@app.post("/api/voice-to-lead")
async def voice_to_lead(audio: UploadFile = File(...)):
    """
    Main endpoint for Airtable Interface Extension.
    Accepts audio file, transcribes, extracts fields, creates lead.

    Flow:
    1. OpenAI Whisper → transcription
    2. Claude → intent classification + field extraction
    3. Airtable API → create lead record
    """
    logger.info(f"Received audio file: {audio.filename}, type: {audio.content_type}")

    # Step 1: Transcribe audio with Whisper
    transcription = await transcribe_audio(audio)

    # Step 2: Classify intent
    intent_result = await classify_intent(transcription)
    logger.info(f"Intent: {intent_result.intent} (confidence: {intent_result.confidence})")

    if intent_result.intent != "create_lead":
        return {
            "success": False,
            "step": "intent_classification",
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "transcription": transcription,
            "message": f"Intent '{intent_result.intent}' not supported. Try describing a new lead."
        }

    # Step 3: Extract lead fields
    extracted = await extract_lead_fields(transcription)
    logger.info(f"Extracted: customer={extracted.customer_name}, phone={extracted.contact_phone}")

    # Step 4: Create Airtable record
    result = await create_airtable_lead(extracted)

    return {
        "success": result.status == "created",
        "record_id": result.record_id,
        "lead_name": result.lead_name,
        "airtable_url": result.airtable_url,
        "transcription": transcription,
        "extracted_fields": {
            "customer_name": extracted.customer_name,
            "contact_phone": extracted.contact_phone,
            "contact_email": extracted.contact_email,
            "property_address": extracted.property_address,
            "lead_source": extracted.lead_source,
            "job_segment": extracted.job_segment,
            "priority": extracted.priority,
            "initial_notes": extracted.initial_notes
        },
        "fields_populated": result.fields_populated,
        "message": result.message
    }


@app.post("/api/transcribe")
async def transcribe_only(audio: UploadFile = File(...)):
    """Transcribe audio without creating a lead. Useful for testing."""
    transcription = await transcribe_audio(audio)
    return {"transcription": transcription}


@app.post("/api/preview-lead")
async def preview_lead(audio: UploadFile = File(...)):
    """
    Preview endpoint: Transcribe and extract fields without creating lead.
    Returns what WOULD be created so user can verify before committing.
    """
    logger.info(f"Preview request: {audio.filename}")

    # Step 1: Transcribe
    transcription = await transcribe_audio(audio)

    # Step 2: Classify intent
    intent_result = await classify_intent(transcription)

    if intent_result.intent != "create_lead":
        return {
            "success": False,
            "step": "intent_classification",
            "intent": intent_result.intent,
            "transcription": transcription,
            "message": f"Didn't sound like a new lead. Try saying customer name, phone, and project details."
        }

    # Step 3: Extract fields
    extracted = await extract_lead_fields(transcription)

    return {
        "success": True,
        "transcription": transcription,
        "extracted_fields": {
            "customer_name": extracted.customer_name,
            "contact_phone": extracted.contact_phone,
            "contact_email": extracted.contact_email,
            "property_address": extracted.property_address,
            "lead_source": extracted.lead_source,
            "job_segment": extracted.job_segment,
            "priority": extracted.priority,
            "initial_notes": extracted.initial_notes
        },
        "message": "Ready to create lead. Review and confirm."
    }


class ConfirmLeadRequest(BaseModel):
    """Request to confirm and create a previewed lead."""
    transcription: str
    extracted_fields: dict


@app.post("/api/confirm-lead")
async def confirm_lead(request: ConfirmLeadRequest):
    """
    Create lead from pre-extracted text (after preview confirmation).
    """
    lead = ExtractedLead(
        customer_name=request.extracted_fields.get("customer_name"),
        contact_phone=request.extracted_fields.get("contact_phone"),
        contact_email=request.extracted_fields.get("contact_email"),
        property_address=request.extracted_fields.get("property_address"),
        lead_source=request.extracted_fields.get("lead_source"),
        job_segment=request.extracted_fields.get("job_segment"),
        priority=request.extracted_fields.get("priority"),
        initial_notes=request.extracted_fields.get("initial_notes"),
        raw_transcription=request.transcription
    )

    result = await create_airtable_lead(lead)

    return {
        "success": result.status == "created",
        "record_id": result.record_id,
        "airtable_url": result.airtable_url,
        "lead_name": lead.customer_name,
        "message": result.message
    }


@app.post("/test/classify")
async def test_classify(payload: WisprWebhook):
    """Test endpoint for intent classification only."""
    result = await classify_intent(payload.transcription)
    return result


@app.post("/test/extract")
async def test_extract(payload: WisprWebhook):
    """Test endpoint for field extraction only."""
    result = await extract_lead_fields(payload.transcription)
    return result


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Starting Voice-to-Airtable server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
