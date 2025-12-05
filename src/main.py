"""
Voice-to-Airtable: EF San Juan Internal Tool
Multi-intent voice CRM interface.
- OpenAI Whisper for transcription
- Claude for intent classification and field extraction
- Airtable API for record creation/updates

Supported intents:
- new_lead: Create a new lead record
- call_note: Log a call/activity against a lead
- status_update: Update a lead's status
- task: Create a follow-up task
"""

import os
import json
import logging
import tempfile
from datetime import datetime, timedelta
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
    description="EF San Juan multi-intent voice CRM interface",
    version="0.3.0"
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
LEADS_TABLE_ID = os.getenv("LEADS_TABLE_ID", "tblK6HaTTVYqeFtd7")
ACTIVITIES_TABLE_ID = os.getenv("ACTIVITIES_TABLE_ID", "tblHVYlHL5UzUzAbB")
TASKS_TABLE_ID = os.getenv("TASKS_TABLE_ID", "tblj9Isash4ukT4Nw")

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
    intent: str  # new_lead, call_note, status_update, task, unknown
    confidence: float
    message: Optional[str] = None
    lead_identifier: Optional[str] = None  # Name/phone of existing lead if referenced


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


class ExtractedCallNote(BaseModel):
    """Extracted call/activity fields from transcription"""
    lead_identifier: Optional[str] = None  # Name or phone to find lead
    activity_type: Optional[str] = None  # Call, Email, Meeting, Site Visit, Voicemail, Text Message, Other
    summary: Optional[str] = None  # Brief summary of the activity
    notes: Optional[str] = None  # Detailed notes
    outcome: Optional[str] = None  # Successful, No Answer, Left Message, Follow-up Required, Cancelled, N/A
    next_follow_up_date: Optional[str] = None  # Date string
    next_steps: Optional[str] = None
    duration_minutes: Optional[int] = None
    raw_transcription: str


class ExtractedStatusUpdate(BaseModel):
    """Extracted status update from transcription"""
    lead_identifier: Optional[str] = None  # Name or phone to find lead
    new_status: Optional[str] = None  # New, Contacted, Qualified, Converted to Opportunity, Lost
    reason: Optional[str] = None  # Why status changed
    raw_transcription: str


class ExtractedTask(BaseModel):
    """Extracted task from transcription"""
    lead_identifier: Optional[str] = None  # Name or phone to find lead (optional)
    task_type: Optional[str] = None  # Call, Email, Meeting, Lead Follow-up, etc.
    title: Optional[str] = None  # Task title/description
    notes: Optional[str] = None  # Additional details
    due_date: Optional[str] = None  # When to complete
    priority: Optional[str] = None  # Low, Medium, High, Critical
    raw_transcription: str


class CreateRecordResponse(BaseModel):
    """Generic response from record creation"""
    status: str
    intent: str
    record_id: Optional[str] = None
    record_name: Optional[str] = None
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
- new_lead: User wants to log a NEW potential customer/lead (first contact, new prospect)
- call_note: User is logging notes from a call/meeting about an EXISTING lead (e.g., "Just talked to John Smith about...")
- status_update: User wants to update an existing lead's status (e.g., "Mark Smith as qualified", "Lost the Henderson deal")
- task: User wants to create a follow-up task (e.g., "Remind me to call back", "Schedule a site visit")
- unknown: The transcription is not related to CRM activities

Key distinction:
- "Got a call from Sarah Johnson, new customer interested in doors" = new_lead (NEW customer)
- "Just talked to Sarah Johnson, she wants to move forward" = call_note (EXISTING customer update)

Respond in JSON format:
{{
  "intent": "new_lead|call_note|status_update|task|unknown",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "lead_identifier": "name or phone mentioned if referencing existing lead, null otherwise"
}}"""

FIELD_EXTRACTION_PROMPT = """You are an AI assistant for EF San Juan, a custom millwork company.
Extract lead information from this voice transcription for creating a CRM record.

Transcription: "{transcription}"

Extract these fields if present (leave null if not mentioned):
- customer_name: Full name of the potential customer
- contact_phone: Phone number (format as XXX-XXX-XXXX if possible)
- contact_email: Email address
- property_address: Property/project address (include city if mentioned)
- lead_source: How they heard about us. Map to: Referral, Website, Walk-in, Repeat Customer, Trade Show, Social Media, Other
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

CALL_NOTE_EXTRACTION_PROMPT = """You are an AI assistant for EF San Juan, a custom millwork company.
Extract activity/call note information from this voice transcription.

Transcription: "{transcription}"

Extract these fields if present (leave null if not mentioned):
- lead_identifier: Name or phone number of the lead this activity is about (REQUIRED - who was contacted)
- activity_type: Map to one of: Call, Email, Meeting, Site Visit, Voicemail, Text Message, Other
- summary: One-line summary of what happened (max 100 chars)
- notes: Detailed notes about the conversation/interaction
- outcome: Map to one of: Successful, No Answer, Left Message, Follow-up Required, Cancelled, N/A
- next_follow_up_date: If a follow-up date was mentioned, extract as YYYY-MM-DD
- next_steps: What needs to happen next
- duration_minutes: If call/meeting duration mentioned, extract as integer

Respond in JSON format only:
{{
  "lead_identifier": "string or null",
  "activity_type": "string or null",
  "summary": "string or null",
  "notes": "string or null",
  "outcome": "string or null",
  "next_follow_up_date": "string or null",
  "next_steps": "string or null",
  "duration_minutes": "integer or null"
}}"""

STATUS_UPDATE_EXTRACTION_PROMPT = """You are an AI assistant for EF San Juan, a custom millwork company.
Extract lead status update information from this voice transcription.

Transcription: "{transcription}"

Extract these fields:
- lead_identifier: Name or phone of the lead to update (REQUIRED)
- new_status: Map to one of these EXACT values:
  - "New" - Just added, not contacted yet
  - "Contacted" - Initial contact made
  - "Qualified" - Confirmed as a real opportunity
  - "Converted to Opportunity" - Moving to quote/project phase
  - "Lost" - Not proceeding (lost to competitor, budget, timing, etc.)
- reason: Brief explanation of why status is changing

Respond in JSON format only:
{{
  "lead_identifier": "string or null",
  "new_status": "string or null",
  "reason": "string or null"
}}"""

TASK_EXTRACTION_PROMPT = """You are an AI assistant for EF San Juan, a custom millwork company.
Extract task/reminder information from this voice transcription.

Transcription: "{transcription}"

Extract these fields if present:
- lead_identifier: Name or phone of related lead (null if general task)
- task_type: Map to one of: Call, Email, Meeting, Lead Follow-up, Scope Follow-up, Proposal Follow-up, Site Visit, Quote Review, Other
- title: Short title for the task (max 100 chars)
- notes: Additional details about the task
- due_date: When to complete. Parse relative dates:
  - "tomorrow" = tomorrow's date
  - "next week" = 7 days from now
  - "in 2 days" = 2 days from now
  - "Monday" = next Monday
  Format as YYYY-MM-DD
- priority: Map to one of: Low, Medium, High, Critical

Respond in JSON format only:
{{
  "lead_identifier": "string or null",
  "task_type": "string or null",
  "title": "string or null",
  "notes": "string or null",
  "due_date": "string or null",
  "priority": "string or null"
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
            message=result.get("reasoning"),
            lead_identifier=result.get("lead_identifier")
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


async def extract_call_note_fields(transcription: str) -> ExtractedCallNote:
    """Use Claude to extract call note fields from transcription."""
    try:
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": CALL_NOTE_EXTRACTION_PROMPT.format(transcription=transcription)
            }]
        )

        result_text = response.content[0].text
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        return ExtractedCallNote(
            lead_identifier=result.get("lead_identifier"),
            activity_type=result.get("activity_type"),
            summary=result.get("summary"),
            notes=result.get("notes"),
            outcome=result.get("outcome"),
            next_follow_up_date=result.get("next_follow_up_date"),
            next_steps=result.get("next_steps"),
            duration_minutes=result.get("duration_minutes"),
            raw_transcription=transcription
        )
    except Exception as e:
        logger.error(f"Call note extraction error: {e}")
        return ExtractedCallNote(raw_transcription=transcription)


async def extract_status_update_fields(transcription: str) -> ExtractedStatusUpdate:
    """Use Claude to extract status update fields from transcription."""
    try:
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": STATUS_UPDATE_EXTRACTION_PROMPT.format(transcription=transcription)
            }]
        )

        result_text = response.content[0].text
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        return ExtractedStatusUpdate(
            lead_identifier=result.get("lead_identifier"),
            new_status=result.get("new_status"),
            reason=result.get("reason"),
            raw_transcription=transcription
        )
    except Exception as e:
        logger.error(f"Status update extraction error: {e}")
        return ExtractedStatusUpdate(raw_transcription=transcription)


async def extract_task_fields(transcription: str) -> ExtractedTask:
    """Use Claude to extract task fields from transcription."""
    try:
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": TASK_EXTRACTION_PROMPT.format(transcription=transcription)
            }]
        )

        result_text = response.content[0].text
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text.strip())

        return ExtractedTask(
            lead_identifier=result.get("lead_identifier"),
            task_type=result.get("task_type"),
            title=result.get("title"),
            notes=result.get("notes"),
            due_date=result.get("due_date"),
            priority=result.get("priority"),
            raw_transcription=transcription
        )
    except Exception as e:
        logger.error(f"Task extraction error: {e}")
        return ExtractedTask(raw_transcription=transcription)


async def find_lead_by_identifier(identifier: str) -> Optional[dict]:
    """Search for a lead by name or phone number."""
    if not all([AIRTABLE_API_KEY, CRM_BASE_ID, LEADS_TABLE_ID]):
        logger.error("Airtable configuration missing for lead search")
        return None

    # Try to find lead by Customer Name or Contact Phone
    # Using SEARCH to find partial matches
    formula = f"OR(SEARCH(LOWER(\"{identifier}\"), LOWER({{Customer Name}})), SEARCH(\"{identifier}\", {{Contact Phone}}))"

    url = f"https://api.airtable.com/v0/{CRM_BASE_ID}/{LEADS_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {
        "filterByFormula": formula,
        "maxRecords": 5
    }

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])
                if records:
                    # Return the first match
                    lead = records[0]
                    logger.info(f"Found lead: {lead.get('id')} for identifier '{identifier}'")
                    return {
                        "id": lead.get("id"),
                        "name": lead.get("fields", {}).get("Customer Name", "Unknown"),
                        "fields": lead.get("fields", {})
                    }
                else:
                    logger.info(f"No lead found for identifier: {identifier}")
                    return None
            else:
                logger.error(f"Lead search failed: {response.status_code} - {response.text}")
                return None

    except Exception as e:
        logger.error(f"Lead search error: {e}")
        return None


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
        # Validate against allowed values
        valid_sources = ["Referral", "Website", "Walk-in", "Repeat Customer", "Trade Show", "Social Media", "Other"]
        if lead.lead_source in valid_sources:
            fields["Lead Source"] = lead.lead_source
            fields_populated.append("Lead Source")
        else:
            fields["Lead Source"] = "Other"
            fields_populated.append("Lead Source (mapped to Other)")

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
        import traceback
        logger.error(f"Airtable request error: {e}\n{traceback.format_exc()}")
        return CreateLeadResponse(
            status="error",
            message=str(e) or "Unknown Airtable error",
            fields_populated=fields_populated
        )


async def create_airtable_activity(note: ExtractedCallNote, lead_id: Optional[str] = None) -> CreateRecordResponse:
    """Create a new Activity record in EF San Juan CRM."""

    if not all([AIRTABLE_API_KEY, CRM_BASE_ID, ACTIVITIES_TABLE_ID]):
        return CreateRecordResponse(
            status="error",
            intent="call_note",
            message="Airtable configuration missing. Check environment variables.",
            fields_populated=[]
        )

    fields = {}
    fields_populated = []

    if note.summary:
        fields["Activity Summary"] = note.summary
        fields_populated.append("Activity Summary")

    if note.activity_type:
        fields["Activity Type"] = note.activity_type
        fields_populated.append("Activity Type")
    else:
        fields["Activity Type"] = "Call"  # Default
        fields_populated.append("Activity Type (default)")

    if note.notes:
        fields["Notes"] = note.notes
        fields_populated.append("Notes")

    if note.outcome:
        fields["Outcome"] = note.outcome
        fields_populated.append("Outcome")

    if note.next_follow_up_date:
        fields["Next Follow-Up Date"] = note.next_follow_up_date
        fields_populated.append("Next Follow-Up Date")

    if note.next_steps:
        fields["Next Steps"] = note.next_steps
        fields_populated.append("Next Steps")

    if note.duration_minutes:
        fields["Duration"] = note.duration_minutes
        fields_populated.append("Duration")

    # Link to lead if we found one
    if lead_id:
        fields["Related Lead"] = [lead_id]  # Linked record field needs array
        fields_populated.append("Related Lead")

    # Add raw transcription to notes
    if fields.get("Notes"):
        fields["Notes"] += f"\n\n---\nVoice transcription ({datetime.now().isoformat()}):\n{note.raw_transcription}"
    else:
        fields["Notes"] = f"Voice transcription ({datetime.now().isoformat()}):\n{note.raw_transcription}"
        fields_populated.append("Notes")

    url = f"https://api.airtable.com/v0/{CRM_BASE_ID}/{ACTIVITIES_TABLE_ID}"
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
                airtable_url = f"https://airtable.com/{CRM_BASE_ID}/{ACTIVITIES_TABLE_ID}/{record_id}"

                return CreateRecordResponse(
                    status="created",
                    intent="call_note",
                    record_id=record_id,
                    record_name=note.summary or "Activity logged",
                    fields_populated=fields_populated,
                    message=f"Successfully logged activity for {note.lead_identifier or 'unknown lead'}",
                    airtable_url=airtable_url
                )
            else:
                logger.error(f"Airtable activity error: {response.status_code} - {response.text}")
                return CreateRecordResponse(
                    status="error",
                    intent="call_note",
                    message=f"Airtable API error: {response.status_code}",
                    fields_populated=fields_populated
                )

    except Exception as e:
        logger.error(f"Airtable activity request error: {e}")
        return CreateRecordResponse(
            status="error",
            intent="call_note",
            message=str(e),
            fields_populated=fields_populated
        )


async def update_airtable_lead_status(update: ExtractedStatusUpdate, lead_id: str, lead_name: str) -> CreateRecordResponse:
    """Update an existing Lead's status in Airtable."""

    if not all([AIRTABLE_API_KEY, CRM_BASE_ID, LEADS_TABLE_ID]):
        return CreateRecordResponse(
            status="error",
            intent="status_update",
            message="Airtable configuration missing.",
            fields_populated=[]
        )

    fields = {}
    fields_populated = []

    if update.new_status:
        # Validate status value
        valid_statuses = ["New", "Contacted", "Qualified", "Converted to Opportunity", "Lost"]
        if update.new_status in valid_statuses:
            fields["Status"] = update.new_status
            fields_populated.append("Status")
        else:
            return CreateRecordResponse(
                status="error",
                intent="status_update",
                message=f"Invalid status '{update.new_status}'. Valid options: {', '.join(valid_statuses)}",
                fields_populated=[]
            )

    # Append reason to notes if provided
    if update.reason:
        timestamp = datetime.now().isoformat()
        status_note = f"\n\n---\nStatus changed to '{update.new_status}' ({timestamp}):\n{update.reason}"

        # Get existing notes first
        url = f"https://api.airtable.com/v0/{CRM_BASE_ID}/{LEADS_TABLE_ID}/{lead_id}"
        headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

        async with httpx.AsyncClient() as http_client:
            get_response = await http_client.get(url, headers=headers)
            if get_response.status_code == 200:
                existing_notes = get_response.json().get("fields", {}).get("Initial Notes", "")
                fields["Initial Notes"] = existing_notes + status_note
                fields_populated.append("Initial Notes")

    url = f"https://api.airtable.com/v0/{CRM_BASE_ID}/{LEADS_TABLE_ID}/{lead_id}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"fields": fields}

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.patch(url, headers=headers, json=payload)

            if response.status_code == 200:
                airtable_url = f"https://airtable.com/{CRM_BASE_ID}/{LEADS_TABLE_ID}/{lead_id}"

                return CreateRecordResponse(
                    status="updated",
                    intent="status_update",
                    record_id=lead_id,
                    record_name=lead_name,
                    fields_populated=fields_populated,
                    message=f"Updated {lead_name} status to '{update.new_status}'",
                    airtable_url=airtable_url
                )
            else:
                logger.error(f"Airtable status update error: {response.status_code} - {response.text}")
                return CreateRecordResponse(
                    status="error",
                    intent="status_update",
                    message=f"Airtable API error: {response.status_code}",
                    fields_populated=fields_populated
                )

    except Exception as e:
        logger.error(f"Airtable status update request error: {e}")
        return CreateRecordResponse(
            status="error",
            intent="status_update",
            message=str(e),
            fields_populated=fields_populated
        )


async def create_airtable_task(task: ExtractedTask, lead_id: Optional[str] = None) -> CreateRecordResponse:
    """Create a new Task record in EF San Juan CRM."""

    if not all([AIRTABLE_API_KEY, CRM_BASE_ID, TASKS_TABLE_ID]):
        return CreateRecordResponse(
            status="error",
            intent="task",
            message="Airtable configuration missing. Check environment variables.",
            fields_populated=[]
        )

    fields = {}
    fields_populated = []

    if task.title:
        fields["Task Name"] = task.title
        fields_populated.append("Task Name")

    if task.task_type:
        fields["Task Type"] = task.task_type
        fields_populated.append("Task Type")
    else:
        fields["Task Type"] = "Lead Follow-up"  # Default
        fields_populated.append("Task Type (default)")

    if task.notes:
        fields["Notes"] = task.notes
        fields_populated.append("Notes")

    if task.due_date:
        fields["Due Date"] = task.due_date
        fields_populated.append("Due Date")
    else:
        # Default to tomorrow
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        fields["Due Date"] = tomorrow
        fields_populated.append("Due Date (default: tomorrow)")

    if task.priority:
        fields["Priority"] = task.priority
        fields_populated.append("Priority")
    else:
        fields["Priority"] = "Medium"
        fields_populated.append("Priority (default)")

    # Set initial status
    fields["Status"] = "Not Started"
    fields_populated.append("Status")

    # Link to lead if found
    if lead_id:
        fields["Leads"] = [lead_id]  # The link field to Leads table
        fields_populated.append("Leads")

    # Add raw transcription to notes
    if fields.get("Notes"):
        fields["Notes"] += f"\n\n---\nVoice transcription ({datetime.now().isoformat()}):\n{task.raw_transcription}"
    else:
        fields["Notes"] = f"Voice transcription ({datetime.now().isoformat()}):\n{task.raw_transcription}"
        fields_populated.append("Notes")

    url = f"https://api.airtable.com/v0/{CRM_BASE_ID}/{TASKS_TABLE_ID}"
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
                airtable_url = f"https://airtable.com/{CRM_BASE_ID}/{TASKS_TABLE_ID}/{record_id}"

                return CreateRecordResponse(
                    status="created",
                    intent="task",
                    record_id=record_id,
                    record_name=task.title or "Task created",
                    fields_populated=fields_populated,
                    message=f"Successfully created task: {task.title or 'Follow-up task'}",
                    airtable_url=airtable_url
                )
            else:
                logger.error(f"Airtable task error: {response.status_code} - {response.text}")
                return CreateRecordResponse(
                    status="error",
                    intent="task",
                    message=f"Airtable API error: {response.status_code}",
                    fields_populated=fields_populated
                )

    except Exception as e:
        logger.error(f"Airtable task request error: {e}")
        return CreateRecordResponse(
            status="error",
            intent="task",
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
        "version": "0.3.0",
        "target": "EF San Juan CRM",
        "supported_intents": ["new_lead", "call_note", "status_update", "task"],
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
    Receive voice transcription from Wispr and process based on intent.
    Supports: new_lead, call_note, status_update, task
    """
    logger.info(f"Received transcription: {payload.transcription[:100]}...")

    # Classify intent
    intent_result = await classify_intent(payload.transcription)
    logger.info(f"Intent: {intent_result.intent} (confidence: {intent_result.confidence})")

    # Route based on intent (same logic as voice-crm endpoint)
    if intent_result.intent == "new_lead":
        extracted = await extract_lead_fields(payload.transcription)
        result = await create_airtable_lead(extracted)
        return {
            "status": result.status,
            "intent": "new_lead",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.lead_name,
            "fields_populated": result.fields_populated,
            "message": result.message,
            "airtable_url": result.airtable_url
        }

    elif intent_result.intent == "call_note":
        extracted = await extract_call_note_fields(payload.transcription)
        lead_id = None
        if extracted.lead_identifier:
            lead = await find_lead_by_identifier(extracted.lead_identifier)
            if lead:
                lead_id = lead["id"]
        result = await create_airtable_activity(extracted, lead_id)
        return {
            "status": result.status,
            "intent": "call_note",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.record_name,
            "fields_populated": result.fields_populated,
            "message": result.message,
            "airtable_url": result.airtable_url
        }

    elif intent_result.intent == "status_update":
        extracted = await extract_status_update_fields(payload.transcription)
        if not extracted.lead_identifier:
            return {"status": "error", "intent": "status_update", "message": "Could not identify lead to update"}
        lead = await find_lead_by_identifier(extracted.lead_identifier)
        if not lead:
            return {"status": "error", "intent": "status_update", "message": f"Lead not found: {extracted.lead_identifier}"}
        result = await update_airtable_lead_status(extracted, lead["id"], lead["name"])
        return {
            "status": result.status,
            "intent": "status_update",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.record_name,
            "fields_populated": result.fields_populated,
            "message": result.message,
            "airtable_url": result.airtable_url
        }

    elif intent_result.intent == "task":
        extracted = await extract_task_fields(payload.transcription)
        lead_id = None
        if extracted.lead_identifier:
            lead = await find_lead_by_identifier(extracted.lead_identifier)
            if lead:
                lead_id = lead["id"]
        result = await create_airtable_task(extracted, lead_id)
        return {
            "status": result.status,
            "intent": "task",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.record_name,
            "fields_populated": result.fields_populated,
            "message": result.message,
            "airtable_url": result.airtable_url
        }

    else:
        return {
            "status": "skipped",
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "message": f"Intent '{intent_result.intent}' not recognized."
        }


@app.post("/api/voice-to-lead")
@app.post("/api/voice-crm")  # New multi-intent endpoint alias
async def voice_to_crm(audio: UploadFile = File(...)):
    """
    Multi-intent voice CRM endpoint for Airtable Interface Extension.
    Accepts audio file, transcribes, classifies intent, extracts fields, performs action.

    Supported intents:
    - new_lead: Create a new lead record
    - call_note: Log an activity/call note for existing lead
    - status_update: Update a lead's status
    - task: Create a follow-up task

    Flow:
    1. OpenAI Whisper → transcription
    2. Claude → intent classification
    3. Claude → intent-specific field extraction
    4. Airtable API → create/update record
    """
    logger.info(f"Received audio file: {audio.filename}, type: {audio.content_type}")

    # Step 1: Transcribe audio with Whisper
    transcription = await transcribe_audio(audio)

    # Step 2: Classify intent
    intent_result = await classify_intent(transcription)
    logger.info(f"Intent: {intent_result.intent} (confidence: {intent_result.confidence}), lead_identifier: {intent_result.lead_identifier}")

    # Step 3: Route based on intent
    if intent_result.intent == "new_lead":
        # Extract lead fields and create new lead
        extracted = await extract_lead_fields(transcription)
        logger.info(f"Extracted new lead: customer={extracted.customer_name}")
        result = await create_airtable_lead(extracted)

        return {
            "success": result.status == "created",
            "intent": "new_lead",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.lead_name,
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

    elif intent_result.intent == "call_note":
        # Extract call note fields
        extracted = await extract_call_note_fields(transcription)
        logger.info(f"Extracted call note for: {extracted.lead_identifier}")

        # Find the lead
        lead_id = None
        lead_name = extracted.lead_identifier or "unknown"
        if extracted.lead_identifier:
            lead = await find_lead_by_identifier(extracted.lead_identifier)
            if lead:
                lead_id = lead["id"]
                lead_name = lead["name"]
            else:
                logger.warning(f"Could not find lead: {extracted.lead_identifier}")

        # Create activity record
        result = await create_airtable_activity(extracted, lead_id)

        return {
            "success": result.status == "created",
            "intent": "call_note",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.record_name,
            "linked_lead": lead_name if lead_id else None,
            "lead_found": lead_id is not None,
            "airtable_url": result.airtable_url,
            "transcription": transcription,
            "extracted_fields": {
                "lead_identifier": extracted.lead_identifier,
                "activity_type": extracted.activity_type,
                "summary": extracted.summary,
                "outcome": extracted.outcome,
                "next_follow_up_date": extracted.next_follow_up_date,
                "next_steps": extracted.next_steps
            },
            "fields_populated": result.fields_populated,
            "message": result.message
        }

    elif intent_result.intent == "status_update":
        # Extract status update fields
        extracted = await extract_status_update_fields(transcription)
        logger.info(f"Extracted status update for: {extracted.lead_identifier} -> {extracted.new_status}")

        # Find the lead (required for status update)
        if not extracted.lead_identifier:
            return {
                "success": False,
                "intent": "status_update",
                "intent_confidence": intent_result.confidence,
                "transcription": transcription,
                "message": "Could not identify which lead to update. Please mention the customer name or phone number."
            }

        lead = await find_lead_by_identifier(extracted.lead_identifier)
        if not lead:
            return {
                "success": False,
                "intent": "status_update",
                "intent_confidence": intent_result.confidence,
                "transcription": transcription,
                "extracted_fields": {
                    "lead_identifier": extracted.lead_identifier,
                    "new_status": extracted.new_status
                },
                "message": f"Could not find lead matching '{extracted.lead_identifier}'. Check the name and try again."
            }

        # Update the lead status
        result = await update_airtable_lead_status(extracted, lead["id"], lead["name"])

        return {
            "success": result.status == "updated",
            "intent": "status_update",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.record_name,
            "new_status": extracted.new_status,
            "airtable_url": result.airtable_url,
            "transcription": transcription,
            "extracted_fields": {
                "lead_identifier": extracted.lead_identifier,
                "new_status": extracted.new_status,
                "reason": extracted.reason
            },
            "fields_populated": result.fields_populated,
            "message": result.message
        }

    elif intent_result.intent == "task":
        # Extract task fields
        extracted = await extract_task_fields(transcription)
        logger.info(f"Extracted task: {extracted.title}")

        # Optionally find linked lead
        lead_id = None
        lead_name = None
        if extracted.lead_identifier:
            lead = await find_lead_by_identifier(extracted.lead_identifier)
            if lead:
                lead_id = lead["id"]
                lead_name = lead["name"]

        # Create task record
        result = await create_airtable_task(extracted, lead_id)

        return {
            "success": result.status == "created",
            "intent": "task",
            "intent_confidence": intent_result.confidence,
            "record_id": result.record_id,
            "record_name": result.record_name,
            "linked_lead": lead_name,
            "airtable_url": result.airtable_url,
            "transcription": transcription,
            "extracted_fields": {
                "lead_identifier": extracted.lead_identifier,
                "task_type": extracted.task_type,
                "title": extracted.title,
                "due_date": extracted.due_date,
                "priority": extracted.priority
            },
            "fields_populated": result.fields_populated,
            "message": result.message
        }

    else:
        # Unknown intent
        return {
            "success": False,
            "intent": intent_result.intent,
            "intent_confidence": intent_result.confidence,
            "transcription": transcription,
            "message": f"Intent '{intent_result.intent}' not recognized. Try:\n- 'New lead from John Smith...' for new leads\n- 'Just talked to John Smith...' for call notes\n- 'Mark John Smith as qualified' for status updates\n- 'Remind me to call John Smith tomorrow' for tasks"
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
