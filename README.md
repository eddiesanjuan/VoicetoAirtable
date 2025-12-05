# Voice-to-Lead

**EF San Juan Internal Tool** - Add CRM leads by voice, directly from Airtable.

## Overview

Sales reps record a voice note describing a new lead. The system automatically:
1. **Transcribes** audio using OpenAI Whisper
2. **Extracts** lead fields using Claude AI
3. **Creates** the record in EF San Juan CRM (Airtable)

```
Voice Recording → OpenAI Whisper → Claude AI → Airtable CRM
```

## Quick Start (Web Recorder)

The fastest way to test:

```bash
# 1. Install dependencies
cd voice-to-airtable
pip install -r requirements.txt

# 2. Start the server
python src/main.py

# 3. Open the web recorder
open web-recorder.html
# Or navigate to: file:///path/to/voice-to-airtable/web-recorder.html
```

Then click "Start Recording", describe a lead, and watch it appear in Airtable.

## Architecture Options

### Option A: Web Recorder (Recommended for Testing)
- Simple HTML page with audio recorder
- Can be opened locally or hosted
- Works immediately

### Option B: Airtable Interface Extension (Best UX)
- Native experience within Airtable
- Requires Airtable developer setup
- See `/airtable-extension/` for scaffold

### Option C: Button Field (Production Alternative)
- Add a Button field in Airtable that opens the web recorder
- One-click from within CRM
- No extension setup needed

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/voice-to-lead` | POST | **Main endpoint** - Audio file → Lead record |
| `/api/transcribe` | POST | Transcribe audio only (for testing) |
| `/webhook/wispr` | POST | Text webhook (for Wispr integration) |
| `/test/classify` | POST | Test intent classification |
| `/test/extract` | POST | Test field extraction |

## Configuration

Environment variables (`.env`):

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for field extraction |
| `OPENAI_API_KEY` | OpenAI API key for Whisper transcription |
| `AIRTABLE_API_KEY` | Airtable Personal Access Token |
| `EF_SANJUAN_CRM_BASE_ID` | EF San Juan CRM base ID |
| `LEADS_TABLE_ID` | Leads table ID |

## Field Extraction

Voice recordings are automatically mapped to CRM fields:

| Spoken Info | CRM Field | Example |
|-------------|-----------|---------|
| Customer name | Customer Name | "John Smith" |
| Phone number | Contact Phone | "555-123-4567" |
| Email | Contact Email | "john@example.com" |
| Property location | Property Address | "123 Seaside Dr, Destin" |
| How they found us | Lead Source | "Referral", "Website" |
| Project type | Job Segment | "RR", "RN", "CR", "CN" |
| Urgency | Priority | "High", "Medium", "Low" |
| Other details | Initial Notes | Combined notes |

## Example Voice Input

> "Got a call from Sarah Johnson at 555-123-4567. She's interested in custom doors for her beach house at 123 Seaside Drive in Destin. Referral from the Hendersons. Needs it done in the next few months."

**Extracted:**
- Customer Name: Sarah Johnson
- Contact Phone: 555-123-4567
- Property Address: 123 Seaside Drive, Destin
- Lead Source: Referral
- Priority: Medium
- Initial Notes: Custom doors, referral from Hendersons, timeline few months

## Cost Per Lead

- OpenAI Whisper: ~$0.006/minute
- Claude API: ~$0.01/request
- **Total: ~$0.02 per voice lead**

## Project Structure

```
voice-to-airtable/
├── src/
│   ├── __init__.py
│   └── main.py              # FastAPI backend
├── airtable-extension/       # Airtable Interface Extension scaffold
│   ├── package.json
│   └── frontend/
│       └── index.js
├── tests/
│   └── test_poc.py          # POC validation tests
├── web-recorder.html         # Standalone web recorder
├── .env                      # Environment configuration
├── .env.example              # Template
├── requirements.txt          # Python dependencies
├── FEATURE_LIST.json         # Feature specifications
├── CLAUDE_PROGRESS.md        # Session tracking
└── README.md                 # This file
```

## Development

```bash
# Run server with auto-reload
uvicorn src.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Test transcription only
curl -X POST http://localhost:8000/api/transcribe \
  -F "audio=@test-recording.webm"

# Test full flow
curl -X POST http://localhost:8000/api/voice-to-lead \
  -F "audio=@test-recording.webm"
```

## Deployment (Railway via GitHub)

### 1. Push to GitHub

```bash
cd voice-to-airtable
git init
git add .
git commit -m "Initial commit: Voice-to-Lead for EF San Juan"
git remote add origin https://github.com/YOUR_USERNAME/voice-to-lead.git
git push -u origin main
```

### 2. Deploy on Railway

1. Go to [railway.app](https://railway.app) and create new project
2. Select "Deploy from GitHub repo"
3. Choose your `voice-to-lead` repository
4. Add environment variables in Railway dashboard:
   - `ANTHROPIC_API_KEY` - Your Anthropic API key
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `AIRTABLE_API_KEY` - Your Airtable Personal Access Token
   - `EF_SANJUAN_CRM_BASE_ID` - `appviHeNrDzt6Lip4`
   - `LEADS_TABLE_ID` - `tblK6HaTTVYqeFtd7`
5. Railway auto-deploys on every push

### 3. Get Your URL

Railway provides a URL like `https://voice-to-lead-production.up.railway.app`

Access the recorder at: `https://YOUR-RAILWAY-URL/recorder`

### 4. Add to Airtable (Optional)

Create a Button field in your Leads table that opens your Railway URL for one-click access

## Airtable Extension Setup

To use the native Airtable experience:

1. Install Airtable CLI:
   ```bash
   npm install -g @airtable/blocks-cli
   ```

2. Initialize in Airtable:
   - Go to Extensions in your base
   - Click "Build a custom extension"
   - Follow setup wizard

3. Link the extension:
   ```bash
   cd airtable-extension
   npm install
   block run
   ```

4. Update `API_URL` in `frontend/index.js` to your deployed backend

## Next Steps

1. Deploy backend to production (Railway recommended)
2. Add Button field in Airtable pointing to web recorder
3. Train sales team on voice input best practices
4. Monitor and refine field extraction prompts
