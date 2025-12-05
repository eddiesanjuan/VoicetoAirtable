# Claude Progress - Voice-to-Lead

> Last Updated: 2025-12-04T12:00:00Z
> Project: Voice-to-Lead (EF San Juan Internal)

## Current Session Summary

Pivoted from Wispr webhook approach to **native Airtable experience**. Built complete voice-to-lead system with:
- OpenAI Whisper for transcription
- Claude for intent classification + field extraction
- Direct Airtable API integration
- Both web recorder and Airtable Interface Extension scaffolds

## Completed This Session

### Backend (v0.2.0)
- Added OpenAI Whisper integration for audio transcription
- Created `/api/voice-to-lead` endpoint (audio → lead record)
- Created `/api/transcribe` endpoint (audio only)
- Added CORS support for Airtable extensions
- Updated all API clients (anthropic_client, openai_client)

### Frontend Options
- **Web Recorder** (`web-recorder.html`) - Standalone HTML page with:
  - Browser audio recording (MediaRecorder API)
  - Submit to backend
  - Success/error display
  - Works immediately (just open in browser)

- **Airtable Interface Extension** (`airtable-extension/`) - Scaffold with:
  - @airtable/blocks SDK setup
  - React component with audio recorder
  - Native Airtable UI components

### Configuration
- Updated `.env` with OpenAI API key (from Quoted)
- Updated `requirements.txt` with openai + python-multipart
- Full README rewrite with new architecture

## Architecture Decision

**User's Vision**: Native Airtable experience for sales reps
- Record voice directly within Airtable
- No external tools (Wispr, etc.)
- OpenAI Whisper for transcription (purpose-built)
- Claude for intelligent field extraction

**Implementation Path**:
1. **Web Recorder** (now) - Validate the flow works
2. **Button Field** (quick win) - One-click from Airtable opens web recorder
3. **Interface Extension** (polished) - Fully native experience

## Cost Analysis

| Component | Cost |
|-----------|------|
| OpenAI Whisper | ~$0.006/minute |
| Claude API | ~$0.01/request |
| **Total per lead** | **~$0.02** |

## Quick State

- **Backend**: v0.2.0 with Whisper + Claude + Airtable
- **Web Recorder**: Complete, ready to test
- **Extension**: Scaffold complete, needs Airtable developer setup
- **Blocking**: Need Airtable Personal Access Token for API calls

## Next Actions (Priority Order)

1. **Test web recorder** - Start server, open HTML, record test lead
2. **Get Airtable PAT** - For actual record creation
3. **Deploy backend** - Railway for production URL
4. **Button field** - Add to Airtable for one-click access
5. **Train team** - Best practices for voice input

## Files Modified This Session

- `src/main.py` - Added Whisper, new endpoints, CORS
- `requirements.txt` - Added openai, python-multipart
- `.env` - Added OPENAI_API_KEY
- `README.md` - Complete rewrite
- `CLAUDE_PROGRESS.md` - This file

## Files Created This Session

- `web-recorder.html` - Standalone voice recorder
- `airtable-extension/package.json` - Extension config
- `airtable-extension/frontend/index.js` - Extension React component

## Architecture Notes

### Flow Diagram
```
┌─────────────────┐
│  Sales Rep      │
│  (in Airtable)  │
└────────┬────────┘
         │ Click "Add by Voice"
         ▼
┌─────────────────┐
│  Web Recorder   │
│  (or Extension) │
└────────┬────────┘
         │ Audio file
         ▼
┌─────────────────────────────────────────────────┐
│                  Backend API                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │   Whisper   │→ │   Claude    │→ │ Airtable ││
│  │ (transcribe)│  │ (extract)   │  │  (create)││
│  └─────────────┘  └─────────────┘  └──────────┘│
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────┐
│  CRM Lead       │
│  Record Created │
└─────────────────┘
```

### Key Design Decisions

1. **OpenAI for transcription** - Whisper is purpose-built, better than Claude for audio
2. **Claude for extraction** - Better reasoning for field mapping and intent
3. **Web-first** - HTML recorder works everywhere, no SDK dependency
4. **Extension as upgrade** - Native UX when time permits
5. **~$0.02/lead** - Extremely cost-effective for high-value leads
