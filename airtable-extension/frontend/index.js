/**
 * Voice-to-Lead Airtable Interface Extension
 * EF San Juan - Add leads by voice
 *
 * Records audio → Sends to backend → OpenAI Whisper → Claude → Airtable record
 */

import {
    initializeBlock,
    Box,
    Button,
    Text,
    Heading,
    Loader,
    Link,
    Icon,
    useBase,
    useRecords
} from '@airtable/blocks/ui';
import React, { useState, useRef, useCallback } from 'react';

// Backend API URL - update this when deployed
const API_URL = 'http://localhost:8000';

function VoiceToLeadExtension() {
    const base = useBase();

    // State
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [audioBlob, setAudioBlob] = useState(null);

    // Refs
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    // Start recording
    const startRecording = useCallback(async () => {
        try {
            setError(null);
            setResult(null);

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                setAudioBlob(blob);

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);

        } catch (err) {
            setError(`Microphone access denied: ${err.message}`);
        }
    }, []);

    // Stop recording
    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    }, [isRecording]);

    // Submit audio to backend
    const submitAudio = useCallback(async () => {
        if (!audioBlob) return;

        setIsProcessing(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');

            const response = await fetch(`${API_URL}/api/voice-to-lead`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                setResult(data);
                setAudioBlob(null);
            } else {
                setError(data.message || 'Failed to create lead');
            }

        } catch (err) {
            setError(`Request failed: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    }, [audioBlob]);

    // Cancel and reset
    const reset = useCallback(() => {
        setAudioBlob(null);
        setResult(null);
        setError(null);
    }, []);

    return (
        <Box padding={3}>
            <Heading size="large" marginBottom={2}>
                Add Lead by Voice
            </Heading>

            <Text textColor="light" marginBottom={3}>
                Record a voice note describing a new lead. The system will automatically extract fields and create a CRM record.
            </Text>

            {/* Error Display */}
            {error && (
                <Box
                    backgroundColor="#FEE2E2"
                    padding={2}
                    borderRadius={2}
                    marginBottom={3}
                >
                    <Text textColor="#991B1B">{error}</Text>
                </Box>
            )}

            {/* Recording Controls */}
            {!result && !isProcessing && (
                <Box marginBottom={3}>
                    {!isRecording && !audioBlob && (
                        <Button
                            onClick={startRecording}
                            icon="microphone"
                            variant="primary"
                            size="large"
                        >
                            Start Recording
                        </Button>
                    )}

                    {isRecording && (
                        <Box display="flex" alignItems="center">
                            <Box
                                width="12px"
                                height="12px"
                                backgroundColor="red"
                                borderRadius="50%"
                                marginRight={2}
                                style={{ animation: 'pulse 1s infinite' }}
                            />
                            <Button
                                onClick={stopRecording}
                                icon="check"
                                variant="danger"
                                size="large"
                            >
                                Stop Recording
                            </Button>
                        </Box>
                    )}

                    {audioBlob && !isRecording && (
                        <Box display="flex" flexDirection="column" gap={2}>
                            <Text marginBottom={2}>
                                Recording ready. Submit to create lead.
                            </Text>
                            <Box display="flex" gap={2}>
                                <Button
                                    onClick={submitAudio}
                                    icon="upload"
                                    variant="primary"
                                >
                                    Create Lead
                                </Button>
                                <Button
                                    onClick={reset}
                                    icon="redo"
                                    variant="secondary"
                                >
                                    Re-record
                                </Button>
                            </Box>
                        </Box>
                    )}
                </Box>
            )}

            {/* Processing State */}
            {isProcessing && (
                <Box
                    display="flex"
                    flexDirection="column"
                    alignItems="center"
                    padding={4}
                >
                    <Loader scale={0.5} marginBottom={2} />
                    <Text>Processing voice recording...</Text>
                    <Text textColor="light" size="small">
                        Transcribing → Extracting → Creating lead
                    </Text>
                </Box>
            )}

            {/* Success Result */}
            {result && result.success && (
                <Box
                    backgroundColor="#D1FAE5"
                    padding={3}
                    borderRadius={2}
                >
                    <Box display="flex" alignItems="center" marginBottom={2}>
                        <Icon name="check" fillColor="#059669" marginRight={1} />
                        <Heading size="small">Lead Created!</Heading>
                    </Box>

                    <Text fontWeight="bold" marginBottom={1}>
                        {result.lead_name}
                    </Text>

                    {result.extracted_fields && (
                        <Box marginTop={2} marginBottom={2}>
                            {result.extracted_fields.customer_name && (
                                <Text size="small">
                                    <strong>Customer:</strong> {result.extracted_fields.customer_name}
                                </Text>
                            )}
                            {result.extracted_fields.contact_phone && (
                                <Text size="small">
                                    <strong>Phone:</strong> {result.extracted_fields.contact_phone}
                                </Text>
                            )}
                            {result.extracted_fields.property_address && (
                                <Text size="small">
                                    <strong>Address:</strong> {result.extracted_fields.property_address}
                                </Text>
                            )}
                            {result.extracted_fields.lead_source && (
                                <Text size="small">
                                    <strong>Source:</strong> {result.extracted_fields.lead_source}
                                </Text>
                            )}
                        </Box>
                    )}

                    <Box marginTop={2} marginBottom={2} padding={2} backgroundColor="#ECFDF5" borderRadius={1}>
                        <Text size="small" textColor="light">
                            <strong>Transcription:</strong> "{result.transcription}"
                        </Text>
                    </Box>

                    <Box display="flex" gap={2} marginTop={2}>
                        {result.airtable_url && (
                            <Link
                                href={result.airtable_url}
                                target="_blank"
                                style={{ textDecoration: 'none' }}
                            >
                                <Button variant="secondary" icon="expand">
                                    Open Lead
                                </Button>
                            </Link>
                        )}
                        <Button onClick={reset} icon="plus" variant="primary">
                            Add Another
                        </Button>
                    </Box>
                </Box>
            )}

            {/* Instructions */}
            {!result && !isProcessing && !isRecording && !audioBlob && (
                <Box
                    marginTop={4}
                    padding={3}
                    backgroundColor="#F3F4F6"
                    borderRadius={2}
                >
                    <Heading size="xsmall" marginBottom={1}>
                        Tips for best results:
                    </Heading>
                    <Text size="small" textColor="light">
                        • Include customer name and contact info
                    </Text>
                    <Text size="small" textColor="light">
                        • Mention property address if known
                    </Text>
                    <Text size="small" textColor="light">
                        • Say how they found us (referral, website, etc.)
                    </Text>
                    <Text size="small" textColor="light">
                        • Describe what they're looking for
                    </Text>
                </Box>
            )}
        </Box>
    );
}

initializeBlock(() => <VoiceToLeadExtension />);
