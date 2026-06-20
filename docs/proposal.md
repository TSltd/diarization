built as a separate component:

diarization.dll

A clean architecture would be something like:

audio/
├─ whisper_transcriber
├─ speaker_embedding
├─ diarization_engine
├─ transcript_merger
└─ output_formatters

Where:

class DiarizationEngine {
public:
std::vector<TranscriptSegment> process(
const std::vector<WhisperSegment>& segments,
const AudioBuffer& audio);
};

and internally it owns:

ISpeakerEmbeddingModel
SpeakerClusterManager
LabelSmoother

Then the CLI simply does:

record audio
↓
run whisper
↓
if diarize:
diarization_engine.process(...)
↓
output

So you get:

unit testing without microphone input
test WAVs can be run directly
future use outside audio record
easier maintenance
easier replacement of speaker models later

The acceptance tests are already written in a way that favors this design because most of them are based on:

--from-wav

rather than live microphone capture.

One thing that probably can't be fully standalone-

The assistant/TTS tagging:

record_assistant_segment(...)

is application-specific.

I'd keep that outside the diarization engine.

The diarization module should only know:

Human speech segment
→ speaker label

Then the application merges:

ASSISTANT
SPEAKER_00
SPEAKER_01

chronologically afterward.
