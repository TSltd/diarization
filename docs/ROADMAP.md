Phase 1

Create DiarizationEngine
Works on WAV files
No live microphone dependencies
No CLI dependencies

Phase 2

Integrate with audio record

Integration Later

The existing command becomes:

audio record
↓
whisper.cpp
↓
DiarizationEngine
↓
formatter
↓
output
