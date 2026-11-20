# Clipforge

Turn long videos into short clips (9:16, 1:1, 16:9) with AI moment detection and burned-in captions.

## Pipeline
yt-dlp / upload → faster-whisper transcript → Gemini picks viral moments → FFmpeg crop + captions

## Requirements
Python 3.10+, FFmpeg, a Gemini API key.

## Run
```
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn main:app --port 8000
```
Open http://localhost:8000

Only process content you own or have the rights to use.
