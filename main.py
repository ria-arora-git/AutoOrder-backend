from fastapi import FastAPI, UploadFile, File, HTTPException
from groq import Groq
import shutil
import json
import os
import uuid

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)


@app.get("/")
def root():
    return {"status": "Backend running"}


@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):

    if not file.filename.endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join("temp", temp_filename)

    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # STEP 1: Transcription
        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo"
            )

        transcript_text = transcription.text

        # STEP 2: Structured extraction
        prompt = f"""
        Extract structured order data from this transcript.
        Return ONLY valid JSON.

        Transcript:
        {transcript_text}

        JSON format:
        {{
          "store": "",
          "items": [
            {{"name": "", "quantity": 0, "unit": ""}}
          ]
        }}
        """

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You extract structured order data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        structured_data = json.loads(completion.choices[0].message.content)

        return structured_data

    except Exception as e:
        return {
            "error": str(e),
            "raw_transcript": transcript_text if "transcript_text" in locals() else ""
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)