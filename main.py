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

    transcript_text = ""

    try:
        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo"
            )

        transcript_text = transcription.text

        prompt = f"""
        You extract FINAL CONFIRMED grocery orders from Hindi phone conversations.

        STRICT RULES:

        1. Only extract items that are FINALIZED at the end of the conversation.
        2. If item was discussed earlier but later changed/rejected → IGNORE.
        3. If quantity unclear → IGNORE item.
        4. Ignore filler talk.
        5. Ignore comparisons.
        6. Ignore cancelled items.

        Think step-by-step internally but output ONLY final JSON.

        Return STRICT JSON:

        {{
        "items":[
        {{
            "name":"",
            "quantity":number,
            "unit":"kg/bori/packet/etc"
        }}
        ]
        }}

        Conversation:

        {transcript_text}
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You extract structured order data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )


        response_text = completion.choices[0].message.content.strip()

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]

        structured_data = json.loads(response_text)

        return {
            "model_used": "llama-3.1-8b-instant",
            "structured": structured_data,
            "transcript": transcript_text
        }
    except Exception as e:
        return {
            "error": str(e),
            "raw_transcript": transcript_text
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)