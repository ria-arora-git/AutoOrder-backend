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
        You extract grocery orders from Hindi phone calls.

        IMPORTANT:

        Extract ALL mentioned items.
        Do NOT remove uncertain items.

        For each item assign:

        • confidence = high / medium / low
        • reason = short explanation

        Rules:

        HIGH → clearly ordered, not corrected  
        MEDIUM → likely ordered but unclear  
        LOW → discussed / rejected / unclear  

        Return STRICT JSON:

        {{
        "items":[
        {{
            "name":"",
            "quantity":number_or_null,
            "unit":"",
            "confidence":"",
            "reason":""
        }}
        ]
        }}

        Conversation:

        {transcript_text}
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Extract final grocery orders."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"} 
        )


        response_text = completion.choices[0].message.content.strip()

        structured_data = json.loads(completion.choices[0].message.content)

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