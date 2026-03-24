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
            model="whisper-large-v3-turbo",
            prompt="""
        This is a Hindi grocery ordering phone call.

        The speaker speaks Hindi or Hinglish.

        Common words:
        bhaiya, kilo, dabba, tel, chini, chawal, rajma, dal, atta

        Do NOT interpret as English.
        Prefer Hindi phonetics.
        """,
            language="hi"
        )

        transcript_text = transcription.text

        prompt = f"""
You extract structured grocery order data from Hindi phone calls.

The conversation may contain:
• store name
• order items
• filler speech

TASKS:

1. Identify STORE NAME if mentioned
2. Extract ALL items

RULES:

• DO NOT guess
• If store name is unclear → return null
• Store name should be a single string

• For items:
  - include raw_text
  - include normalized_name (only if confident)
  - include quantity and unit

Return STRICT JSON:

{{
  "store_name": "",
  "items":[
    {{
      "raw_text":"",
      "normalized_name":"",
      "quantity": null,
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
                {"role": "system", "content": "You extract structured grocery order data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

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