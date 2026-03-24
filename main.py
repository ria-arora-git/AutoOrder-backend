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
Hindi grocery order conversation.

STRICT RULES:
- Language is Hindi only
- Do NOT output English sentences
- Do NOT translate into English
- Preserve Hindi words phonetically

Examples:
bhaiya, kilo, dabba, tel, chini, chawal, rajma, dal

Bad output example:
"metal tape bars" ❌

Good output:
"4 dabba tel" ✅
""",
    language="hi"
)

        transcript_text = transcription.text

        prompt = f"""
You are NOT allowed to guess.

You ONLY extract items that are EXPLICITLY spoken.

STRICT RULES:

- Do NOT infer missing items
- Do NOT replace unknown words with known grocery items
- Do NOT translate into English if unclear
- If word unclear → keep raw_text and mark LOW confidence
- If something sounds like noise → keep as raw_text, do NOT interpret

IMPORTANT:
If transcript is unclear or noisy, return items with:
- normalized_name = null
- confidence = low

Return JSON:

{{
 "store_name": null_or_name_if_clearly_mentioned,
 "items":[
   {{
     "raw_text":"",
     "normalized_name":null_or_string,
     "quantity":number_or_null,
     "unit":null_or_string,
     "confidence":"high/medium/low",
     "reason":""
   }}
 ]
}}

Transcript:
{transcript_text}
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You extract structured grocery order data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=0.1,
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