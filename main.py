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
                model="whisper-large-v3",
                language="hi",
                prompt="""
Hindi grocery phone order.

STRICT:
- Hindi or Hinglish only
- No random English words
- No foreign language output
- Preserve spoken words as-is

Examples:
bhaiya, kilo, dabba, tel, chini, chawal, rajma

Bad:
metal tape bars

Good:
4 dabba tel
"""
            )

        transcript_text = transcription.text.strip()

        if len(transcript_text.split()) < 2:
            return {
                "error": "Audio too short / unclear",
                "transcript": transcript_text
            }

        print("TRANSCRIPT:", transcript_text)

        extraction_prompt = f"""
You extract grocery orders from Hindi phone calls.

STRICT RULES:

- Do NOT guess anything
- Do NOT add items not present in transcript
- Do NOT convert unclear words into known grocery items
- If unclear → keep raw_text and mark LOW confidence
- Do NOT force English translation

Return ONLY JSON:

{{
 "store_name": null_or_string,
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
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            top_p=0.1,
            response_format={"type": "json_object"},
        )

        response_text = completion.choices[0].message.content.strip()

        try:
            structured_data = json.loads(response_text)
        except Exception:
            return {
                "error": "JSON parsing failed",
                "raw_llm_output": response_text,
                "transcript": transcript_text
            }

        cleaned_items = []

        for item in structured_data.get("items", []):
            if item.get("confidence") == "low":
                item["normalized_name"] = None
            cleaned_items.append(item)

        structured_data["items"] = cleaned_items

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