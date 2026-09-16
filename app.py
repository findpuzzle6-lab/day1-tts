import os, io
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModel

app = FastAPI()
model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5", trust_remote_code=True,
    token=os.environ.get("HF_TOKEN")
)

REF_AUDIO = "reference.wav"
REF_TEXT = "ఇక్కడ మీ ఆడియో లో మీరు చెప్పిన వాక్యం అచ్చుగా టైప్ చేయండి"

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def tts(req: TTSRequest):
    audio = model(req.text, ref_audio_path=REF_AUDIO, ref_text=REF_TEXT)
    buf = io.BytesIO()
    sf.write(buf, np.array(audio, dtype=np.float32), samplerate=24000, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")

@app.get("/")
def health():
    return {"status": "ok"}
