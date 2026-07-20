import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from inference import inference
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

origins = os.getenv("CORS")

app = FastAPI(title="coleridge-kaggle-app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "App is running ✅"}

class RequestBody(BaseModel):
    text : str

@app.post("/predict")
async def predict(req : RequestBody):
    res = inference( req.text)

    if not res:
        res.append("No Dataset Found 👎🏻")

    return {"ner_list": f"{res}"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)