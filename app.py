import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from inference import inference

app = FastAPI(title="coleridge-kaggle-app")

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