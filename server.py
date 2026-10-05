from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import tempfile
import os

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name

        # Run YOLO
        results = model(tmp_path)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()
                })

        os.remove(tmp_path)

        return JSONResponse({"detections": detections})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def health():
    return {"status": "yolo server running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
