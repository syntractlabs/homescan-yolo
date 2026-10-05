from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import uvicorn
import io
from PIL import Image

app = FastAPI()
model = YOLO("yolov8n.pt")  # You can switch to yolov11n.pt later

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = model(img)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })

    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
