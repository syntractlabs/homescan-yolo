from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import tempfile
import uvicorn

app = FastAPI()
model = YOLO("yolov8n.pt")


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    tmp_path = None

    try:
        image_bytes = await image.read()

        if not image_bytes:
            return JSONResponse(
                {"error": "The uploaded image was empty."},
                status_code=400,
            )

        original_name = image.filename or "damage.jpg"
        extension = os.path.splitext(original_name)[1].lower()

        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            extension = ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            tmp_path = tmp.name

        results = model.predict(source=tmp_path)

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append(
                    {
                        "class": int(box.cls.item()),
                        "confidence": float(box.conf.item()),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )

        return JSONResponse({"detections": detections})

    except Exception as error:
        return JSONResponse(
            {"error": str(error)},
            status_code=500,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/")
def health():
    return {"status": "yolo server running", "backend_version": "temp-extension-fix-v2"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    uvicorn.run(app, host="0.0.0.0", port=port)

