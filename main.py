import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException

from predict import predict_audio


app = FastAPI(
    title="AI Audio Detection API",
    description="Detects whether an audio file is real or AI-generated.",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message": "AI Audio Detection API is running",
        "status": "online"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------
    # Check file
    # --------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )


    # --------------------------------------------------
    # Check extension
    # --------------------------------------------------

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".ogg"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()


    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, MP3, FLAC, M4A or OGG."
            )
        )


    temp_path = None


    try:

        # --------------------------------------------------
        # Create temporary file
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file
            )


        # --------------------------------------------------
        # Run ML pipeline
        # --------------------------------------------------

        result = predict_audio(
            temp_path
        )


        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {

            "filename": file.filename,

            "prediction": result["prediction"],

            "probability": result["probability"],

            "xgboost_probability": result[
                "xgb_probability"
            ],

            "cnn_probability": result[
                "cnn_probability"
            ],

            "threshold": result["threshold"]

        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # --------------------------------------------------
        # Delete temporary audio file
        # --------------------------------------------------

        if temp_path is not None:

            try:

                os.remove(temp_path)

            except OSError:

                pass