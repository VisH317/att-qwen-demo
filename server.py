from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
from run_qwen import run_qwen

app = FastAPI()

@app.post("/qwen")
def qwen_handler(text: str = Annotated[str, Form()], images: list[UploadFile] = File(), videos: list[UploadFile] = File()):
    image_names = [image.filename for image in images]
    video_names = [video.filename for video in videos]

    output_text = run_qwen(text, image_names, video_names)
