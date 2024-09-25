from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
from run_qwen import run_qwen, create_model
import base64
from io import BytesIO
from contextlib import asynccontextmanager
from PIL import Image
import tempfile
from typing import Optional, Union, List
from torchvision.io import VideoReader

context = {}

# @asynccontextmanager
# async def lifetime(app: FastAPI):
#     print("loading model...")
#     context["model"] = create_model()
#     yield
#     context.clear()


app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("Loading model...")
    context["model"] = create_model()
    print("Model loaded.")

# Event triggered on application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    print("Clearing model context...")
    context.clear()


@app.get("/test")
def test():
    return "hi!"

# , videos: list[UploadFile] = File()
#text: str = Annotated[str, Form()]

@app.post("/qwen")
async def qwen_handler(text: str, images: Optional[UploadFile] = File(None), videos: Optional[UploadFile] = File(None)):
    # print(images.file)
    if images:
        file_bytes = await images.read()
        # Convert bytes into a BytesIO object
        image_stream = BytesIO(file_bytes)

        # Open the image using PIL
        image = [Image.open(image_stream)]
    else: image = []
    
    if videos:
        video_bytes = await videos.read()
        video_stream = BytesIO(video_bytes)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_stream.getvalue())

            print(temp_video_file.name)
            output_text = run_qwen(context["model"], text, image, [temp_video_file.name])
            return output_text

    else:
        output_text = run_qwen(context["model"], text, image, [])
        return output_text


    # return [text]

    # video_names = [video.filename for video in videos]
    # video_names = []

    # output_text = run_qwen(context["model"], text, image, video_names)
    # return output_text
    return ["dezz nuts"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)

