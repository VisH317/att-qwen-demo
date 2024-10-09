from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
from run_qwen import run_qwen, create_model
import base64
import cv2
import os
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
            tempfile_name = temp_video_file.name
        
            # Open the video using OpenCV
            cap = cv2.VideoCapture(tempfile_name)
            
            # Get video properties
            frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Width of the frames
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4

            # Create a VideoWriter object to write the output video

            with tempfile.NamedTemporaryFile(suffix=".mp4") as file:
                out = cv2.VideoWriter(file.name, fourcc, frame_rate, (width, height))

                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Write every 30th frame to the output video
                    if frame_count % 30 == 0:
                        out.write(frame)
                        print("count: ", frame_count)

                    frame_count += 1

                # Release the resources
                cap.release()
                out.release()

                print(file.name)
                output_text = run_qwen(context["model"], text, image, [file.name])
                return output_text

        # Remove the temporary input video file
        os.remove(tempfile_name)

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

