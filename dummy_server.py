from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
def test():
    return "hi!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dummy_server:app", host="127.0.0.1", port=8000, reload=True)
