import uvicorn

from gradio_frontend import Frontend


if __name__ == "__main__":
    frontend = Frontend()
    uvicorn.run(frontend.app, host="0.0.0.0", port=8080)
