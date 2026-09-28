"""Start Invariant Sentinel:  python run.py   then open http://127.0.0.1:8000"""
import threading
import webbrowser

import uvicorn

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    print(f"Invariant Sentinel is starting at {url}  (Ctrl+C to stop)")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, log_level="warning")
