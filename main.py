"""Modal deployment — Seoul Next-Day Maximum Temperature Predictor (Streamlit).

Jalankan:
    pip install modal
    modal serve main.py     # dev, hot-reload
    modal deploy main.py    # production

Endpoint: https://<workspace>--ac-seoul-temp-v2-proxy.us-east.modal.run
Cloudflare Worker (subdomain suhu-v2) meneruskan request ke endpoint ini.
"""

import subprocess

import modal

APP_NAME = "ac-seoul-temp-v2-proxy"
PORT = 8501

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path="/root",
        ignore=[".git", "__pycache__", ".venv", "*.png"],
    )
)


@app.function(
    image=image,
    timeout=3600,
    # Streamlit = long-lived server; biarkan Modal yang mengatur lifecycle
    min_containers=0,
    max_containers=5,
)
@modal.web_server(port=PORT, startup_timeout=180)
def serve() -> None:
    """Jalankan Streamlit di dalam container Modal."""
    subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.port", str(PORT),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.enableWebsocketCompression", "false",
        ],
        cwd="/root",
    )
