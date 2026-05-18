import threading
import time
import webview
from app import app


def run_flask():
    app.run(port=5001, debug=False, use_reloader=False)


if __name__ == '__main__':
    # Start Flask in a background thread
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()

    # Give Flask a moment to start before opening the window
    time.sleep(1.5)

    # Open the app in a native macOS window
    webview.create_window(
        'Farmer AI Advisor',
        'http://localhost:5001',
        width=1100,
        height=800,
        resizable=True,
        min_size=(800, 600)
    )
    webview.start()
