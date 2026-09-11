"""Start the bgutil-pot server and keep it running until SIGTERM."""
import subprocess
import sys
import time
import os
import signal

BUN = os.path.expanduser("~/.bun/bin/bun")
SERVER_DIR = "/home/z/my-project/scripts/bgutil-ytdlp-pot-provider/server"
LOG_FILE = "/tmp/bgutil-pot.log"
PID_FILE = "/tmp/bgutil-pot.pid"

def is_running():
    if not os.path.exists(PID_FILE):
        return False
    try:
        pid = int(open(PID_FILE).read().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False

if is_running():
    print("bgutil-pot already running")
    sys.exit(0)

# Start in foreground (we'll keep it alive via this script)
print("Starting bgutil-pot server...", flush=True)
with open(LOG_FILE, "w") as log:
    proc = subprocess.Popen(
        [BUN, "run", "src/main.ts", "--port", "4416"],
        cwd=SERVER_DIR,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach from this shell
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    print(f"Started PID {proc.pid}, waiting for ready...", flush=True)
    # Wait for it to bind
    for i in range(15):
        time.sleep(1)
        try:
            import urllib.request
            r = urllib.request.urlopen("http://127.0.0.1:4416/ping", timeout=2)
            if r.status == 200:
                print(f"Server is UP after {i+1}s: {r.read().decode()[:100]}")
                sys.exit(0)
        except Exception:
            continue
    print("Server failed to start", flush=True)
    sys.exit(1)
