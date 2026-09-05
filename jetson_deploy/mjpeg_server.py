"""
mjpeg_server.py
---------------
Tiny MJPEG-over-HTTP streamer so you can watch what the car sees from any browser on the
same network - laptop or phone - with nothing to install on the viewing device.

    http://<jetson-ip>:8080/

Why this rather than X11 forwarding: the Jetson runs headless over SSH, so there is no
display for cv2.imshow. X11 forwarding would need XQuartz on macOS and streams video
poorly. A browser tab needs no setup at all and works equally well from a phone, which is
handy when you are walking beside the car and the laptop is on a bench.

Runs its own daemon thread, so it never blocks the control loop. Frames are dropped rather
than queued if the network cannot keep up - a laggy viewer must never slow the car's
control loop down.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frameboundary"

# The page pulls single JPEGs in a loop rather than relying on multipart streaming.
# Safari handles multipart/x-mixed-replace unreliably - it often just spins - whereas
# fetching one frame at a time works in every browser. /stream.mjpg is still served for
# clients that prefer it.
_PAGE = b"""<!doctype html>
<html><head><title>RC Car - live view</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { background:#111; color:#eee; font-family:-apple-system,system-ui,sans-serif;
         margin:0; display:flex; flex-direction:column; align-items:center; }
  h3   { font-weight:500; margin:12px 0 8px; font-size:16px; }
  img  { max-width:100vw; height:auto; image-rendering:pixelated; border-radius:6px;
         background:#222; min-height:200px; }
  p    { color:#888; font-size:13px; margin:10px; }
  #fps { color:#6c6; font-variant-numeric:tabular-nums; }
</style></head>
<body>
  <h3>RC Car &mdash; live camera + segmentation</h3>
  <img id="v" alt="waiting for first frame...">
  <p><span id="fps">connecting...</span> &nbsp;&middot;&nbsp; telemetry is drawn into the frame</p>
<script>
  const img = document.getElementById('v');
  const fps = document.getElementById('fps');
  let n = 0, t0 = Date.now(), busy = false;

  async function tick() {
    if (!busy) {
      busy = true;
      try {
        const r = await fetch('/snapshot.jpg?t=' + Date.now(), {cache: 'no-store'});
        if (r.ok) {
          const b = await r.blob();
          const u = URL.createObjectURL(b);
          const old = img.src;
          img.src = u;
          if (old.startsWith('blob:')) URL.revokeObjectURL(old);
          n++;
          const dt = (Date.now() - t0) / 1000;
          if (dt >= 1) { fps.textContent = (n/dt).toFixed(1) + ' fps'; n = 0; t0 = Date.now(); }
        }
      } catch (e) { fps.textContent = 'disconnected'; }
      busy = false;
    }
    setTimeout(tick, 40);
  }
  tick();
</script>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "RCCarStream/1.0"
    protocol_version = "HTTP/1.1"   # keep-alive; needed for repeated snapshot fetches

    def log_message(self, fmt, *args):
        pass  # Silence per-request logging - it would drown the telemetry output.

    def do_GET(self):
        path = self.path.split("?", 1)[0]   # ignore the cache-busting query string

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.end_headers()
            self.wfile.write(_PAGE)
            return

        # Single frame - works in every browser, unlike multipart streaming.
        if path == "/snapshot.jpg":
            jpg = self.server.frame_source.latest()
            if jpg is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)
            return

        if path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        stream = self.server.frame_source
        try:
            while True:
                jpg = stream.wait_for_frame()
                if jpg is None:
                    break
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # Viewer closed the tab - normal.


class MjpegStreamer:
    """
    Call publish(frame_bgr) as often as you like; connected browsers get the latest frame.
    Encoding happens on the publishing thread, so keep jpeg_quality modest and scale down
    if the control loop rate starts to suffer.
    """

    def __init__(self, port=8080, jpeg_quality=70, scale=1.0):
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.scale = scale

        self._latest = None
        self._condition = threading.Condition()
        self._running = True

        self._server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        self._server.daemon_threads = True
        self._server.frame_source = self

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def publish(self, frame_bgr):
        if not self._running:
            return
        if self.scale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, None, fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self._condition:
            self._latest = buf.tobytes()
            self._condition.notify_all()

    def latest(self):
        """Most recent frame, or None if none has been produced yet."""
        with self._condition:
            return self._latest

    def wait_for_frame(self, timeout=5.0):
        with self._condition:
            if not self._condition.wait(timeout):
                return self._latest  # timed out; resend the last frame to keep the stream alive
            return self._latest

    def close(self):
        self._running = False
        with self._condition:
            self._latest = None
            self._condition.notify_all()
        try:
            self._server.shutdown()
        except Exception:
            pass
