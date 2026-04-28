from __future__ import annotations

import shutil
import sys
import urllib.request


def main() -> int:
    if shutil.which("gst-inspect-1.0") is None:
        print("skip: gst-inspect-1.0 unavailable")
        return 0
    offer = b"v=0\r\no=smoke 0 0 IN IP4 127.0.0.1\r\ns=smoke\r\nt=0 0\r\n"
    req = urllib.request.Request(
        "http://localhost:8080/webrtc/whep/cam0/ui",
        data=offer,
        method="POST",
        headers={"Content-Type": "application/sdp"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            body = response.read().decode("utf-8", "replace")
            assert response.status in (201, 503, 404), response.status
            print(body.splitlines()[0] if body else response.status)
    except OSError as exc:
        print(f"runner unavailable: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
