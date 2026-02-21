"""
Brain script: reads serial from Arduino Uno, triggers play on the server
so the iPhone (web app) plays the requested audio. Run the server first. 
"""
import argparse
import logging
import sys
import time

import requests
import serial
import serial.tools.list_ports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_BAUD = 115200


def find_arduino_port():
    """Find Arduino Uno (or similar) by USB VID/PID or name."""
    for port in serial.tools.list_ports.comports():
        if port.vid is not None and port.pid is not None:
            # Arduino Uno R3: 0x2341 / 0x0043 or 0x0043; CH340: 0x1a86 / 0x7523
            if (port.vid, port.pid) in ((0x2341, 0x0043), (0x2341, 0x0001), (0x1a86, 0x7523)):
                return port.device
        if "usbmodem" in (port.device or "").lower() or "arduino" in (port.description or "").lower():
            return port.device
    return None


def request_play(server_base: str, file: str | None = None, url: str | None = None) -> bool:
    """Tell the server to send a play command to connected clients."""
    try:
        body = {}
        if file:
            body["file"] = file
        elif url:
            body["url"] = url
        else:
            return False
        r = requests.post(f"{server_base}/api/play", json=body, timeout=5)
        if r.ok and r.json().get("ok"):
            logger.info("Play requested: %s", body)
            return True
        logger.warning("Play request failed: %s %s", r.status_code, r.text)
        return False
    except Exception as e:
        logger.error("Play request error: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Arduino brain: serial -> play on iPhone")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if not set)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate")
    parser.add_argument("--map", default="EVENT:button1=alarm.wav", help="Comma-separated EVENT:name=file.wav mappings")
    args = parser.parse_args()

    # Parse event -> file mapping
    event_to_file = {}
    for part in args.map.split(","):
        part = part.strip()
        if "=" in part:
            ev, f = part.split("=", 1)
            event_to_file[ev.strip()] = f.strip()

    port = args.port or find_arduino_port()
    if not port:
        logger.error("No Arduino port found. Plug in the board or set --port.")
        sys.exit(1)
    logger.info("Using serial port: %s", port)

    try:
        ser = serial.Serial(port, args.baud, timeout=0.1)
    except Exception as e:
        logger.error("Could not open serial: %s", e)
        sys.exit(1)

    line_buf = b""
    while True:
        try:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                time.sleep(0.01)
                continue
            line_buf += chunk
            while b"\n" in line_buf or b"\r" in line_buf:
                sep = b"\n" if b"\n" in line_buf else b"\r"
                line, line_buf = line_buf.split(sep, 1)
                line = line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # e.g. EVENT:button1 or SENSOR:42
                if line in event_to_file:
                    request_play(args.server, file=event_to_file[line])
                elif line.startswith("EVENT:") and "EVENT:" in args.map:
                    request_play(args.server, file=event_to_file.get("EVENT:button1", "alarm.wav"))
        except serial.SerialException as e:
            logger.error("Serial error: %s", e)
            time.sleep(1)
        except KeyboardInterrupt:
            break
    ser.close()


if __name__ == "__main__":
    main()
