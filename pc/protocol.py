"""UART framing between the PC and the PS: one raw 70x70 uint8 grayscale image per request, one reply with a record per path."""
from __future__ import annotations

import struct
import time

FRAME_START = b"\xA6\x6A"
REPLY_START = b"\xA5"
REPLY_END = b"\x5A"
IMAGE_BYTES = 70 * 70
PATHS = ("cnn", "snn", "cascade")
CLASSES = ("circle", "square", "triangle")   # NN output order; must match the training label order
CLASS_COUNT = len(CLASSES)
# Reply: the time the PS needed to receive the image payload (us), then one record per path:
# prediction, flags (bit 0: a head ran, bits 2:1: which head = the class the selector excluded), network slot load / run, head slot load / run
# and start-to-done time (all ns, measured by cycle counters in the PL), and the three scores.
PREFIX = struct.Struct("<I")
RESULT = struct.Struct("<BBIIIIIhhh")
REPLY_BYTES = PREFIX.size + len(PATHS) * RESULT.size


def encode_image(payload: bytes) -> bytes:
    """Frame: start bytes, payload length (u16), raw grayscale pixels (the lecture frame header, but uint8 pixels instead of Q8.8)."""
    if len(payload) != IMAGE_BYTES:
        raise ValueError(f"Expected {IMAGE_BYTES} raw grayscale uint8 pixels")
    return FRAME_START + struct.pack("<H", len(payload)) + payload


def read_reply(port, timeout: float = 30.0) -> bytes:
    """Read the reply from the serial port, skipping any text the PS prints before the start byte."""
    deadline = time.monotonic() + timeout

    def exact(count):
        data = bytearray()
        while len(data) < count:
            if time.monotonic() >= deadline:
                raise TimeoutError("FPGA reply timed out; check PS firmware and UART settings")
            data.extend(port.read(count - len(data)))
        return bytes(data)

    while exact(1) != REPLY_START:
        pass
    payload = exact(REPLY_BYTES)
    if exact(1) != REPLY_END:
        raise ValueError("Bad FPGA reply end: expected 0x5A")
    return payload


def decode_result(payload: bytes) -> dict:
    """Decode the receive time and the three records (cnn, snn, cascade)."""
    if len(payload) != REPLY_BYTES:
        raise ValueError(f"Expected {REPLY_BYTES} FPGA result bytes, got {len(payload)}")
    (uart_rx_us,) = PREFIX.unpack_from(payload)
    results = {}
    for index, name in enumerate(PATHS):
        prediction, flags, net_load_ns, net_run_ns, head_load_ns, head_run_ns, total_ns, *logits = \
            RESULT.unpack_from(payload, PREFIX.size + index * RESULT.size)
        if prediction == 0xFF:
            raise ValueError(f"FPGA {name} failed (prediction 0xFF)")
        if prediction >= CLASS_COUNT:
            raise ValueError(f"Invalid FPGA {name} class ID: {prediction}")
        results[name] = {
            "prediction": prediction,
            "head_used": bool(flags & 1),
            "excluded": (flags >> 1) & 3 if flags & 1 else None,   # class the selector excluded = which head ran
            "net_load_ns": net_load_ns,
            "net_run_ns": net_run_ns,
            "head_load_ns": head_load_ns,
            "head_run_ns": head_run_ns,
            "total_ns": total_ns,
            "logits": tuple(logits),
        }
    return {"uart_rx_us": uart_rx_us, "results": results}


def send_image(device: str, baud: int, image_gray8: bytes) -> dict:
    import serial

    frame = encode_image(image_gray8)
    # A short read timeout lets read_reply enforce its overall deadline.
    with serial.Serial(device, baudrate=baud, timeout=0.1, write_timeout=5,
                       xonxoff=False, rtscts=False, dsrdtr=False) as port:
        time.sleep(0.2)  # Let the port settle, as in the lecture sender.
        port.reset_input_buffer()
        if port.write(frame) != len(frame):
            raise OSError("Incomplete UART write")
        port.flush()
        return decode_result(read_reply(port))
