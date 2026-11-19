#!/usr/bin/env python3
"""Draw a shape and send it as a raw 70x70 grayscale image to the FPGA over UART; shows the reply of the CNN, the SNN and the cascade."""
from __future__ import annotations

import argparse
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np

from protocol import CLASSES, PATHS, send_image

SIZE = 280

def us(nanoseconds):
    return f"{nanoseconds / 1000:.2f}"


def head_text(row):
    """Whether the cascade ran a head, and which one (a head is named after the class the selector excluded)."""
    if row["head_used"]:
        return f'yes ({CLASSES[row["excluded"]]} excluded)'
    return "no (selector sure)" if row["name"] == "cascade" else "—"


def format_reply(result):
    lines = []
    for name in PATHS:
        row = dict(result["results"][name], name=name)
        prediction = row["prediction"]
        lines.append(f"{name.upper()}: {CLASSES[prediction]} (class {prediction})")
        if name == "cascade":
            lines.append(f"  Head: {head_text(row)}")
        lines.extend((
            f'  PL time: {us(row["total_ns"])} µs from start to done',
            f'    run: network {us(row["net_run_ns"])} µs, head {us(row["head_run_ns"])} µs;'
            f' load: network {us(row["net_load_ns"])} µs, head {us(row["head_load_ns"])} µs',
            f'  Logits (raw int16): {row["logits"]}',
        ))
    lines.append(f'Image payload received by the PS in {result["uart_rx_us"]} µs')
    return "\n".join(lines)


def prepare_drawing(raw):
    if raw.shape != (SIZE, SIZE) or raw.dtype != np.uint8:
        raise ValueError("Expected 280x280 uint8 drawing")
    if not np.any(raw < 220):
        raise ValueError("Draw a shape first")
    image = cv2.resize(raw, (70, 70), interpolation=cv2.INTER_AREA)
    if image.shape != (70, 70) or image.dtype != np.uint8:
        raise ValueError("Expected raw 70x70 uint8 grayscale image")
    return image, image.tobytes(order="C")


class DrawingApp:
    def __init__(self, root, port, baud):
        self.root = root
        self.raw = np.full((SIZE, SIZE), 255, np.uint8)
        self.last = None
        self.request_id = 0
        self.messages = queue.Queue()
        self.port = tk.StringVar(value=port)
        self.baud = tk.StringVar(value=str(baud))
        self.status = tk.StringVar(value="Draw a black shape on white, then send it to the FPGA.")
        root.title("Assignment 2 — FPGA shape inference")
        root.columnconfigure(0, weight=1)
        ttk.Label(root, text="Draw one shape. The same image goes to the CNN, the SNN and the cascade (selector + head).").grid(
            row=0, column=0, padx=12, pady=8)
        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="white", cursor="crosshair")
        self.canvas.grid(row=1, column=0, padx=12, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.down)
        self.canvas.bind("<B1-Motion>", self.move)
        self.canvas.bind("<ButtonRelease-1>", self.up)
        controls = ttk.Frame(root)
        controls.grid(row=2, column=0, padx=12, pady=8)
        ttk.Button(controls, text="Clear (Esc)", command=self.clear).grid(row=0, column=0, padx=4)
        ttk.Button(controls, text="Export PNG + RAW", command=self.export).grid(row=0, column=1, padx=4)
        self.send_button = ttk.Button(controls, text="Send to FPGA", command=self.send)
        self.send_button.grid(row=0, column=2, padx=4)
        connection = ttk.Frame(root)
        connection.grid(row=3, column=0, padx=12, pady=8)
        ttk.Label(connection, text="Serial port").grid(row=0, column=0)
        ttk.Entry(connection, textvariable=self.port, width=24).grid(row=0, column=1, padx=6)
        ttk.Label(connection, text="Baud").grid(row=0, column=2)
        ttk.Entry(connection, textvariable=self.baud, width=10).grid(row=0, column=3, padx=6)
        columns = (("label", "Prediction", 90), ("head", "Head used", 210), ("net_run", "Network µs", 90),
                   ("head_run", "Head µs", 80), ("load", "Load µs", 80), ("total", "Total µs", 80))
        self.table = ttk.Treeview(root, columns=[key for key, _, _ in columns], height=3)
        self.table.heading("#0", text="Path")
        self.table.column("#0", width=80)
        for key, title, width in columns:
            self.table.heading(key, text=title)
            self.table.column(key, width=width)
        for name in PATHS:
            self.table.insert("", "end", iid=name, text=name, values=("—",) * len(columns))
        self.table.grid(row=4, column=0, padx=12, pady=8, sticky="ew")
        ttk.Label(root, textvariable=self.status, wraplength=620).grid(row=5, column=0, padx=12, pady=8)
        ttk.Label(root, text="FPGA reply").grid(row=6, column=0, padx=12, sticky="w")
        self.reply = ScrolledText(root, width=76, height=10, state="disabled", wrap="word")
        self.reply.grid(row=7, column=0, padx=12, pady=8, sticky="nsew")
        root.rowconfigure(7, weight=1)
        root.bind("<Escape>", lambda _: self.clear())
        root.after(100, self.poll)

    def point(self, event):
        return (max(0, min(SIZE - 1, event.x)), max(0, min(SIZE - 1, event.y)))

    def down(self, event):
        self.last = self.point(event)
        x, y = self.last
        cv2.circle(self.raw, self.last, 4, 0, -1, cv2.LINE_AA)
        self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black", outline="black")

    def move(self, event):
        point = self.point(event)
        if self.last is not None:
            cv2.line(self.raw, self.last, point, 0, 8, cv2.LINE_AA)
            self.canvas.create_line(*self.last, *point, fill="black", width=8, capstyle=tk.ROUND)
            self.last = point

    def up(self, event):
        self.move(event)
        self.last = None

    def clear(self):
        self.raw.fill(255)
        self.canvas.delete("all")
        self.last = None

    def export(self):
        try:
            image, payload = prepare_drawing(self.raw)
            target = filedialog.asksaveasfilename(
                initialdir=str(Path(__file__).resolve().parents[1]),
                defaultextension=".png", filetypes=[("PNG image", "*.png")])
            if not target:
                return
            path = Path(target)
            raw_path = path.with_suffix(".gray8")
            if raw_path.exists() and not messagebox.askyesno("Replace raw image?", f"Replace {raw_path}?"):
                return
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Cannot write {path}")
            raw_path.write_bytes(payload)
            self.status.set(f"Saved {path.name} + {raw_path.name} (4900 bytes).")
        except (ValueError, OSError, cv2.error) as error:
            messagebox.showerror("Export failed", str(error))

    def send(self):
        try:
            _, payload = prepare_drawing(self.raw)
            device = self.port.get().strip()
            baud = int(self.baud.get())
            if not device or baud <= 0:
                raise ValueError("Set serial port and positive baud rate")
        except ValueError as error:
            messagebox.showerror("Cannot send", str(error))
            return
        self.request_id = (self.request_id + 1) & 0xFFFFFFFF
        request_id = self.request_id
        self.send_button.state(["disabled"])
        for name in self.table.get_children():
            self.table.item(name, values=("pending",) + ("—",) * 5)
        self.status.set(f"Request {request_id}: waiting for FPGA; up to 30 s.")
        self.show_reply("Waiting for FPGA…")

        def work():
            try:
                result = send_image(device, baud, payload)
                self.messages.put((request_id, result, None))
            except Exception as error:
                self.messages.put((request_id, None, str(error)))

        threading.Thread(target=work, daemon=True).start()

    def show_reply(self, text):
        self.reply.configure(state="normal")
        self.reply.delete("1.0", "end")
        self.reply.insert("1.0", text)
        self.reply.configure(state="disabled")

    def poll(self):
        try:
            request_id, result, error = self.messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self.send_button.state(["!disabled"])
            if error is not None:
                for name in self.table.get_children():
                    self.table.item(name, values=("error",) + ("—",) * 5)
                self.status.set(error)
                self.show_reply(error)
            else:
                for name, row in result["results"].items():
                    row = dict(row, name=name)
                    self.table.item(name, values=(
                        CLASSES[row["prediction"]], head_text(row), us(row["net_run_ns"]), us(row["head_run_ns"]),
                        us(row["net_load_ns"] + row["head_load_ns"]), us(row["total_ns"])))
                self.status.set(f"Request {request_id} complete.")
                reply = format_reply(result)
                self.show_reply(reply)
                print(reply, flush=True)
        self.root.after(100, self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=4000000)
    args = parser.parse_args()
    root = tk.Tk()
    DrawingApp(root, args.port, args.baud)
    root.mainloop()


if __name__ == "__main__":
    main()
