"""Live APPRCA perception bridge for the Jackal ZED WebSocket server.

This process runs on the GPU/server computer. It receives the Jackal's RGB
stream, accepts object goals from the browser, performs Grounding DINO (+ SAM
when a checkpoint is available), publishes annotated frames, and sends semantic
target guidance back to the robot. It never publishes robot velocity directly.
"""

import json
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import websocket
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    SamPredictor = None
    sam_model_registry = None


SERVER = os.environ.get("SERVER", "ws://127.0.0.1:8050").rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN", "")
CAMERA_ID = os.environ.get("CAMERA_ID", "jackal-zed2i")
DINO_REPO = os.environ.get("DINO_REPO", "IDEA-Research/grounding-dino-tiny")
SAM_CKPT = Path(os.environ.get("SAM_CKPT", Path(__file__).with_name("sam_vit_b_01ec64.pth")))
DINO_BOX_THR = float(os.environ.get("DINO_BOX_THR", "0.12"))
TAU_HIGH = float(os.environ.get("TAU_HIGH", "0.40"))
TAU_LOW = float(os.environ.get("TAU_LOW", "0.15"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STOP = threading.Event()
FRAME_Q = queue.Queue(maxsize=1)
GOAL_LOCK = threading.Lock()
GOAL = {"target": "", "active": False, "completed": False}
CONTROL_LOCK = threading.Lock()
LATEST_CONTROL = {"type": "stop", "state": "IDLE", "target": ""}


def ws_headers():
    return [f"Authorization: Bearer {API_TOKEN}"] if API_TOKEN else None


def put_latest(q, value):
    try:
        q.put_nowait(value)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(value)


class ReconnectingSender:
    def __init__(self, url, binary=False):
        self.url, self.binary = url, binary
        self.q = queue.Queue(maxsize=1)
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, payload):
        put_latest(self.q, payload)

    def _run(self):
        while not STOP.is_set():
            ws = None
            try:
                ws = websocket.create_connection(self.url, timeout=5, header=ws_headers())
                while not STOP.is_set():
                    try:
                        payload = self.q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    (ws.send_binary if self.binary else ws.send)(payload)
            except Exception as exc:
                print(f"[WS] sender reconnect: {exc}")
                STOP.wait(1.0)
            finally:
                if ws:
                    ws.close()


def receive_loop(url, callback):
    while not STOP.is_set():
        ws = None
        try:
            ws = websocket.create_connection(url, timeout=5)
            ws.settimeout(1.0)
            while not STOP.is_set():
                try:
                    callback(ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
        except Exception as exc:
            print(f"[WS] receiver reconnect: {exc}")
            STOP.wait(1.0)
        finally:
            if ws:
                ws.close()


def on_frame(data):
    if isinstance(data, bytes):
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            put_latest(FRAME_Q, frame)


def on_goal(raw):
    msg = json.loads(raw)
    with GOAL_LOCK:
        if msg.get("type") == "goal":
            GOAL.update(target=msg["target"], active=True, completed=False)
            print(f"[GOAL] Find and approach: {msg['target']}")
        elif msg.get("type") == "stop":
            GOAL.update(active=False, completed=False)
            set_control(type="stop", state="STOPPED", target=GOAL["target"])


def on_status(raw):
    msg = json.loads(raw)
    if msg.get("state") == "ARRIVED":
        with GOAL_LOCK:
            if GOAL["active"] and msg.get("target") == GOAL["target"]:
                GOAL.update(active=False, completed=True)
                set_control(type="stop", state="COMPLETED", target=GOAL["target"])
                print(f"[GOAL] Completed: {GOAL['target']}")


def set_control(**values):
    with CONTROL_LOCK:
        LATEST_CONTROL.clear()
        LATEST_CONTROL.update(values)


def control_heartbeat(sender):
    while not STOP.wait(0.25):
        with CONTROL_LOCK:
            payload = dict(LATEST_CONTROL)
        payload["timestamp"] = time.time()
        sender.send(json.dumps(payload))


def port_state(confidence):
    if confidence >= TAU_HIGH:
        return "verified"
    if confidence >= TAU_LOW:
        return "tentative"
    return "rejected"


def main():
    base_query = f"camera_id={CAMERA_ID}"
    annotated_sender = ReconnectingSender(
        f"{SERVER}/ws/push?{base_query}&eye=apprca", binary=True)
    control_sender = ReconnectingSender(
        f"{SERVER}/ws/control/push?{base_query}")

    threading.Thread(target=receive_loop,
                     args=(f"{SERVER}/ws/view?{base_query}&eye=left", on_frame), daemon=True).start()
    threading.Thread(target=receive_loop,
                     args=(f"{SERVER}/ws/goal?{base_query}", on_goal), daemon=True).start()
    threading.Thread(target=receive_loop,
                     args=(f"{SERVER}/ws/status?{base_query}", on_status), daemon=True).start()
    threading.Thread(target=control_heartbeat, args=(control_sender,), daemon=True).start()

    print(f"[DINO] Loading {DINO_REPO} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(DINO_REPO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_REPO).to(DEVICE).eval()

    sam_predictor = None
    if SamPredictor and SAM_CKPT.is_file():
        print(f"[SAM] Loading {SAM_CKPT}")
        sam_predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT)).to(DEVICE))
    else:
        print("[SAM] Checkpoint unavailable; bounding-box visualization only")

    print(f"[READY] Open {SERVER.replace('ws://', 'http://').replace('wss://', 'https://')} and submit a goal")
    try:
        while not STOP.is_set():
            frame = FRAME_Q.get()
            with GOAL_LOCK:
                goal = dict(GOAL)
            display = frame.copy()
            if not goal["active"]:
                label = "GOAL COMPLETED" if goal["completed"] else "WAITING FOR OBJECT GOAL"
                cv2.putText(display, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (40, 220, 80), 2, cv2.LINE_AA)
                ok, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    annotated_sender.send(jpg.tobytes())
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            prompt = goal["target"].lower().strip() + "."
            inputs = processor(images=pil, text=prompt, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                outputs = model(**inputs)
            sizes = torch.tensor([pil.size[::-1]], device=DEVICE)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=DINO_BOX_THR,
                text_threshold=0.15, target_sizes=sizes)[0]

            if len(result["scores"]) == 0:
                set_control(type="guidance", state="searching", target=goal["target"])
                cv2.putText(display, f"SEARCHING: {goal['target']}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 255), 2)
            else:
                best = int(torch.argmax(result["scores"]).item())
                confidence = float(result["scores"][best])
                box = result["boxes"][best].detach().cpu().numpy().astype(int)
                x1, y1, x2, y2 = box.tolist()
                state = port_state(confidence)
                mask = None
                if sam_predictor and state != "rejected":
                    sam_predictor.set_image(rgb)
                    masks, _, _ = sam_predictor.predict(box=box.astype(np.float32), multimask_output=False)
                    mask = masks[0]
                    overlay = np.zeros_like(display)
                    overlay[mask] = (60, 190, 70)
                    display = cv2.addWeighted(display, 1.0, overlay, 0.35, 0)
                color = (40, 220, 80) if state == "verified" else (0, 200, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                cv2.putText(display, f"{goal['target']} {confidence:.2f} {state}",
                            (max(5, x1), max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)
                h, w = frame.shape[:2]
                command_state = "verified" if state == "verified" else "searching"
                set_control(type="guidance", state=command_state, target=goal["target"],
                            confidence=confidence,
                            bbox_norm=[x1 / w, y1 / h, x2 / w, y2 / h])

            ok, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                annotated_sender.send(jpg.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        set_control(type="stop", state="STOPPED", target="")
        control_sender.send(json.dumps({"type": "stop", "state": "STOPPED", "timestamp": time.time()}))
        time.sleep(0.3)
        STOP.set()


if __name__ == "__main__":
    main()
