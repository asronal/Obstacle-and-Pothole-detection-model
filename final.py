import cv2
import numpy as np
import time
from datetime import datetime
import os
import sys
import argparse
import threading
import queue

# ---------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------
MODEL_PATH        = "best.onnx"
INPUT_SIZE        = 320
CONF_THRESHOLD    = 0.40
NMS_THRESHOLD     = 0.40
CLASSES           = ["obstacle", "pothole"]

SAVE_DIR          = "detections"
COOLDOWN_SECONDS  = 1
DRAW_BOXES        = True
SPATIAL_THRESHOLD = 100
TEST_OUTPUT_DIR   = "test"   # for --image/--video bbox outputs
# ---------------------------------------------------------

saved_locations = {"obstacle": [], "pothole": []}
last_save_time  = {"obstacle": 0,  "pothole": 0}
total_saved     = {"obstacle": 0,  "pothole": 0}

os.makedirs(os.path.join(SAVE_DIR, "obstacle"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "pothole"),  exist_ok=True)
print(f"[INFO] Saving detections to: {os.path.abspath(SAVE_DIR)}")


# -- Helpers -----------------------------------------------
def auto_white_balance(img_rgb):
    img = img_rgb.astype(np.float32)
    avg_r, avg_g, avg_b = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    img[:,:,0] *= avg_gray / (avg_r + 1e-6)
    img[:,:,1] *= avg_gray / (avg_g + 1e-6)
    img[:,:,2] *= avg_gray / (avg_b + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)

def stabilize_color(img_rgb):
    img_rgb = auto_white_balance(img_rgb)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def letterbox(img, new_shape=(320, 320)):
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
    top  = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, top, left

def calculate_distance(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def is_new_location(class_name, cx, cy):
    for saved in saved_locations[class_name]:
        if calculate_distance((cx, cy), saved) < SPATIAL_THRESHOLD:
            return False
    return True


# -- YOLOv8 post-processing --------------------------------
def parse_yolov8(outputs, scale, pad_top, pad_left):
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0].T    # [N, 4+C]

    boxes, confidences, class_ids = [], [], []
    for det in pred:
        class_scores = det[4:]
        class_id     = int(np.argmax(class_scores))
        conf         = float(class_scores[class_id])

        if conf < CONF_THRESHOLD:
            continue
        if class_id >= len(CLASSES):
            continue

        cx, cy, w, h = det[:4]
        x = int((cx - w / 2 - pad_left) / scale)
        y = int((cy - h / 2 - pad_top)  / scale)
        w = int(w / scale)
        h = int(h / scale)

        boxes.append([x, y, w, h])
        confidences.append(conf)
        class_ids.append(class_id)

    return boxes, confidences, class_ids


# -- Save helper -------------------------------------------
def save_detection(frame_bgr, boxes, confidences, class_ids, flat_indices):
    current_time  = time.time()
    saved_classes = []

    for i in flat_indices:
        if i >= len(boxes) or i >= len(class_ids) or i >= len(confidences):
            continue
        class_id = class_ids[i]
        if class_id >= len(CLASSES):
            continue

        class_name = CLASSES[class_id]
        x, y, w, h = boxes[i]
        cx, cy = x + w // 2, y + h // 2

        if not is_new_location(class_name, cx, cy):
            continue
        if current_time - last_save_time[class_name] < COOLDOWN_SECONDS:
            continue

        last_save_time[class_name] = current_time
        saved_locations[class_name].append((cx, cy))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filepath  = os.path.join(SAVE_DIR, class_name,
                                 f"{class_name}_{timestamp}.jpg")

        img_to_save = frame_bgr.copy()
        if DRAW_BOXES:
            for j in flat_indices:
                if j >= len(boxes) or j >= len(class_ids):
                    continue
                bx, by, bw, bh = boxes[j]
                cid   = class_ids[j]
                label = CLASSES[cid] if cid < len(CLASSES) else f"cls{cid}"
                conf  = confidences[j]
                color = (0, 0, 255) if label == "pothole" else (0, 165, 255)
                cv2.rectangle(img_to_save, (bx, by), (bx+bw, by+bh), color, 2)
                cv2.putText(img_to_save, f"{label} {conf*100:.1f}%",
                            (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img_to_save,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (10, frame_bgr.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imwrite(filepath, img_to_save)
        saved_classes.append(class_name)

    return saved_classes


# -- Argument parsing --------------------------------------
parser = argparse.ArgumentParser(description="YOLOv8n detector - source selector")
group  = parser.add_mutually_exclusive_group()
group.add_argument("--webcam", type=int, nargs="?", const=-1, metavar="INDEX",
                   help="USB webcam - auto-detect if no index given (e.g. --webcam or --webcam 2)")
group.add_argument("--video",  type=str,    metavar="PATH",
                   help="Path to a video file (e.g. --video clip.mp4)")
group.add_argument("--image",  type=str,    metavar="PATH",
                   help="Path to a single image (e.g. --image frame.jpg)")
# default (no flag) -> picamera2
args = parser.parse_args()

SOURCE = "picam"
if args.webcam is not None: SOURCE = "webcam"
elif args.video is not None: SOURCE = "video"
elif args.image is not None: SOURCE = "image"

# Create test output dir for --image/--video modes
if SOURCE in ("image", "video"):
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Test outputs will be saved to: {os.path.abspath(TEST_OUTPUT_DIR)}")

# Video writer for --video mode
video_writer       = None
output_video_path  = None
if SOURCE == "video":
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(TEST_OUTPUT_DIR, f"{base_name}_{timestamp}_output.mp4")
    print(f"[INFO] Output video will be saved to: {output_video_path}")

# -- Main loop ---------------------------------------------
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cv2.setNumThreads(4)          # all 4 Pi4 cores for inference
cv2.ocl.setUseOpenCL(False)   # disable OpenCL overhead on Pi
print("[INFO] YOLOv8n ONNX model loaded (4 threads)")

# -- Source initialisation ---------------------------------

def find_usb_webcam():
    """Scan indices 0-9, return first working webcam index, or -1 if none found."""
    for index in range(10):
        cap_test = cv2.VideoCapture(index)
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            cap_test.release()
            if ret:
                return index
    return -1

picam2 = None
cap    = None

if SOURCE == "picam":
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    picam2.set_controls({
        "AwbEnable": True, "AeEnable": True,
        "Brightness": 0.0, "Contrast": 1.0,
        "Saturation": 1.0, "Sharpness": 1.0,
    })
    print("[INFO] Source: Pi Camera (picamera2)")

elif SOURCE == "webcam":
    webcam_index = args.webcam if args.webcam != -1 else find_usb_webcam()
    if webcam_index == -1:
        print("[ERROR] No USB webcam detected. Plug one in or specify --webcam INDEX")
        sys.exit(1)
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {webcam_index}")
        sys.exit(1)
    print(f"[INFO] Source: USB webcam (index {webcam_index})")

elif SOURCE == "video":
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {args.video}")
        sys.exit(1)
    print(f"[INFO] Source: Video file ({args.video})")

elif SOURCE == "image":
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    print(f"[INFO] Source: Image file ({args.image})")

print("[INFO] Starting inference loop (CTRL+C to stop)")
print("-" * 80)

frame_count = 0
start_time  = time.time()
fps         = 0.0


def get_frame():
    """Return next BGR frame or None when source is exhausted."""
    if SOURCE == "picam":
        raw = picam2.capture_array()
        if raw is None or raw.size == 0:
            return None
        # picamera2 -> BGR2RGB -> stabilize -> RGB2BGR (matches original working flow)
        frame_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        frame_rgb = stabilize_color(frame_rgb)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    elif SOURCE in ("webcam", "video"):
        ret, frame = cap.read()
        if not ret:
            return None
        return frame                    # cap.read() already returns BGR, no processing needed

    elif SOURCE == "image":
        frame = cv2.imread(args.image)  # imread already returns BGR
        return frame


def write_video_frame(output_frame):
    """Initialize video writer on first call then write every frame."""
    global video_writer
    if video_writer is None:
        h, w = output_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
        print(f"[INFO] Video writer initialized: {w}x{h}")
    video_writer.write(output_frame)


# -- Threaded frame buffer so capture overlaps with inference ------
frame_queue  = queue.Queue(maxsize=2)
capture_done = threading.Event()

def capture_thread():
    """Continuously grab frames into a small buffer in a separate thread."""
    while not capture_done.is_set():
        frame = get_frame()
        if frame is None:
            frame_queue.put(None)   # signal end of source
            break
        try:
            frame_queue.put(frame, timeout=1)
        except queue.Full:
            pass                    # drop frame if inference can't keep up
        # Image mode: only one frame needed, stop after putting it
        if SOURCE == "image":
            frame_queue.put(None)   # signal done after single frame
            break

t_capture = threading.Thread(target=capture_thread, daemon=True)
t_capture.start()


try:
    while True:
        try:
            frame_bgr = frame_queue.get(timeout=5)
        except queue.Empty:
            print("\n[WARN] No frame received for 5s, exiting.")
            break

        # Video ended -> stop; image -> run once then stop
        if frame_bgr is None:
            if SOURCE in ("video", "image"):
                print(f"\n[INFO] Source finished.")
            break

        img, scale, pad_top, pad_left = letterbox(frame_bgr, (INPUT_SIZE, INPUT_SIZE))
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        net.setInput(blob)

        t0       = time.time()
        outputs  = [net.forward()]
        infer_ms = (time.time() - t0) * 1000

        boxes, confidences, class_ids = parse_yolov8(
            outputs, scale, pad_top, pad_left
        )
        indices  = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        flat     = indices.flatten().tolist() if len(indices) > 0 else []

        saved_info = ""
        if flat:
            saved_classes = save_detection(frame_bgr, boxes, confidences, class_ids, flat)
            if saved_classes:
                for cls in saved_classes:
                    total_saved[cls] += 1
                saved_info = f" [SAVED: {', '.join(saved_classes).upper()}]"

        frame_count += 1
        fps = frame_count / (time.time() - start_time)

        # Draw bboxes on frame for --image/--video modes
        if SOURCE in ("image", "video"):
            output_frame = frame_bgr.copy()
            for i in flat:
                if i >= len(boxes) or i >= len(class_ids):
                    continue
                bx, by, bw, bh = boxes[i]
                cid   = class_ids[i]
                conf  = confidences[i]
                label = CLASSES[cid] if cid < len(CLASSES) else f"cls{cid}"
                color = (0, 0, 255) if label == "pothole" else (0, 165, 255)
                cv2.rectangle(output_frame, (bx, by), (bx+bw, by+bh), color, 2)
                cv2.putText(output_frame, f"{label} {conf*100:.1f}%",
                            (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add overlays
            cv2.putText(output_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        (10, frame_bgr.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(output_frame, f"FPS:{fps:.1f} Det:{len(flat)}",
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            # Save based on mode
            if SOURCE == "image":
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                test_path = os.path.join(TEST_OUTPUT_DIR, f"{base_name}_{timestamp}_output.jpg")
                cv2.imwrite(test_path, output_frame)
                print(f"\n[INFO] Saved output image to: {test_path}")
            
            elif SOURCE == "video":
                write_video_frame(output_frame)

        det_str = ", ".join(
            f"{CLASSES[class_ids[i]]}:{confidences[i]*100:.0f}%"
            for i in flat if i < len(class_ids) and class_ids[i] < len(CLASSES)
        )
        tracked    = f" | Tracked: O={len(saved_locations['obstacle'])} P={len(saved_locations['pothole'])}"
        save_count = f" | Saved: O={total_saved['obstacle']} P={total_saved['pothole']}"

        sys.stdout.write("\033[K")
        print(f"\rFPS:{fps:5.2f} | Infer:{infer_ms:5.1f}ms | Det:{len(flat)}"
              + (f" | {det_str}" if det_str else "")
              + tracked + save_count + saved_info,
              end="")
        sys.stdout.flush()

        # Image mode: single inference, no loop needed
        if SOURCE == "image":
            print()   # newline after stats
            break

except KeyboardInterrupt:
    print("\n" + "-" * 80)
    print(f"[INFO] Stopping...")

finally:
    capture_done.set()          # stop capture thread
    print(f"[INFO] Unique tracked  - Obstacles: {len(saved_locations['obstacle'])}, "
          f"Potholes: {len(saved_locations['pothole'])}")
    print(f"[INFO] Total saved     - Obstacles: {total_saved['obstacle']}, "
          f"Potholes: {total_saved['pothole']}")
    print(f"[INFO] Average FPS: {fps:.2f}")
    if picam2:
        picam2.stop()
    if cap:
        cap.release()
    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved to: {output_video_path}")
    print("[INFO] Done.")
