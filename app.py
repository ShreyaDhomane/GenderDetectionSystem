# ============================================================
# ORG GENDER DETECTION
# SINGLE FILE VERSION
# MASK DETECTION REMOVED
# ESP32 INTEGRATED
# ============================================================

import os
import cv2
import time
import math
import threading
import numpy as np
import requests

from flask import (
    Flask,
    Response,
    jsonify,
    send_from_directory
)

from insightface.app import FaceAnalysis


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ---------------- Camera ---------------- #

CAMERA_SOURCE = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

WINDOW_NAME = "OrgGenderDetection"


# ---------------- Models ---------------- #

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FACE_MODEL = os.path.join(
    MODELS_DIR,
    "face",
    "face_detection_yunet_2023mar.onnx"
)


# ---------------- Detection Settings ---------------- #

REQUIRED_STABLE_FRAMES = 5
COOLDOWN_SECONDS = 10
PADDING = 80


# ---------------- Face Database ---------------- #

FACE_SIMILARITY_THRESHOLD = 0.75


# ---------------- Capture Folder ---------------- #

CAPTURE_FOLDER = os.path.join(
    BASE_DIR,
    "captures"
)

# Dedicated folder for actual male photos
MALE_CAPTURE_FOLDER = os.path.join(
    CAPTURE_FOLDER,
    "male"
)

os.makedirs(
    MALE_CAPTURE_FOLDER,
    exist_ok=True
)


# ============================================================
# ESP32 CONFIGURATION
# ============================================================

ESP32_IP = "10.144.35.105"

ESP32_URL = (
    f"http://{ESP32_IP}:80"
)

# IMPORTANT: This must match the POST endpoint implemented on the ESP32.
# Keep "/" if your ESP32 handles POST requests on the root endpoint.
ESP32_TRIGGER_PATH = "/"

ESP32_TRIGGER_URL = (
    ESP32_URL.rstrip("/") + "/"
    if ESP32_TRIGGER_PATH == "/"
    else ESP32_URL.rstrip("/") + "/" + ESP32_TRIGGER_PATH.lstrip("/")
)

# Reconnection check interval
ESP32_RECHECK_INTERVAL = 30

# Same person ke liye repeat ESP32 request cooldown
ESP32_PERSON_COOLDOWN = 3


# ESP32 state
esp32_connected = False

last_esp32_check = 0

last_esp32_sent = {}


# ============================================================
# CAMERA
# ============================================================

class Camera:

    def __init__(self):

        self.cap = cv2.VideoCapture(
            CAMERA_SOURCE
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Cannot open camera"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            FPS
        )


    def read(self):

        ret, frame = self.cap.read()

        if not ret:

            return None

        return frame


    def release(self):

        if self.cap:

            self.cap.release()


# ============================================================
# INSIGHTFACE ANALYZER
# ============================================================

class FaceAnalyzer:

    def __init__(self):

        self.app = FaceAnalysis(
            name="buffalo_s",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )


    def analyze(self, frame):

        return self.app.get(frame)


# ============================================================
# YUNET FACE DETECTOR
# ============================================================

class FaceDetector:

    def __init__(self):

        self.detector = cv2.FaceDetectorYN.create(
            model=FACE_MODEL,
            config="",
            input_size=(640, 480),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=5000
        )


    def detect(self, frame):

        h, w = frame.shape[:2]

        self.detector.setInputSize(
            (w, h)
        )

        _, faces = self.detector.detect(
            frame
        )

        return faces


# ============================================================
# CENTROID TRACKER
# ============================================================

class CentroidTracker:

    def __init__(
        self,
        max_distance=50
    ):

        self.next_id = 0

        self.objects = {}

        self.max_distance = (
            max_distance
        )


    def update(self, boxes):

        updated_objects = {}

        for box in boxes:

            x, y, w, h = box

            cx = x + w // 2
            cy = y + h // 2

            matched = False

            for (
                object_id,
                centroid
            ) in self.objects.items():

                distance = math.sqrt(
                    (cx - centroid[0]) ** 2
                    +
                    (cy - centroid[1]) ** 2
                )

                if (
                    distance
                    < self.max_distance
                ):

                    updated_objects[
                        object_id
                    ] = (
                        cx,
                        cy
                    )

                    matched = True

                    break


            if not matched:

                updated_objects[
                    self.next_id
                ] = (
                    cx,
                    cy
                )

                self.next_id += 1


        self.objects = (
            updated_objects
        )

        return self.objects


# ============================================================
# FACE DATABASE
# ============================================================

class FaceDatabase:

    def __init__(self):

        self.faces = {}

        self.next_id = 1


    def cosine_similarity(
        self,
        emb1,
        emb2
    ):

        emb1 = (
            emb1
            /
            np.linalg.norm(emb1)
        )

        emb2 = (
            emb2
            /
            np.linalg.norm(emb2)
        )

        return np.dot(
            emb1,
            emb2
        )


    def get_person_id(
        self,
        embedding
    ):

        best_id = None

        best_score = 0

        for (
            person_id,
            saved_embedding
        ) in self.faces.items():

            score = (
                self.cosine_similarity(
                    embedding,
                    saved_embedding
                )
            )

            if score > best_score:

                best_score = score

                best_id = person_id


        if (
            best_score
            >= FACE_SIMILARITY_THRESHOLD
        ):

            return best_id


        person_id = self.next_id

        self.faces[
            person_id
        ] = embedding

        self.next_id += 1

        return person_id


# ============================================================
# ESP32 COMMUNICATION
# ============================================================

def test_esp32_connection():

    global esp32_connected
    global last_esp32_check

    print(
        "\n============================================"
    )

    print(
        "       TESTING ESP32 CONNECTION"
    )

    print(
        "============================================"
    )

    print(
        f"ESP32 IP       : {ESP32_IP}"
    )

    print(
        f"ESP32 Health   : {ESP32_URL}"
    )

    print(
        f"ESP32 Trigger  : {ESP32_TRIGGER_URL}"
    )

    try:

        response = requests.get(
            ESP32_URL,
            timeout=3
        )

        last_esp32_check = time.time()

        print(
            f"Health status  : {response.status_code}"
        )

        if response.text:

            print(
                f"Health response: {response.text[:300]}"
            )

        if 200 <= response.status_code < 300:

            esp32_connected = True

            print(
                "ESP32 CONNECTED!"
            )

            print(
                "============================================\n"
            )

            return True

        print(
            f"ESP32 responded with status: "
            f"{response.status_code}"
        )

    except requests.exceptions.Timeout:

        print(
            "ESP32 CONNECTION TIMEOUT"
        )

        print(
            "Check whether ESP32 is powered ON."
        )

    except requests.exceptions.ConnectionError:

        print(
            "ESP32 NOT REACHABLE"
        )

        print(
            "Check whether ESP32 and PC "
            "are on the same network."
        )

    except Exception as error:

        print(
            f"ESP32 ERROR: {error}"
        )

    esp32_connected = False

    print(
        "============================================\n"
    )

    return False


def send_esp32_detection(
    person_id,
    gender,
    confidence
):

    global esp32_connected
    global last_esp32_check
    global last_esp32_sent

    current_time = time.time()

    # --------------------------------------------------------
    # ONLY MALE DETECTION
    # --------------------------------------------------------

    if gender != "Male":

        return False

    # --------------------------------------------------------
    # PERIODIC RECONNECTION
    # --------------------------------------------------------

    if (
        not esp32_connected
        and
        current_time - last_esp32_check
        >= ESP32_RECHECK_INTERVAL
    ):

        print(
            "[ESP32] Attempting reconnection..."
        )

        test_esp32_connection()

    # --------------------------------------------------------
    # ESP32 NOT CONNECTED
    # --------------------------------------------------------

    if not esp32_connected:

        print(
            "[ESP32] Signal NOT sent: ESP32 is offline."
        )

        return False

    # --------------------------------------------------------
    # SAME PERSON COOLDOWN
    # --------------------------------------------------------

    if (
        person_id in last_esp32_sent
        and
        current_time
        -
        last_esp32_sent[person_id]
        <
        ESP32_PERSON_COOLDOWN
    ):

        print(
            f"[ESP32] Cooldown active for person {person_id}."
        )

        return False

    # --------------------------------------------------------
    # DATA TO SEND
    # --------------------------------------------------------

    payload = {
        "gender": "Male",
        "confidence": int(confidence),
        "id": int(person_id)
    }

    print(
        "\n[ESP32] Sending captured male detection..."
    )

    print(
        f"         URL        : {ESP32_TRIGGER_URL}"
    )

    print(
        f"         Payload    : {payload}"
    )

    # --------------------------------------------------------
    # SEND WITH ONE RETRY
    # --------------------------------------------------------

    for attempt in range(1, 3):

        try:

            response = requests.post(
                ESP32_TRIGGER_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json"
                },
                timeout=3
            )

            print(
                f"[ESP32] Attempt {attempt} "
                f"status: {response.status_code}"
            )

            if response.text:

                print(
                    f"[ESP32] Response: "
                    f"{response.text[:500]}"
                )

            if 200 <= response.status_code < 300:

                last_esp32_sent[person_id] = time.time()

                esp32_connected = True
                last_esp32_check = time.time()

                print(
                    "[ESP32] Male detection signal SENT successfully."
                )

                print(
                    f"         ID         : {person_id}"
                )

                print(
                    f"         Confidence : "
                    f"{int(confidence)}%"
                )

                return True

            if 400 <= response.status_code < 500:

                print(
                    "[ESP32] Client/endpoint error. "
                    "Check ESP32_TRIGGER_PATH and ESP32 code."
                )

                last_esp32_check = time.time()

                return False

            if attempt == 1:

                time.sleep(0.2)
                continue

            print(
                "[ESP32] Server returned an error after retry."
            )

            last_esp32_check = time.time()

            return False

        except requests.exceptions.Timeout:

            print(
                f"[ESP32] Request timeout on attempt {attempt}."
            )

            if attempt == 1:

                continue

            esp32_connected = False
            last_esp32_check = time.time()

            return False

        except requests.exceptions.ConnectionError:

            print(
                f"[ESP32] Connection error on attempt {attempt}."
            )

            if attempt == 1:

                continue

            esp32_connected = False
            last_esp32_check = time.time()

            return False

        except Exception as error:

            print(
                f"[ESP32] Error on attempt {attempt}: {error}"
            )

            if attempt == 1:

                continue

            esp32_connected = False
            last_esp32_check = time.time()

            return False

    return False


# ============================================================
# MAIN ORG GENDER DETECTOR
# ============================================================

class OrgGenderDetector:

    def __init__(self):

        # ----------------------------------------------------
        # CAPTURE FOLDER
        # ----------------------------------------------------

        os.makedirs(
            MALE_CAPTURE_FOLDER,
            exist_ok=True
        )

        # All captured male photos are stored here:
        # captures/male/
        self.capture_folder = (
            MALE_CAPTURE_FOLDER
        )


        # ----------------------------------------------------
        # INITIALIZE COMPONENTS
        # ----------------------------------------------------

        self.camera = Camera()

        self.analyzer = FaceAnalyzer()

        self.face_database = (
            FaceDatabase()
        )


        # ----------------------------------------------------
        # OPTIONAL YUNET DETECTOR
        # ----------------------------------------------------

        self.face_detector = None

        if os.path.exists(
            FACE_MODEL
        ):

            try:

                self.face_detector = (
                    FaceDetector()
                )

            except Exception as error:

                print(
                    "[WARNING] YuNet detector "
                    f"could not be loaded: {error}"
                )


        # ----------------------------------------------------
        # TRACKER
        # ----------------------------------------------------

        self.tracker = (
            CentroidTracker()
        )


        # ----------------------------------------------------
        # CAPTURE STATE
        # ----------------------------------------------------

        self.stable_frames = 0

        self.last_capture_time = 0

        self.capture_count = (
            self.get_existing_capture_count()
        )


        # ----------------------------------------------------
        # CURRENT DATA
        # ----------------------------------------------------

        self.gender = "Waiting"

        self.person_id = "--"

        self.status = (
            "Waiting for male face..."
        )

        self.last_capture = "--"


        # ----------------------------------------------------
        # LOCK
        # ----------------------------------------------------

        self.lock = threading.Lock()


    # ========================================================
    # EXISTING CAPTURE COUNT
    # ========================================================

    def get_existing_capture_count(
        self
    ):

        files = [

            file

            for file in os.listdir(
                MALE_CAPTURE_FOLDER
            )

            if file.lower().endswith(
                ".jpg"
            )

        ]

        return len(files)


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self):

        with self.lock:

            frame = self.camera.read()

            if frame is None:

                return None


            # ------------------------------------------------
            # CLEAN COPY
            # ------------------------------------------------

            original_frame = (
                frame.copy()
            )


            # ------------------------------------------------
            # FACE ANALYSIS
            # ------------------------------------------------

            faces = (
                self.analyzer.analyze(
                    frame
                )
            )


            # ------------------------------------------------
            # MALE FACES ONLY
            # ------------------------------------------------

            male_faces = [

                face

                for face in faces

                if face.gender == 1

            ]


            # ------------------------------------------------
            # STABILITY
            # ------------------------------------------------

            if len(male_faces) > 0:

                self.stable_frames += 1

            else:

                self.stable_frames = 0


            # ------------------------------------------------
            # RESET CURRENT INFORMATION
            # ------------------------------------------------

            self.gender = "Waiting"

            self.person_id = "--"


            # ------------------------------------------------
            # PROCESS ALL FACES
            # ------------------------------------------------

            boxes_for_tracker = []


            for face in faces:

                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                h, w = (
                    frame.shape[:2]
                )


                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    w,
                    x2
                )

                y2 = min(
                    h,
                    y2
                )


                # --------------------------------------------
                # TRACKER BOX
                # --------------------------------------------

                boxes_for_tracker.append(

                    (
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    )

                )


                # --------------------------------------------
                # PERSON ID
                # --------------------------------------------

                embedding = (
                    face.embedding
                )

                person_id = (
                    self.face_database
                    .get_person_id(
                        embedding
                    )
                )


                # --------------------------------------------
                # GENDER
                # --------------------------------------------

                gender = (

                    "Male"

                    if face.gender == 1

                    else "Female"

                )


                # --------------------------------------------
                # DRAW FACE BOX
                # --------------------------------------------

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2

                )


                # --------------------------------------------
                # LABEL
                # --------------------------------------------

                label = (

                    f"ID: {person_id} | "

                    f"{gender}"

                )


                cv2.putText(

                    frame,

                    label,

                    (

                        x1,

                        max(

                            25,

                            y1 - 10

                        )

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 0),

                    2

                )


                # --------------------------------------------
                # STORE MALE INFORMATION
                # --------------------------------------------

                if face.gender == 1:

                    self.gender = gender

                    self.person_id = (
                        person_id
                    )


            # ------------------------------------------------
            # UPDATE TRACKER
            # ------------------------------------------------

            if boxes_for_tracker:

                self.tracker.update(
                    boxes_for_tracker
                )


            # ------------------------------------------------
            # CURRENT TIME
            # ------------------------------------------------

            current_time = (
                time.time()
            )


            # ------------------------------------------------
            # COOLDOWN
            # ------------------------------------------------

            cooldown_active = (

                current_time

                -

                self.last_capture_time

                <

                COOLDOWN_SECONDS

            )


            # ------------------------------------------------
            # AUTO CAPTURE MALE
            # ------------------------------------------------

            if (

                len(male_faces) > 0

                and

                self.stable_frames
                >=
                REQUIRED_STABLE_FRAMES

                and

                not cooldown_active

            ):

                # --------------------------------------------
                # FIRST MALE FACE
                # --------------------------------------------

                male_face = (
                    male_faces[0]
                )


                # --------------------------------------------
                # PERSON ID
                # --------------------------------------------

                embedding = (
                    male_face.embedding
                )

                male_person_id = (
                    self.face_database
                    .get_person_id(
                        embedding
                    )
                )


                # --------------------------------------------
                # CAPTURE NUMBER
                # --------------------------------------------

                self.capture_count += 1


                # --------------------------------------------
                # MALE BOUNDING BOX
                # --------------------------------------------

                x1, y1, x2, y2 = (

                    male_face
                    .bbox
                    .astype(int)

                )


                h, w = (
                    original_frame
                    .shape[:2]
                )


                # --------------------------------------------
                # PADDING
                # --------------------------------------------

                x1 = max(
                    0,
                    x1 - PADDING
                )

                y1 = max(
                    0,
                    y1 - PADDING
                )

                x2 = min(
                    w,
                    x2 + PADDING
                )

                y2 = min(
                    h,
                    y2 + PADDING
                )


                # --------------------------------------------
                # CLEAN MALE CROP
                # --------------------------------------------

                male_crop = (

                    original_frame[

                        y1:y2,

                        x1:x2

                    ]

                )


                if male_crop.size > 0:

                    # ----------------------------------------
                    # SAVE INSIDE captures/male/
                    # ----------------------------------------
                    # IMPORTANT:
                    # self.capture_folder points to:
                    # captures/male/

                    filename = os.path.join(

                        self.capture_folder,

                        f"male_"

                        f"{self.capture_count:03d}"

                        ".jpg"

                    )


                    success = cv2.imwrite(

                        filename,

                        male_crop

                    )


                    if success:

                        self.last_capture = (

                            os.path.basename(
                                filename
                            )

                        )

                        self.last_capture_time = (

                            current_time

                        )

                        self.stable_frames = 0


                        print(

                            "[CAPTURE] "

                            "Male photo saved: "

                            f"{filename}"

                        )


                        # ------------------------------------
                        # SEND MALE DETECTION TO ESP32
                        # ------------------------------------

                        send_esp32_detection(

                            person_id=male_person_id,

                            gender="Male",

                            confidence=(

                                male_face.det_score
                                * 100

                            )

                        )


                    else:

                        print(

                            "[ERROR] "

                            "Could not save image."

                        )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if cooldown_active:

                remaining = max(

                    0,

                    int(

                        COOLDOWN_SECONDS

                        -

                        (

                            current_time

                            -

                            self.last_capture_time

                        )

                    )

                )

                self.status = (

                    f"Cooldown: "

                    f"{remaining}s"

                )


            elif len(male_faces) > 0:

                self.status = (

                    f"Male detected: "

                    f"{self.stable_frames}/"

                    f"{REQUIRED_STABLE_FRAMES}"

                )


            elif len(faces) > 0:

                self.status = (

                    "Female detected - Ignored"

                )


            else:

                self.status = (

                    "Waiting for male face..."

                )


            # ------------------------------------------------
            # DRAW STATUS
            # ------------------------------------------------

            cv2.putText(

                frame,

                self.status,

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 255),

                2

            )


            # ------------------------------------------------
            # DRAW ESP32 STATUS
            # ------------------------------------------------

            esp_status = (

                "ESP32: CONNECTED"

                if esp32_connected

                else

                "ESP32: DISCONNECTED"

            )


            esp_color = (

                (0, 255, 0)

                if esp32_connected

                else

                (0, 0, 255)

            )


            cv2.putText(

                frame,

                esp_status,

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                esp_color,

                2

            )


            return frame


    # ========================================================
    # GET STATUS
    # ========================================================

    def get_status(self):

        with self.lock:

            current_time = (
                time.time()
            )


            cooldown_remaining = max(

                0,

                int(

                    COOLDOWN_SECONDS

                    -

                    (

                        current_time

                        -

                        self.last_capture_time

                    )

                )

            )


            return {

                "gender":
                    self.gender,

                "person_id":
                    self.person_id,

                "stability":
                    self.stable_frames,

                "required_frames":
                    REQUIRED_STABLE_FRAMES,

                "status":
                    self.status,

                "capture_count":
                    self.capture_count,

                "last_capture":
                    self.last_capture,

                "cooldown":
                    cooldown_remaining,

                "esp32_connected":
                    esp32_connected

            }


    # ========================================================
    # RELEASE CAMERA
    # ========================================================

    def release(self):

        with self.lock:

            self.camera.release()


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__
)


# ============================================================
# INITIALIZE DETECTOR
# ============================================================

print(
    "\n============================================"
)

print(
    "   ORG GENDER DETECTION"
)

print(
    "   Initializing AI models..."
)

print(
    "============================================\n"
)

detector = OrgGenderDetector()

print(
    "\n[READY] Detector initialized successfully."
)


# ============================================================
# INITIAL ESP32 CONNECTION TEST
# ============================================================

print(
    "\n[ESP32] Checking connection..."
)

esp32_connected = test_esp32_connection()


# ============================================================
# LIVE CAMERA GENERATOR
# ============================================================

def generate_frames():

    while True:

        frame = (
            detector.process_frame()
        )

        if frame is None:

            break


        success, buffer = (

            cv2.imencode(

                ".jpg",

                frame

            )

        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            +

            frame_bytes

            +

            b"\r\n"

        )


# ============================================================
# HTML + CSS + JAVASCRIPT
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>OrgGenderDetection</title>


<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}


body {

    font-family: Arial, sans-serif;

    background: #0b0f14;

    color: #ffffff;

}


.navbar {

    height: 70px;

    padding: 0 30px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    background: #111720;

    border-bottom: 1px solid #222b36;

}


.logo {

    font-size: 21px;

    font-weight: bold;

}


.status {

    color: #5cff9d;

    font-size: 13px;

    font-weight: bold;

}


.dot {

    display: inline-block;

    width: 9px;

    height: 9px;

    margin-right: 7px;

    background: #5cff9d;

    border-radius: 50%;

}


.dashboard {

    padding: 28px;

    max-width: 1500px;

    margin: auto;

}


.section-title {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 14px;

}


.section-title h2 {

    font-size: 18px;

}


.section-title span {

    font-size: 11px;

    padding: 6px 10px;

    background: #5cff9d;

    color: #07100b;

    border-radius: 5px;

    font-weight: bold;

}


.camera-section {

    margin-bottom: 22px;

}


.camera-box {

    width: 100%;

    height: 520px;

    background: #05080c;

    border: 1px solid #252e39;

    border-radius: 12px;

    overflow: hidden;

    display: flex;

    align-items: center;

    justify-content: center;

}


.camera-feed {

    width: 100%;

    height: 100%;

    object-fit: contain;

    display: block;

}


.info-section {

    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 18px;

}


.card {

    background: #111720;

    border: 1px solid #222b36;

    border-radius: 12px;

    padding: 21px;

}


.card h3 {

    margin-bottom: 17px;

    font-size: 15px;

}


.info-row {

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 12px 0;

    border-bottom: 1px solid #222b36;

    color: #8995a3;

}


.info-row strong {

    color: #ffffff;


}


.capture-status {

    padding: 15px;

    background: #0b1118;

    border-radius: 8px;

    color: #5cff9d;

    margin-bottom: 15px;

    min-height: 48px;

    display: flex;

    align-items: center;

}


.progress {

    height: 8px;

    background: #252e39;

    border-radius: 10px;

    overflow: hidden;

}


.progress-bar {

    width: 0%;

    height: 100%;

    background: #5cff9d;

    transition: width 0.2s ease;

}


.card p {

    margin-top: 10px;

    color: #8995a3;

    font-size: 13px;

}


.stats {

    display: grid;

    grid-template-columns: repeat(3, 1fr);

    gap: 18px;

    margin-top: 22px;

}


.stat-card {

    background: #111720;

    border: 1px solid #222b36;

    border-radius: 12px;

    padding: 20px;

}


.stat-card span {

    color: #8995a3;

    font-size: 13px;

}


.stat-card strong {

    display: block;

    margin-top: 10px;

    font-size: 28px;

}


.online {

    color: #5cff9d;

    font-size: 18px !important;

}


.recent {

    margin-top: 25px;

}


.images {

    display: grid;

    grid-template-columns:
        repeat(auto-fill, minmax(180px, 1fr));

    gap: 18px;

}


.capture-item {

    background: #111720;

    border: 1px solid #222b36;

    border-radius: 10px;

    overflow: hidden;

}


.capture-item img {

    width: 100%;

    height: 190px;

    object-fit: cover;

    display: block;

    background: #05080c;

}


.capture-name {

    padding: 10px;

    font-size: 12px;

    color: #8995a3;

    text-align: center;

}


.image-error {

    display: none;


}


.image-placeholder {

    min-height: 150px;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #111720;

    border: 1px solid #222b36;

    border-radius: 10px;

    color: #56616d;

    font-size: 12px;

}


@media (max-width: 900px) {

    .info-section {

        grid-template-columns: 1fr;

    }


    .stats {

        grid-template-columns: 1fr;

    }


    .camera-box {

        height: 400px;

    }

}


@media (max-width: 600px) {

    .dashboard {

        padding: 15px;

    }


    .navbar {

        padding: 0 15px;

    }


    .logo {

        font-size: 17px;

    }


    .camera-box {

        height: 300px;

    }

}

</style>

</head>


<body>


<header class="navbar">

    <div class="logo">
        🛡️ OrgGenderDetection
    </div>


    <div class="status">

        <span
            class="dot"
            id="status-dot"
        ></span>

        <span id="system-status">
            SYSTEM ONLINE
        </span>

    </div>

</header>


<main class="dashboard">


<section class="camera-section">

    <div class="section-title">

        <h2>Live Camera</h2>

        <span>LIVE</span>

    </div>


    <div class="camera-box">

        <img
            src="/video_feed"
            class="camera-feed"
            alt="Live Camera Feed"
        >

    </div>

</section>


<section class="info-section">


    <div class="card">

        <h3>Detection Status</h3>


        <div class="info-row">

            <span>Gender</span>

            <strong id="gender">
                Waiting
            </strong>

        </div>


        <div class="info-row">

            <span>Person ID</span>

            <strong id="person-id">
                --
            </strong>

        </div>

    </div>


    <div class="card">

        <h3>Capture Status</h3>


        <div
            class="capture-status"
            id="capture-status"
        >
            Waiting for male face...
        </div>


        <div class="progress">

            <div
                class="progress-bar"
                id="progress-bar"
            ></div>

        </div>


        <p id="stability">
            Stability: 0 / 5
        </p>

    </div>

</section>


<section class="stats">


    <div class="stat-card">

        <span>Total Captures</span>

        <strong id="total-captures">
            0
        </strong>

    </div>


    <div class="stat-card">

        <span>Male Captures</span>

        <strong id="male-captures">
            0
        </strong>

    </div>


    <div class="stat-card">

        <span>ESP32 Status</span>

        <strong
            id="system-status-card"
            class="online"
        >
            CHECKING
        </strong>

    </div>


</section>


<section class="recent">


    <div class="section-title">

        <h2>Recent Male Captures</h2>

    </div>


    <div
        class="images"
        id="capture-gallery"
    >

        <div class="image-placeholder">

            No male captures yet

        </div>


    </div>

</section>


</main>


<script>

let lastCapture = "";


async function updateDashboard() {

    try {

        const response =
            await fetch("/status");


        if (!response.ok) {

            throw new Error(
                "Status request failed"
            );

        }


        const data =
            await response.json();


        // ==========================================
        // GENDER
        // ==========================================

        document.getElementById(
            "gender"
        ).textContent =
            data.gender;


        // ==========================================
        // PERSON ID
        // ==========================================

        document.getElementById(
            "person-id"
        ).textContent =
            data.person_id;


        // ==========================================
        // CAPTURE STATUS
        // ==========================================

        document.getElementById(
            "capture-status"
        ).textContent =
            data.status;


        // ==========================================
        // STABILITY
        // ==========================================

        const stability =
            data.stability || 0;


        const required =
            data.required_frames || 5;


        document.getElementById(
            "stability"
        ).textContent =
            `Stability: ${stability} / ${required}`;


        // ==========================================
        // PROGRESS
        // ==========================================

        const percentage =
            Math.min(
                (stability / required) * 100,
                100
            );


        document.getElementById(
            "progress-bar"
        ).style.width =
            percentage + "%";


        // ==========================================
        // TOTAL CAPTURES
        // ==========================================

        document.getElementById(
            "total-captures"
        ).textContent =
            data.capture_count;


        // ==========================================
        // MALE CAPTURES
        // ==========================================

        document.getElementById(
            "male-captures"
        ).textContent =
            data.capture_count;


        // ==========================================
        // ESP32 STATUS
        // ==========================================

        const esp32Online =
            data.esp32_connected === true;


        document.getElementById(
            "system-status-card"
        ).textContent =
            esp32Online
                ? "CONNECTED"
                : "DISCONNECTED";


        document.getElementById(
            "system-status"
        ).textContent =
            esp32Online
                ? "SYSTEM ONLINE"
                : "ESP32 OFFLINE";


        // ==========================================
        // LAST CAPTURE
        // ==========================================

        if (

            data.last_capture !== "--"

            &&

            data.last_capture !== lastCapture

        ) {

            lastCapture =
                data.last_capture;

            loadCaptures();

        }

    }

    catch (error) {

        console.error(
            "Dashboard error:",
            error
        );

    }

}


// ==========================================
// LOAD CAPTURES
// ==========================================

async function loadCaptures() {

    try {

        const response =
            await fetch("/captures");


        if (!response.ok) {

            throw new Error(
                "Could not load captures"
            );

        }


        const files =
            await response.json();


        const gallery =
            document.getElementById(
                "capture-gallery"
            );


        if (!files.length) {

            gallery.innerHTML = `

                <div class="image-placeholder">

                    No male captures yet

                </div>

            `;

            return;

        }


        gallery.innerHTML =

            files.map(file => {

                return `

                    <div class="capture-item">

                        <img

                            src="/captures/${encodeURIComponent(file)}"

                            alt="Captured Male Photo"

                            loading="lazy"

                            onerror="this.parentElement.classList.add('image-error')"

                        >

                        <div class="capture-name">

                            ${file}

                        </div>

                    </div>

                `;

            }).join("");

    }

    catch (error) {

        console.error(
            "Gallery error:",
            error
        );

    }

}


// ==========================================
// STATUS EVERY 500ms
// ==========================================

setInterval(
    updateDashboard,
    500
);


// ==========================================
// CAPTURES EVERY 2 SECONDS
// ==========================================

setInterval(
    loadCaptures,
    2000
);


// ==========================================
// INITIAL LOAD
// ==========================================

updateDashboard();

loadCaptures();

</script>


</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return HTML_PAGE


# ============================================================
# LIVE VIDEO
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=(

            "multipart/x-mixed-replace; "

            "boundary=frame"

        )

    )


# ============================================================
# LIVE STATUS API
# ============================================================

@app.route("/status")
def status():

    return jsonify(
        detector.get_status()
    )


# ============================================================
# MALE CAPTURE IMAGE
# ============================================================

@app.route(
    "/captures/<filename>"
)

def captured_image(filename):

    return send_from_directory(

        detector.capture_folder,

        filename

    )


# ============================================================
# GET ALL MALE CAPTURES
# ============================================================

@app.route("/captures")

def captures():

    capture_folder = (
        detector.capture_folder
    )


    os.makedirs(

        capture_folder,

        exist_ok=True

    )


    files = [

        file

        for file in os.listdir(
            capture_folder
        )

        if file.lower().endswith(
            ".jpg"
        )

    ]


    files.sort(

        key=lambda file:

        os.path.getmtime(

            os.path.join(

                capture_folder,

                file

            )

        ),

        reverse=True

    )


    return jsonify(
        files
    )


# ============================================================
# TEST INSIGHTFACE
# ============================================================

def test_insightface():

    print(
        "\n============================================"
    )

    print(
        "       TESTING INSIGHTFACE"
    )

    print(
        "============================================\n"
    )


    test_app = FaceAnalysis(

        name="buffalo_s",

        providers=[
            "CPUExecutionProvider"
        ]

    )


    test_app.prepare(
        ctx_id=0
    )


    print(
        "InsightFace Loaded Successfully"
    )


# ============================================================
# OPTIONAL DESKTOP MODE
# ============================================================

def run_desktop_mode():

    print(
        "\n============================================"
    )

    print(
        "       DESKTOP MODE"
    )

    print(
        "       Press Q to exit"
    )

    print(
        "============================================\n"
    )


    while True:

        frame = (
            detector.process_frame()
        )


        if frame is None:

            break


        cv2.imshow(

            WINDOW_NAME,

            frame

        )


        if (

            cv2.waitKey(1)

            & 0xFF

            == ord("q")

        ):

            break


    detector.release()

    cv2.destroyAllWindows()


# ============================================================
# COMMAND LINE OPTIONS
# ============================================================

def print_help():

    print(

        """
============================================================
ORG GENDER DETECTION
============================================================

Normal Flask Dashboard:
    python app.py

Test InsightFace:
    python app.py test

Desktop Camera Mode:
    python app.py desktop

============================================================
ESP32:
    IP      : 10.144.35.105
    Port    : 80
    Trigger : Male capture
============================================================
"""

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import sys

    try:

        if len(sys.argv) > 1:

            command = (
                sys.argv[1].lower()
            )


            if command == "test":

                detector.release()

                test_insightface()


            elif command == "desktop":

                run_desktop_mode()


            elif command in (

                "help",

                "-h",

                "--help"

            ):

                print_help()


            else:

                print(

                    f"Unknown command: "

                    f"{command}"

                )

                print_help()


        else:

            print(
                "\n============================================"
            )

            print(
                "   ORG GENDER DETECTION DASHBOARD"
            )

            print(
                "============================================"
            )

            print(
                "   Open: http://localhost:5000"
            )

            print(
                f"   ESP32: "
                f"{'CONNECTED' if esp32_connected else 'DISCONNECTED'}"
            )

            print(
                "============================================\n"
            )


            app.run(

                host="0.0.0.0",

                port=5000,

                debug=False,

                threaded=True

            )


    finally:

        detector.release()