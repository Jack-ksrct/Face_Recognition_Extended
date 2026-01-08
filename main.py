import cv2
import face_recognition
import numpy as np
import os
from collections import deque


# ---------------- BEARD DETECTION ----------------
def detect_beard(face_img):
    """
    Robust BEARD detection (ignores mustache & shadows)
    """

    h, w, _ = face_img.shape

    # Ignore very small faces
    if h < 120 or w < 120:
        return False

    # Beard-only region (jaw + chin)
    y1, y2 = int(h * 0.70), int(h * 0.95)
    x1, x2 = int(w * 0.25), int(w * 0.75)

    beard_region = face_img[y1:y2, x1:x2]
    if beard_region.size == 0:
        return False

    gray = cv2.cvtColor(beard_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Hair density
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    hair_ratio = cv2.countNonZero(thresh) / thresh.size

    # Texture (beard is rough)
    texture = gray.var()

    # Edge density (hair strands)
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = cv2.countNonZero(edges) / edges.size

    return (
        hair_ratio > 0.30 and
        texture > 350 and
        edge_ratio > 0.035
    )


# ---------------- SETUP ----------------
os.makedirs("faces", exist_ok=True)
name = input("Enter the name: ")

cap = cv2.VideoCapture(0)

# Temporal smoothing buffer
beard_history = deque(maxlen=7)


# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        top, left = max(0, top), max(0, left)
        bottom, right = min(frame.shape[0], bottom), min(frame.shape[1], right)

        face_img = frame[top:bottom, left:right]
        if face_img.size == 0:
            continue

        # -------- BEARD WITH SMOOTHING --------
        raw_beard = detect_beard(face_img)
        beard_history.append(raw_beard)
        stable_beard = sum(beard_history) >= 4  # majority vote

        label = "Beard Detected" if stable_beard else "No Beard"
        color = (0, 255, 0) if stable_beard else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Press 'c' to Capture | 'q' to Quit", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------- CAPTURE & ENCODE (FIXED) --------
    if key == ord('c') and face_locations:
        encodings = face_recognition.face_encodings(
            rgb_frame,
            [(top, right, bottom, left)]
        )

        if encodings:
            face_img = frame[top:bottom, left:right]
            cv2.imwrite(f"faces/{name}.jpg", face_img)
            np.save(f"faces/{name}_encoding.npy", encodings[0])
            print(f"Face & encoding saved for {name}")
        else:
            print("Encoding failed – face not clear")

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
