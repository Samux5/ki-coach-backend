import cv2
import mediapipe as mp
import json
import os

# --- 1. Initialisierung ---
mp_pose = mp.solutions.pose
# Initialisiere Pose mit den notwendigen Parametern
pose = mp_pose.Pose(
    static_image_mode=False,        # Video-Modus
    model_complexity=1,             # Standard-Modell
    enable_segmentation=False,      # Wir brauchen keine Maske
    min_detection_confidence=0.5    # Standard-Konfidenz
)
mp_drawing = mp.solutions.drawing_utils

# --- 2. Eingabe- und Ausgabedateien ---
# Stelle sicher, dass diese Datei im selben Ordner liegt
INPUT_VIDEO = 'kniebeuge.mp4'
OUTPUT_VIDEO = 'analyse_output.mp4' # Video mit Skelett
OUTPUT_JSON = 'keypoints.json'      # Unsere Rohdaten

# --- 3. Video verarbeiten ---
cap = cv2.VideoCapture(INPUT_VIDEO)

# Video-Metadaten für das Output-Video holen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# VideoWriter-Objekt zum Speichern des Videos
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# Liste zum Speichern aller Keypoints
all_frames_keypoints = []

print(f"Verarbeite Video: {INPUT_VIDEO}")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Video-Ende erreicht oder Fehler beim Lesen.")
        break

    # Bild von BGR zu RGB konvertieren (MediaPipe braucht RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pose-Erkennung durchführen
    results = pose.process(image_rgb)

    # Keypoints extrahieren und speichern
    frame_keypoints = {}
    if results.pose_landmarks:
        # Gehe durch alle 33 Landmarks
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            # Wir speichern den Namen (über den Index) und die (x, y, z, visibility) Koordinaten
            landmark_name = mp_pose.PoseLandmark(i).name
            frame_keypoints[landmark_name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        all_frames_keypoints.append(frame_keypoints)

        # Skelett auf das Originalbild (BGR) zeichnen
        mp_drawing.draw_landmarks(
            image, # Das BGR-Bild zum Zeichnen
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

    # Frame in die Ausgabedatei schreiben
    out.write(image)

# --- 4. Aufräumen und Speichern ---
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()

# Keypoints als JSON-Datei speichern
with open(OUTPUT_JSON, 'w') as f:
    json.dump(all_frames_keypoints, f, indent=4)

print(f"Verarbeitung abgeschlossen!")
print(f"Analysiertes Video gespeichert als: {os.path.abspath(OUTPUT_VIDEO)}")
print(f"Keypoint-Daten gespeichert als: {os.path.abspath(OUTPUT_JSON)}")