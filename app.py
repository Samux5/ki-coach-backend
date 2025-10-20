import cv2
import mediapipe as mp
import json
import math
import os
from flask import Flask, request, jsonify
import tempfile
from flask_cors import CORS

# --- Initialisiere Flask Server ---
app = Flask(__name__)
CORS(app)

# --- KONSTANTEN ---
MIN_VISIBILITY = 0.3 
RUECKEN_WINKEL_SCHWELLE = 35 
DEBUG_VIDEO_FILENAME = 'debug_output.mp4' 

# --- MEDIA PIPE INITIALISIERUNG ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_processor = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# --- HILFSFUNKTIONEN ---
def calculate_angle(a, b, c):
    """Berechnet den Winkel zwischen drei Gelenken."""
    try:
        a_x, a_y = a['x'], a['y']
        b_x, b_y = b['x'], b['y']
        c_x, c_y = c['x'], c['y']
        angle_rad = math.atan2(c_y - b_y, c_x - b_x) - math.atan2(a_y - b_y, a_x - b_x)
        angle_deg = math.degrees(angle_rad)
        angle_deg = abs(angle_deg)
        if angle_deg > 180: angle_deg = 360 - angle_deg
        return angle_deg
    except: return None

def calculate_line_angle_with_horizontal(p1, p2):
    """Berechnet den Winkel einer Linie (p1-p2) zur Horizontalen."""
    try:
        delta_y = p2['y'] - p1['y']
        delta_x = p2['x'] - p1['x']
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    except: return None

# --- API ENDPUNKT ---
@app.route("/analyse", methods=["POST"])
def analyse_video_endpoint():
    if 'video' not in request.files: return jsonify({"error": "Keine Videodatei."}), 400
    video_file = request.files['video']
    if video_file.filename == '': return jsonify({"error": "Keine Datei ausgewählt."}), 400

    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_video:
        video_file.save(temp_video.name)
        all_frames_keypoints = []
        cap = cv2.VideoCapture(temp_video.name)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DEBUG_VIDEO_FILENAME, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_processor.process(image_rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(image)
            frame_keypoints = {}
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    frame_keypoints[mp_pose.PoseLandmark(i).name] = {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                all_frames_keypoints.append(frame_keypoints)
        cap.release()
        out.release()

    if not all_frames_keypoints: return jsonify({"error": "Keine Posen erkannt."}), 400

    max_hip_y = 0
    tiefster_frame_index = -1

    for i, frame in enumerate(all_frames_keypoints):
        try:
            hip = frame.get('LEFT_HIP') or frame.get('RIGHT_HIP')
            if hip and hip['visibility'] > MIN_VISIBILITY:
                if hip['y'] > max_hip_y:
                    max_hip_y = hip['y']
                    tiefster_frame_index = i
        except: continue
    
    if tiefster_frame_index == -1:
        return jsonify({"error": "Konnte Hüfte im Video nicht zuverlässig tracken."}), 400

    # ANALYSIERE ALLE WERTE AN GENAU DIESEM EINEN, KORREKTEN FRAME
    bewertung_tiefe = "Nicht analysiert"
    bewertung_ruecken = "Nicht analysiert"
    hueft_unter_knie = False
    debug_info = {}
    
    try:
        tiefster_frame = all_frames_keypoints[tiefster_frame_index]
        shoulder = tiefster_frame.get('LEFT_SHOULDER') or tiefster_frame.get('RIGHT_SHOULDER')
        hip = tiefster_frame.get('LEFT_HIP') or tiefster_frame.get('RIGHT_HIP')
        knee = tiefster_frame.get('LEFT_KNEE') or tiefster_frame.get('RIGHT_KNEE')
        ankle = tiefster_frame.get('LEFT_ANKLE') or tiefster_frame.get('RIGHT_ANKLE')

        if hip: debug_info['hip_visibility'] = round(hip.get('visibility', 0), 2)
        if knee: debug_info['knee_visibility'] = round(knee.get('visibility', 0), 2)

        # FINALE ANALYSE DER TIEFE (Hüfte vs. Knie)
        can_analyse_depth = hip and knee and hip.get('visibility', 0) > MIN_VISIBILITY and knee.get('visibility', 0) > MIN_VISIBILITY
        if can_analyse_depth:
            hueft_unter_knie = hip['y'] > knee['y']
            bewertung_tiefe = "Gut" if hueft_unter_knie else "Zu hoch"
        else:
            if not knee:
                debug_info['depth_error'] = "Knie-Gelenk wurde am tiefsten Punkt nicht gefunden."
            elif knee and knee.get('visibility', 0) <= MIN_VISIBILITY:
                debug_info['depth_error'] = f"Knie-Sichtbarkeit ({debug_info.get('knee_visibility')}) war unter dem Schwellenwert von {MIN_VISIBILITY}."

        # Analyse des Rückens
        can_analyse_back = shoulder and hip and knee and ankle and all(p.get('visibility', 0) > MIN_VISIBILITY for p in [shoulder, hip, knee, ankle])
        if can_analyse_back:
            winkel_oberkoerper = calculate_line_angle_with_horizontal(shoulder, hip)
            winkel_schienbein = calculate_line_angle_with_horizontal(knee, ankle)
            if winkel_oberkoerper is not None and winkel_schienbein is not None:
                ruecken_schienbein_differenz = abs(abs(winkel_oberkoerper) - abs(winkel_schienbein))
                bewertung_ruecken = "Gut" if ruecken_schienbein_differenz < RUECKEN_WINKEL_SCHWELLE else "Vorgebeugt"
                debug_info['ruecken_schienbein_differenz_grad'] = round(ruecken_schienbein_differenz, 2)
        
    except Exception as e:
        debug_info['general_error'] = str(e)

    # Bereite die finale Antwort vor
    response_data = {
        "bewertung_tiefe": bewertung_tiefe,
        "bewertung_ruecken": bewertung_ruecken,
        "huefte_war_unter_knie": hueft_unter_knie,
        "analysierter_frame_index": tiefster_frame_index,
        "debug_info": debug_info
    }
    return jsonify(response_data), 200

# --- Startseite bleibt gleich ---
@app.route("/")
def index():
    return "<h1>KI-Form-Coach API</h1><p>Der Server läuft! 🚀</p>"

# Der app.run Block wurde für die Produktion entfernt.






















