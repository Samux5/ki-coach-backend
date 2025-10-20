import json
import math
import os

# --- HILFSFUNKTION: Winkel berechnen ---
def calculate_angle(a, b, c):
    """Berechnet den Winkel zwischen drei Punkten (a, b, c), wobei b der Scheitelpunkt ist."""
    try:
        # Koordinaten extrahieren
        a_x, a_y = a['x'], a['y']
        b_x, b_y = b['x'], b['y']
        c_x, c_y = c['x'], c['y']
        
        # Vektoren berechnen
        v1_x, v1_y = a_x - b_x, a_y - b_y
        v2_x, v2_y = c_x - b_x, c_y - b_y
        
        # Winkelberechnung mit atan2
        angle_rad = math.atan2(v2_y, v2_x) - math.atan2(v1_y, v1_x)
        angle_deg = math.degrees(angle_rad)
        
        # Winkel auf 0-180 Grad normalisieren
        angle_deg = abs(angle_deg)
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
            
        return angle_deg
    except Exception as e:
        # Falls Koordinaten fehlen (z.B. Punkt nicht sichtbar)
        return None

# --- HAUPTPROGRAMM ---
JSON_FILE = 'keypoints.json'
MIN_VISIBILITY = 0.5 # Wie sicher muss sich die KI sein (0.0 - 1.0)
TIEFEN_SCHWELLE = 95 # Unser "Business-Logik": unter 95 Grad = tief

# --- 1. Daten laden ---
if not os.path.exists(JSON_FILE):
    print(f"FEHLER: Die Datei '{JSON_FILE}' wurde nicht gefunden.")
    print("Bitte führe zuerst 'python3 analyse.py' aus.")
    exit()

try:
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
except json.JSONDecodeError:
    print(f"FEHLER: Die Datei '{JSON_FILE}' ist beschädigt oder leer.")
    exit()

if not data:
    print("FEHLER: Die JSON-Datei ist leer. Die Analyse war nicht erfolgreich.")
    exit()

print(f"Analysiere {len(data)} Frames aus {JSON_FILE}...")

min_hip_angle = 180  # Startwert (gestreckte Hüfte)
min_knee_angle = 180 # Startwert (gestrecktes Knie)
valid_frames = 0

# Wir gehen davon aus, dass die Person seitlich steht.
# Wir versuchen BEIDE Seiten (links/rechts) zu finden, falls eine verdeckt ist.

for frame in data:
    try:
        # Versuche LINKE Seite
        shoulder = frame['LEFT_SHOULDER']
        hip = frame['LEFT_HIP']
        knee = frame['LEFT_KNEE']
        ankle = frame['LEFT_ANKLE']
    except KeyError:
        try:
            # Wenn linke Seite fehlt, versuche RECHTE Seite
            shoulder = frame['RIGHT_SHOULDER']
            hip = frame['RIGHT_HIP']
            knee = frame['RIGHT_KNEE']
            ankle = frame['RIGHT_ANKLE']
        except KeyError:
            # Wenn beide Seiten fehlen -> Frame überspringen
            continue 

    # Nur Frames verwenden, in denen alle Gelenke gut sichtbar sind
    if all(p['visibility'] > MIN_VISIBILITY for p in [shoulder, hip, knee, ankle]):
        
        valid_frames += 1
        
        # 2. Winkel berechnen
        current_hip_angle = calculate_angle(shoulder, hip, knee)
        current_knee_angle = calculate_angle(hip, knee, ankle)
        
        if current_hip_angle is None or current_knee_angle is None:
            continue
            
        # 3. Tiefsten Punkt finden
        if current_hip_angle < min_hip_angle:
            min_hip_angle = current_hip_angle
        
        if current_knee_angle < min_knee_angle:
            min_knee_angle = current_knee_angle


# --- 4. Ergebnis ausgeben ---
print("\n--- ERGEBNIS DER ANALYSE (TIEFSTER PUNKT) ---")

if valid_frames == 0:
    print("FEHLER: Konnte keine gültige Haltung finden.")
    print(f"Mögliche Gründe: Person nicht seitlich zur Kamera oder 'MIN_VISIBILITY' (aktuell {MIN_VISIBILITY}) ist zu hoch eingestellt.")
elif min_hip_angle == 180:
    print("Konnte Haltung analysieren, aber keine Beugung erkannt.")
else:
    print(f"Tiefster Hüft-Winkel (Tiefe): {min_hip_angle:.2f} Grad")
    print(f"Kleinster Knie-Winkel:       {min_knee_angle:.2f} Grad")
    print(f"(Basierend auf {valid_frames} gültigen Frames)")

    # Hier ist unsere eigentliche "KI-Bewertung"
    if min_hip_angle < TIEFEN_SCHWELLE:
        print("\n✅ BEWERTUNG: Tiefe ist 'Gut' (unter {TIEFEN_SCHWELLE} Grad).")
    else:
        print(f"\n❌ BEWERTUNG: Tiefe ist 'Zu hoch' (über {TIEFEN_SCHWELLE} Grad).")