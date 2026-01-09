import cv2
import numpy as np
from deepface import DeepFace

# --- CONFIGURATION ---
VIDEO_PATH = 'video-phase4.mp4' 
OUTPUT_REPORT = 'resumo_atividades.txt'
ANOMALY_THRESHOLD = 15000     
EMOTION_INTERVAL = 5          

def generate_report(total_frames, anomaly_count, emotion_log, activity_log):
    if emotion_log:
        dominant_emotion = max(set(emotion_log), key=emotion_log.count)
    else:
        dominant_emotion = "Nenhuma detectada"

    report_content = (
        "--- RELATÓRIO TECH CHALLENGE - FASE 4 ---\n\n"
        f"1. Total de frames analisados: {total_frames}\n"
        f"2. Número de anomalias detectadas (Movimentos Bruscos): {anomaly_count}\n"
        f"3. Emoção predominante no vídeo: {dominant_emotion}\n"
        f"4. Resumo de Atividades:\n"
        f"   - O vídeo apresentou {len(activity_log)} momentos distintos de atividade.\n"
    )

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\nRelatório salvo em: {OUTPUT_REPORT}")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    total_frames = 0
    anomaly_count = 0
    prev_frame_gray = None
    current_emotion = "Analisando..."
    emotion_history = []
    activity_log = []

    print("Iniciando... (Aguarde o download dos pesos do DeepFace na primeira execução)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        target_width = 640
        height, width = frame.shape[:2]
        scaling_factor = target_width / float(width)
        if scaling_factor < 1:
            new_height = int(height * scaling_factor)
            frame = cv2.resize(frame, (target_width, new_height))

        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        activity_status = "Estatico" 
        color = (0, 255, 0)

        if prev_frame_gray is not None:
            diff = cv2.absdiff(prev_frame_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_magnitude = np.sum(thresh) // 255
            
            if motion_magnitude > ANOMALY_THRESHOLD:
                activity_status = "ANOMALIA"
                anomaly_count += 1
                color = (0, 0, 255)
            elif motion_magnitude > 1000:
                activity_status = "Movimento"
                color = (0, 255, 255)
                activity_log.append("Normal")
            
            cv2.putText(frame, f"Atividade: {activity_status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        prev_frame_gray = gray.copy()

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if total_frames % EMOTION_INTERVAL == 0:
                try:
     
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size == 0:
                        continue

                    result = DeepFace.analyze(img_path = face_roi, 
                                              actions=['emotion'], 
                                              enforce_detection=False, 
                                              detector_backend='opencv', 
                                              align=False)
                    
                    if isinstance(result, list):
                        current_emotion = result[0]['dominant_emotion']
                    else:
                        current_emotion = result['dominant_emotion']
                        
                    emotion_history.append(current_emotion)
                    
                except Exception as e:
                    print(f"!!! Erro DeepFace: {e}")
                    current_emotion = "Erro"

            cv2.putText(frame, f"Emocao: {current_emotion}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Tech Challenge', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    generate_report(total_frames, anomaly_count, emotion_history, activity_log)

if __name__ == "__main__":
    main()