from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import threading
import time
import os
import numpy as np
from scipy.spatial import procrustes

# Flask app setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*",async_mode='threading') 
# 為何我用async_mode用其他 就 沒辦法socketemit數據到前端 只有用"threading"才能
# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 計算動作好壞
def angle_between(p1, p2, p3):
    v1 = np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z])
    v2 = np.array([p3.x, p3.y, p3.z]) - np.array([p2.x, p2.y, p2.z])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def evaluate_throw_pose(landmarks):
    try:
        shoulder = landmarks[12]  # 右肩膀
        elbow = landmarks[14]     # 右手肘
        wrist = landmarks[16]     # 右手腕
        angle = angle_between(shoulder, elbow, wrist)
        if 70 <= angle <= 110:
            return {"elbow_angle": angle, "status": "✅ 良好"}
        else:
            return {"elbow_angle": angle, "status": "⚠️ 應注意：胱部角度異常"}
    except Exception as e:
        return {"error": str(e)}

def extract_pose_from_image(image_path):
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError(f"No landmarks detected in {image_path}")
        return results.pose_landmarks.landmark

reference_pose_start = extract_pose_from_image("templates/pose_start.JPG")
reference_pose_end = extract_pose_from_image("templates/pose_end.JPG")

def compute_similarity(pose1, pose2, alpha=1.0):
    pose1 = np.array([[lm.x, lm.y, lm.z] for lm in pose1])
    pose2 = np.array([[lm.x, lm.y, lm.z] for lm in pose2])
    _, _, disparity = procrustes(pose1, pose2)
    score = np.exp(-alpha * disparity)
    return score

def Standard_action_comparison(landmarks):
    try:
        sim_start = compute_similarity(landmarks, reference_pose_start)
        sim_end = compute_similarity(landmarks, reference_pose_end)
        return {
            "similarity_to_start": round(sim_start, 3),
            "similarity_to_end": round(sim_end, 3)
        }
    except Exception as e:
        return {"error": str(e)}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_path = None
processing_thread = None
processing = False
selected_joint_idx = 0

joint_names = {i: name for i, name in enumerate([
    "鼻子", "左眼內角", "左眼", "左眼外角", "右眼內角",
    "右眼", "右眼外角", "左耳", "右耳", "嘴巴左側", "嘴巴右側",
    "左肩膀", "右肩膀", "左手肘", "右手肘",
    "左手腕", "右手腕", "左手小指", "右手小指",
    "左手抽指", "右手抽指", "左手掌心", "右手掌心",
    "左魄", "右魄", "左膝蓋", "右膝蓋",
    "左腳跑", "右腳跑", "左腳後跑", "右腳後跑",
    "左腳抽指", "右腳抽指"])}

@app.route('/')
def index():
    return render_template('index.html', joint_names=joint_names)

@app.route('/upload', methods=['POST'])
def upload():
    global video_path, processing, processing_thread
    f = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, 'uploaded.mp4')
    f.save(video_path)
    if processing_thread is None or not processing_thread.is_alive():
        processing = True
        processing_thread = threading.Thread(target=process_video)
        processing_thread.start()
    return {"success": True}

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('select_joint')
def handle_select_joint(data):
    global selected_joint_idx
    selected_joint_idx = int(data['joint'])
    print(f"Selected joint changed to: {selected_joint_idx}")

def gen_frames():
    global video_path, processing
    cap = cv2.VideoCapture(video_path) if video_path else None

    with mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while processing and cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            ret2, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

def process_video():
    global video_path, processing, selected_joint_idx
    cap = cv2.VideoCapture(video_path) if video_path else None
    if not cap or not cap.isOpened():
        return
    with mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                j = landmarks[selected_joint_idx]
                data = {'x': j.x, 'y': j.y, 'z': j.z}
                eval_result = evaluate_throw_pose(landmarks)
                comparison_result = Standard_action_comparison(landmarks)
                print(data)
            else:
                data = {'x': None, 'y': None, 'z': None}
                eval_result = {"error": '沒有偵測到關節'}
                comparison_result = {"error": '沒有偵測到關節'}
            socketio.emit('joint_data', data)
            socketio.emit('pose_feedback', eval_result)
            socketio.emit('standard_action_comparison', comparison_result)
        cap.release()
        socketio.emit('video_finished')

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)