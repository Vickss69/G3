import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time

st.info("📝 Note: This comparison is based only on the image, not real-life appearance.")
st.warning("⚠️ DISCLAIMER: For image comparison only. Misuse is not allowed.")

# Cache the MediaPipe Face Mesh to avoid reloading
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

@st.cache_resource
def load_haar_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

# Set page title and description
st.title("Know who's more Beautiful")
st.write("Upload two face images to see which scores higher.")

# Preload the detectors
face_mesh = load_face_mesh()
face_cascade, eye_cascade = load_haar_cascades()

def calculate_face_shape(landmarks, image_shape):
    h, w = image_shape[:2]
    # Jawline: approximate using MediaPipe landmarks (e.g., chin to jaw sides)
    jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67]
    jaw = [landmarks[i] for i in jaw_indices]

    # Forehead width: distance between temples (approximated)
    forehead_width = np.linalg.norm(np.array(landmarks[103]) - np.array(landmarks[332]))
    # Face height: chin to forehead top
    face_height = np.linalg.norm(np.array(landmarks[152]) - np.array(landmarks[10]))
    # Cheek width: distance between cheek landmarks
    cheek_width = np.linalg.norm(np.array(landmarks[234]) - np.array(landmarks[454]))

    # Estimate forehead height using landmarks above eyebrows
    forehead_height = (landmarks[151][1] + landmarks[9][1]) / 2 - landmarks[1][1]
    total_face_height = face_height + forehead_height

    # Ratios for classification
    aspect_ratio = total_face_height / forehead_width
    cheek_to_jaw_ratio = cheek_width / forehead_width

    # Classification based on geometric ratios
    if aspect_ratio < 1.3 and cheek_to_jaw_ratio < 0.9:
        return 8
    elif aspect_ratio >= 1.3 and cheek_to_jaw_ratio < 0.9:
        return 18
    elif cheek_to_jaw_ratio >= 1.0:
        return 11
    elif cheek_to_jaw_ratio < 0.8 and aspect_ratio > 1.5:
        return 15
    elif landmarks[234][0] - landmarks[454][0] > cheek_width / 2 and landmarks[152][1] > landmarks[234][1]:
        return 26
    elif cheek_width < forehead_width and aspect_ratio > 1.4:
        return 0
    else:
        return 22

def detect_face_shape(image_path, face_mesh=face_mesh):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                face_shape = calculate_face_shape(landmarks, image.shape)
                return face_shape * 100 / 26
        return 0
    except Exception as e:
        st.error(f"Error in face shape detection: {e}")
        return 0

def get_average_skin_color(image_path, face_mesh=face_mesh):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                # Approximate face bounding box using landmarks
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                roi = image[y_min:y_max, x_min:x_max]
                step = max(1, min(roi.shape[0], roi.shape[1]) // 20)
                skin_pixels = [roi[i, j] for i in range(0, roi.shape[0], step) for j in range(0, roi.shape[1], step)]
                avg_bgr = np.mean(skin_pixels, axis=0)
                S = np.sqrt(sum(avg_bgr ** 2))
                return S
        return 0
    except Exception as e:
        st.error(f"Error in skin color detection: {e}")
        return 0

def rate_jawline(image_path, face_mesh=face_mesh):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = img.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                # Jawline indices from chin to sides
                jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67]
                jawline = np.array([landmarks[i] for i in jaw_indices])

                left_jawline = jawline[:len(jawline)//2]
                right_jawline = jawline[len(jawline)//2:]
                center_point = jawline[len(jawline)//2]
                left_distances = [np.linalg.norm(pt - center_point) for pt in left_jawline]
                right_distances = [np.linalg.norm(pt - center_point) for pt in right_jawline[::-1]]
                symmetry_score = 100 - np.mean(np.abs(np.array(left_distances) - np.array(right_distances)))
                symmetry_score = max(0, min(100, symmetry_score))

                def calculate_angle(a, b, c):
                    ba = a - b
                    bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    cosine_angle = min(1.0, max(-1.0, cosine_angle))
                    return np.degrees(np.arccos(cosine_angle))

                angles = [calculate_angle(jawline[i-1], jawline[i], jawline[i+1]) for i in range(1, len(jawline)-1)]
                sharpness_score = 100 - np.mean(np.abs(np.array(angles) - 120))
                sharpness_score = max(0, min(100, sharpness_score))

                jawline_rating = 0.6 * symmetry_score + 0.4 * sharpness_score
                return jawline_rating
        return 0
    except Exception as e:
        st.error(f"Error in jawline detection: {e}")
        return 0

def calculate_eye_shape(eye_landmarks):
    width = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    height = (np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5])) +
              np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))) / 2
    aspect_ratio = height / width
    if aspect_ratio < 0.25:
        return 38
    elif 0.25 <= aspect_ratio <= 0.35:
        return 28
    elif aspect_ratio > 0.35:
        return 22
    else:
        return 12

def detect_eyes_shape(image_path, face_mesh=face_mesh):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        height, width = img.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = img.shape[:2]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                # Left eye landmarks (approximate 36-41 from Dlib)
                left_eye_indices = [33, 246, 161, 160, 159, 158]
                right_eye_indices = [362, 398, 384, 385, 386, 387]
                left_eye = [landmarks[i] for i in left_eye_indices]
                right_eye = [landmarks[i] for i in right_eye_indices]
                left_eye_shape = calculate_eye_shape(left_eye)
                right_eye_shape = calculate_eye_shape(right_eye)
                return ((left_eye_shape + right_eye_shape) * 100 / 72)
        return 0
    except Exception as e:
        st.error(f"Error in eye shape detection: {e}")
        return 0

def classify_eye_color(rgb_values):
    r, g, b = rgb_values
    if r > 100 and g < 70 and b < 40:
        return 5
    elif r > 140 and g > 100 and b < 60:
        return 19
    elif r < 100 and g < 100 and b > 120:
        return 29
    elif r < 100 and g > 120 and b < 100:
        return 14
    elif r > 100 and g > 80 and b < 60:
        return 9
    elif r < 100 and g < 100 and b < 80:
        return 24
    else:
        return 15

def get_eye_rgb_value(eye_image):
    return np.mean(eye_image, axis=(0, 1))

def detect_eye_colors(image_path, face_cascade=face_cascade, eye_cascade=eye_cascade):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return 0
        (x, y, w, h) = faces[0]
        face_region = image[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
        if len(eyes) < 2:
            return 0
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        left_eye = face_region[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
        right_eye = face_region[eyes[1][1]:eyes[1][1]+eyes[1][3], eyes[1][0]:eyes[1][0]+eyes[1][2]]
        left_eye_rgb = get_eye_rgb_value(left_eye)
        right_eye_rgb = get_eye_rgb_value(right_eye)
        left_eye_color = classify_eye_color(left_eye_rgb)
        right_eye_color = classify_eye_color(right_eye_rgb)
        S = (left_eye_color + right_eye_color) / 2
        return S * 100 / 30
    except Exception as e:
        st.error(f"Error in eye color detection: {e}")
        return 0

def calculate_hair_color_score(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0, 0, 0
        height, width = image.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hair_region = image_rgb[:int(image.shape[0] * 0.4), :]
        step = max(1, min(hair_region.shape[0], hair_region.shape[1]) // 30)
        samples = [hair_region[i, j] for i in range(0, hair_region.shape[0], step) for j in range(0, hair_region.shape[1], step)]
        avg_rgb = np.mean(samples, axis=0)
        return avg_rgb[0], avg_rgb[1], avg_rgb[2]
    except Exception as e:
        st.error(f"Error in hair color calculation: {e}")
        return 0, 0, 0

def calculate_hair_density_and_baldness(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0, 0
        max_dimension = 500
        height, width = image.shape
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        edges = cv2.Canny(image, 100, 200)
        mask = np.zeros_like(edges)
        mask[:int(image.shape[0] * 0.4), :] = 255
        total_pixels = np.sum(mask > 0)
        if total_pixels == 0:
            return 0, 0
        hair_pixels = np.sum(cv2.bitwise_and(edges, mask) > 0)
        S1 = (hair_pixels / total_pixels) * 100
        scalp_pixels = np.sum(cv2.bitwise_and(edges, mask) == 0)
        S2 = (scalp_pixels / total_pixels) * 100
        return S1, S2
    except Exception as e:
        st.error(f"Error in hair density calculation: {e}")
        return 0, 0

def calculate_final_score(image_path):
    try:
        a, b, c = calculate_hair_color_score(image_path)
        S1, S2 = calculate_hair_density_and_baldness(image_path)
        if a == 0 and b == 0 and c == 0:
            return 0
        color_score = ((256 * 1.74 - (a**2 + b**2 + c**2)**0.5) * 100 / (256 * 1.74))
        S = (color_score + S1 + S2) / 3
        return S
    except Exception as e:
        st.error(f"Error in final hair score calculation: {e}")
        return 0

def mark_winner(image_path, is_winner=True):
    try:
        image = Image.open(image_path)
        if is_winner:
            draw = ImageDraw.Draw(image)
            text = "Hott ONE"
            try:
                font = ImageFont.truetype("arial.ttf", 50)
            except IOError:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            image_width, image_height = image.size
            position_x = (image_width - text_width) // 2
            position_y = image_height - text_height - 20
            rectangle_position = (position_x - 10, position_y - 5, position_x + text_width + 10, position_y + text_height + 5)
            draw.rectangle(rectangle_position, fill="white")
            draw.text((position_x, position_y), text, fill="black", font=font)
        return image
    except Exception as e:
        st.error(f"Error marking winner: {e}")
        return Image.open(image_path)

def analyze_image(image_path, progress_callback=None):
    try:
        metrics = {}
        if progress_callback:
            progress_callback(0, "Detecting face shape...")
        metrics['face_shape'] = detect_face_shape(image_path)
        if metrics['face_shape'] == 0:
            metrics.update({'skin_color': 0, 'skin_score': 0, 'jawline': 0, 'eye_shape': 0, 'eye_color': 0, 'hair_score': 0, 'final_score': 0})
            if progress_callback:
                progress_callback(100, "No face detected!")
            return metrics

        if progress_callback:
            progress_callback(20, "Analyzing skin color...")
        metrics['skin_color'] = get_average_skin_color(image_path)
        metrics['skin_score'] = (1.5 * 100 * metrics['skin_color'] / (256 * (3 ** 0.5)))

        if progress_callback:
            progress_callback(35, "Analyzing jawline...")
        metrics['jawline'] = rate_jawline(image_path)

        if progress_callback:
            progress_callback(50, "Analyzing eye shape...")
        metrics['eye_shape'] = detect_eyes_shape(image_path)

        if progress_callback:
            progress_callback(65, "Analyzing eye color...")
        metrics['eye_color'] = detect_eye_colors(image_path)

        if progress_callback:
            progress_callback(80, "Analyzing hair...")
        metrics['hair_score'] = calculate_final_score(image_path)

        metrics['final_score'] = (
            metrics['face_shape'] * 25 + metrics['skin_score'] * 40 + metrics['jawline'] * 15 +
            metrics['eye_shape'] * 10 + metrics['eye_color'] * 10 + metrics['hair_score'] * 20
        )
        if progress_callback:
            progress_callback(100, "Analysis complete!")
        return metrics
    except Exception as e:
        st.error(f"Error in image analysis: {e}")
        return {k: 0 for k in ['face_shape', 'skin_color', 'skin_score', 'jawline', 'eye_shape', 'eye_color', 'hair_score', 'final_score']}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Face 1")
    uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="file1")
with col2:
    st.subheader("Face 2")
    uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="file2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    st.write("Processing images... This may take a moment.")
    temp_file1 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file1.write(uploaded_file1.getvalue())
    temp_file1.close()
    temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file2.write(uploaded_file2.getvalue())
    temp_file2.close()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        def update_progress(percent, message=""):
            progress_bar.progress(percent / 100)
            if message:
                status_text.text(message)

        status_text.text("Analyzing first image...")
        metrics1 = analyze_image(temp_file1.name, lambda percent, msg: update_progress(percent / 2, f"Image 1: {msg}"))
        status_text.text("Analyzing second image...")
        metrics2 = analyze_image(temp_file2.name, lambda percent, msg: update_progress(50 + percent / 2, f"Image 2: {msg}"))
        status_text.text("Comparing results...")

        s1, s2 = metrics1['final_score'], metrics2['final_score']
        winner_image = temp_file1.name if s1 >= s2 else temp_file2.name
        winner_pil = mark_winner(winner_image, True)

        status_text.empty()
        st.subheader("Results")
        col3, col4 = st.columns(2)
        with col3:
            st.image(uploaded_file1, caption="Image 1")
            st.metric("Score", f"{s1:.2f}")
            with st.expander("Detailed Scores for Image 1"):
                for key, value in metrics1.items():
                    if key != 'skin_color' and key != 'final_score':
                        st.write(f"{key.replace('_', ' ').title()}: {value:.2f}")
        with col4:
            st.image(uploaded_file2, caption="Image 2")
            st.metric("Score", f"{s2:.2f}")
            with st.expander("Detailed Scores for Image 2"):
                for key, value in metrics2.items():
                    if key != 'skin_color' and key != 'final_score':
                        st.write(f"{key.replace('_', ' ').title()}: {value:.2f}")
        st.subheader("Hott One ⚡")
        st.image(winner_pil, caption=f"Face {'1' if s1 >= s2 else '2'} Wins!")
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        try:
            os.unlink(temp_file1.name)
            os.unlink(temp_file2.name)
        except:
            pass
else:
    st.info("Please upload both images to compare.")
