import streamlit as st
import cv2
import numpy as np
import PIL
import openvino as ov
from io import BytesIO

# ---- Load Models ----
core = ov.Core()

model_face = core.read_model(model='models/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model=model_face, device_name="CPU")
input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

model_emo = core.read_model(model='models/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")
input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

model_age = core.read_model(model='models/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model=model_age, device_name="CPU")
input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output

# ---- Helper Functions ----
def preprocess(image, input_layer):
    N, C, H, W = input_layer.shape
    resized_image = cv2.resize(image, (W, H))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image

def find_faceboxes(image, results, threshold):
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= threshold]
    image_h, image_w, _ = image.shape
    face_boxes = face_boxes * np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes

def draw_age_gender_emotion(face_boxes, image, threshold, box_thickness=4, box_color=(255, 0, 0)):
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    show_image = image.copy()

    for box in face_boxes:
        xmin, ymin, xmax, ymax = box
        face = image[ymin:ymax, xmin:xmax]

        emo_input = preprocess(face, input_layer_emo)
        emo_output = compiled_model_emo([emo_input])[output_layer_emo].squeeze()
        emo_index = np.argmax(emo_output)

        ag_input = preprocess(face, input_layer_ag)
        ag_result = compiled_model_ag([ag_input])
        age = int(np.squeeze(ag_result[1]) * 100)
        gender_scores = np.squeeze(ag_result[0])
        gender = 'female' if gender_scores[0] > 0.65 else 'male' if gender_scores[1] > 0.55 else 'unknown'

        label = f"{gender} {age} {EMOTION_NAMES[emo_index]}"
        font_scale = max(image.shape[1] / 900, 0.6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
        text_w, text_h = text_size

        cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
        cv2.rectangle(show_image, (xmin, ymin - text_h - 10), (xmin + text_w + 10, ymin), box_color, -1)
        cv2.putText(show_image, label, (xmin + 5, ymin - 5), font, font_scale, (255, 255, 255), 2)

    return show_image

def predict_image(image, conf_threshold, box_thickness=4, box_color=(255, 0, 0)):
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes = find_faceboxes(image, results, conf_threshold)
    return draw_age_gender_emotion(face_boxes, image, conf_threshold, box_thickness, box_color)

def convert_image_to_bytes(image_cv):
    is_success, buffer = cv2.imencode(".jpg", image_cv)
    if is_success:
        return BytesIO(buffer.tobytes())
    return None

# ---- Streamlit UI ----
st.set_page_config(page_title="Age/Gender/Emotion", page_icon="🤓", layout="centered")
st.title("Age/Gender/Emotion Project 🤓")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Confidence Threshold (%)", 10, 100, 20)) / 100

# ---- Webcam ----
def play_live_camera():
    image_data = st.camera_input("Take a photo using your webcam")
    if image_data is not None:
        image = PIL.Image.open(image_data)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = predict_image(image_cv, conf_threshold, box_thickness=2, box_color=(255, 0, 0))
        st.image(result, channels="BGR")

        img_bytes = convert_image_to_bytes(result)
        if img_bytes:
            st.download_button("📥 Download Webcam Capture", data=img_bytes, file_name="webcam_image.jpg", mime="image/jpeg")

# ---- Image Upload ----
if source_radio == "IMAGE":
    uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded:
        image = PIL.Image.open(uploaded)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = predict_image(image_cv, conf_threshold, box_thickness=4, box_color=(255, 0, 0))
        st.image(result, channels="BGR")

        img_bytes = convert_image_to_bytes(result)
        if img_bytes:
            st.download_button("📥 Download Result Image", data=img_bytes, file_name="processed_image.jpg", mime="image/jpeg")
    else:
        st.image("assets/sample_image.jpg")
        st.info("Upload an image to try it out.")

# ---- Video Upload ----
elif source_radio == "VIDEO":
    uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4"])
    if uploaded:
        with open("upload.mp4", "wb") as f:
            f.write(uploaded.read())
        cap = cv2.VideoCapture("upload.mp4")
        st_frame = st.empty()
        last_result = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = predict_image(frame, conf_threshold, box_thickness=4, box_color=(255, 0, 0))
            last_result = result
            st_frame.image(result, channels="BGR")

        cap.release()

        if last_result is not None:
            img_bytes = convert_image_to_bytes(last_result)
            if img_bytes:
                st.download_button("📥 Download Last Video Frame", data=img_bytes, file_name="video_frame.jpg", mime="image/jpeg")
    else:
        st.video("assets/sample_video.mp4")

# ---- Webcam ----
elif source_radio == "WEBCAM":
    play_live_camera()
