mport streamlit as st
import PIL
import cv2
import numpy as np
import openvino as ov
import io

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

# ---- Preprocess ----
def preprocess(image, input_layer):
    N, C, H, W = input_layer.shape
    resized_image = cv2.resize(image, (W, H))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image

# ---- Face Detection ----
def find_faceboxes(image, results, threshold):
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= threshold]
    scores = scores[scores >= threshold]

    h, w, _ = image.shape
    face_boxes = face_boxes * np.array([w, h, w, h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes, scores

# ---- Drawing & Inference ----
def draw_age_gender_emotion(face_boxes, image, input_layer_ag):
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    show_image = image.copy()

    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]

        # Emotion
        input_image = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image])[output_layer_emo].squeeze()
        emotion_index = np.argmax(results_emo)

        # Age & Gender
        input_image_ag = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age = int(np.squeeze(results_ag[1]) * 100)
        gender_data = np.squeeze(results_ag[0])
        gender = 'female' if gender_data[0] >= 0.65 else 'male' if gender_data[1] >= 0.55 else 'unknown'
        box_color = (200, 200, 0) if gender == 'female' else (0, 200, 200) if gender == 'male' else (200, 200, 200)

        # Display
        text = f"{gender} {age} {EMOTION_NAMES[emotion_index]}"
        font_scale = max(image.shape[1] / 900, 0.5)
        cv2.putText(show_image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 0), 1)
        cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), box_color, 1)

    return show_image

# ---- Prediction ----
def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, _ = find_faceboxes(image, results, conf_threshold)
    return draw_age_gender_emotion(face_boxes, image, input_layer_ag)

# ---- Webcam Preview ----
def play_live_camera():
    image_data = st.camera_input("Take a picture")

    if image_data is not None:
        image = PIL.Image.open(image_data)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(predict_image(image_cv, conf_threshold), channels="BGR")


# ---- Streamlit UI ----
st.set_page_config(page_title="Age/Gender/Emotion", page_icon=":nerd_face:", layout="centered")
st.title("Age/Gender/Emotion Project :nerd_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Confidence Threshold (%)", 10, 100, 20)) / 100

# ---- Image Mode ----
if source_radio == "IMAGE":
    input_img = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if input_img:
        image = PIL.Image.open(input_img)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(predict_image(image_cv, conf_threshold), channels="BGR")
    else:
        st.image("assets/sample_image.jpg")
        st.write("Upload an image from the sidebar.")

# ---- Video Mode ----
elif source_radio == "VIDEO":
    input_vid = st.sidebar.file_uploader("Upload a video", type=["mp4"])
    if input_vid:
        with open("upload.mp4", "wb") as f:
            f.write(input_vid.read())
        cap = cv2.VideoCapture("upload.mp4")
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output = predict_image(frame, conf_threshold)
            st_frame.image(output, channels="BGR")
        cap.release()
    else:
        st.video("assets/sample_video.mp4")

# ---- Webcam Mode ----
elif source_radio == "WEBCAM":
    play_live_camera()






