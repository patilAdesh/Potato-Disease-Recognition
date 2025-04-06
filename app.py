import streamlit as st
import numpy as np
import cv2
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import time
import platform
import requests
from indian_disease_names import get_indian_disease_name, indian_disease_names
import pandas as pd
import altair as alt


def translate_text(text, dest_language):
    
    try:
        url = 'https://translate.googleapis.com/translate_a/single'
        params = {
            'client': 'gtx',
            'sl': 'auto',
            'tl': dest_language,
            'dt': 't',
            'q': text
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            translated_text = ''
            for sentence in result[0]:
                if sentence[0]:
                    translated_text += sentence[0]
            return translated_text
        else:
            return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


st.set_page_config(
    page_title="Potato Disease Recognition",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"  
)


IMG_SIZE = (224, 224)
MODEL_PATH = "potato_disease_model_best.keras"
TFLITE_MODEL_PATH = "potato_disease_model.tflite"
CLASS_INDICES_PATH = "class_indices.json"


@st.cache_data(ttl=86400, persist="disk")  
def get_cached_translation(text, dest_language):
    return translate_text(text, dest_language)


@st.cache_data(ttl=86400, persist="disk")  
def get_text(language_code):
    base_texts = {
        'title': '🥔 Potato Disease Recognition',
        'subtitle': 'Detect diseases in potato plants using a trained deep learning model',
        'files_error': 'Required model files not found. Please make sure the following files are in the current directory:',
        'image_header': 'Image Upload Detection',
        'video_header': 'Video Upload Detection',
        'realtime_header': 'Real-time Detection',
        'choose_image': 'Choose an image...',
        'choose_video': 'Choose a video...',
        'original_image': 'Original Image',
        'prediction_result': 'Prediction Result',
        'predicted_class': 'Predicted Class',
        'confidence': 'Confidence',
        'class_probabilities': 'Class Probabilities',
        'analysis_complete': 'Analysis complete!',
        'analyzing_image': 'Analyzing image...',
        'early_blight_info': 'Early Blight is caused by the fungus Alternaria solani. Symptoms include dark brown spots with concentric rings that appear on lower leaves first.',
        'late_blight_info': 'Late Blight is caused by Phytophthora infestans. It\'s a serious disease that can destroy entire fields rapidly in wet weather.',
        'healthy_info': 'This plant appears healthy with no signs of disease.',
        'start_camera': 'Start Camera',
        'stop_camera': 'Stop Camera',
        'camera_started': 'Camera started. Point at a potato plant leaf.',
        'camera_stopped': 'Camera stopped.',
        'process_video': 'Process Video',
        'process_every_n': 'Process every N frames',
        'processing_video': 'Processing video...',
        'processing_complete': 'Processing complete.',
        'video_analysis': 'Video Analysis Summary',
        'most_common': 'Most common prediction:',
        'about_title': 'About',
        'about_info': 'This application uses deep learning to detect diseases in potato plants. Upload an image or video of a potato plant, or use your webcam for real-time detection.',
        'input_mode_title': 'Input Mode',
        'select_mode': 'Select detection mode:'
    }
    if language_code == 'en':
        return base_texts
    translated_texts = {}
    for key, text in base_texts.items():
        try:
            translated_text = get_cached_translation(text, language_code)
            translated_texts[key] = translated_text
        except Exception as e:
            translated_texts[key] = text
            print(f"Translation error for {key}: {e}")
    return translated_texts



st.sidebar.title("# Language options")
languages = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Punjabi': 'pa',
    'Oriya': 'or',
    'Assamese': 'as',
    'Urdu': 'ur'
}


selected_language = st.sidebar.selectbox(
    "Select Language",
    list(languages.keys()),
    index=0,  
    key="language_selector",
    disabled=False,  
    label_visibility="visible",
    on_change=None  
)


st.markdown("""
<style>
    div[data-testid="stSelectbox"] > div > div > div {
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)


language_code = languages[selected_language]
sidebar_text = get_text(language_code)


st.sidebar.title(sidebar_text['about_title'])
st.sidebar.info(sidebar_text['about_info'])


st.sidebar.title(sidebar_text.get('input_mode_title', 'Input Mode'))


input_modes = ["Image Upload", "Video Upload", "Real-time Detection"]
translated_modes = []

for mode in input_modes:
    try:
        if language_code != 'en':
            translated_text = get_cached_translation(mode, language_code)
            translated_modes.append(translated_text)
        else:
            translated_modes.append(mode)
    except:
        translated_modes.append(mode)

input_mode = st.sidebar.radio(
    sidebar_text.get('select_mode', 'Select detection mode:'),
    translated_modes
)


selected_mode_index = translated_modes.index(input_mode)
input_mode = input_modes[selected_mode_index]


@st.cache_resource
def load_class_indices():
    try:
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
        return class_names
    except Exception as e:
        st.error(f"Error loading class indices: {e}")
        return None


@st.cache_resource(show_spinner=False)  
def load_keras_model():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        model = load_model(MODEL_PATH)
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None


@st.cache_resource(show_spinner=False)  
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None


@st.cache_data(ttl=3600, max_entries=100)  
def preprocess_image(img):
    if img.shape[0] != IMG_SIZE[0] or img.shape[1] != IMG_SIZE[1]:
        img = cv2.resize(img, IMG_SIZE)
    if img.shape[2] == 3 and not np.array_equal(img[0,0,0], img[0,0,2]):  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@st.cache_data(ttl=3600, max_entries=100, show_spinner=False)  
def predict_keras(img, model, class_names):
    try:
        with tf.device('/CPU:0'):  
            predictions = model.predict(img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        class_name = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        return class_name, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error", 0.0, np.zeros(len(class_names))


def draw_prediction(img, class_name, confidence):
    img_copy = img.copy()
    cv2.rectangle(img_copy, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    text = f"{class_name}: {confidence:.2f}"
    cv2.putText(img_copy, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img_copy


def real_time_detection():
    global current_text
    st.subheader(current_text['realtime_header'])
    model = load_keras_model()
    class_names = load_class_indices()
    if model is None or class_names is None:
        st.error("Could not load the model or class indices. Please check the files.")
        return
    system_platform = platform.system()
    st.sidebar.subheader("Performance Settings")
    frame_skip = st.sidebar.slider("Process every N frames", min_value=1, max_value=5, value=2, help="Higher values improve performance but reduce smoothness")
    resolution_scale = st.sidebar.slider("Resolution Scale", min_value=0.5, max_value=1.0, value=0.75, step=0.05, help="Lower values improve performance")
    start_button = st.sidebar.button(current_text['start_camera'])
    if start_button:
        video_placeholder = st.empty()
        status_text = st.empty()
        stop_button = st.sidebar.button(current_text['stop_camera'])
        try:
            status_text.text("Opening camera...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open webcam. Please check your camera connection.")
                return
            width = int(640 * resolution_scale)
            height = int(480 * resolution_scale)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            status_text.text(current_text['camera_started'])
            frame_count = 0
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Could not read frame from webcam.")
                    break
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                try:
                    processed_img = preprocess_image(frame)
                    class_name, confidence, _ = predict_keras(processed_img, model, class_names)
                    if languages[selected_language] != 'en':
                        indian_name = get_indian_disease_name(class_name, languages[selected_language])
                        if indian_name != class_name:  
                            display_class = indian_name
                        else:
                            try:
                                display_class = get_cached_translation(class_name, languages[selected_language])
                            except:
                                display_class = class_name
                    else:
                        display_class = class_name
                    result_frame = draw_prediction(frame, display_class, confidence)
                    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    display_height = 400  
                    frame_aspect = result_frame.shape[1] / result_frame.shape[0]
                    display_width = int(display_height * frame_aspect)
                    display_frame = cv2.resize(result_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True, output_format='JPEG', quality=80)
                except Exception as e:
                    st.error(f"Error processing frame: {e}")
                    break
                time.sleep(max(0.01, 0.05 * frame_skip))
            cap.release()
            status_text.text(current_text['camera_stopped'])
        except Exception as e:
            st.error(f"Error with webcam: {e}")


def image_upload_detection():
    global current_text
    st.subheader(current_text['image_header'])
    model = load_keras_model()
    class_names = load_class_indices()
    if model is None or class_names is None:
        st.error("Could not load the model or class indices. Please check the files.")
        return
    uploaded_file = st.sidebar.file_uploader(current_text['choose_image'], type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            col1, col2 = st.columns([3, 2])
            with col1:
                display_height = 300  
                img_aspect = img.shape[1] / img.shape[0]
                display_width = int(display_height * img_aspect)
                if img.shape[0] > display_height or img.shape[1] > display_width:
                    display_img = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_AREA)
                else:
                    display_img = img
                display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                st.image(display_img_rgb, use_container_width=True)
            processed_img = preprocess_image(img)
            with st.spinner(current_text['analyzing_image']):
                class_name, confidence, predictions = predict_keras(processed_img, model, class_names)
            st.success(current_text['analysis_complete'])
            with col2:
                if languages[selected_language] != 'en':
                    indian_name = get_indian_disease_name(class_name, languages[selected_language])
                    if indian_name != class_name:  
                        pred_text = f"<h4>{current_text['predicted_class']}: {indian_name} ({confidence:.2%})</h4>"
                    else:
                        try:
                            translated_class = get_cached_translation(class_name, languages[selected_language])
                            pred_text = f"<h4>{current_text['predicted_class']}: {translated_class} ({confidence:.2%})</h4>"
                        except:
                            pred_text = f"<h4>{current_text['predicted_class']}: {class_name} ({confidence:.2%})</h4>"
                else:
                    pred_text = f"<h4>{current_text['predicted_class']}: {class_name} ({confidence:.2%})</h4>"
                st.markdown(pred_text, unsafe_allow_html=True)
                probs_df = {class_names[i]: float(predictions[i]) for i in range(len(predictions))}
                display_names = {}
                if languages[selected_language] != 'en':
                    for class_name_key in probs_df.keys():
                        try:
                            indian_name = get_indian_disease_name(class_name_key, languages[selected_language])
                            if indian_name != class_name_key:  
                                display_names[class_name_key] = indian_name
                            else:
                                display_names[class_name_key] = get_cached_translation(class_name_key, languages[selected_language])
                        except Exception as e:
                            display_names[class_name_key] = class_name_key
                else:
                    display_names = {k: k for k in probs_df.keys()}
                chart_data = pd.DataFrame({
                    'Disease': [display_names[k] for k in probs_df.keys()],
                    'Probability': list(probs_df.values()),
                    'Original': list(probs_df.keys())  
                })
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Disease:N', title='Disease'),
                    y=alt.Y('Probability:Q', title='Probability', scale=alt.Scale(domain=[0, 1.0])),
                    tooltip=['Disease', 'Probability', 'Original']  
                ).properties(
                    height=300
                )
                st.altair_chart(chart, use_container_width=True)
            st.markdown("<hr style='margin: 0.5em 0; padding: 0'>", unsafe_allow_html=True)
            col3, col4 = st.columns([1, 1])
            if class_name == "Early Blight":
                if languages[selected_language] != 'en':
                    disease_name = get_indian_disease_name(class_name, languages[selected_language])
                    if disease_name == class_name:  
                        try:
                            disease_name = get_cached_translation(class_name, languages[selected_language])
                        except:
                            disease_name = class_name
                else:
                    disease_name = class_name
                st.info(current_text['early_blight_info'].replace('Early Blight', disease_name))
            elif class_name == "Late Blight":
                if languages[selected_language] != 'en':
                    disease_name = get_indian_disease_name(class_name, languages[selected_language])
                    if disease_name == class_name:  
                        try:
                            disease_name = get_cached_translation(class_name, languages[selected_language])
                        except:
                            disease_name = class_name
                else:
                    disease_name = class_name
                st.warning(current_text['late_blight_info'].replace('Late Blight', disease_name))
            elif class_name == "Healthy":
                if languages[selected_language] != 'en':
                    disease_name = get_indian_disease_name(class_name, languages[selected_language])
                    if disease_name == class_name:  
                        try:
                            disease_name = get_cached_translation(class_name, languages[selected_language])
                        except:
                            disease_name = class_name
                else:
                    disease_name = class_name
                st.success(current_text['healthy_info'].replace('Healthy', disease_name))
        except Exception as e:
            st.error(f"Error processing image: {e}")


def video_upload_detection():
    global current_text
    st.subheader(current_text['video_header'])
    model = load_keras_model()
    class_names = load_class_indices()
    if model is None or class_names is None:
        st.error("Could not load the model or class indices. Please check the files.")
        return
    uploaded_file = st.sidebar.file_uploader(current_text['choose_video'], type=["mp4", "avi", "mov"], label_visibility="collapsed")
    st.sidebar.subheader("Performance Settings")
    frame_skip = st.sidebar.slider(current_text['process_every_n'], min_value=1, max_value=20, value=10, help="Higher values improve performance")
    resolution_scale = st.sidebar.slider("Resolution Scale", min_value=0.5, max_value=1.0, value=0.75, step=0.05, help="Lower values improve performance")
    process_button = st.sidebar.button(current_text['process_video'])
    if uploaded_file is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            st.write(f"FPS: {fps}")
            st.write(f"Total frames: {frame_count}")
            st.write(f"Duration: {duration:.2f} seconds")
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            if process_button:
                stop_button = st.sidebar.button("Stop Processing")
                frame_idx = 0
                status_text.text(current_text['processing_video'])
                class_counts = {class_name: 0 for class_name in class_names.values()}
                resize_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resolution_scale)
                resize_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resolution_scale)
                batch_size = 5  
                frames_to_process = []
                frame_indices = []
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        status_text.text("End of video.")
                        break
                    progress = int(frame_idx / frame_count * 100)
                    progress_bar.progress(progress)
                    if frame_idx % frame_skip == 0:
                        if resolution_scale < 1.0:
                            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
                        frames_to_process.append(frame)
                        frame_indices.append(frame_idx)
                        if len(frames_to_process) >= batch_size or frame_idx + frame_skip >= frame_count:
                            try:
                                for i, (batch_frame, batch_idx) in enumerate(zip(frames_to_process, frame_indices)):
                                    processed_img = preprocess_image(batch_frame)
                                    class_name, confidence, _ = predict_keras(processed_img, model, class_names)
                                    class_counts[class_name] += 1
                                    if i == len(frames_to_process) - 1:
                                        @st.cache_data(ttl=86400)
                                        def get_translated_class(class_name, lang_code):
                                            if lang_code == 'en':
                                                return class_name
                                            try:
                                                return get_cached_translation(class_name, lang_code)
                                            except:
                                                return class_name
                                                
                                        if languages[selected_language] != 'en':
                                            display_class = get_translated_class(class_name, languages[selected_language])
                                        else:
                                            display_class = class_name
                                        result_frame = draw_prediction(batch_frame, display_class, confidence)
                                        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                                        display_height = 400  
                                        frame_aspect = result_frame.shape[1] / result_frame.shape[0]
                                        display_width = int(display_height * frame_aspect)
                                        display_frame = cv2.resize(result_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                                        video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error processing frame {frame_idx}: {e}")
                            frames_to_process = []
                            frame_indices = []
                    frame_idx += 1
                if frame_idx > 0:
                    st.subheader(current_text['video_analysis'])
                    total_processed = sum(class_counts.values())
                    class_percentages = {class_name: count/total_processed for class_name, count in class_counts.items() if total_processed > 0}
                    st.bar_chart(class_percentages)
                    most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
                    @st.cache_data(ttl=86400)
                    def get_translated_class(class_name, lang_code):
                        if lang_code == 'en':
                            return class_name
                        try:
                            return get_cached_translation(class_name, lang_code)
                        except:
                            return class_name
                            
                    translated_class = get_translated_class(most_common_class, languages[selected_language])
                    st.success(f"{current_text['most_common']} {translated_class}")
                cap.release()
                os.unlink(tfile.name)
                status_text.text(current_text['processing_complete'])
        except Exception as e:
            st.error(f"Error processing video: {e}")


current_text = {}


def main():
    global current_text
    language_code = languages[selected_language]
    with st.spinner("Loading language..."):
        current_text = get_text(language_code)
    
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 0;'>{current_text['title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; margin-top: 0;'>{current_text['subtitle']}</p>", unsafe_allow_html=True)
    
    
    files_exist = all(os.path.exists(f) for f in [MODEL_PATH, CLASS_INDICES_PATH])
    
    if not files_exist:
        st.error(
            f"{current_text['files_error']}\n"
            f"- {MODEL_PATH}\n"
            f"- {CLASS_INDICES_PATH}"
        )
        return
    
    
    if input_mode == "Image Upload":
        image_upload_detection()
    elif input_mode == "Video Upload":
        video_upload_detection()
    else:  
        real_time_detection()

if __name__ == "__main__":
    main() 