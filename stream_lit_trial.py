import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings
import streamlit.components.v1 as component
from mtcnn.mtcnn import MTCNN
from base64 import b64decode, b64encode
import numpy as np
import tensorflow as tf
import cv2
import av
from typing import Union, NamedTuple, List
import threading
import time
import tensorflow.keras.backend as tfkb
from tensorflow.keras.models import load_model as ld_m
from tensorflow.keras.models import model_from_json as mfj
import queue
import PIL

# sets the pages configuration function
st.set_page_config(layout='wide', initial_sidebar_state='collapsed')


def main():
    # title
    st.title("""
            &#x1f332;
    Computer vision app to help improve Working from Home, by monitoring your emotions
    """)

    FER2013 = ("Predicts 7 different types of emotions")

    Happy_House = ("Predict 2 different types of emotions")

    about = ("About the project")

    app_mode = st.sidebar.selectbox(
        "Choose model",
        [
            FER2013,
            Happy_House,
            # Upload,
            about
        ],
    )
    # Information bar
    with st.expander("See how to build better WFH habits"):
        st.subheader(
            "To improve your well being this app will help you take regular breaks, all you have to do is keep the plant growing")
        st.subheader(app_mode)

    # Choose live stream or image prediction
    if app_mode == FER2013:
        live_stream()
    elif app_mode == Happy_House:
        upload_image()
    else:
        about_project()


c2, c3, c4 = st.columns((25, 25, 15))

# path to models
model_path = 'models/fer2013_ownnetwork00012lr_deeper_w_tensorboardv20210904-195946.pt'
model_fer = 'models/happy_house_training_inceptionv3.pt'


def about_project():
    st.text("Summer project 2021 Silvester Stephens")


# Loading realtime prediction model and cache
@st.cache(allow_output_mutation=True)
def load_model():
    model = ld_m(model_path)
    model.summary()  # going to take this out
    return model

# Loading image model and cache


@st.cache(allow_output_mutation=True)
def model_hh():
    model = ld_m(model_fer)
    model.summary()
    return model


# Real time function
def live_stream():
    # Container
    with c2:
        # Video transformer and prediction
        # prediction model has to be done within the class because it is threaded
        # Frames can't be passed outside the class
        class VideoTransformer(VideoProcessorBase):
            result_queue: "queue.Queue[List[Prediction]]"
            _predicted: Union[List[str], None]
            _predicted_lock: threading.Lock

            def __init__(self) -> None:
                self.i = 0
                self._predicted = None
                self._predicted_lock = threading.Lock()
                self.result_queue = queue.Queue()

            @property
            def predicted(self) -> Union[List[str], None]:
                with self._predicted_lock:
                    return self._predicted

            # prediction function
            def _predict_image(self, image):

                result = []
                imageresized = cv2.cvtColor(cv2.resize(
                    image, (64, 64)), cv2.COLOR_RGB2GRAY)
                imageresized = np.array(imageresized)
                imageresized = imageresized/255.0
                imagexdim = np.expand_dims(imageresized, axis=(0, -1))
                emotions_fer = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                                3: 'Happy', 4: 'Sad', 5: 'Suprise', 6: 'Neutral'}
                model = load_model()
                predicton = model.predict(imagexdim)
                print(np.argmax(predicton))
                emotion_pred = emotions_fer[np.argmax(predicton[0])]

                result.append(emotion_pred)
                return result

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                label = self._predict_image(img)

                with self._predicted_lock:
                    self._predicted = label
                return av.VideoFrame.from_ndarray(img, format='bgr24')

        webrtc_rtx = webrtc_streamer(
            key='loop', video_processor_factory=VideoTransformer, sendback_audio=False, mode=WebRtcMode.SENDRECV, sendback_video=VideoTransformer, async_processing=True)
    # Container
    with c4:
        # placeholders on screen which will be updated
        text_placeholder = st.empty()
        prediction_placeholder = st.empty()
    # Container
    with c3:
        if webrtc_rtx.state.playing:
            print('playing....')
            labels_placeholdes = st.empty()
            while True:
                if webrtc_rtx.video_processor:
                    x = webrtc_rtx.video_processor.predicted
                    print(x)
                    try:
                        # Changes the state of the GIF depending on prediction
                        with labels_placeholdes.container():
                            if x[0] == 'Happy' or x[0] == 'Neutral':

                                text_placeholder.text(
                                    'Thank you for looking after you')
                                prediction_placeholder.text(x[0])
                                labels_placeholdes.image(
                                    'gif/slowfullgrowth.gif')
                            elif x[0] == 'Disgust' or x[0] == 'Sad' or x[0] == 'Fear':

                                text_placeholder.text(
                                    'Hello please look after yourself and take a break')
                                prediction_placeholder.text(x[0])
                                labels_placeholdes.image(
                                    'gif/slowdeathfromhalfgrowth.gif')
                            elif x[0] == 'Angry':

                                text_placeholder.text(
                                    'Please keep me alive and take a moment away from the screen')
                                prediction_placeholder.text(x[0])
                                labels_placeholdes.image(
                                    'gif/slowdeathfromquatergrowth.gif')
                            elif x[0] == 'Suprise':

                                prediction_placeholder.text(x[0])
                                labels_placeholdes.image(
                                    'gif/andbreath.gif')
                            else:

                                labels_placeholdes.image(
                                    'gif/startstage-removebg-preview.png')
                                prediction_placeholder.text(x[0])
                    except TypeError:
                        print("None type")
                    x = ''
                    time.sleep(8)

        else:
            print('false')

# Image prediction function


def upload_image():
    image = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
    if image is not None:
        img = PIL.Image.open(image)
        img_array = np.asfarray(img)
        img_array = cv2.resize(img_array, (150, 150))
        img_array = np.array(img_array)
        img_array = img_array/255.0
        img_array = np.expand_dims(img_array, axis=0)
        model = model_hh()
        prediction = model.predict(img_array)
        print(prediction[0][0])
        emotions = {0: 'Not Smiling', 1: 'Smiling'}
        if prediction > 0.7:
            pred_emotion = emotions.__getitem__(1)
        else:
            pred_emotion = emotions.__getitem__(0)

        p = pred_emotion + "  " + str(round(prediction[0][0], 3) * 100) + "%"
        st.image(img, p, 350, 350)


if __name__ == "__main__":
    main()
