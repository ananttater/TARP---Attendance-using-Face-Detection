import streamlit as st
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import time
import cv2
import pickle
import face_recognition
from Preparing_local import prepare_test_img, test

t0= time.time()
path = "db"


def main():
    def load_model():
        with open ('encoded_faces.pickle', 'rb') as f_in:
            encoded_trains = pickle.load(f_in)
        return encoded_trains
    
    encoded_trains = load_model()

    # Start of the project
    st.title("Attendance Project")
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Training", "Attend Live"])


    if app_mode == "Training":
        st.subheader('Training Steps:')
        st.markdown("1. Get a photo of every student with **only one face** in the picture.")
        st.markdown('2. Put all the photos in the **db** folder')
        st.markdown("3. Press **Train The Model** Button")

        if st.button("Train The Model"):
            import Training
            encoded_trains, images = Training.training(path)
            st.write(images)
            st.write(len(encoded_trains))
            output_file = 'encoded_faces.pickle'

            with open(output_file, 'wb') as f_out:
                pickle.dump(encoded_trains, f_out)
            

    elif app_mode == "Attend Live":
        st.title("Webcam Live Feed")
        attendance_file = st.file_uploader("Choose attendance file",type =['csv'])

        if attendance_file is not None:
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)

            while run:
                _, test_img = camera.read()
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
                test_img_small = cv2.resize(test_img,(0,0),None,0.5,0.5)

                face_test_locations = face_recognition.face_locations(test_img_small, model = "hog")
                encoded_tests = face_recognition.face_encodings(test_img_small)
                df = test(encoded_tests, face_test_locations, test_img, encoded_trains, attendance_file)
                #st.image(test_img)
                FRAME_WINDOW.image(test_img)
                #st.write(df)
            else:
                st.write('Stopped')



def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)



if __name__=='__main__':
    main()
    