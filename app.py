import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle 
from PIL import Image

 
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="Image Classification Web App",page_icon="",layout="centered",initial_sidebar_state="expanded")
st.title('Image Classifier using Machine Learning')
st.subheader('by Amlan Mohanty ')
st.text('Upload the Image')

model = pickle.load(open('img_model.p','rb'))

st.markdown("""
<style>
body {
    color: #000099;
    background-color: #669999;
    etc. 
}
</style>
    """, unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image')

  if st.button('PREDICT'):
    CATEGORIES = ['apple fruit','lemon fruit']
    st.write('Result...')
    flat_data=[]
    img = np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.title(f' PREDICTED OUTPUT: {y_out}')
    q = model.predict_proba(flat_data)
    for index, item in enumerate(CATEGORIES):
      st.write(f'{item} : {q[0][index]*100}%')


st.sidebar.subheader("About App")

st.sidebar.info("This web app is made as part of Image Classification Project")
st.sidebar.info("Browse the image you want to upload or simply drag and drop the image")
st.sidebar.info("Click on the 'Predict' button to check whether the uploaded image is of 'Apple' or 'Lemon' ")
st.sidebar.info("Don't forget to rate this app")



feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=10,step=1)

if feedback:
  st.header("Thank you for rating the app!")