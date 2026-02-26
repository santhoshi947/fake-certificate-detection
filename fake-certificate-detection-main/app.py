import streamlit as st
from predict import predict_certificate

st.set_page_config(page_title="Fake Certificate Detection")

st.title("🎓 Fake Certificate Detection System")

uploaded_file = st.file_uploader("Upload Certificate Image", type=["jpg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("temp.jpg", caption="Uploaded Certificate", use_column_width=True)

    result = predict_certificate("temp.jpg")

    if result == "Fake":
        st.error("❌ This Certificate is Fake")
    else:
        st.success("✅ This Certificate is Genuine")