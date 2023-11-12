import streamlit as st
from PIL import Image

st.set_page_config(
        layout="centered",
        initial_sidebar_state="collapsed")

st.markdown("# NTT DATA Datathon FME-UPC 💻")
st.sidebar.markdown("# Home 🎈")


image = Image.open("LogoOptiFlow.jpeg")

st.subheader("Welcome to our project!")

st.image(image, use_column_width=True)

st.write("---")

st.subheader("About Us")
team_members = [
    {
        "name": "Alex Serrano",
        "degree": "Mathematics and Data Science and Engineering",
        "linkedin": "https://www.linkedin.com/in/alexste/",
    },
    {
        "name": "Jan Tarrats",
        "degree": "Mathematics and Computer Engineering",
        "linkedin": "https://www.linkedin.com/in/jan-tarrats-castillo/",
    },
    {
        "name": "Joel Solé",
        "degree": "Mathematics and Data Science and Engineering",
        "linkedin": "https://www.linkedin.com/in/joel-solé-casale-4a917b1b7/",
    },
    {
        "name": "Nathaniel Mitrani",
        "degree": "Mathematics and Data Science and Engineering",
        "linkedin": "https://www.linkedin.com/in/nathaniel-mitrani-hadida-031b4021a/",
    }
]

# Set up columns for horizontal layout
col1, col2 = st.columns(2)
cols = [col1, col2, col1, col2]

for member, col in list(zip(team_members, cols)):
    with col:
        c = st.container()
        c.write(f"### {member['name']}")
        c.markdown(f"<u>Degrees</u>: {member['degree']}", unsafe_allow_html=True)
        c.write(f"Linkedin: [{member['name']}]({member['linkedin']})")

st.write("---")

st.subheader("Abstract")
st.write(
    "In this project, we propose a novel manner to optimize the robustness of a supply chain through stochastic modeling of its disruptions."
)

#st.info('Download ', icon="ℹ️")

with open("../Paper/documentation.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Dowload paper",
                    data=PDFbyte,
                    file_name="Optiflow_NTTData_Datathon_2023.pdf",
                    mime='application/octet-stream')


st.write("---")

st.subheader("Github")
st.info('See out [Github](https://github.com/JoelSoleCasale/JNA-Datathon2023) repository', icon="ℹ️")
