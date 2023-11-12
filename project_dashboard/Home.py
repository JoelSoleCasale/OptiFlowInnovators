import streamlit as st
from PIL import Image


st.set_page_config(
        layout="centered",
        initial_sidebar_state="collapsed")

st.markdown("# NTT DATA Datathon FME-UPC 💻")
st.sidebar.markdown("# Home 🎈")

logo = Image.open("LogoOptiFlow.jpeg")

st.subheader("Streamlining Inventory Management for Storable Healthcare Products in a Unified Demand Environment")
st.info("Welcome to our project! ⛩️", icon="⛩️")

st.write(" ")
st.write("---")
st.write(" ")

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


st.write(" ")
st.write("---")
st.write(" ")

st.image(logo, use_column_width=True)

st.write(" ")
st.write("---")
st.write(" ")

st.subheader("Abstract")
st.write(
    "In this project, we propose a mathematical model and implementation based on a collaborative scheme designed to optimize the storage and distribution of medical products to hospitals given historical data.")

with open("../Paper/NTTStorableSupplyPlanner/pdf/main.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Dowload paper",
                    data=PDFbyte,
                    file_name="Optiflow_NTTData_Datathon_2023.pdf",
                    mime='application/octet-stream')


st.write("---")

st.subheader("Github")
st.info('See our [Github](https://github.com/JoelSoleCasale/JNA-Datathon2023) repository', icon="ℹ️")

st.write(" ")
st.write(" ")
st.write("---")
st.write(" ")
st.write(" ")

st.markdown("# Unified demand model 🌐")
st.sidebar.markdown("# Unified demand model 🌐")


video_file = open('../Graphics/animation0.mp4', 'rb')
video_bytes = video_file.read()

st.write("Simulation of the flow of products between providers and a central storage center that supplies all hospitals in a region:")
st.video(video_bytes)
st.caption("Generated with pygame")

st.write(" ")
st.write("---")
st.write(" ")

st.markdown("# Model results 📋")
st.sidebar.markdown("# Model results 📋")

st.write('We have used different values for', r'$\beta$', ' and ', r'$P_{max}$',' to observe the effects of different environmental and robustness restrictions on the optimal cost of storage.')
heatmap = Image.open("heatmap.jpg")
st.image(heatmap, use_column_width=True)
st.caption("Heatmap of optimal costs in terms of "+r'$\beta$'+" and "+ r'$P_{max}$'+" for product 70130 (APÓSITO DE HIDROCOLOIDE-7)")
st.write(' As expected, the more robust and the fewer orders allowed (i.e. the less environmental impact) lead to increased optimal costs. We also observe that it is significantly harder to have a lesser environmental impact than to be more robust.')
