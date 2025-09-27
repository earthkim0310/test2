import streamlit as st
import sys

st.set_page_config(page_title="Sanity Check", layout="centered")

st.title("✅ Streamlit Cloud Sanity Check")
st.write("If you can see this, your app wiring (repo/main-file/requirements) is correct.")

# Show Python and package versions (if installed)
st.subheader("Environment")
st.write("Python:", sys.version)

try:
    import streamlit as _st
    st.write("streamlit:", _st.__version__)
except Exception as e:
    st.write("streamlit: (not found)", e)

try:
    import numpy as _np
    st.write("numpy:", _np.__version__)
except Exception as e:
    st.write("numpy: (not found)", e)

try:
    import plotly as _pl
    st.write("plotly:", _pl.__version__)
    import plotly.express as px
    st.subheader("Plotly quick test")
    fig = px.line(x=[0,1,2,3], y=[0,1,0,1], title="It Works!")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.write("plotly: (not found)", e)

st.caption("If this page loads, but your main app doesn't, the issue is inside your app code.")
