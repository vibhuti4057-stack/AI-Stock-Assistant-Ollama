import streamlit as st

st.set_page_config(page_title="AI Financial Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.6); }
    70% { box-shadow: 0 0 0 25px rgba(0, 212, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0); }
}
.avatar-circle {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00D4FF, #1E2761);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 60px;
    animation: pulse 2s infinite;
    margin-bottom: 20px;
}
</style>
<div class="avatar-circle">🤖</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#00D4FF;'>Beyond the Chart — AI Agent</h1>", unsafe_allow_html=True)
st.write("If you can see a pulsing glowing circle above this text, the avatar works correctly.")