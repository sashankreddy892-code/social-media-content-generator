import asyncio
import streamlit as st

from socialmedia import generate_content


st.set_page_config(
    page_title="Social Media Content Generator",
    page_icon="📱",
    layout="wide",
)

st.title("📱 Social Media Content Generator")

st.write(
    "Generate social media content from a YouTube video transcript "
    "using Gemini and the OpenAI Agents SDK."
)

st.header("🎥 YouTube Video")

video_id = st.text_input(
    "YouTube Video ID",
    placeholder="Example: dQw4w9WgXcQ",
)

st.caption(
    "For https://www.youtube.com/watch?v=dQw4w9WgXcQ, "
    "the video ID is dQw4w9WgXcQ."
)

query = st.text_area(
    "Your Query",
    placeholder=(
        "Generate a professional LinkedIn post and "
        "an engaging Instagram caption."
    ),
    height=120,
)

st.header("📢 Select Platforms")

col1, col2, col3 = st.columns(3)

with col1:
    linkedin = st.checkbox("LinkedIn", value=True)

with col2:
    instagram = st.checkbox("Instagram", value=True)

with col3:
    twitter = st.checkbox("Twitter/X", value=False)


if st.button(
    "🚀 Generate Content",
    type="primary",
    use_container_width=True,
):

    if not video_id.strip():
        st.error("Please enter a YouTube Video ID.")
        st.stop()

    platforms = []

    if linkedin:
        platforms.append("LinkedIn")

    if instagram:
        platforms.append("Instagram")

    if twitter:
        platforms.append("Twitter/X")

    if not platforms:
        st.error("Please select at least one platform.")
        st.stop()

    with st.spinner(
        "Fetching transcript and generating content..."
    ):

        try:
            result = asyncio.run(
                generate_content(
                    video_id=video_id.strip(),
                    query=query,
                    platforms=platforms,
                )
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("Content generated successfully!")

    st.header("✨ Generated Content")

    output = result.final_output

    st.markdown(output)

    st.download_button(
        label="⬇️ Download Content",
        data=output,
        file_name="social_media_content.txt",
        mime="text/plain",
    )

st.markdown("---")

st.caption(
    "Powered by Gemini + OpenAI Agents SDK + YouTube Transcript API"
)