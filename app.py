import streamlit as st
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from gait_preprocessing import extract_frames, extract_silhouettes, create_gei


# Custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;500&display=swap');
    
    html {
        scroll-behavior: smooth;
    }
    
    .header {
        font-size: 2.5rem !important;
        color: #2E86C1 !important;
        text-align: center;
        padding: 1rem;
        font-family: 'Roboto', sans-serif;
        border-bottom: 3px solid #2E86C1;
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .processing-animation {
        text-align: center;
        padding: 2rem;
        background: #F8F9FA;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .result-success {
        color: #145A32;
        background: #D5F5E3;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: fadeIn 0.5s;
    }
    
    .result-failure {
        color: #922B21;
        background: #FADBD8;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # App Header
    st.markdown('<div class="header">👟 Gait Recognition Authentication</div>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("System Overview")
        st.markdown("""
        🔍 This system analyzes human gait patterns through:
        - Video processing
        - Silhouette extraction
        - Gait Energy Image (GEI) creation
        - Pattern matching
        """)
        st.markdown("---")
        st.header("Settings")
        match_threshold = st.slider("Matching Threshold", 0.5, 1.0, 0.75)
        st.markdown("---")
        st.info("ℹ️ Upload a 5-10 second walking video for best results")
    
    # Main content area
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # File uploader section
            with st.expander("📤 UPLOAD VIDEO", expanded=True):
                uploaded_file = st.file_uploader(
                    "Select a video file", 
                    type=["mp4", "avi", "mov"],
                    label_visibility="collapsed"
                )
                
                if uploaded_file:
                    st.video(uploaded_file)
                    uploaded_file.seek(0)

        # Processing and results section
        if uploaded_file:
            with st.spinner(""):
                # Create processing container
                with st.empty():
                    processing_container = st.container()
                    
                    with processing_container:
                        # Save temporary files
                        temp_video_path = "temp_video.mp4"
                        with open(temp_video_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Processing animation
                        with st.container():
                            st.markdown('<div class="processing-animation">'
                                       '🔍 ANALYZING GAIT PATTERNS<br>'
                                       '<div class="loader"></div>'
                                       '</div>', unsafe_allow_html=True)
                        
                        # Process frames
                        temp_frames_folder = "temp_frames"
                        temp_silhouettes_folder = "temp_silhouettes"
                        
                        # Step 1: Extract frames
                        extract_frames(temp_video_path, temp_frames_folder)
                        
                        # Step 2: Extract silhouettes
                        extract_silhouettes(temp_frames_folder, temp_silhouettes_folder)
                        
                        # Step 3: Create GEI
                        uploaded_gei = create_gei(temp_silhouettes_folder)
                        
                        # Display processing results
                        processing_container.empty()
                        
                        # Show intermediate results
                        with st.container():
                            st.markdown("### Processing Stages")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Sample Frame**")
                                frame_files = os.listdir(temp_frames_folder)
                                if frame_files:
                                    sample_frame = cv2.imread(os.path.join(temp_frames_folder, frame_files[0]))
                                    st.image(sample_frame, use_column_width=True)
                            
                            # with col2:
                            #     st.markdown("**Silhouette**")
                            #     silhouette_files = os.listdir(temp_silhouettes_folder)
                            #     if silhouette_files:
                            #         sample_silhouette = cv2.imread(os.path.join(temp_silhouettes_folder, silhouette_files[0]))
                            #         st.image(sample_silhouette, use_column_width=True)
                            
                            with col3:
                                st.markdown("**Gait Energy Image**")
                                if uploaded_gei is not None:
                                    gei_normalized = cv2.normalize(uploaded_gei, None, 0, 255, cv2.NORM_MINMAX)
                                    gei_uint8 = gei_normalized.astype('uint8')
                                    st.image(gei_uint8, use_container_width=True)
                        
                        # Perform comparison
                        result = is_person_enrolled(temp_video_path, "gei", match_threshold)
                        
                        # Show final result
                        st.markdown("## Verification Result")
                        if result:
                            st.markdown(
                                '<div class="result-success">'
                                '✅ VERIFIED<br>'
                                '<span style="font-size: 1.2rem;">Identity confirmed with gait pattern match</span>'
                                '</div>', 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="result-failure">'
                                '❌ ACCESS DENIED<br>'
                                '<span style="font-size: 1.2rem;">No matching gait pattern found</span>'
                                '</div>', 
                                unsafe_allow_html=True
                            )
                        
                        # Cleanup
                        os.remove(temp_video_path)
                        for folder in [temp_frames_folder, temp_silhouettes_folder]:
                            if os.path.exists(folder):
                                for file in os.listdir(folder):
                                    os.remove(os.path.join(folder, file))
                                os.rmdir(folder)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<footer>'
        'Gait Authentication System • Powered by Streamlit • '
        '<a href="#top" style="color: #2E86C1; text-decoration: none;">↑ Back to Top</a>'
        '</footer>',
        unsafe_allow_html=True
    )
# Function to compare GEIs using SSIM
def compare_geis(gei1, gei2, threshold=0.75):  # Adjusted threshold
    gei1_resized = cv2.resize(gei1, (128, 128))
    gei2_resized = cv2.resize(gei2, (128, 128))

    # Compute SSIM (returns similarity score between -1 and 1)
    similarity = ssim(gei1_resized, gei2_resized, data_range=255)
    print(f"SSIM Similarity: {similarity}")

    return similarity >= threshold


# Function to check if a person is enrolled
def is_person_enrolled(uploaded_video, enrolled_geis_folder, match_threshold=0.75, reject_threshold=0.5):
    # Step 1: Preprocess the uploaded video
    temp_frames_folder = "temp_frames"
    temp_silhouettes_folder = "temp_silhouettes"
    os.makedirs(temp_frames_folder, exist_ok=True)
    os.makedirs(temp_silhouettes_folder, exist_ok=True)

    # Extract frames
    extract_frames(uploaded_video, temp_frames_folder)

    # Extract silhouettes
    extract_silhouettes(temp_frames_folder, temp_silhouettes_folder)

    # Create GEI for the uploaded video
    uploaded_gei = create_gei(temp_silhouettes_folder)

    if uploaded_gei is None:
        print("Error: Failed to create GEI for the uploaded video.")
        return False

    # Step 2: Compare with enrolled GEIs
    max_similarity = -1
    for person_folder in os.listdir(enrolled_geis_folder):
        person_path = os.path.join(enrolled_geis_folder, person_folder)
        if not os.path.isdir(person_path):
            continue  # Skip if it's not a folder

        for view_folder in os.listdir(person_path):
            view_path = os.path.join(person_path, view_folder)
            if not os.path.isdir(view_path):
                continue  # Skip if it's not a folder

            for gei_file in os.listdir(view_path):
                enrolled_gei_path = os.path.join(view_path, gei_file)
                enrolled_gei = cv2.imread(enrolled_gei_path, cv2.IMREAD_GRAYSCALE)

                if enrolled_gei is None:
                    print(f"Error: Failed to load enrolled GEI from {enrolled_gei_path}.")
                    continue

                similarity = compare_geis(uploaded_gei, enrolled_gei, match_threshold)
                if similarity > max_similarity:
                    max_similarity = similarity

    # Step 3: Decide based on similarity
    if max_similarity >= match_threshold:
        return True  # Person is enrolled
    elif max_similarity < reject_threshold:
        return False  # Person is not enrolled
    else:
        return False  # Ambiguous case (e.g., similarity between reject_threshold and match_threshold)


if __name__ == "__main__":
    main()