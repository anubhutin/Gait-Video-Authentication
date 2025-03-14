import streamlit as st
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from gait_preprocessing import extract_frames, extract_silhouettes, create_gei

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


# Streamlit App
def main():
    st.title("Gait Authentication System")
    st.write("Upload a video to check if the person is enrolled.")

    # Upload video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Check if the person is enrolled
        enrolled_geis_folder = "gei"  # Folder containing enrolled GEIs
        result = is_person_enrolled(temp_video_path, enrolled_geis_folder, match_threshold=0.75, reject_threshold=0.5)
        if result:
            st.success("Person is enrolled!")
        else:
            st.error("Person is not enrolled.")

        # Clean up temporary files
        os.remove(temp_video_path)
        if os.path.exists("temp_frames"):
            for file in os.listdir("temp_frames"):
                os.remove(os.path.join("temp_frames", file))
            os.rmdir("temp_frames")
        if os.path.exists("temp_silhouettes"):
            for file in os.listdir("temp_silhouettes"):
                os.remove(os.path.join("temp_silhouettes", file))
            os.rmdir("temp_silhouettes")


if __name__ == "__main__":
    main()
