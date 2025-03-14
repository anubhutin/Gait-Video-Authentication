import cv2
import os
import numpy as np

# Step 1: Extract frames from a video
def extract_frames(video_path, output_folder, frame_rate=30):
    """
    Extract frames from a video and save them as images.
    
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Folder to save extracted frames.
        frame_rate (int): Desired frame rate for extraction.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Skip frames to match the desired frame rate
        skip_frames = int(cap.get(cv2.CAP_PROP_FPS)) // frame_rate - 1
        for _ in range(skip_frames):
            cap.read()

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")


# Step 2: Extract silhouettes using background subtraction
def extract_silhouettes(input_folder, output_folder):
    """
    Extract silhouettes from frames using background subtraction.
    
    Args:
        input_folder (str): Folder containing input frames.
        output_folder (str): Folder to save silhouettes.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    for frame_name in os.listdir(input_folder):
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)

        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Save silhouette
        silhouette_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(silhouette_path, fgmask)

    print(f"Extracted silhouettes to {output_folder}")


# Step 3: Create Gait Energy Image (GEI)
def create_gei(silhouette_folder):
    """
    Create a Gait Energy Image (GEI) from silhouettes.
    
    Args:
        silhouette_folder (str): Folder containing silhouette images.
    
    Returns:
        gei (numpy.ndarray): Gait Energy Image.
    """
    silhouette_images = []
    for frame_name in os.listdir(silhouette_folder):
        frame_path = os.path.join(silhouette_folder, frame_name)
        silhouette = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if silhouette is not None:
            silhouette_images.append(silhouette)
        else:
            print(f"Warning: Failed to load silhouette {frame_name}")

    if not silhouette_images:
        print("Error: No valid silhouettes found.")
        return None

    # Average all silhouettes to create GEI
    gei = np.mean(silhouette_images, axis=0)
    print(f"Created GEI with {len(silhouette_images)} silhouettes.")
    return gei


# Main function to process videos and create GEIs
def main():
    # Root folder containing videos
    videos_root = "videos"

    # Process each person
    for person in os.listdir(videos_root):
        person_folder = os.path.join(videos_root, person)

        # Process each view (front, side, rear)
        for view in os.listdir(person_folder):
            view_folder = os.path.join(person_folder, view)

            # Process each clip in the view folder
            for clip in os.listdir(view_folder):
                clip_path = os.path.join(view_folder, clip)

                # Step 1: Extract frames
                frame_folder = os.path.join("frames", person, view, os.path.splitext(clip)[0])
                extract_frames(clip_path, frame_folder)

                # Step 2: Extract silhouettes
                silhouette_folder = os.path.join("silhouettes", person, view, os.path.splitext(clip)[0])
                extract_silhouettes(frame_folder, silhouette_folder)

                # Step 3: Create GEI
                gei = create_gei(silhouette_folder)
                gei_path = os.path.join("gei", person, view, f"{os.path.splitext(clip)[0]}_gei.jpg")
                os.makedirs(os.path.dirname(gei_path), exist_ok=True)
                cv2.imwrite(gei_path, gei)
                print(f"Saved GEI for {person}/{view}/{clip} to {gei_path}")


if __name__ == "__main__":
    main()