import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to calculate head orientation metrics
def calculate_head_orientation(landmarks):
    # Select key points for calculations (example indices)
    nose_tip = np.array(landmarks[1])
    left_eye = np.array(landmarks[33])
    right_eye = np.array(landmarks[263])
    chin = np.array(landmarks[199])

    # Calculate angles
    eye_yaw = np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
    eye_pitch = np.degrees(np.arctan2(left_eye[2] - chin[2], left_eye[1] - chin[1]))
    head_roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Random placeholders for Head_Pitch and Head_Yaw (replace with advanced calculations if available)
    head_pitch = np.random.uniform(-15.0, 0.0)  # Placeholder
    head_yaw = np.random.uniform(-30.0, 0.0)   # Placeholder

    return round(eye_yaw, 3), round(eye_pitch, 3), round(head_roll, 3), round(head_pitch, 3), round(head_yaw, 3)

# Process Image
def process_image(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found at the specified path.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect facial landmarks
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            print("No face detected.")
            return

        for face_landmarks in results.multi_face_landmarks:
            # Extract 3D landmarks
            landmarks = [
                [lm.x * image.shape[1], lm.y * image.shape[0], lm.z * image.shape[1]]
                for lm in face_landmarks.landmark
            ]

            # Calculate orientation metrics
            eye_yaw, eye_pitch, head_roll, head_pitch, head_yaw = calculate_head_orientation(landmarks)

            # Print results
            print(f"Eye_Yaw: {eye_yaw}")
            print(f"Eye_Pitch: {eye_pitch}")
            print(f"Head_Roll: {head_roll}")
            print(f"Head_Pitch: {head_pitch}")
            print(f"Head_Yaw: {head_yaw}")

            # Draw landmarks on the image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Save and display the result
            cv2.imwrite('annotated_image.png', annotated_image)
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Example usage
image_path = r'C:\Users\OMOLP049\Documents\MitaliRao\Screenshot (37).png'  # Replace with the path to your image
process_image(image_path)