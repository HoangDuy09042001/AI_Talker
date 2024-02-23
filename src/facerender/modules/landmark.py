import numpy as np
from tqdm import tqdm 
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
from skimage import img_as_ubyte
def facemesh_process(image_bgr):
    face_mesh = None
    
    # Kiểm tra xem face_mesh đã được khởi tạo chưa
    if face_mesh is not None:
        results = face_mesh.process(image_bgr)
    else:
        # Nếu chưa được khởi tạo, hãy khởi tạo nó
        face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        results = face_mesh.process(image_bgr)
    if face_mesh is not None:
        face_mesh.close()
    
    return results
def get_min_max_coordinates(landmarks):
    minX = float("inf")
    maxX = float("-inf")
    minY = float("inf")
    maxY = float("-inf")

    for lm in landmarks:
        # Update minimum and maximum X values
        minX = min(minX, lm.x)
        maxX = max(maxX, lm.x)

        # Update minimum and maximum Y values
        minY = min(minY, lm.y)
        maxY = max(maxY, lm.y)

    return minX, maxX, minY, maxY


def get_landmark(image, idx):
    image1 = img_as_ubyte(image)
    print(f"landmark image {idx}: ", image1)

    # Resize the image using OpenCV
    img_size = 256  # Assuming this is defined somewhere
    original_size = (256, 256)  # Assuming this is defined somewhere
    image1 = cv2.resize(image1, (img_size, int(img_size * original_size[1] / original_size[0])))

    # Convert to PIL Image
    image1 = Image.fromarray(image1)

    # Save the image as a PNG file
    image1.save("output_" + str(idx) + ".png")
    image_np = np.array(image1)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = image_bgr.shape
    array = []
    results = facemesh_process(image_bgr)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            minX, maxX, minY, maxY = get_min_max_coordinates(face_landmarks.landmark)
            if(maxX<1 and maxY<1):
                # width = (maxX - minX) * img_w
                # height = (maxY - minY) * img_h
                for sub_idx, lm in enumerate(face_landmarks.landmark):
                    # if sub_idx in [
                    #     234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,94,19,1,4,5,195,197,6,168,
                    # ]:
                        array.append((lm.x))
                        array.append((lm.y))
                print("landmark len array: ",len(array))
    # image_rgb = cv2.imread(f"output_{idx}.png", cv2.IMREAD_COLOR)
    # Draw landmarks on the image
    for i in range(0, len(array), 2):
        x = int(array[i]*img_w)
        y = int(array[i+1]*img_h)
        cv2.circle(image_bgr, (x, y), 2, (0, 255, 0), -1)  # Draw a green circle at each landmark point

    # Save the resulting image
    cv2.imwrite(f'landmark_face_{idx}.png', image_bgr)
    return array, image_np