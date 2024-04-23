import cv2
import numpy as np
import matplotlib.pyplot as plt

def morph_operation(matinput):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    morph = cv2.erode(matinput, kernel, iterations=1)
    morph = cv2.dilate(morph, kernel, iterations=2)
    morph = cv2.erode(morph, kernel, iterations=1)
    morph = cv2.dilate(morph, kernel, iterations=1)
    
    return morph

def analyze_blob(matblobs, display_frame):
    _, blobs = cv2.findContours(matblobs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = []

    print("Number of Blobs:", len(blobs))
    
    for blob in blobs:
        if len(blob) < 5:  # Ensure there are enough points for minAreaRect
            continue
        # Ensure the blob is correctly typed and shaped for minAreaRect
        blob = blob.astype(np.float32).reshape(-1, 1, 2)
        rot_rect = cv2.minAreaRect(blob)
        (cx, cy), (sw, sh), angle = rot_rect
        
        # Check area constraints
        if sw * sh < 75000 or sw * sh > 200000:
            continue
        # Check aspect ratio constraints
        rect_ratio = max(sw, sh) / min(sw, sh)
        if rect_ratio < 1 or rect_ratio > 3.5:
            continue
        # Check fill ratio
        on_count = cv2.contourArea(blob)
        total_count = sw * sh
        if total_count <= 0 or on_count / total_count < 0.4:
            continue
        # Check brightness constraint
        if display_frame[int(cy), int(cx), 0] > 75:
            continue

        # Drawing valid blob
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        cv2.drawContours(display_frame, [box], 0, (0, 0, 255), 2)
        valid_blobs.append(blob)

    print("Number of Valid Blobs:", len(valid_blobs))
    return valid_blobs

def main_process():
    img = cv2.imread('testV2/test11.jpg')  # Make sure the path is correct
    if img is None:
        print("Image not found")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the same edge detection and thresholding techniques
    edged = cv2.Canny(gray, 44, 194)
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 3, 2)    
    # Apply morphological operations to close the gaps
    thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.erode(thresh, None, iterations=5)

    # Now use the morphed image for blob analysis
    display_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    valid_blobs = analyze_blob(thresh, display_color)   
    for b in valid_blobs:
        cv2.drawContours(display_color, [b], -1, (0, 255, 255), 2)
    cv2.imshow("Display Color", display_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_process()
