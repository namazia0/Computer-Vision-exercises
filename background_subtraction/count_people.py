import numpy as np
import cv2
import background_subtraction

'''
This code implements a Mixture of Gaussians (MOG) background subtraction algorithm and tracks detected people across frames and counts the total number of people in the video.
'''

def object_tracking(dict, distance_threshold):
    """
    Track and count detected people across frames
    """
    # Initialize with detections from frame 3
    centers_of_people = dict["3"]

    # Track centers across remaining frames
    for i in range(4, len(dict.keys())+1):
        for k in range(len(dict[str(i)])):
            new_person = True
            potential_new_center = dict[str(i)][k]
            
            # Find nearest existing center
            for j, center in enumerate(centers_of_people):
                distance = np.linalg.norm(np.array(potential_new_center) - np.array(center))
                if distance < distance_threshold:
                    current_nearest_center = j
                    new_person = False
            
            # Update or add center
            if not new_person:
                centers_of_people[current_nearest_center] = potential_new_center
            else:
                centers_of_people.append(potential_new_center)
    print("number of people: ", len(centers_of_people))


if __name__ == '__main__':
    distance_threshold = 20
    dict = {}
    img = cv2.imread('imgs/0001.jpg')
    mog = background_subtraction.MOG(img.shape[0], img.shape[1], 3, 0.3, 0.3)

    # Process frames and detect people
    for i in range(1, 15+1):
        dict[str(i)] = []
        img = cv2.imread('imgs/{:04d}.jpg'.format(i))

        # Background subtraction
        label_img = mog.updateParam(img, np.ones(img.shape[:2]))
        label_img = (label_img > 0).astype(np.uint8) * 255
        cv2.imwrite('output/label{:04d}.jpg'.format(i), label_img)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        label_img = cv2.morphologyEx(label_img, cv2.MORPH_OPEN, kernel, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        label_img = cv2.morphologyEx(label_img, cv2.MORPH_CLOSE, kernel, kernel, iterations=1)
        cv2.imwrite('output/post_processing{:04d}.jpg'.format(i), label_img)

        # Extract person centers from contours
        contours, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_vis = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)

        vis = img.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if area > 30 and h > 15 and h / max(w, 1) > 1.5:
                center = (x + w // 2, y + h // 2)
                dict[str(i)].append(center)

                cv2.rectangle(
                    vis,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        cv2.imwrite(f"output/bbox_{i:04d}.jpg", vis)

    # count people across all frames
    object_tracking(dict, distance_threshold)