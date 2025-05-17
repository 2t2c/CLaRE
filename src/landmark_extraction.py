"""
This script extracts facial features from an image using the face_recognition library.
"""
from PIL import Image, ImageDraw
import face_recognition
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch

def extract_rois(image, merge_landmarks=False):
    """
    Extracts rois from the image based on facial landmarks.
    """
    # convert torch tensor to numpy image
    if isinstance(image, str):
        image = face_recognition.load_image_file(image)
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().detach().numpy()  # (3, H, W)
        image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)
    print("Found {} face(s) in this photograph.".format(len(face_landmarks_list)))
    # saving rois
    rois = []

    def crop_patch(points, pad=5):
        # Get bounding box around the feature
        xs, ys = zip(*points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        min_x = max(min_x - pad, 0)
        min_y = max(min_y - pad, 0)
        max_x = min(max_x + pad, image.shape[1])
        max_y = min(max_y + pad, image.shape[0])

        patch = [min_x, min_y, max_x, max_y]

        return patch

    # loop over each face
    for face_idx, face_landmarks in enumerate(face_landmarks_list):
        if merge_landmarks:
            if "left_eye" in face_landmarks and "right_eye" in face_landmarks:
                # Combine both eyes' points
                points = face_landmarks["left_eye"] + face_landmarks["right_eye"]
                patch = crop_patch(points)
                rois.append((patch, "eyes"))
            if "left_eyebrow" in face_landmarks and "right_eyebrow" in face_landmarks:
                # Combine both eyebrows points
                points = face_landmarks["left_eyebrow"] + face_landmarks["right_eyebrow"]
                patch = crop_patch(points)
                rois.append((patch, "eyebrows"))
            if "top_lip" in face_landmarks and "bottom_lip" in face_landmarks:
                # Combine both lips points
                points = face_landmarks["top_lip"] + face_landmarks["bottom_lip"]
                patch = crop_patch(points)
                rois.append((patch, "lips"))
            if "nose_bridge" in face_landmarks and "nose_tip" in face_landmarks:
                # Combine nose bridge and nose tip points
                points = face_landmarks["nose_bridge"] + face_landmarks["nose_tip"]
                patch = crop_patch(points)
                rois.append((patch, "nose"))
            if "chin" in face_landmarks:
                points = face_landmarks["chin"]
                patch = crop_patch(points)
                rois.append((patch, "chin"))
    else:
        for feature_name, points in face_landmarks.items():
            patch = crop_patch(points)
            rois.append((patch, feature_name))

    return rois