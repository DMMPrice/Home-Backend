import cv2
import dlib
import mediapipe as mp
import numpy as np
import os
import bz2
import urllib.request
from skimage import feature
from scipy import ndimage
import math
from skimage.feature import graycomatrix, graycoprops

def describe_lbp(image, num_points, radius, eps=1e-7):
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist, lbp

def extract_all_features(image_path):
    """
    Extracts 201 facial features from an image and returns them as a single vector.
    """
    try:
        # --- Load Image & Models ---
        if not os.path.exists(image_path): return []
        image = cv2.imread(image_path)
        if image is None: return []
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        if not os.path.exists(predictor_path):
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            urllib.request.urlretrieve(url, predictor_path + ".bz2")
            with bz2.BZ2File(predictor_path + ".bz2", 'rb') as f_in:
                with open(predictor_path, 'wb') as f_out: f_out.write(f_in.read())
            os.remove(predictor_path + ".bz2")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')
        left_ear_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_leftear.xml')
        right_ear_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_rightear.xml')

        features = {}

        # --- Run Detections ---
        results_mp = face_mesh.process(img_rgb)
        results_dlib = detector(img_gray)
        faces_haar = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

        # --- Feature Extraction ---

        # Feature 1 & 2: Delaunay Triangulation Area and Centroid Length
        total_area, total_length = 0.0, 0.0
        if results_mp.multi_face_landmarks:
            face_landmarks_mp = results_mp.multi_face_landmarks[0]
            mp_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks_mp.landmark]
            rect = (0, 0, w, h)
            subdiv = cv2.Subdiv2D(rect)
            if mp_landmarks:
                for p in mp_landmarks:
                    if 0 <= p[0] < w and 0 <= p[1] < h:
                        subdiv.insert(p)
                triangles = subdiv.getTriangleList()
                if triangles is not None:
                    landmarks_np_mp = np.array(mp_landmarks, dtype=np.int32).reshape((-1, 1, 2))
                    centroids = []
                    for t in triangles:
                        pt1_tuple, pt2_tuple, pt3_tuple = (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))
                        if cv2.pointPolygonTest(landmarks_np_mp, pt1_tuple, False) >= 0 and \
                           cv2.pointPolygonTest(landmarks_np_mp, pt2_tuple, False) >= 0 and \
                           cv2.pointPolygonTest(landmarks_np_mp, pt3_tuple, False) >= 0:
                            total_area += 0.5 * abs(pt1_tuple[0]*(pt2_tuple[1]-pt3_tuple[1]) + pt2_tuple[0]*(pt3_tuple[1]-pt1_tuple[1]) + pt3_tuple[0]*(pt1_tuple[1]-pt2_tuple[1]))
                            centroids.append(((t[0]+t[2]+t[4])//3, (t[1]+t[3]+t[5])//3))
                    for i in range(len(centroids) - 1):
                        total_length += np.linalg.norm(np.array(centroids[i]) - np.array(centroids[i+1]))
        features['total_area'] = np.round(total_area, 2)
        features['total_length'] = np.round(total_length, 2)

        # Features 3-34: 16x16 Grayscale Matrix Column Means and Stds
        img_resized = cv2.resize(img_gray, (16, 16)).astype(np.float32) / 255.0
        features['col_means'] = np.round(np.mean(img_resized, axis=0), 2).tolist()
        features['col_stds'] = np.round(np.std(img_resized, axis=0), 2).tolist()

        # Feature 35: Pupil Distance
        d_pupil = 0.0
        if results_dlib and len(results_dlib) > 0:
            landmarks_np_dlib = np.array([[p.x, p.y] for p in predictor(img_gray, results_dlib[0]).parts()])
            if landmarks_np_dlib.shape[0] >= 48:
                left_pupil_center_dlib = np.mean(landmarks_np_dlib[36:42], axis=0).astype(int)
                right_pupil_center_dlib = np.mean(landmarks_np_dlib[42:48], axis=0).astype(int)
                d_pupil = np.linalg.norm(left_pupil_center_dlib - right_pupil_center_dlib)
            elif results_mp and results_mp.multi_face_landmarks:
                 mp_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results_mp.multi_face_landmarks[0].landmark]
                 if 474 < len(mp_landmarks) and 469 < len(mp_landmarks):
                      left_pupil_mp = np.array(mp_landmarks[474])
                      right_pupil_mp = np.array(mp_landmarks[469])
                      d_pupil = np.linalg.norm(left_pupil_mp - right_pupil_mp)
        features['d_pupil'] = np.round(d_pupil, 2)

        # Feature 36 & 37: Face Area & Perimeter (from Haar Cascade)
        face_area, face_perimeter, fw, fh = 0.0, 0.0, 0, 0
        if len(faces_haar) > 0:
             (fx, fy, fw, fh) = faces_haar[0]
             face_area, face_perimeter = float(fw * fh), float(2 * (fw + fh))
        features['face_area'] = face_area
        features['face_perimeter'] = face_perimeter

        # Feature 38: Triangle area (pupils & nose) (using dlib)
        tri_area = 0.0
        if results_dlib and len(results_dlib) > 0 and 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 68:
             left_pupil_center_dlib = np.mean(landmarks_np_dlib[36:42], axis=0)
             right_pupil_center_dlib = np.mean(landmarks_np_dlib[42:48], axis=0)
             nose_tip_dlib = landmarks_np_dlib[30]
             tri_area = 0.5 * abs(left_pupil_center_dlib[0]*(right_pupil_center_dlib[1]-nose_tip_dlib[1]) + right_pupil_center_dlib[0]*(nose_tip_dlib[1]-left_pupil_center_dlib[1]) + nose_tip_dlib[0]*(left_pupil_center_dlib[1]-right_pupil_center_dlib[1]))
        features['tri_area'] = np.round(tri_area, 2)

        # Features 39-42: Additional Distances (d1, d2, d3, d4)
        d1, d2, d3, d4 = 0.0, 0.0, 0.0, 0.0
        if fw > 0 and fh > 0:
             d4 = fh
             d1 = 0.9 * fw
             if not mouth_cascade.empty():
                  mouths = mouth_cascade.detectMultiScale(img_gray[fy+fh//2:fy+fh, fx:fx+fw], 1.1, 5)
                  d3 = mouths[0][2] if len(mouths) > 0 else 0.8 * fw
             else: d3 = 0.8 * fw
             if not left_ear_cascade.empty() and not right_ear_cascade.empty():
                 left_ears = left_ear_cascade.detectMultiScale(img_gray[fy:fy+fh, fx:fx+fw], 1.1, 5)
                 right_ears = right_ear_cascade.detectMultiScale(img_gray[fy:fy+fh, fx:fx+fw], 1.1, 5)
                 if len(left_ears)>0 and len(right_ears)>0:
                      left_ear_center = (fx + left_ears[0][0] + left_ears[0][2] // 2, fy + left_ears[0][1] + left_ears[0][3] // 2)
                      right_ear_center = (fx + right_ears[0][0] + right_ears[0][2] // 2, fy + right_ears[0][1] + right_ears[0][3] // 2)
                      d2 = np.linalg.norm(np.array(right_ear_center) - np.array(left_ear_center))
                 else: d2 = 1.1 * fw
             else: d2 = 1.1 * fw
        features['d1'], features['d2'], features['d3'], features['d4'] = np.round(d1,2), np.round(d2,2), np.round(d3,2), np.round(d4,2)

        # Feature 43: Eye-to-eye distance to face width ratio
        eye_to_face_width_ratio = 0.0
        if fw > 0 and d_pupil > 0: eye_to_face_width_ratio = d_pupil / fw
        features['eye_to_face_width_ratio'] = np.round(eye_to_face_width_ratio, 4)

        # Feature 44: Nose bridge to face height ratio (using dlib)
        nose_bridge_ratio = 0.0
        if results_dlib and len(results_dlib) > 0 and 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 31:
             face_dlib_height = results_dlib[0].height()
             nose_top_dlib = landmarks_np_dlib[27]
             nose_tip_dlib = landmarks_np_dlib[30]
             nose_bridge_length_dlib = np.linalg.norm(nose_top_dlib - nose_tip_dlib)
             if face_dlib_height > 0: nose_bridge_ratio = nose_bridge_length_dlib / face_dlib_height
        features['nose_bridge_ratio'] = np.round(nose_bridge_ratio, 4)

        # Feature 45: Interocular distance / nose width (using dlib)
        interocular_ratio = 0.0
        if results_dlib and len(results_dlib) > 0 and 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 43:
             left_eye_inner_dlib = landmarks_np_dlib[39]
             right_eye_inner_dlib = landmarks_np_dlib[42]
             interocular_dist_dlib = np.linalg.norm(left_eye_inner_dlib - right_eye_inner_dlib)
             nose_left_dlib = landmarks_np_dlib[31]
             nose_right_dlib = landmarks_np_dlib[35]
             nose_width_dlib = np.linalg.norm(nose_left_dlib - nose_right_dlib)
             if nose_width_dlib > 0: interocular_ratio = interocular_dist_dlib / nose_width_dlib
        features['interocular_ratio'] = np.round(interocular_ratio, 4)

        # Feature 46: Inner Canthus to Outer Canthus Distance (MediaPipe)
        inner_canthus_dist = 0.0
        if results_mp and results_mp.multi_face_landmarks:
            mp_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results_mp.multi_face_landmarks[0].landmark]
            if all(idx < len(mp_landmarks) for idx in [33, 133]):
                inner_canthus_dist = math.dist(mp_landmarks[33], mp_landmarks[133])
        features['inner_canthus_dist'] = np.round(inner_canthus_dist, 2)

        # Feature 47: Eye Aspect Ratio (EAR) (MediaPipe)
        ear = 0.0
        if results_mp and results_mp.multi_face_landmarks:
            mp_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results_mp.multi_face_landmarks[0].landmark]
            if all(idx < len(mp_landmarks) for idx in [33,160,144,133]):
                dist_h = np.linalg.norm(np.array(mp_landmarks[33]) - np.array(mp_landmarks[133]))
                if dist_h > 0: ear = np.linalg.norm(np.array(mp_landmarks[160]) - np.array(mp_landmarks[144])) / dist_h
        features['ear'] = np.round(ear, 2)

        # Feature 48: Distance from eye to eyebrow (MediaPipe)
        eye_eyebrow_dist = 0.0
        if results_mp and results_mp.multi_face_landmarks:
            mp_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results_mp.multi_face_landmarks[0].landmark]
            if all(idx < len(mp_landmarks) for idx in [159, 70]):
                eye_eyebrow_dist = math.dist(mp_landmarks[159], mp_landmarks[70])
        features['eye_eyebrow_dist'] = np.round(eye_eyebrow_dist, 2)

        # Feature 49: Nose tip to chin distance / face height (dlib)
        nose_chin_face_height_ratio = 0.0
        if results_dlib and len(results_dlib) > 0 and 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 68:
             nose_tip_dlib = landmarks_np_dlib[30]
             chin_dlib = landmarks_np_dlib[8]
             face_dlib_height = results_dlib[0].height()
             nose_to_chin_dist = np.linalg.norm(nose_tip_dlib - chin_dlib)
             if face_dlib_height > 0: nose_chin_face_height_ratio = nose_to_chin_dist / face_dlib_height
        features['nose_chin_face_height_ratio'] = np.round(nose_chin_face_height_ratio, 4)

        # Feature 50: Mouth width to nose width ratio (MediaPipe)
        mouth_nose_ratio = 0.0
        if results_mp and results_mp.multi_face_landmarks:
            mp_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results_mp.multi_face_landmarks[0].landmark]
            if all(idx < len(mp_landmarks) for idx in [61,291,102,331]):
                mouth_w_mp = math.dist(mp_landmarks[61], mp_landmarks[291])
                nose_w_mp = math.dist(mp_landmarks[102], mp_landmarks[331])
                if nose_w_mp > 0: mouth_nose_ratio = mouth_w_mp / nose_w_mp
        features['mouth_nose_ratio'] = np.round(mouth_nose_ratio, 2)

        # Feature 51: Mouth center to chin distance (MediaPipe)
        mouth_chin_dist = 0.0
        if results_mp and results_mp.multi_face_landmarks:
            mp_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results_mp.multi_face_landmarks[0].landmark]
            if all(idx < len(mp_landmarks) for idx in [13,14,152]):
                mc_pt = ((mp_landmarks[13][0]+mp_landmarks[14][0])//2, (mp_landmarks[13][1]+mp_landmarks[14][1])//2)
                mouth_chin_dist = math.dist(mc_pt, mp_landmarks[152])
        features['mouth_chin_dist'] = np.round(mouth_chin_dist, 2)

        # Features 52-53: Facial Symmetry
        h_sym_score, e_sym_score = 0.0, 0.0
        if results_mp and results_mp.multi_face_landmarks:
            landmarks = [(lm.x * w, lm.y * h) for lm in results_mp.multi_face_landmarks[0].landmark]
            central_x = np.mean([landmarks[i][0] for i in [152, 4, 1] if i < len(landmarks)]) if [i for i in [152,4,1] if i < len(landmarks)] else w/2
            h_discrepancies, e_discrepancies = [], []
            symmetric_pairs = [(33, 263), (133, 362), (61, 291), (234, 454)] # Simplified pairs
            for l_idx, r_idx in symmetric_pairs:
                if all(i < len(landmarks) for i in [l_idx, r_idx]):
                    l_pt, r_pt = landmarks[l_idx], landmarks[r_idx]
                    h_discrepancies.append(abs(abs(l_pt[0] - central_x) - abs(r_pt[0] - central_x)))
                    mirrored_x = central_x + (central_x - l_pt[0])
                    e_discrepancies.append(math.dist(r_pt, (mirrored_x, l_pt[1])))
            face_w = faces_haar[0][2] if len(faces_haar) > 0 else w
            if face_w > 0:
                h_sym_score = 1 - (np.mean(h_discrepancies) / face_w) if h_discrepancies else 1.0
                e_sym_score = 1 - (np.mean(e_discrepancies) / face_w) if e_discrepancies else 1.0
        features['horizontal_symmetry_score'] = np.round(h_sym_score, 3)
        features['euclidean_symmetry_score'] = np.round(e_sym_score, 2)

        # Features 54-79: LBP of Forehead
        lbp_hist_forehead = [0.0] * 26
        if len(faces_haar) > 0:
            (fx,fy,fw,fh) = faces_haar[0]
            forehead_roi = img_gray[fy:fy+int(fh*0.3), fx:fx+fw]
            if forehead_roi.size > 0:
                lbp_hist_forehead, _ = describe_lbp(forehead_roi, 24, 8)
                lbp_hist_forehead = lbp_hist_forehead.tolist()
        features['lbp_hist_forehead'] = lbp_hist_forehead

        # Features 80-105: LBP of region between eyebrows
        lbp_hist_eyebrows = [0.0] * 26
        if results_dlib and len(results_dlib) > 0:
            landmarks_np_dlib = np.array([[p.x, p.y] for p in predictor(img_gray, results_dlib[0]).parts()])
            if landmarks_np_dlib.shape[0] >= 48:
                l_eb_outer, r_eb_outer = landmarks_np_dlib[17], landmarks_np_dlib[26] # Outer eyebrow points
                l_eb_inner, r_eb_inner = landmarks_np_dlib[21], landmarks_np_dlib[22] # Inner eyebrow points

                # Define ROI based on inner eyebrow points
                eb_roi_x1, eb_roi_x2 = min(l_eb_inner[0], r_eb_inner[0]), max(l_eb_inner[0], r_eb_inner[0])
                eb_roi_y1 = min(l_eb_inner[1], r_eb_inner[1]) - 10 # Extend slightly above
                eb_roi_y2 = max(l_eb_inner[1], r_eb_inner[1]) + 10 # Extend slightly below

                # Ensure ROI is within image bounds
                eb_roi_x1 = max(0, eb_roi_x1)
                eb_roi_y1 = max(0, eb_roi_y1)
                eb_roi_x2 = min(w, eb_roi_x2)
                eb_roi_y2 = min(h, eb_roi_y2)

                eyebrow_region = img_gray[eb_roi_y1:eb_roi_y2, eb_roi_x1:eb_roi_x2]

                if eyebrow_region.size > 0:
                    lbp_hist_eyebrows, _ = describe_lbp(eyebrow_region, 24, 8)
                    lbp_hist_eyebrows = lbp_hist_eyebrows.tolist()
        features['lbp_hist_eyebrows'] = lbp_hist_eyebrows

        # Features 106-137: Gabor Filters on Irises
        left_gabor, right_gabor = [0.0]*16, [0.0]*16
        if results_mp and results_mp.multi_face_landmarks:
            landmarks_iris = results_mp.multi_face_landmarks[0].landmark
            def get_gabor_features(iris_indices, img, w, h, kernels):
                feats = []
                if all(idx < len(landmarks_iris) for idx in iris_indices):
                    center = np.mean([(landmarks_iris[i].x*w, landmarks_iris[i].y*h) for i in iris_indices], axis=0)
                    radius = 15
                    y1, y2 = int(center[1]-radius), int(center[1]+radius)
                    x1, x2 = int(center[0]-radius), int(center[0]+radius)
                    y1,y2,x1,x2 = max(0,y1), min(h,y2), max(0,x1), min(w,x2) # Clamp bounds
                    iris_region = img[y1:y2, x1:x2]
                    if iris_region.size > 0:
                        for kern in kernels:
                            f_img = ndimage.convolve(iris_region.astype(float), kern)
                            feats.extend([np.mean(f_img), np.std(f_img), np.max(f_img), np.min(f_img)])
                return feats if len(feats) == 16 else [0.0]*16

            kernels = [cv2.getGaborKernel((21,21),4,t,10,0.5,0) for t in [0,np.pi/4,np.pi/2,3*np.pi/4]]
            left_gabor = get_gabor_features(range(474,478), img_gray, w, h, kernels)
            right_gabor = get_gabor_features(range(469,473), img_gray, w, h, kernels)

        features['left_gabor'] = left_gabor
        features['right_gabor'] = right_gabor

        # Features 138-161: GLCM of Temple
        glcm_feats_list = [0.0]*24
        if len(faces_haar) > 0:
            (fx,fy,fw,fh) = faces_haar[0]
            # Approximate temple ROI (right temple)
            temple_roi_x1 = fx + int(fw * 0.7)
            temple_roi_y1 = fy + int(fh * 0.1)
            temple_roi_w = int(fw * 0.2)
            temple_roi_h = int(fh * 0.3)

            temple_roi_x2 = temple_roi_x1 + temple_roi_w
            temple_roi_y2 = temple_roi_y1 + temple_roi_h

            # Ensure ROI is within image boundaries
            temple_roi_x1, temple_roi_y1, temple_roi_x2, temple_roi_y2 = max(0, temple_roi_x1), max(0, temple_roi_y1), min(w, temple_roi_x2), min(h, temple_roi_y2)

            temple_image = img_gray[temple_roi_y1:temple_roi_y2, temple_roi_x1:temple_roi_x2]

            if temple_image.size > 0:
                glcm = graycomatrix(temple_image.astype(np.uint8), [1], [0,np.pi/4,np.pi/2,3*np.pi/4], 256, symmetric=True, normed=True)
                props = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
                glcm_feats_list = [item for prop in props for item in graycoprops(glcm, prop).ravel()]
        features['glcm'] = glcm_feats_list

        # Features 162-170: Facial Ratios (based on Haar and Dlib)
        features['forehead_face_ratio'], features['midface_face_ratio'], features['lower_face_ratio_haar'] = 0.0, 0.0, 0.0
        features['forehead_midface_ratio'], features['midface_lower_face_ratio'] = 0.0, 0.0
        features['mouth_face_width_ratio'] = 0.0
        features['nose_bridge_top_len_ratio'], features['nose_bridge_mid_len_ratio'], features['nose_bridge_mid_top_ratio'] = 0.0, 0.0, 0.0

        if len(faces_haar) > 0:
            (fx,fy,fw,fh) = faces_haar[0]
            if fh > 0:
                features['forehead_face_ratio'] = np.round((fh*0.3)/fh, 2)
                features['midface_face_ratio'] = np.round((fh*0.4)/fh, 2)
                features['lower_face_ratio_haar'] = np.round((fh*0.3)/fh, 2)
            if fh*0.4 > 0: features['forehead_midface_ratio'] = np.round((fh*0.3)/(fh*0.4), 2)
            if fh*0.3 > 0: features['midface_lower_face_ratio'] = np.round((fh*0.4)/(fh*0.3), 2)

        if results_dlib and len(results_dlib) > 0:
            face_dlib = results_dlib[0]
            landmarks_np_dlib = np.array([[p.x, p.y] for p in predictor(img_gray, face_dlib).parts()])
            if landmarks_np_dlib.shape[0] >= 68:
                face_w_dlib = face_dlib.width()
                mouth_w_dlib = np.linalg.norm(landmarks_np_dlib[48] - landmarks_np_dlib[54])
                if face_w_dlib > 0: features['mouth_face_width_ratio'] = np.round(mouth_w_dlib/face_w_dlib, 2)
                nose_top = landmarks_np_dlib[27]
                nose_tip = landmarks_np_dlib[30]
                nose_len = np.linalg.norm(nose_top - nose_tip)
                nose_bridge_top_left = landmarks_np_dlib[21]
                nose_bridge_top_right = landmarks_np_dlib[22]
                nose_bridge_top_w = np.linalg.norm(nose_bridge_top_left - nose_bridge_top_right)
                nose_bridge_mid_left = landmarks_np_dlib[32]
                nose_bridge_mid_right = landmarks_np_dlib[34]
                nose_bridge_mid_w = np.linalg.norm(nose_bridge_mid_left - nose_bridge_mid_right)

                if nose_len > 0: features['nose_bridge_top_len_ratio'] = np.round(nose_bridge_top_w/nose_len, 2)
                if nose_len > 0: features['nose_bridge_mid_len_ratio'] = np.round(nose_bridge_mid_w/nose_len, 2)
                if nose_bridge_top_w > 0: features['nose_bridge_mid_top_ratio'] = np.round(nose_bridge_mid_w/nose_bridge_top_w, 2)


        # Features 171-185: Jaw Curvature
        jaw_curvature_features = [0.0] * 15
        if results_dlib and len(results_dlib) > 0:
            landmarks_np_dlib = np.array([[p.x, p.y] for p in predictor(img_gray, results_dlib[0]).parts()])
            if landmarks_np_dlib.shape[0] >= 17:
                jawline_pts = landmarks_np_dlib[0:17]
                for i in range(1, len(jawline_pts) - 1):
                    p_prev, p_curr, p_next = jawline_pts[i-1], jawline_pts[i], jawline_pts[i+1]
                    v1, v2 = p_prev - p_curr, p_next - p_curr
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        jaw_curvature_features[i-1] = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        features['jaw_curvature'] = [np.round(j, 2) for j in jaw_curvature_features]

        # Features 186-190: Mouth Symmetry
        mouth_symmetry_scores = [0.0] * 5
        if results_dlib and len(results_dlib) > 0 and 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 68:
            mouth_pts = landmarks_np_dlib[48:68]
            left_corner, right_corner = mouth_pts[0], mouth_pts[6]
            mouth_center_x, mouth_center_y = (left_corner[0] + right_corner[0]) / 2, (left_corner[1] + right_corner[1]) / 2
            mouth_symmetry_scores[0] = abs(left_corner[1] - right_corner[1])
            if mouth_pts.shape[0] >= 5: # Check if necessary points exist
                upper_lip_left_y, upper_lip_right_y = mouth_pts[2][1], mouth_pts[4][1]
                dist_upper_left = abs(upper_lip_left_y - mouth_center_y)
                dist_upper_right = abs(upper_lip_right_y - mouth_center_y)
                if dist_upper_right > 0: mouth_symmetry_scores[1] = dist_upper_left / dist_upper_right
            if mouth_pts.shape[0] >= 11: # Check if necessary points exist
                lower_lip_left_y, lower_lip_right_y = mouth_pts[10][1], mouth_pts[8][1]
                dist_lower_left = abs(lower_lip_left_y - mouth_center_y)
                dist_lower_right = abs(lower_lip_right_y - mouth_center_y)
                if dist_lower_right > 0: mouth_symmetry_scores[2] = dist_lower_left / dist_lower_right
            if mouth_pts.shape[0] >= 6: # Check if necessary points exist
                dist_horiz_upper_left = abs(mouth_pts[1][0] - mouth_center_x)
                dist_horiz_upper_right = abs(mouth_pts[5][0] - mouth_center_x)
                if dist_horiz_upper_right > 0: mouth_symmetry_scores[3] = dist_horiz_upper_left / dist_horiz_upper_right
            if mouth_pts.shape[0] >= 12: # Check if necessary points exist
                dist_horiz_lower_left = abs(mouth_pts[11][0] - mouth_center_x)
                dist_horiz_lower_right = abs(mouth_pts[7][0] - mouth_center_x)
                if dist_horiz_upper_right > 0: mouth_symmetry_scores[4] = dist_horiz_lower_left / dist_horiz_lower_right
        features['mouth_symmetry'] = [np.round(m, 2) for m in mouth_symmetry_scores]

        # Feature 191: Jaw Angle at Center
        jaw_center_angle = 0.0
        if results_dlib and len(results_dlib) > 0 and 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 10:
            p_prev, p_curr, p_next = landmarks_np_dlib[7], landmarks_np_dlib[8], landmarks_np_dlib[9]
            v1, v2 = p_prev - p_curr, p_next - p_curr
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                jaw_center_angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        features['jaw_center_angle'] = np.round(jaw_center_angle, 2)

        # Features 192-200: Jaw Symmetry
        jaw_symmetry_scores = [0.0] * 9
        if results_dlib and len(results_dlib) > 0 and 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 31:
            chin_point = landmarks_np_dlib[8]
            symmetric_jaw_pairs = [(7, 9), (6, 10), (5, 11), (4, 12), (3, 13), (2, 14), (1, 15), (0, 16)]
            distance_ratios = []
            horizontal_ratios = []
            if landmarks_np_dlib.shape[0] >= 31: # Check if nose tip landmark exists
                nose_tip_point = landmarks_np_dlib[30]
                midline_x = (chin_point[0] + nose_tip_point[0]) / 2
                for i, (left_idx, right_idx) in enumerate(symmetric_jaw_pairs):
                    if all(idx < landmarks_np_dlib.shape[0] for idx in [left_idx, right_idx]):
                        left_point, right_point = landmarks_np_dlib[left_idx], landmarks_np_dlib[right_idx]
                        dist_left, dist_right = np.linalg.norm(chin_point - left_point), np.linalg.norm(chin_point - right_point)
                        if dist_right > 0: distance_ratios.append(dist_left / dist_right)
                        dist_left_midline, dist_right_midline = abs(left_point[0] - midline_x), abs(right_point[0] - midline_x)
                        if dist_right_midline > 0: horizontal_ratios.append(dist_left_midline / dist_right_midline)
            jaw_symmetry_scores[:len(distance_ratios)] = [np.round(r, 3) for r in distance_ratios] # Round ratios
            if horizontal_ratios: jaw_symmetry_scores[8] = np.round(np.mean(horizontal_ratios), 3) # Round mean
        features['jaw_symmetry'] = jaw_symmetry_scores


        # Feature 201: Number of Golden Ratios
        golden_ratio_count = 0
        GOLDEN_RATIO = 1.618
        TOLERANCE = 0.1
        if results_dlib and len(results_dlib) > 0:
            face_dlib = results_dlib[0]
            face_height = face_dlib.height()
            face_width = face_dlib.width()
            if face_width > 0:
                r1 = face_height / face_width
                if abs(r1 - GOLDEN_RATIO) <= TOLERANCE:
                    golden_ratio_count += 1

            if 'landmarks_np_dlib' in locals() and landmarks_np_dlib.shape[0] >= 68:
                nose_top = landmarks_np_dlib[27]
                nose_tip = landmarks_np_dlib[30]
                nose_length = np.linalg.norm(nose_top - nose_tip)
                nose_left = landmarks_np_dlib[31]
                nose_right = landmarks_np_dlib[35]
                nose_width = np.linalg.norm(nose_left - nose_right)
                if nose_width > 0:
                    r2 = nose_length / nose_width
                    if abs(r2 - GOLDEN_RATIO) <= TOLERANCE:
                        golden_ratio_count += 1

                mouth_left = landmarks_np_dlib[48]
                mouth_right = landmarks_np_dlib[54]
                mouth_width = np.linalg.norm(mouth_left - mouth_right)
                if nose_width > 0:
                    r3 = mouth_width / nose_width
                    if abs(r3 - GOLDEN_RATIO) <= TOLERANCE:
                        golden_ratio_count += 1

                left_eye_pts_dlib = landmarks_np_dlib[36:42]
                right_eye_pts_dlib = landmarks_np_dlib[42:48]
                left_pupil = np.mean(left_eye_pts_dlib, axis=0)
                right_pupil = np.mean(right_eye_pts_dlib, axis=0)
                pupil_dist = np.linalg.norm(left_pupil - right_pupil)
                if mouth_width > 0:
                    r4 = pupil_dist / mouth_width
                    if abs(r4 - GOLDEN_RATIO) <= TOLERANCE:
                        golden_ratio_count += 1

        features['golden_ratio_count'] = golden_ratio_count


        # --- Assemble Final Vector ---
        final_feature_vector = [
            features.get('total_area', 0.0),
            features.get('total_length', 0.0),
            *features.get('col_means', [0.0]*16),
            *features.get('col_stds', [0.0]*16),
            features.get('d_pupil', 0.0),
            features.get('face_area', 0.0),
            features.get('face_perimeter', 0.0),
            features.get('tri_area', 0.0),
            features.get('d1', 0.0),
            features.get('d2', 0.0),
            features.get('d3', 0.0),
            features.get('d4', 0.0),
            features.get('eye_to_face_width_ratio', 0.0),
            features.get('nose_bridge_ratio', 0.0),
            features.get('interocular_ratio', 0.0),
            features.get('inner_canthus_dist', 0.0),
            features.get('ear', 0.0),
            features.get('eye_eyebrow_dist', 0.0),
            features.get('nose_chin_face_height_ratio', 0.0),
            features.get('mouth_nose_ratio', 0.0),
            features.get('mouth_chin_dist', 0.0),
            features.get('horizontal_symmetry_score', 0.0),
            features.get('euclidean_symmetry_score', 0.0),
            *features.get('lbp_hist_forehead', [0.0]*26),
            *features.get('lbp_hist_eyebrows', [0.0]*26),
            *features.get('left_gabor', [0.0]*16),
            *features.get('right_gabor', [0.0]*16),
            *features.get('glcm', [0.0]*24),
            features.get('forehead_face_ratio', 0.0),
            features.get('midface_face_ratio', 0.0),
            features.get('lower_face_ratio_haar', 0.0),
            features.get('forehead_midface_ratio', 0.0),
            features.get('midface_lower_face_ratio', 0.0),
            features.get('mouth_face_width_ratio', 0.0),
            features.get('nose_bridge_top_len_ratio', 0.0),
            features.get('nose_bridge_mid_len_ratio', 0.0),
            features.get('nose_bridge_mid_top_ratio', 0.0),
            *features.get('jaw_curvature', [0.0]*15),
            *features.get('mouth_symmetry', [0.0]*5),
            features.get('jaw_center_angle', 0.0),
            *features.get('jaw_symmetry', [0.0]*9),
            features.get('golden_ratio_count', 0)
        ]

        if len(final_feature_vector) > 201:
            final_feature_vector = final_feature_vector[:201]
        elif len(final_feature_vector) < 201:
            final_feature_vector.extend([0.0] * (201 - len(final_feature_vector)))

        return final_feature_vector

    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return []
