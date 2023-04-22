from flask import Flask, request, jsonify
import torchvision.ops.boxes as bops
import base64
import cv2
import torch
import torch
import numpy as np
import pandas as pd
import time
import sys
from Alignedreid_demo import Aligned_Reid_class
from PIL import Image
import logging
import json


# Initializations
app = Flask(__name__)
tracker = cv2.legacy.TrackerKCF_create()
ReId = Aligned_Reid_class()
init_bb = None
suspect_features = None
flag = False
frame_data = None
has_exit_count = 0
currently_selected_src = None
has_suspect_exit = False
exit_iou_vector = []
Reid_flag = False
position = None
camera_id = None
num_of_suspect_features = None
first_call = True


#Load the YOLOv5 model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if sys.modules.get('models') is not None:
    sys.modules.pop('models')

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device)

def torch_normalizer(tensor):
    normalized_tensor = tensor / tensor.norm(dim=1, keepdim=True)
    return normalized_tensor

def process_predictions(predictions, image):
    predict = []
    x_shape, y_shape = image.shape[1], image.shape[0]
    for box in predictions:
        x1, y1, x2, y2 = int(box[0] * x_shape), int(box[1] * y_shape), int(box[2] * x_shape), int(box[3] * y_shape)
        predict.append([x1, y1, x2, y2])
    return predict

def xyxytoxywh(coordinates):
    coordinates[2] = coordinates[2]-coordinates[0]
    coordinates[3] = coordinates[3]-coordinates[1]
    return coordinates

def xywhtoxyxy(coordinates):
    coordinates[2] = coordinates[2]+coordinates[0]
    coordinates[3] = coordinates[3]+coordinates[1]
    return coordinates

def iou_check(predictions=None, csrt_box=None):
    IOU = None
    IOU_vector = [None]*len(predictions)
    csrt_x, csrt_y, csrt_w, csrt_h = csrt_box
    box2 = torch.tensor([[csrt_x, csrt_y, csrt_x + csrt_w, csrt_y + csrt_h]], dtype=torch.float)

    for index,yolo_bbox  in enumerate(predictions):
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
        box1 = torch.tensor([[yolo_x1, yolo_y1, yolo_x2, yolo_y2]], dtype=torch.float)
        IOU = bops.box_iou(box1, box2)*100
        IOU_vector[index] = int(IOU)
    return IOU_vector, box2

def fetch_inference_results(inference_df):
    predictions = inference_df[inference_df['name'].isin(['person', 'backpack', 'handbag', 'suitcase'])].iloc[:,:4].values
    labels = inference_df[inference_df['name'].isin(['person', 'backpack', 'handbag', 'suitcase'])].iloc[:,6].values
    #predictions, labels = inference_df.iloc[:,:4].values, inference_df.iloc[:,6].values

    return predictions, labels

def fetch_request(request):
    image = request.json['image']
    suspect_img_encoded = request.json['suspect_img']
    suspect_coordinates = None
    suspect_feat_img_encoded = request.json['suspect_feat_img']
    return image,suspect_img_encoded,suspect_coordinates, suspect_feat_img_encoded

def fetch_reid_request(request):
    frames_data_dict = request.json['image_data']
    suspect_img_encoded = request.json['suspect_img']
    suspect_coordinates = None
    suspect_feat_img_encoded = request.json['suspect_feat_img']
    return frames_data_dict,suspect_img_encoded,suspect_coordinates, suspect_feat_img_encoded

def decode_img(image,count = 0):
    if count == 0:
        image = cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_UNCHANGED)
        return image
    else:
        image_list = {}
        for key,value in image.items():
            index = int(key)
            image_list[index] = cv2.imdecode(np.frombuffer(base64.b64decode(value), np.uint8), cv2.IMREAD_UNCHANGED)
        return image_list

def get_frame_data(frame, predictions):
    # this function iterates over all the predictions
    # crop the images with the help of prediction
    # convert images into pill image
    # then return bbox and pil image
    frame_m_data = []
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i, box in enumerate(predictions):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        image = frame[y1:y2, x1:x2]
        bounding_box = [x1, y1, x2, y2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        frame_m_data.append([bounding_box, image])
    return np.array(frame_m_data,dtype=object)


def dtw(dist_mat):
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]

    return dist[-1,-1]/sum(dist.shape), dist


def distance_cal(a1,a2):
    dist = np.zeros((8, 8))
    for i in range(8):
        temp_feat1 = a1[i]
        for j in range(8):
            temp_feat2 = a2[j]
            dist[i][j] = torch.norm(temp_feat1 - temp_feat2)
    d,__ = dtw(dist)
    return d

def has_person_exit(iou_vector, count):
    flag = True
    for item in iou_vector:
        if item != 0:
            flag = False
            break
    if flag is True:
        count += 1
        return flag, count
    return flag, count

def iou_check2(yolo_bbox, csrt_box):
    csrt_x1, csrt_y1, csrt_x2, csrt_y2 = csrt_box
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
    box1 = torch.tensor([[yolo_x1, yolo_y1, yolo_x2, yolo_y2]], dtype=torch.float)
    box2 = torch.tensor([[csrt_x1, csrt_y1, csrt_x2, csrt_y2]], dtype=torch.float)
    IOU = bops.box_iou(box1, box2)
    return IOU

def joint_reid(frame_m_data, result_list, features, bounding_box,feat_tensors):
    features_reid_score = []
    for i, b in enumerate(frame_m_data[:, 0]):
        IOU = iou_check2(b, bounding_box)
        if IOU * 100 > 0 and IOU != 1 and result_list[2] * 100 < 60:
            a1 = feat_tensors[1] # list of features....0 for person 1 for bag 2 for watch etc...these are features captured during roi draw
            a2 = features[i] # features on that image...can be person bag etc.... so we have to check with all of them.
            d = distance_cal(a1, a2)
            if d * 100 < 60:
                print('inside joint reid: ', d)
                features_reid_score.append(d)
    return features_reid_score # a list which contains scores.....

def detect_occlusion(iou_vector):
    count = 0
    max_iou = iou_vector.index(max(iou_vector))
    # print('IOU Vector..:', iou_vector)
    for iou_score in iou_vector:
        if iou_score != max_iou and iou_score >= 10:
            count +=1

    if count > 1:
        print('Occlusion:', True)
    return count

def return_reid_results(result_dict, resulted_features, images_list, predictions_dict, num_of_suspect_features):
    features_reid_score = []
    for key, res in  resulted_features.items():
        features = [torch_normalizer(feat) for feat in res]
        frame_data = get_frame_data(images_list[key], predictions_dict[key])
        a1 = suspect_features[0]
        dist_matrix = [distance_cal(a1, a2) for a2 in features]
        for idx, bounding_box in enumerate(frame_data[:,0]):
            result_list = [False,bounding_box,dist_matrix[idx]]

            if num_of_suspect_features > 1:
                features_reid_score = joint_reid(frame_data, result_list, features, bounding_box, suspect_features)

            if len(features_reid_score) != 0:
                arr = np.array(features_reid_score)
                result_list.append(np.min(arr))

            if len(result_list) == 4:
                join_reid_score = result_list[2] * result_list[3]
                result_list[2] = join_reid_score
                print('Result list: ',result_list)
                print('Joint reid score: ', join_reid_score, 'Camera id: ', key)
            else:
                join_reid_score = result_list[2]

            result_dict[key] += [result_list]
            if num_of_suspect_features > 1:
                if join_reid_score * 100 <= 35:
                    idx = len(result_dict[key]) - 1
                    result_dict[key][idx][0] = True

            if num_of_suspect_features == 1:
                if join_reid_score * 100 <= 60:
                    idx = len(result_dict[key]) - 1
                    result_dict[key][idx][0] = True
    for key in result_dict.keys():
        for i in range(len(result_dict[key])):
            print('Camera id: ', key, result_dict[key][i])
    return result_dict


@app.route('/predict', methods=['GET','POST'])
def predict():
    global init_bb
    global tracker
    global suspect_features
    global flag
    global frame_data
    global exit_iou_vector
    global has_exit_count
    global currently_selected_src
    global has_suspect_exit
    global Reid_flag
    global position
    global camera_id
    global num_of_suspect_features
    global first_call

    if request.method == 'POST':
        try:
            if Reid_flag is True:
                print('*****************Loading Re-id Module*************************')
                if init_bb is None and has_suspect_exit is True:
                    begin = time.time()
                    images, suspect_img_encoded,suspect_coordinates,suspect_feat_img_encoded = fetch_reid_request(request) # images is a dictionary
                    images_list = decode_img(images, len(images))
                    curr_length = 0
                    predictions_dict = {}
                    result_dict = {}
                    pill_objects = {}
                    resulted_features = {}
                    images = []
                    features_reid_score = []
                # task is to run the parallel inference for all images on multiple gpu instances..

                # block for yolo predictions..
                    for key, image in images_list.items():
                        result_dict[key] = []
                        images.append(image)
                    s = time.time()
                    results = model(images)
                    torch.cuda.synchronize()
                    e = time.time()
                    print('yolo: ', e - s)

                # setting up inference results from predictions obtained from yolo
                    for i, key in enumerate(images_list.keys()):
                        predictions, _ = fetch_inference_results(results.pandas().xyxy[i])
                        predictions_dict[key] = predictions
                        frame_data = get_frame_data(images[i], predictions)
                        if len(frame_data) != 0:
                            result_dict[key] = []
                            pill_objects[key] = frame_data[:, -1]
                # setting pill images for reid inference.
                    img = []
                    # if there are persons detected atleast one the follow below block
                    # else just return reid to be true to check in next set of frames..
                    if len(pill_objects) != 0:
                        for key in pill_objects:
                            for data in pill_objects[key]:
                                img.append(data)
                        s = time.time()
                        x_features = ReId.inference(persons=img)
                        torch.cuda.synchronize()
                        e = time.time()
                        print('reid: ', e-s)
                        idx = 0
                        curr_length = 0

                        for key in pill_objects:
                            f = []
                            for data in pill_objects[key]:
                                f.append(x_features.pop(0))
                            resulted_features[key] = f
                        result_dict = return_reid_results(result_dict,resulted_features,images_list,predictions_dict,num_of_suspect_features)

                        for key,_ in images_list.items():
                            for i in range(len(result_dict[key])):
                                if result_dict[key][i][0] is True:
                                    init_bb = result_dict[key][i][1]
                                    position = init_bb.copy()
                                    init_bb[2] = init_bb[2] - init_bb[0]
                                    init_bb[3] = init_bb[3] - init_bb[1]
                                    init_bb = tuple(init_bb)
                                    tracker = cv2.legacy.TrackerKCF_create()
                                    tracker.init(images_list[key], init_bb)
                                    camera_id = key

                        if position is not None and camera_id is not None:
                            # this block starts intra camera tracking from next frame...
                            Reid_flag = False
                            has_suspect_exit = False
                            json_dict = {camera_id: position}
                            torch.cuda.synchronize()
                            print('Elapsed time: True',time.time()-begin)
                            return json_dict
                        else:
                            has_suspect_exit = True
                            init_bb = None
                            Reid_flag = True
                            json_dict = {"ReidStatus": True}
                            torch.cuda.synchronize()
                            print('Elapsed time: True',time.time()-begin)
                            return json_dict
                    else:
                        has_suspect_exit = True
                        init_bb = None
                        Reid_flag = True
                        json_dict = {"ReidStatus": True}
                        torch.cuda.synchronize()
                        print('Elapsed time: True', time.time() - begin)
                        return json_dict

        except Exception as e:
            json_dict = {"ReidStatus": True}
            torch.cuda.synchronize()
            print('Exception was: ', e)
            print('Elapsed time: True',time.time()-begin)
            return json_dict

        if Reid_flag is False:
            begin = time.time()
            image, suspect_img_encoded, suspect_coordinates, suspect_feat_img_encoded = fetch_request(request)
            image = decode_img(image)

            # this block will be called on first call to api,
            # therefore we will extract suspect features only once
            # finally saves a vector which contains suspect features obtained after inference.
            if init_bb is None and has_suspect_exit is False:
                # setting this flag because when we call this module first time we want to get it's current possition..
                flag = True
                feat = []
                suspect_img_decoded = decode_img(suspect_img_encoded)
                suspect_img_decoded = cv2.cvtColor(suspect_img_decoded, cv2.COLOR_BGR2RGB)
                suspect_img_decoded = Image.fromarray(suspect_img_decoded)
                feat.append(suspect_img_decoded)

                # if it's not none then only extract.
                if suspect_feat_img_encoded is not None:
                    suspect_feat_img_encoded = decode_img(suspect_feat_img_encoded)
                    suspect_feat_img_encoded = cv2.cvtColor(suspect_feat_img_encoded, cv2.COLOR_BGR2RGB)
                    suspect_feat_img_encoded = Image.fromarray(suspect_feat_img_encoded)
                    feat.append(suspect_feat_img_encoded)

                res = ReId.inference2(persons=feat)
                suspect_features = [torch_normalizer(r) for r in res]  # 0 for suspect , 1 for bag so on..
                num_of_suspect_features = len(suspect_features)
                init_bb = 'Not None'

            if init_bb is not None and has_suspect_exit is False:
                a = time.time()
                results = model(image)
                torch.cuda.synchronize()
                b = time.time()
                print('Yolo took: ', b-a)
                predictions, labels = fetch_inference_results(results.pandas().xyxy[0])
                features_reid_score = []
                # if there are no bounding boxes in stream.
                # initiate global reid to check presence in other stream.
                if len(predictions) == 0:
                    tracker = cv2.legacy.TrackerKCF_create() # resetting the tracker.
                    print('No object found in stream...Initiating Global Re-identification')
                    has_suspect_exit = True
                    init_bb = None
                    Reid_flag = True
                    has_exit_count = 0  # unset to zero. because we will now find in which stream suspect is present.
                    json_dict = {"ReidStatus": True}
                    return json_dict
                else:
                    if flag is True:
                        current_cam_dict = {}
                        current_cam_dict[0] = []
                        frame_data = get_frame_data(image, predictions)
                        pill_objects = frame_data[:,-1]
                        res = ReId.inference(persons=pill_objects)
                        features = [torch_normalizer(feat) for feat in res]
                        a1 = suspect_features[0]  # 0 signifies it's a person
                        dist_matrix = [distance_cal(a1,a2) for a2 in features]

                        for idx, bounding_box in enumerate(frame_data[:, 0]):
                            result_list = [False, bounding_box, dist_matrix[idx]]
                            if num_of_suspect_features > 1:
                                features_reid_score = joint_reid(frame_data, result_list, features, bounding_box, suspect_features)
                            if len(features_reid_score) != 0:
                                arr = np.array(features_reid_score)
                                result_list.append(np.min(arr))
                            if len(result_list) == 4:
                                join_reid_score = result_list[2] * result_list[3]
                                result_list[2] = join_reid_score
                            else:
                                join_reid_score = result_list[2]
                            current_cam_dict[0] += [result_list]
                            if num_of_suspect_features > 1:
                                if join_reid_score * 100 <= 35:
                                    idx = len(current_cam_dict[0]) - 1
                                    current_cam_dict[0][idx][0] = True
                            if num_of_suspect_features == 1:
                                if join_reid_score * 100 <= 60:
                                    idx = len(current_cam_dict[0]) - 1
                                    current_cam_dict[0][idx][0] = True

                        for i in range(len(current_cam_dict[0])):
                            if current_cam_dict[0][i][0] is True:
                                init_bb = current_cam_dict[0][i][1]
                                position = init_bb.copy()
                                init_bb[2] = init_bb[2]-init_bb[0]
                                init_bb[3] = init_bb[3]-init_bb[1]
                                init_bb = tuple(init_bb)
                                tracker = cv2.legacy.TrackerKCF_create()
                                print(type(init_bb), init_bb, type(image))
                                tracker.init(image, init_bb)

                        print(current_cam_dict)
                    flag = False
                    success, coords = tracker.update(image)
                    IOU_vector,box2 = iou_check(predictions,coords)
                    pos = IOU_vector.index(max(IOU_vector))
                    if IOU_vector[pos] >= 30:
                        # when IOU is >= 30 we can have two case
                        # case1: occlusion which will cause tracker to deviate
                        # case2: perfect no occlusion just update the tracker with new bbox
                        print('IOU',IOU_vector[pos])
                        count = detect_occlusion(IOU_vector)
                        # case 1: occlusion
                        if count > 1:
                            flag = True
                            coords = pd.DataFrame(predictions[pos]).to_json()
                            torch.cuda.synchronize()
                            print('Elapsed time: False',time.time()-begin)
                            return coords
                        # case2 : No occlusion
                        else:
                            suspect_new_coords = xyxytoxywh(predictions[pos])
                            tracker = cv2.legacy.TrackerKCF_create() # overwrite the previous tracker..to handle scale change issue.
                            tracker.init(image,suspect_new_coords)
                            coords = xywhtoxyxy(suspect_new_coords)
                            has_exit_count = 0
                            coords = pd.DataFrame(coords).to_json()
                            torch.cuda.synchronize()
                            print('Elapsed time: False',time.time()-begin)
                            return coords
                    else:
                        # less than < 30 iou signifies low confidence person might exit
                        # or occlusion
                        is_roi_not_present, has_exit_count = has_person_exit(IOU_vector,has_exit_count)
                        print('Exit since', has_exit_count,'frame', is_roi_not_present, IOU_vector)

                        # case when person exits from last 3 consecutive frames.
                        # IOU will be [0,0,0] means person is not there, moved away from stream
                        if is_roi_not_present and has_exit_count >= 2:
                            tracker = cv2.legacy.TrackerKCF_create()  # resetting the tracker.
                            has_suspect_exit = True
                            init_bb = None
                            Reid_flag = True
                            has_exit_count = 0 # unset to zero..because we will now find in which stream suspect is present..
                            json_dict = {"ReidStatus": True}
                            torch.cuda.synchronize()
                            print('Elapsed time: False',time.time()-begin)
                            return json_dict
                        # occlusion case once again
                        # low IOU score person might get occluded
                        elif IOU_vector[pos] != 0:
                            # person can be occluded by other objects or tracker might drift..
                            # therefore intra camera reid.
                            flag = True
                            coords = xywhtoxyxy(list(coords))
                            coords = pd.DataFrame(coords).to_json()
                            torch.cuda.synchronize()
                            print('Elapsed time: False',time.time()-begin)
                            return coords
                        else:
                            coords = xywhtoxyxy(list(coords))
                            coords = pd.DataFrame(coords).to_json()
                            torch.cuda.synchronize()
                            print('Elapsed time: False',time.time()-begin)
                            return coords


if __name__ == '__main__':
    app.run('0.0.0.0')
