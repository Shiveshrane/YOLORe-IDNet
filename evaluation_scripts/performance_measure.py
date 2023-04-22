import numpy as np
import math

class ObjectTrackingEvaluator:
    def __init__(self, ground_truth_file, predictions_file):
        self.ground_truth_file = ground_truth_file
        self.path = '/home/vipin/projects/video_dataset/OTB-100/Results/'
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.ground_truth = self.read_otb_ground_truth_file(ground_truth_file)
        self.predictions = self.read_tracker_predictions(predictions_file, len(self.ground_truth))
        self.precision = None
        self.recall = None
        self.accuracy = None
        self.f1_score = None
        self.IOU_data = []
        self.mean_iou = None
        self.thresholds = None
        self.threshold = 0.4
        self.performance = 0
        self.sre = 0 # success rate evaluation means of successes
        self.ope = 0 # one pass evaluation- mean of ious..
        self.calculate_precision_recall_accuracy()
        self.calculate_f1_score()
        self.IOU_performance()

    def read_otb_ground_truth_file(self,filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        frame_data = {}
        for i in range(len(lines)):
            data = lines[i].strip().split(',')
            frame_number = i+1
            bbox = tuple(map(int, data[0].split('\t')))
            #bbox = tuple(map(int, data))

            if frame_number in frame_data:
                frame_data[frame_number].append(list(bbox))
            else:
                frame_data[frame_number] = list(bbox)
        return frame_data

    def read_tracker_predictions(self, file_path, frames_count):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        frame_data = {}
        for i in range(len(lines)):
            data = lines[i].strip().replace('{', '').replace('}', '').split(':')
            frame_number = int(float(data[0].strip()))
            bbox_data = tuple(map(int, data[1].strip().replace('(', '').replace(')', '').split(',')))
            frame_data[frame_number] = list(bbox_data)
        
        frame_data = {k: v for k, v in sorted(frame_data.items(), key=lambda item: item[0])}
        return frame_data

    def calculate_success_precision(self,gt_boxes, pred_boxes):
        thresholds = np.arange(0, 1.05, 0.05)
        self.thresholds = thresholds
        success_curve = np.zeros(len(thresholds))
        precision_curve = np.zeros(len(thresholds))

        for i, thresh in enumerate(thresholds):
            # calculate success
            successes = 0
            for frame_num, gt_box in gt_boxes.items():
                pred_box = pred_boxes[frame_num]
                if pred_box is not None and self.IoU(gt_box, pred_box) > thresh:
                    successes += 1
            success_curve[i] = float(successes) / len(gt_boxes)

            # calculate precision
            num_boxes = len(pred_boxes)
            if num_boxes > 0:
                tp = 0
                for frame_num, pred_box in pred_boxes.items():
                    gt_box = gt_boxes.get(frame_num)

                    if gt_box is not None and self.IoU(pred_box, gt_box) > thresh:
                        tp += 1
                precision_curve[i] = tp / num_boxes
            else:
                precision_curve[i] = 0

        return success_curve, precision_curve

    def calculate_precision_recall_accuracy(self):
        distance_sum = 0
        valid_frames = 0

        for frame_id, gt in self.ground_truth.items():
            if frame_id not in self.predictions:
                self.false_negatives += 1
            else:
                pd = self.predictions[frame_id]
                iou = self.IoU(gt, pd)
                self.IOU_data.append(iou)

                if pd == [0,0,0,0]:
                    if gt == [0,0,0,0]:
                        self.true_negatives += 1
                    else:
                        self.false_negatives += 1
                else:
                    pd_center = [(pd[0] + pd[2])/2, (pd[1] + pd[3])/2]
                    gt_center = [(gt[0] + gt[2])/2, (gt[1] + gt[3])/2]
                    distance = math.sqrt((pd_center[0] - gt_center[0])**2 + (pd_center[1] - gt_center[1])**2)
                    distance_sum += distance
                    valid_frames += 1

                    if iou >= self.threshold:
                        self.true_positives += 1
                    else:
                        self.false_positives += 1
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.accuracy = (self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)
        self.sre = self.true_positives / len(self.ground_truth)
        self.ope = distance_sum / valid_frames
        self.mean_iou = np.mean(self.IOU_data)


    def calculate_f1_score(self):
        if self.precision + self.recall == 0:
            return 0
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

    def IOU_performance(self):
        hits = [iou for iou in self.IOU_data if iou > self.threshold]
        performance = len(hits)/len(self.IOU_data)
        self.performance = performance*100

    def IoU(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def generate_results(self, sequence_name):
        self.precision = round(self.precision,2)
        self.recall = round(self.recall,2)
        self.accuracy = round(self.accuracy,2)
        self.f1_score = round(self.f1_score,2)
        self.sre = round(self.sre,2)
        self.ope = round(self.ope,2)
        path = self.path + sequence_name + '.txt'
        with open(path, 'w') as f:
            f.write("Precision: {}\n".format(self.precision))
            f.write("Recall: {}\n".format(self.recall))
            f.write("Accuracy: {}\n".format(self.accuracy))
            f.write("F1 Score: {}\n".format(self.f1_score))
            f.write("Success rate evaluation(TB-100 metric): {}\n".format(self.sre))
            f.write("Mean IOU: {}\n".format(self.mean_iou))
            f.write("One pass evaluation avg iou(TB-100 metric): {}\n".format(self.ope))

        print("Precision: {}".format(self.precision))
        print("Recall: {}".format(self.recall))
        print("Accuracy: {}".format(self.accuracy))
        print("F1 Score: {}".format(self.f1_score))
        print("Mean IOU: {}".format(self.mean_iou))
        print("Success rate evaluation(TB-100 metric): {}%".format(self.sre*100))
        print("One pass evaluation avg iou(TB-100 metric): {}".format(self.ope))
