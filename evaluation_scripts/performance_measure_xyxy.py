import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
import math

class ObjectTrackingEvaluator:
    def __init__(self, ground_truth_file, predictions_file):
        self.ground_truth_file = ground_truth_file
        self.path = '/home/vipin/projects/video_dataset/IIT-Goa-dataset-1/predictions_without_OD/results/'
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
        self.thresholds = None
        self.threshold = 0.4
        self.performance = 0
        self.mean_iou = None
        self.sre = 0 # success rate evaluation means of successes
        self.ope = 0 # one pass evaluation- mean of ious..
        self.calculate_precision_recall_accuracy()
        self.calculate_f1_score()
        self.IOU_performance()
        self.generate_pr_plot(self.ground_truth, self.predictions)

    def read_otb_ground_truth_file(self,filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        frame_data = {}
        for line in lines:
            data = eval(line.strip())
            frame_number = list(data.keys())[0]
            bbox = data[frame_number]
            if frame_number in frame_data:
                frame_data[frame_number].append(bbox)
            else:
                frame_data[frame_number] = [bbox]
        return frame_data

    def read_tracker_predictions(self, file_path, frames_count):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        frame_data = {}
        for i in range(len(lines)):
            data = lines[i].strip().replace('{', '').replace('}', '').split(':')
            frame_number = int(float(data[0].strip()))
            bbox_data = tuple(map(int, data[1].strip().replace('[', '').replace(']', '').split(',')))
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
                for box in gt_box:
                    gt_box = box
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
                for box in gt:
                    gt = box
                iou = self.IoU(gt, pd)
                self.IOU_data.append(iou)

                if gt == [0,0,0,0]:
                    if pd == [0,0,0,0]:
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
        #self.recall = 0
        self.accuracy = (self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)
        self.sre = self.true_positives / len(self.ground_truth)
        self.ope = distance_sum / valid_frames
        self.mean_iou = np.mean(self.IOU_data)
        print('TP:', self.true_positives)
        print('FP:', self.false_positives)
        print('TN:', self.true_negatives)
        print('FN:', self.false_negatives)
    def calculate_f1_score(self):

        if self.precision + self.recall == 0:
            self.f1_score = 0
        else:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)


    def IOU_performance(self):
        hits = [iou for iou in self.IOU_data if iou > self.threshold]
        performance = len(hits)/len(self.IOU_data)
        self.performance = performance*100

    def IoU(self, boxa, boxb):
        x_left = max(boxa[0], boxb[0])
        y_top = max(boxa[1], boxb[1])
        x_right = min(boxa[2], boxb[2])
        y_bottom = min(boxa[3], boxb[3])

        # Calculate area of intersection rectangle
        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        # Calculate areas of the two bounding boxes
        boxa_area = (boxa[2] - boxa[0] + 1) * (boxa[3] - boxa[1] + 1)
        boxb_area = (boxb[2] - boxb[0] + 1) * (boxb[3] - boxb[1] + 1)

        # Calculate IOU
        iou = intersection_area / float(boxa_area + boxb_area - intersection_area)

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

    def generate_pr_plot(self, y_true, y_pred):
        """
        Generates a precision-recall plot and saves it to the current directory.

        Args:
            y_true: true labels
            y_pred: predicted labels
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve')

        plt.savefig('precision_recall_plot.png')
