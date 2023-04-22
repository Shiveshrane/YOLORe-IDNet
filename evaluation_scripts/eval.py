from performance_measure_xyxy import ObjectTrackingEvaluator
import matplotlib.pyplot as plt
import numpy as np


gt = '/home/vipin/projects/video_dataset/IIT-Goa-dataset-1/ground_truth/8_gt.txt'
pd = '/home/vipin/projects/video_dataset/IIT-Goa-dataset-1/predictions_without_OD/8.txt'
#pd = '/home/vipin/projects/video_dataset/IIT-Goa-dataset-1/predictions_with_OD/8.txt'

measure = ObjectTrackingEvaluator(gt, pd)
#success_curve, precision_curve = measure.calculate_success_precision(measure.ground_truth, measure.predictions)
measure.generate_results('8')
