Base pytorch yolov3 obj detection code cloned from https://github.com/ayooshkathuria/pytorch-yolo-v3

Utility Codes:

**Calculate_iou_2obj.py** --> used to calculate the accuracy of object detection given ground-truth and predicted file name

**locality_for_all_dataset.py** with conjunction with **locality_meas.py** --> used to get the spatial, temporal locality and I,P frame size

**ground_truth_expr.py** --> used for apply the preprocessing algorithms (like: encoding, JPEG compression, Downsizing for different parameter values) in batch and also store the detection output given a network bandwidth limit
