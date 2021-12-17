import os
import cv2
import time
import math
import onnxruntime
import numpy as np
from tool.utils import load_class_names, nms_cpu
from halo import Halo
import torch
import argparse
from utils import getMeanAndVar, load_models, to_numpy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

detections_map = {}

"""
predictions_map format:

label: [Array<Array<Original Pos Tuple>, Array<Predicted Pos Tuple>>]
"""
predictions_map = {}


# covert the output from model to bounding boxes


def post_processing(img, conf_thresh, nms_thresh, output, verbose):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                                  ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    t3 = time.time()

    if verbose:
        print('-----------------------------------')
        print('       max and argmax : %f' % (t2 - t1))
        print('                  nms : %f' % (t3 - t2))
        print('Post processing total : %f' % (t3 - t1))
        print('-----------------------------------')

    return bboxes_batch, (t3 - t2)

# helper function to draw bounding boxes


def plot_boxes_cv2(img, boxes, class_names, verbose, color=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [
                      1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            if verbose:
                print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(
                img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)

    return img

# run detection on the image/frame using the given onnx session


def detect(session, image_src, class_names, verbose):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H),
                         interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    if verbose:
        print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    start = time.time()
    obj_detect_start = time.time()
    outputs = session.run(None, {input_name: img_in})
    obj_detect_end = time.time()

    boxes, nms_time = post_processing(img_in, 0.4, 0.6, outputs, verbose)

    bbox_start = time.time()
    image_src = plot_boxes_cv2(image_src, boxes[0], class_names, verbose)
    bbox_end = time.time()

    end = time.time()
    # return image_src, (end - start), (post_end - post_start)
    return image_src, boxes, (end - start), (obj_detect_end - obj_detect_start), nms_time, (bbox_end - bbox_start)

# loading the transformer model
def getTrajectory(frame, input_width, input_height, bboxes, class_names):
    

    def getBoxCenters():
        centers = []
        for box in bboxes:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            centers.append((center_x, center_y, box[6]))
        return centers

    detections = getBoxCenters()
    for (x, y, c) in detections:
        if class_names[c] in detections_map:
            detections_map[class_names[c]][0].append([x, y])
        else:
            detections_map[class_names[c]] = [[[x, y]]]

        predictions_count = 0
        predictions_time = 0

    for label in detections_map:
        if label in detections_map and len(detections_map[label][0]) >= 9:
            arr = np.array(detections_map[label], dtype=np.float32)
            delta = arr[:, 1:, 0:2] - arr[:, :-1, 0:2]

            input_torch = torch.from_numpy(delta).to(device)
            if label == "end_effector":
                input_torch = (input_torch.to(
                    device) - mean_end_effector.to(device)) / std_end_effector.to(device)
            elif label == "problance":
                input_torch = (input_torch.to(
                    device) - mean_problance.to(device)) / std_problance.to(device)
            elif label == "probe":
                input_torch = (input_torch.to(device) -
                               mean_probe.to(device)) / std_probe.to(device)
            elif label == "person":
                input_torch = (input_torch.to(device) -
                               mean_person.to(device)) / std_person.to(device)
            else:
                input_torch = (input_torch.to(device) - mean_window.to(device)) / std_window.to(device)

            # src_att = torch.ones(
            #     (input_torch.shape[0], 1, input_torch.shape[1])).to(device)
            # start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(
            #     1).repeat(input_torch.shape[0], 1, 1).to(device)
            # dec_input = start_of_seq.to(device)

            start = time.time()
            predictions = []
            
            if label == "end_effector":
                ort_inputs = {end_effector_session.get_inputs()[0].name: to_numpy(input_torch)}
                out = end_effector_session.run(None, ort_inputs)
            elif label == "problance":
                ort_inputs = {problance_session.get_inputs()[0].name: to_numpy(input_torch)}
                out = problance_session.run(None, ort_inputs)
            elif label == "probe":
                ort_inputs = {probe_session.get_inputs()[0].name: to_numpy(input_torch)}
                out = probe_session.run(None, ort_inputs)
            elif label == "person":
                ort_inputs = {person_session.get_inputs()[0].name: to_numpy(input_torch)}
                out = person_session.run(None, ort_inputs)
            else:
                ort_inputs = {window_session.get_inputs()[0].name: to_numpy(input_torch)}
                out = person_session.run(None, ort_inputs)
            dec_input = torch.cat([dec_input, out[:, -1:, :]], 1)

            end = time.time()
            predictions_time += (end - start)
            predictions_count += 1
            
            if label == "end_effector":
                preds = (dec_input[:, 1:, 0:2]*std_end_effector.to(device) + mean_end_effector.to(
                    device)).detach().cpu().numpy().cumsum(1) + arr[:, -1, 0:2]
            elif label == "problance":
                preds = (dec_input[:, 1:, 0:2]*std_problance.to(device) + mean_problance.to(device)).detach(
                ).cpu().numpy().cumsum(1) + arr[:, -1, 0:2]
            elif label == "probe":
                preds = (dec_input[:, 1:, 0:2]*std_probe.to(device) + mean_probe.to(device)).detach(
                ).cpu().numpy().cumsum(1) + arr[:, -1, 0:2]
            elif label == "person":
                preds = (dec_input[:, 1:, 0:2]*std_person.to(device) + mean_person.to(device)).detach(
                ).cpu().numpy().cumsum(1) + arr[:, -1, 0:2]
            else:
                preds = (dec_input[:, 1:, 0:2]*std_window.to(device) + mean_window.to(device)).detach(
                ).cpu().numpy().cumsum(1) + arr[:, -1, 0:2]
                
            predictions.append(preds)
            predictions = np.concatenate(predictions, 0)
            # print(predictions)
            detections_map[label][0].pop(0)
            predictions_map[label] = [[], []]

            for j in range(11):
                pp1 = (int(preds[0, j, 0] * input_width),
                       int(preds[0, j, 1] * input_height))
                pp2 = (int(preds[0, j + 1, 0] * input_width),
                       int(preds[0, j + 1, 1] * input_height))
                predictions_map[label][1].append([pp1, pp2])

            for j in range(7):
                op1 = (int(arr[0, j, 0] * input_width),
                       int(arr[0, j, 1] * input_height))
                op2 = (int(arr[0, j + 1, 0] * input_width),
                       int(arr[0, j + 1, 1] * input_height))
                predictions_map[label][0].append([op1, op2])

    if predictions_count > 0:
        return frame, predictions_time / predictions_count

    return frame, 0 


def runInference(model, video_path, output_path, num_frames, class_names, verbose):

    session = onnxruntime.InferenceSession(
        model, providers=["CUDAExecutionProvider", ])
    video = cv2.VideoCapture(video_path)

    if (video.isOpened() == False):
        print("Error reading video file")

    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_size = (int(frame_width), int(frame_height))

    yolo_input_res = (608, 608)
    size = (frame_width, frame_height)

    if os.path.exists(output_path):
        print("clearing the output path")
        os.remove(output_path)

    input_fps = int(video.get(cv2.CAP_PROP_FPS))
    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'MP4V'), input_fps, frame_size)

    frame_count = 0
    total = {"inference": 0, "traj": 0, "obj": 0, "nms": 0, "bbox": 0}
    frame_freq = 10
    traj_count = 0
    past_trajectory_fps = 0
    with Halo(spinner="dots", text="Loading the frames") as sp:
        while(True):
            ret, frame = video.read()
            frame_count += 1

            if ret == True:
                # print(f"Frame {frame_count}")

                start = time.time()
                detection, bboxes, inference_time, obj_det, nms_time, bbox_time = detect(
                    session, frame, class_names, verbose)
                
                trajectory_time = 0
                if frame_count % frame_freq == 0 and len(bboxes) > 0:
                    detection, trajectory_time = getTrajectory(
                        detection, frame_width, frame_height, bboxes[0], class_names)
                end = time.time()
                if frame_count > 1:
                    total["inference"] += inference_time
                    total["obj"] += obj_det
                    total["nms"] += nms_time
                    total["bbox"] += bbox_time
                    if trajectory_time > 0:
                        total["traj"] += trajectory_time
                        traj_count += 1

                inference_fps = round(1 / inference_time, 2)
                total_time = end - start
                if trajectory_time > 0:
                    trajectory_fps = round(1 / (trajectory_time), 2)
                    past_trajectory_fps = trajectory_fps 
                    total_time += trajectory_time
                else:
                    trajectory_fps = past_trajectory_fps
                    if past_trajectory_fps > 0:
                        total_time += 1/past_trajectory_fps
                
                
                total_fps = round(1 / total_time, 2)
                sp.text = f"Frame {frame_count} Inference FPS: {inference_fps} Total FPS: {total_fps}"

                cv2.putText(detection, f"Input FPS: {input_fps} | Inference FPS: {inference_fps} | Trajectory FPS: {trajectory_fps} | Total Fps: {round(1 / (trajectory_time + inference_time), 2)}", (
                    50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                for label in predictions_map:
                    for op1, op2 in predictions_map[label][0]:
                        detection = cv2.line(
                            detection, op1, op2, (0, 0, 255), 2)
                    for pp1, pp2 in predictions_map[label][1]:
                        detection = cv2.line(
                            detection, pp1, pp2, (0, 255, 0), 2)

                result.write(detection)

                if num_frames is not None and frame_count == num_frames:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
    print(
        f"[Average Object Detection Time]: {total['obj'] / (frame_count - 1)}")
    print(f"[Average NMS Time]: {total['nms'] / (frame_count - 1) }")
    print(f"[Average BBOX Time]: {total['bbox'] / (frame_count - 1) }")
    print(
        f"[Average Inference Time]: {total['inference'] / (frame_count - 1)}")
    print(f"[Average FPS]: {1 / (total['inference'] / (frame_count - 1))}")
    if traj_count > 0:
        print(
            f"[Average Trajectory Prediction Time]: {total['traj'] / (traj_count)}")
    video.release()
    result.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        required=True, help="Path to the yolo model file")
    parser.add_argument('-i', '--input', type=str,
                        required=True, help="Path to the video file")
    parser.add_argument('-o', '--output', type=str,
                        required=True, help="Path to the output video")
    parser.add_argument('-f', '--frame_count', type=int,
                        help="Number of frames to run the video")
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help="Enable more details")
    parser.add_argument('-c', '--classes', help="Number of classes model trained on")
    parser.add_argument('-d', '--dataset', help="Path to the dataset file")
    parser.add_argument('-t', '--trajectory', help="Path to onnx model directory")
    parser.add_argument('-p', '--pytorch_model', help="Pytorch Model used RNN / LSTM / GRU")
    args = parser.parse_args()
    # print(args)

    class_names = load_class_names(args.classes)
    
    mean_end_effector, std_end_effector = getMeanAndVar(args.dataset, 0)
    mean_person, std_person = getMeanAndVar(args.dataset, 1)
    mean_probe, std_probe = getMeanAndVar(args.dataset, 2)
    mean_problance, std_problance = getMeanAndVar(args.dataset, 3)
    mean_window, std_window = getMeanAndVar(args.dataset, 4)
    
    end_effector_session, person_session, probe_session, problance_session, window_session = load_models(args.trajectory, args.classes, args.pytorch_model)
    
    runInference(args.model, args.input, args.output,
                 args.frame_count, class_names, args.verbose)
