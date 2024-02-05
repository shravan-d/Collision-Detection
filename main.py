import matplotlib.pyplot as plt
from yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox
from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
from deep_sort import build_tracker
import math
import argparse
import os
import time
import numpy as np
import warnings
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))
cudnn.benchmark = True


def topleft_bottomleft(arr):
    arr_copy = [-1, -1]
    arr_copy[0] = int((arr[0] + arr[2]) / 2)
    arr_copy[1] = arr[3]
    return np.array(arr_copy)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def compensate_relative_motion(prev_frame, current_frame, coordinates):
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_points = cv2.goodFeaturesToTrack(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), mask=None, **feature_params)
    current_points, _, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None)
    E, mask = cv2.findEssentialMat(prev_points, current_points)
    E = E[:3, :]
    _, R, t, mask = cv2.recoverPose(E, prev_points, current_points)
    T_camera = np.eye(3)
    T_camera[:2, :2] = R[:2, :2]
    T_camera[:2, 2] = t[:2].squeeze()
    object_position_homogeneous = np.append(coordinates, 1)
    T_camera_inv = np.linalg.inv(T_camera)
    object_position_compensated_ = np.dot(T_camera_inv, object_position_homogeneous)
    object_position_compensated = object_position_compensated_[:2] / object_position_compensated_[2]
    object_position_compensated = np.array([int(ele) for ele in object_position_compensated])
    object_position_compensated[1] = coordinates[1]
    if abs(object_position_compensated[0] - coordinates[0] - 0.2*current_frame.shape[1]) > 0:
        object_position_compensated[0] = coordinates[0]
    return object_position_compensated


class VideoTracker(object):
    def __init__(self, args):
        print('Initialize DeepSORT & YOLO-V5')
        self.args = args

        self.img_size = args.img_size                   # image size in detector, default is 640
        self.frame_interval = args.frame_interval       # frequency

        self.device = select_device(args.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # create video capture ****************
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        # ***************************** initialize DeepSORT **********************************
        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)

        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

        # ***************************** initialize YOLO-V5 **********************************
        self.detector = torch.load(args.weights, map_location=self.device)['model'].float()  # load to FP32
        self.detector.to(self.device).eval()
        if self.half:
            self.detector.half()  # to FP16

        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

        print('Done..')
        if self.device == 'cpu':
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

    def __enter__(self):
        # ************************* Load video from camera *************************
        if self.args.cam != -1:
            print('Camera ...')
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ************************* Load video from file *************************
        else:
            assert os.path.isfile(self.args.input_path), "Path error"
            self.vdo.open(self.args.input_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
            print('Done. Load video file ', self.args.input_path)

        # ************************* create output *************************
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.mp4")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
            print('Done. Create output file ', self.save_video_path)

        if self.args.save_txt:
            os.makedirs(self.args.save_txt, exist_ok=True)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        yolo_time, sort_time, avg_fps = [], [], []
        t_start = time.time()

        idx_frame = 0
        last_out = None
        times = []
        position_track = np.array([[-1, -1]])
        previous_time, previous_frame = -1, []
        time_interval, time_intervals = 0, []
        while self.vdo.grab():
            t0 = time.time()
            _, img0 = self.vdo.retrieve()

            if idx_frame % self.args.frame_interval == 0:
                outputs, yt, st = self.image_track(img0)        # (#ID, 5) x1,y1,x2,y2,id
                last_out = outputs
                yolo_time.append(yt)
                sort_time.append(st)
                print('Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
            else:
                outputs = last_out  # directly use prediction in last frames
            t1 = time.time()
            avg_fps.append(t1 - t0)

            outputs = np.array([o for o in outputs if o[-1] == 4])
            time_interval += 1

            if len(outputs) > 0:
                time_intervals.append(time_interval)
                time_interval = 0
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                text_scale = max(1, img0.shape[1] // 1600)
                img0 = draw_boxes(img0, bbox_xyxy, identities)

                point_of_interest = topleft_bottomleft(outputs[0])
                compensated_point_of_interest = compensate_relative_motion(previous_frame, img0, point_of_interest)

                cv2.circle(img0, point_of_interest, 2, (0, 255, 0))
                cv2.circle(img0, compensated_point_of_interest, 2, (0, 0, 255))

                car_line = [(300, 480), (440, 320), (600, 480), (540, 320)]
                cv2.line(img0, car_line[0], car_line[1], (0, 255, 0), 1)
                cv2.line(img0, car_line[2], car_line[3], (0, 255, 0), 1)


                position_track = np.vstack([position_track, compensated_point_of_interest])
                frame_count = 30
                if point_of_interest[0] < 570:
                    cv2.putText(img0, 'Time: 0.0s',
                                (outputs[0][2]-80, outputs[0][1]-20), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), 2)
                    if len(position_track) == frame_count:
                        position_track = position_track[1:]
                else:
                    if len(position_track) == frame_count:
                        delta_positions = np.diff(position_track[:, :2], axis=0)
                        time_intervals = np.array(time_intervals[len(time_intervals)-(frame_count-1):])
                        velocity = delta_positions / time_intervals[:, None]
                        time_intervals = []
                        velocity = np.mean(velocity, axis=0)
                        print('Velocity', velocity)
                        if abs(velocity[0]) > 0.6 or abs(velocity[1]) > 0.8:
                            lspace = np.linspace(0, img0.shape[0] - 1, img0.shape[0])
                            z = np.polyfit(position_track[:, 0], position_track[:, 1], 1)
                            i_x, i_y = line_intersection((car_line[2], car_line[3]), ((int(-z[1]/z[0]), 0), (int((480 - z[1])/z[0]), 480)))
                            if 480 > i_y > 50:
                                cv2.line(img0, (int(-z[1]/z[0]), 0), (int((480 - z[1])/z[0]), 480), (255, 0, 0), 1)
                            distance = math.sqrt((i_x - position_track[-1, 0]) ** 2 + (i_y - position_track[-1, 1]) ** 2)
                            relative_speed = math.sqrt(velocity[0] ** 2)
                            time_to_reach = distance / relative_speed / 30
                            times.append(time_to_reach)
                            if len(times) >= 3:
                                time_to_reach = np.mean(times[:-3])
                            print('Time: ', distance, relative_speed, time_to_reach)
                            cv2.putText(img0, 'Time: %.2f ' % time_to_reach,
                                        (outputs[0][2]-80, outputs[0][1]-20), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), 2)

                        position_track = position_track[1:]

            if self.args.display:
                cv2.imshow("test", img0)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(img0)

            idx_frame += 1
            previous_frame = img0

        print('Avg YOLO time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),
                                                            sum(sort_time)/len(sort_time)))
        t_end = time.time()
        print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

    def image_track(self, im0):
        """
        :param im0: original image, BGR format
        :return:
        """
        # preprocess ************************************************************
        # Padded resize
        img = letterbox(im0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # numpy to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        s = '%gx%g ' % img.shape[2:]    # print string

        # Detection time *********************************************************
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.detector(img, augment=self.args.augment)[0]  # list: bz * [ (#obj, 6)]

        # Apply NMS and filter object other than person (cls:0)
        pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,
                                   classes=self.args.classes, agnostic=self.args.agnostic_nms)
        t2 = time_synchronized()

        # get all obj ************************************************************
        det = pred[0]  # for video, bz is 1
        if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls

            # Rescale boxes from img_size to original im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results. statistics of number of each obj
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

            bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
            confs = det[:, 4:5].cpu()

            # ****************************** deepsort ****************************
            outputs = self.deepsort.update(bbox_xywh, confs, im0)
            # (#ID, 5) x1,y1,x2,y2,track_ID
        else:
            outputs = torch.zeros((0, 5))

        t3 = time.time()
        return outputs, t2-t1, t3-t2


# 7, 19, 23
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_path', type=str, default='LondonTestMini3.mp4', help='source')
    parser.add_argument('--save_path', type=str, default='output/', help='output folder')  # output folder
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_txt', default='output/predict/', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # camera only
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

    # YOLO-V5 parameters
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deepsort parameters
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with VideoTracker(args) as vdo_trk:
        vdo_trk.run()

