import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from yolov5.license_plate import main
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.deep_sort.opt import LicenceOpt
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from _collections import deque
pts = [deque() for _ in range(10000)]
pts_d = [deque() for _ in range(10000)]
speed_dict = {}

counter = []
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

def speed(pts_d, sample_frames, sample_fps, id, DISTANCE):
    sample_duration = sample_frames / sample_fps
    detection_duration = (len(pts_d[id]) * sample_duration)/sample_frames
    velocity = int((DISTANCE / detection_duration) * 3.6)
    speed_dict[id] = [velocity, False]
    pts_d[id] = deque()


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, detection_track, detection_bbox = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate, opt.track, opt.bbox
    UPPER_HEIGHT, LOWER_HEIGHT, MOTORCYCLE_MAX_SPEED, CAR_MAX_SPEED, TRUCK_MAX_SPEED, BUS_MAX_SPEED, DISTANCE = opt.upper, opt.lower, opt.motorcycle, opt.car, opt.truck, opt.bus, opt.distance


    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    
    licese_plate_device = select_device('cpu')
    license_plate_model = attempt_load('./yolov5/runs/train/license_plate/weights/best.pt', map_location=licese_plate_device)  # load FP32 model

    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img_plate = img
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        main(LicenceOpt(model= license_plate_model, img = img_plate, im0 = im0s))
        # Inference
        
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()
        fps = 1./(t2-t1)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            
            im1 = im0.copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                       
                        color = colors[int(id) % len(colors)]
                        color = [i * 255 for i in color]

                        center = (int(((output[0]) + (output[2]))/2), int(((output[1])+(output[3]))/2))
                        pts[id].append(center)
                        
                        if detection_track:
                            for j in range(1, len(pts[id])):
                                if pts[id][j-1] is None or pts[id][j] is None:
                                    continue
                                thickness = int(np.sqrt(64/float(j+1))*2)
                                cv2.line(im0, (pts[id][j-1]), (pts[id][j]), (0, 255, 0), thickness)

                        height, width, _ = im0.shape
                        cv2.line(im0, (0, UPPER_HEIGHT), (width, UPPER_HEIGHT), (0, 255, 255), thickness=2)
                        cv2.line(im0, (0, LOWER_HEIGHT), (width, LOWER_HEIGHT), (0, 255, 0), thickness=2)
                    

                        center_y = int(((output[1])+(output[3]))/2)
                        if center_y <= UPPER_HEIGHT + 1 and center_y >= LOWER_HEIGHT - 1:
                            pts_d[id].append([False, center_y, time.time(), fps])
                            if names[c] == 'car' or names[c] == 'truck' or names[c] == 'motorcycle' or names[c] == 'bus':
                                counter.append(int(id))

                        
                        if len(pts_d[id]) > 0:
                            try:
                                if pts_d[id][0][1] <= (LOWER_HEIGHT + UPPER_HEIGHT)/2 and center_y >= UPPER_HEIGHT:
                                    pts_d[id].append([False, center_y, time.time(), fps])
                                    speed(pts_d, dataset.frames, dataset.fps, id, DISTANCE)
                                elif pts_d[id][0][1] >= (LOWER_HEIGHT + UPPER_HEIGHT)/2 and center_y <= LOWER_HEIGHT:
                                    pts_d[id].append([False, center_y, time.time(), fps])
                                    speed(pts_d, dataset.frames, dataset.fps, id, DISTANCE)
                            except Exception as e:
                                print(e)
                        
                        try:
                            if id in speed_dict.keys():
                                label = f'{names[c]} {speed_dict[id][0]} KM/H'
                                
                                if names[c] == 'car' and speed_dict[id][0] >= CAR_MAX_SPEED and speed_dict[id][1] == False:
                                    speed_dict[id][1] = True
                                    frame = im1[output[1]-5:output[3]+5, output[0]-5:output[2]+5]
                                    cv2.imshow('frame', frame)
                                    filename = os.path.join(os.getcwd(), 'inference')
                                    filename = os.path.join(filename, 'tracked')
                                    filename = os.path.join(filename, f"{names[c]}{speed_dict[id][0]}{time.time()}.jpg")
                                    cv2.imwrite(filename,  frame)
                                elif names[c] == 'car' and speed_dict[id][0] >= CAR_MAX_SPEED:
                                    cv2.rectangle(im0, (int(output[0]),int(output[1])), (int(output[2]),int(output[3])), (0, 0, 255), 2)
                                elif names[c] == 'truck' and speed_dict[id][0] >= TRUCK_MAX_SPEED and speed_dict[id][1] == False:
                                    speed_dict[id][1] = True
                                    frame = im1[output[1]-5:output[3]+5, output[0]-5:output[2]+5]
                                    cv2.imshow('frame', frame)
                                    filename = os.path.join(os.getcwd(), 'inference')
                                    filename = os.path.join(filename, 'tracked')
                                    filename = os.path.join(filename, f"{names[c]}{speed_dict[id][0]}{time.time()}.jpg")
                                    cv2.imwrite(f'{names[c]}-{speed_dict[id][0]}-{datetime.now()}.jpg', frame)
                                elif names[c] == 'truck' and speed_dict[id][0] >= TRUCK_MAX_SPEED:
                                    cv2.rectangle(im0, (int(output[0]),int(output[1])), (int(output[2]),int(output[3])), (0, 0, 255), 2)
                                elif names[c] == 'motorcycle' and speed_dict[id][0] >= MOTORCYCLE_MAX_SPEED and speed_dict[id][1] == False:
                                    speed_dict[id][1] = True
                                    frame = im1[output[1]-5:output[3]+5, output[0]-5:output[2]+5]
                                    cv2.imshow('frame', frame)
                                    filename = os.path.join(os.getcwd(), 'inference')
                                    filename = os.path.join(filename, 'tracked')
                                    filename = os.path.join(filename, f"{names[c]}{speed_dict[id][0]}{time.time()}.jpg")
                                    cv2.imwrite(f'{names[c]}-{speed_dict[id][0]}-{datetime.now()}.jpg', frame)
                                elif names[c] == 'motorcycle' and speed_dict[id][0] >= MOTORCYCLE_MAX_SPEED:
                                    cv2.rectangle(im0, (int(output[0]),int(output[1])), (int(output[2]),int(output[3])), (0, 0, 255), 2)
                                elif names[c] == 'bus' and speed_dict[id][0] >= BUS_MAX_SPEED and speed_dict[id][1] == False:
                                    speed_dict[id][1] = True
                                    frame = im1[output[1]-5:output[3]+5, output[0]-5:output[2]+5]
                                    cv2.imshow('frame', frame)
                                    filename = os.path.join(os.getcwd(), 'inference')
                                    filename = os.path.join(filename, 'tracked')
                                    
                                    filename = os.path.join(filename, f"{names[c]}{speed_dict[id][0]}{time.time()}.jpg")
                                    cv2.imwrite(f'{names[c]}-{speed_dict[id][0]}-{datetime.now()}.jpg', frame)
                                elif names[c] == 'bus' and speed_dict[id][0] >= BUS_MAX_SPEED:
                                    cv2.rectangle(im0, (int(output[0]),int(output[1])), (int(output[2]),int(output[3])), (0, 0, 255), 2)
                            else:
                                label = f'{names[c]}'
                        except Exception as e:
                                label = f'{names[c]}'

                        if detection_bbox:
                            cv2.rectangle(im0, (int(output[0]),int(output[1])), (int(output[2]),int(output[3])), color, 2)
                        cv2.rectangle(im0, (int(output[0]), int(output[1]-10)), (int(output[0])+(len(label)
                                    +len(str(id)))*7, int(output[1])), color, -1)
                        cv2.putText(im0, label, (int(output[0]), int(output[1]-1)), 0, 0.4,
                                    (255, 255, 255), 1)
                        

                        total_count = len(set(counter))

                        cv2.putText(im0, "Total Vehicle Count: " + str(total_count), (0,80), 0, 1, (0,0,255), 2)

                        
                        cv2.putText(im0, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.resizeWindow('output', 1024, 768)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    raise StopIteration


            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default = [2, 3, 5, 7], help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--upper', type=int, default=700, help='Upper detection limit')
    parser.add_argument('--lower', type=int, default=350, help='Lower detection limit')
    parser.add_argument('--motorcycle', type=int, default=60, help='Max Motorcycle Speed Limit')
    parser.add_argument('--car', type=int, default=100, help='Max Car Speed Limit')
    parser.add_argument('--truck', type=int, default=80, help='Max Truck Speed Limit')
    parser.add_argument('--bus', type=int, default=80, help='Max Bus Speed Limit')
    parser.add_argument('--distance', type=int, default=5, help="Distance between two limits(m)")
    parser.add_argument('--track', action='store_true', help="show the detection track")
    parser.add_argument('--bbox', action='store_true', help="show the detection box")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
