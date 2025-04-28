import argparse
import sys
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from utils.general import check_img_size, is_ascii, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device

from PIL import Image
import numpy as np 
import re
import easyocr
try:
    reader = easyocr.Reader(['en'], gpu='cuda')
except:
    reader = easyocr.Reader(['en'])

def extract_number(image, block):
    image = Image.fromarray(image)
    image = image.crop((block[0].item(), block[1].item(), block[2].item(), block[3].item())).convert('RGB') 
    image = image.resize((image.size[0]*3, image.size[1]*3))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # img = cv2.GaussianBlur(img, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(
        img, 255, 1, 1, 55, 5)  # APPLY ADAPTIVE THRESHOLD

    # cv2.imshow('frame', imgThreshold)
    # if cv2.waitKey(0):
    #     cv2.destroyAllWindows()
    y = reader.readtext(imgThreshold)
    y = ''.join(e[1] for e in y if e[1])
    y = "".join(re.findall("[A-Z0-9]+", y))
    return y

@torch.no_grad()
def run(model,  # model.pt path(s)
        img,  # Frame
        im0, 
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        ):
    
   
    device = select_device(device)
    half &= device.type != 'cpu'
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    classify, pt = False, True
    stride, names = 64, [f'class{i}' for i in range(1000)]
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()
    if classify:
        modelc = load_classifier(name='resnet50', n=2)
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    imgsz = check_img_size(imgsz, s=stride)
    ascii = is_ascii(names)

    
    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once    
    pred = model(img, augment=augment, visualize=False)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    # Process predictions
    for i, det in enumerate(pred):
        annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
            for *xyxy, conf, cls in reversed(det):            
                c = int(cls)
                number = extract_number(im0, xyxy)
                annotator.box_label(xyxy, number, color=colors(c, True))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5/runs/train/license_plate/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='./aa.avi', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
