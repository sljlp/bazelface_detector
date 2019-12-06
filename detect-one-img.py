import test
from test import FaceDetector
from test import detect
from test import cropImg
import cv2
import numpy as np
from test import expandBox

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input' , required=True)
    parser.add_argument('-c', '--cropped', default=None)
    parser.add_argument('-d', '--drew', default=None)
    parser.add_argument('-f', '--expanding_factor', default=0.3, type=float)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    det = FaceDetector('models/face_detection_front.tflite',
                       'data/anchors.csv')
    print ('reading image %s' % args.input)
    img = cv2.imread(args.input)
    boxes, keyps = detect(img,det,1.0)
    for i in range(len(boxes)):
        boxes[i] = expandBox(boxes[i],args.expanding_factor)
    print boxes
    print keyps
    if args.cropped is not None:
        if len(boxes) == 1:
            box = boxes.reshape([4])
            cropped = cropImg(img,box)
            cv2.imwrite(args.cropped, cropped)
        else:
            cropped_name = args.cropped[:-4]
            cropped_suffix = args.cropped[-3:]
            if not (cropped_suffix == 'png' or cropped_suffix =='jpg'):
                cropped_suffix = 'jpg'
            for i, box in enumerate(boxes):
                cropped = cropImg(img,box)
                cv2.imwrite(cropped_name+'_%03d.' % (i+1) +cropped_suffix,cropped)
    if args.drew is not None:
        for box in boxes:
            box = np.reshape(box,[2,2]).astype(int)
            cv2.rectangle(img,tuple(box[0]), tuple(box[1]),(0,255,0),)
        cv2.imwrite(args.drew, img)







