import numpy as np
import cv2
from face_detector import FaceDetector

import os
import copy


lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

#
# def expand(box,factor):
#     box = box.reshape([2,2])
#     center = (box[1] + box[0]+1)*0.5
#     box = box + (box - center) * factor
#     return box.reshape([4])

def expandBox(box, fac=0.2):
        box = np.reshape(box, [2, 2]).astype(np.float)
        center = np.mean(box, axis=0)
        box = box + (box - center) * fac
        return box.reshape([4]).astype(int)


def cropImg(img, box):
    box = np.reshape(box, [2, 2]).astype(int)
    h, w, c = img.shape
    bw, bh = box[1] - box[0]
    start_cropped = box[0].copy()
    start_cropped = np.where(start_cropped < 0, -start_cropped, 0)
    # print start_cropped
    start_raw = box[0]
    start_raw = np.where(start_raw < 0, 0, start_raw)
    end_cropped = np.where(box[1] < [w, h], [bw, bh], [w, h] - box[0])
    end_raw = end_cropped - start_cropped + start_raw

    cropped_img = np.zeros((bh, bw, 3), dtype=np.uint8)
    # print start_raw , end_raw ,end_raw - start_raw
    # print start_cropped, end_cropped, end_cropped - start_cropped
    # print
    cropped_img[start_cropped[1]:end_cropped[1], start_cropped[0]:end_cropped[0]] = img[start_raw[1]:end_raw[1],
                                                                                        start_raw[0]:end_raw[0]]

    # cv2.imshow('cropped',cropped_img)
    # cv2.waitKey(100)
    return cropped_img


class LmkDetector():
    def __init__(self, pb_path):
        self.model = cv2.dnn.readNetFromTensorflow(pb_path)

    def predict(self, img):
            img = cv2.resize(img,(112,112))
            # img2 = img.copy()
            img2 = img[:, :, ::-1]
            blob = cv2.dnn.blobFromImage(img2) / 256.0
            assert isinstance(self.model, cv2.dnn_Net)
            self.model.setInput(blob)
            print blob.shape
            # output = np.zeros((212,),dtype=float)
            output = self.model.forward()
            return output


def detect(img_in, det, resize_factor):
    # print img.shape
    img = cv2.resize(img_in, (0, 0), fx=resize_factor, fy=resize_factor)
    det.reset_cost_time()
    rgb = img[:, :, ::-1]
    list_keypoints, list_bbox = det(rgb)

    # ps = int(np.ceil(min(rgb.shape[0], rgb.shape[1]) / 256))
    boxes = np.array([], dtype=float)
    keypts = np.array([], dtype=float)
    if list_keypoints is not None:
        boxes = np.zeros((len(list_keypoints), 4), dtype=float)
        keypts = np.zeros((len(list_keypoints), 6, 2), dtype=float)
        for idx in range(len(list_keypoints)):
            keypoints = list_keypoints[idx]
            bbox = list_bbox[idx]
            x0 = np.round(bbox[0] - bbox[2] / 2)
            y0 = np.round(bbox[1] - bbox[3] / 2)
            x1 = np.round(bbox[0] + bbox[2] / 2)
            y1 = np.round(bbox[1] + bbox[3] / 2)
            boxes[idx, :] = [x0, y0, x1, y1]
            for i in range(len(keypoints)):
                p = [keypoints[i, 0] + 0.5, keypoints[i, 1] + 0.5]
                keypts[idx, i, :] = p
    print ('cost time : %f' % det.get_cost_time())

    if len(boxes) <= 0 or len(keypts) <= 0:
        return boxes.astype(int), keypts.astype(int)

    b_shape = boxes.shape
    p_shape = keypts.shape
    boxes = boxes.reshape([-1, 2]) / [resize_factor, resize_factor]
    keypts = keypts.reshape([-1, 2]) / [resize_factor, resize_factor]

    return boxes.reshape(b_shape).astype(int), keypts.reshape(p_shape).astype(int)


def detectAll(img, BoxDet, LmkDet,box = None):
    if box is None or len(box) == 0:
        box,_ = detect(img,BoxDet,1)
    # box = box.astype(np.float32)
    lmk = np.zeros((len(box),212),dtype=int)
    for i,b in enumerate(box):
        b = expandBox(b,0.3)
        cim = cropImg(img,b)
        assert isinstance(LmkDet,LmkDetector)
        pts = LmkDet.predict(cim)
        pts = pts.reshape([-1,2])
        pts = pts * (b[2:] - b[:2]+1) + b[:2]
        lmk[i] = pts.reshape([212]).astype(int)
    return box.astype(int),lmk 

        

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgset-dir')
    parser.add_argument('--dataset')
    parser.add_argument('--new-dataset')
    parser.add_argument('--img-size')
    parser.add_argument('--lmk-model')
    parser.add_argument('--predicted-file')
    # parser.add_argument('--cropped-dir')

    return parser.parse_args()

    
def getBox(pts):
        pts = np.reshape(pts, [106, 2]).astype(np.float32)
        box = np.zeros((4,), dtype=np.float32)
        box[:2] = pts.min(axis=0)
        box[2:] = pts.max(axis=0)
        return box

def iou(box1, box2):
    box1 = np.reshape(box1, [2, 2]).astype(np.float32)
    box2 = np.reshape(box2, [2, 2]).astype(np.float32)

    wh1 = box1[1] - box1[0] + 1
    wh2 = box2[1] - box2[0] + 1

    a1 = wh1[1] * wh1[0]
    a2 = wh2[1] * wh2[0]

    inter = np.zeros((2, 2), dtype=np.float32)
    inter[0] = np.max([box1[0], box2[0]], axis=0)
    inter[1] = np.min([box1[1], box2[1]], axis=0)

    wh_inter = inter[1] - inter[0] + 1
    a_inter = wh_inter[0] * wh_inter[1]

    a1 = 0 if a1 < 0 else a1
    a2 = 0 if a2 < 0 else a2
    a_inter = 0 if a_inter < 0 else a_inter

    assert a1 > 0 or a2 > 0

    return a_inter / (a1 + a2 - a_inter)

def getBestBox(gtBox, boxes):
    gtBox = gtBox.reshape([4])
    boxes = boxes.reshape([-1, 4])
    bestIou = 0
    bestindex = -1
    for i, b in enumerate(boxes):
        tiou = iou(gtBox, b)
        if tiou > bestIou:
            bestIou = tiou
            bestindex = i
    return bestIou, bestindex
def testSmoothByP(frames,lmks,patch):
    def diffs(lmk1,lmk2):
        '''
        input shape (212) (212)
        return shape(106)
        '''
        lmk1 = lmk1.reshape([-1,106,2])
        lmk2 = lmk2.reshape([-1,106,2])
        box1 = np.zeros([len(lmk1),2,2],dtype=np.float32)
        box2 = box1.copy()
        box1[:,0] = np.min(lmk1,axis=1)
        box1[:,1] = np.max(lmk1,axis=1)
        box2[:,0] = np.min(lmk2,axis=1)
        box2[:,1] = np.max(lmk2,axis=1)
        wh1 = box1[:,1] - box1[:,0] + 1
        wh2 = box2[:,1] - box2[:,0] + 1
        wh = (wh1 + wh2) / 2
        # print wh
        # input('wh')
        edge_size = np.sqrt(wh[:,0] * wh[:,1]) 
        return np.sqrt(np.sum(np.square(lmk1 - lmk2),axis=2)).reshape([-1,106]) / edge_size
    half_patch = patch // 2
    flmk = np.zeros([len(frames),patch,106,2],dtype=np.float32)
    valid = np.ones([len(frames),patch,106],dtype=int)
    fblmk = np.copy(flmk)
    midlmk = np.copy(fblmk)
    for i in range(half_patch, len(frames)-half_patch):
        # fblmk = np.zeros([patch,106,2],dtype=np.float32)
        for k in range(-half_patch+i, half_patch + i + 1):
            if k != i:
                pframe = frames[k]
                nframe = frames[i]
                fpts = lmks[k]
                temppts = np.zeros(fpts.shape,dtype=np.float32)
                fbpts = np.copy(temppts)
                print fpts.shape
                # for ti , fpt in enumerate(fpts):
                fpts = np.reshape(fpts,[-1,1,2])
                temppts, st, err = cv2.calcOpticalFlowPyrLK(pframe,nframe,fpts,None)
                # print temppts
                fbpts, st, err =  cv2.calcOpticalFlowPyrLK(nframe,pframe,temppts,None)
                assert fbpts.shape == (len(fpts),1,2)
                fbpts = np.reshape(fbpts,[-1,2]).astype(np.float32)
                fpts = np.reshape(fpts,[-1,2])
                print fpts , fbpts
                # input()
                    # temppts[i] = 
                tempdiffs  = diffs(fpts,fbpts).reshape([106])
                flmk[i,k +half_patch-i] = fpts
                fblmk[i,k +half_patch-i] = fbpts
                midlmk[i,k +half_patch-i] = temppts.reshape([106,2])
                
            else:
                tempdiffs = np.array([0.0]*106,dtype=np.float32)
                flmk[i,k+half_patch-i] = lmks[k]
                fblmk[i,k+half_patch-i] = lmks[k]
                midlmk[i,k + half_patch - i] = lmks[i]
            valid[i,k+half_patch-i,:] = np.where(tempdiffs<0.001,1,0).astype(int)
    weighted_sum_lmks = np.zeros([len(lmks),106,2],dtype=np.float32)

    for i in range(half_patch, len(frames)-half_patch):
        tvalid = valid[i] # patch x 106
        # tmidlmk = np.transpose(midlmk[i].copy(),[2,0,1]) # 2 x patch x 106
        tmidlmk = midlmk[i].copy() #patch x 106 x 2
        tmidlmk *= tvalid.reshape([patch,106,1]) 
        # tflmk = np.transpose(tflmk,[1,2,0]) # patch x 106 x 2
        sum_tmidlmk = np.sum(tmidlmk,axis=0) # 106 x 2
        sum_tvalid = np.sum(tvalid,axis=0).astype(np.float32) # 106 
        weighted_sum_lmks[i] = sum_tmidlmk / sum_tvalid.reshape([106,1])

    return frames[half_patch:-half_patch] , weighted_sum_lmks[half_patch:-half_patch], flmk[half_patch:-half_patch] , fblmk[half_patch:-half_patch], midlmk, valid[half_patch:-half_patch]
def testSmoothByP_v3(frames,lmks_src,patch):
    # print lmks_src[patch]
    # input()

    lmks = lmks_src.copy()
    def diffs(lmk1,lmk2):
        '''
        input shape (212) (212)
        return shape(106)
        '''
        lmk1 = lmk1.reshape([-1,106,2])
        lmk2 = lmk2.reshape([-1,106,2])
        box1 = np.zeros([len(lmk1),2,2],dtype=np.float32)
        box2 = box1.copy()
        box1[:,0] = np.min(lmk1,axis=1)
        box1[:,1] = np.max(lmk1,axis=1)
        box2[:,0] = np.min(lmk2,axis=1)
        box2[:,1] = np.max(lmk2,axis=1)
        wh1 = box1[:,1] - box1[:,0] + 1
        wh2 = box2[:,1] - box2[:,0] + 1
        wh = (wh1 + wh2) / 2
        # print wh
        # input('wh')
        edge_size = np.sqrt(wh[:,0] * wh[:,1]) 
        return np.sqrt(np.sum(np.square(lmk1 - lmk2),axis=2)).reshape([-1,106]) / edge_size
    half_patch = patch // 2
    flmk = np.zeros([len(frames),patch,106,2],dtype=np.float32)
    valid = np.ones([len(frames),patch,106],dtype=int)
    fblmk = np.copy(flmk)
    midlmk = np.copy(fblmk)
    for i in range(half_patch, len(frames)-half_patch):
        # fblmk = np.zeros([patch,106,2],dtype=np.float32)
        for k in range(-half_patch+i, half_patch + i + 1):
            if k != i:
                pframe = frames[k]
                nframe = frames[i]
                fpts = lmks[k]
                temppts = np.zeros(fpts.shape,dtype=np.float32)
                fbpts = np.copy(temppts)
                print fpts.shape
                # for ti , fpt in enumerate(fpts):
                fpts = np.reshape(fpts,[-1,1,2])
                temppts, st, err = cv2.calcOpticalFlowPyrLK(pframe,nframe,fpts,None)
                # print temppts
                fbpts, st, err =  cv2.calcOpticalFlowPyrLK(nframe,pframe,temppts,None)
                assert fbpts.shape == (len(fpts),1,2)
                fbpts = np.reshape(fbpts,[-1,2]).astype(np.float32)
                fpts = np.reshape(fpts,[-1,2])
                print fpts , fbpts
                # input()
                    # temppts[i] = 
                tempdiffs  = diffs(fpts,fbpts).reshape([106])
                flmk[i,k +half_patch-i] = fpts
                fblmk[i,k +half_patch-i] = fbpts
                midlmk[i,k +half_patch-i] = temppts.reshape([106,2])
                
            else:
                tempdiffs = np.array([0.0]*106,dtype=np.float32)
                flmk[i,k+half_patch-i] = lmks[k]
                fblmk[i,k+half_patch-i] = lmks[k]
                midlmk[i,k + half_patch - i] = lmks[k]
                print midlmk
            valid[i,k+half_patch-i,:] = np.where(tempdiffs<0.001,1,0).astype(int)
        valid[i,patch//2+1:] = 0
        tmidlmk = midlmk[i].copy() # patch x 106 x 2
        # print midlmk
        # input()
        # valid[i] patch x 106
        tmidlmk *= valid[i].reshape([patch,106,1])
        # print tmidlmk[patch//2] - lmks[i]
        
        # input()
        lmks[i] = tmidlmk.sum(axis=0) / valid[i].sum(axis=0).reshape([106,1])
    # weighted_sum_lmks = np.zeros([len(lmks),106,2],dtype=np.float32)

    # for i in range(half_patch, len(frames)-half_patch):
    #     tvalid = valid[i] # patch x 106
    #     tmidlmk = np.transpose(midlmk[i].copy(),[2,0,1]) # 2 x patch x 106
    #     tmidlmk *= tvalid
    #     # tflmk = np.transpose(tflmk,[1,2,0]) # patch x 106 x 2
    #     sum_tmidlmk = np.sum(tmidlmk,axis=1) # 2 x 106
    #     sum_tvalid = np.sum(tvalid,axis=0).astype(np.float32) # 106 
    #     weighted_sum_lmks[i] = np.transpose(sum_tmidlmk / sum_tvalid,[1,0])

    return frames[half_patch:-half_patch] , lmks[half_patch:-half_patch], flmk[half_patch:-half_patch] , fblmk[half_patch:-half_patch], midlmk, valid[half_patch:-half_patch]

def testSmoothByP_v2(frames,lmks,patch):
    '''
    v2 use only pre patch videos
    and update lmk every one frame
    '''
    def diffs(lmk1,lmk2):
        '''
        input shape (212) (212)
        return shape(106)
        '''
        lmk1 = lmk1.reshape([-1,106,2])
        lmk2 = lmk2.reshape([-1,106,2])
        box1 = np.zeros([len(lmk1),2,2],dtype=np.float32)
        box2 = box1.copy()
        box1[:,0] = np.min(lmk1,axis=1)
        box1[:,1] = np.max(lmk1,axis=1)
        box2[:,0] = np.min(lmk2,axis=1)
        box2[:,1] = np.max(lmk2,axis=1)
        wh1 = box1[:,1] - box1[:,0] + 1
        wh2 = box2[:,1] - box2[:,0] + 1
        wh = (wh1 + wh2) / 2
        # print wh
        # input('wh')
        edge_size = np.sqrt(wh[:,0] * wh[:,1]) 
        return np.sqrt(np.sum(np.square(lmk1 - lmk2),axis=2)).reshape([-1,106]) / edge_size
    half_patch = patch // 2
    flmk = np.zeros([len(frames),patch,106,2],dtype=np.float32)
    valid = np.ones([len(frames),patch,106],dtype=np.float32)
    fblmk = np.copy(flmk)
    midlmk = np.copy(fblmk)
    # weighted_sum_lmks = lmks.copy()
    weight = np.array(range(patch),dtype=np.float32)+5
    weight /= np.sum(weight)
    weight = weight.astype(np.float32)
    for i in range(patch, len(frames)):
        # fblmk = np.zeros([patch,106,2],dtype=np.float32)
        for k in range(i-patch, i):
            if k != i-1:
                pframe = frames[k]
                nframe = frames[i]
                fpts = lmks[k]
                temppts = np.zeros(fpts.shape,dtype=np.float32)
                fbpts = np.copy(temppts)
                print fpts.shape
                # for ti , fpt in enumerate(fpts):
                fpts = np.reshape(fpts,[-1,1,2])
                temppts, st, err = cv2.calcOpticalFlowPyrLK(pframe,nframe,fpts,None)
                # print temppts
                fbpts, st, err =  cv2.calcOpticalFlowPyrLK(nframe,pframe,temppts,None)
                assert fbpts.shape == (len(fpts),1,2)
                fbpts = np.reshape(fbpts,[-1,2]).astype(np.float32)
                fpts = np.reshape(fpts,[-1,2])
                print fpts , fbpts
                # input()
                    # temppts[i] = 
                tempdiffs  = diffs(fpts,fbpts).reshape([106])
                flmk[i,k -i+patch] = fpts
                fblmk[i,k +patch-i] = fbpts
                midlmk[i,k +patch-i] = temppts.reshape([106,2])
                
            else:
                tempdiffs = np.array([0.0]*106,dtype=np.float32)
                flmk[i,k+patch-i] = lmks[k]
                fblmk[i,k+patch-i] = lmks[k]
                midlmk[i,k + patch - i] = lmks[i]
            print tempdiffs
            # input()
            valid[i,k+patch-i,:] = np.where(tempdiffs<0.001,1.0,0.0).astype(np.float32)
            # print valid[i,k+patch-i,:]
            # print (i, k+patch-i)
            # input()
            valid[i,k+patch-i] *= weight[k+patch - i]
        
        # print midlmk.shape
        # print valid[0,:,0]
        # print valid[5,0,0:100]
        # input()
        print valid[i].sum(axis=0).shape
        print valid[i].shape
        valid[i,:] = valid[i] / (valid[i].sum(axis=0)).reshape([106])
        print valid[i,:,0].shape
        print '----------'
        print np.sum(valid[i,:,:],axis=-2)
        input()
        midlmk[i] *= valid[i].reshape([patch,106,1])
        lmks[i] = midlmk[i].sum(axis=0)
        # lmks[i] = np.where( valid[i].reshape([, midlmk[i].sum(axis=0),lmks[0])
    # for i in range(patch, len(frames)):
    #     tvalid = valid[i] # patch x 106
    #     tmidlmk = np.transpose(midlmk[i].copy(),[2,0,1]) # 2 x patch x 106
    #     tmidlmk *= tvalid
    #     # tflmk = np.transpose(tflmk,[1,2,0]) # patch x 106 x 2
    #     sum_tmidlmk = np.sum(tmidlmk,axis=1) # 2 x 106
    #     sum_tvalid = np.sum(tvalid,axis=0).astype(np.float32) # 106 
    #     weighted_sum_lmks[i] = np.transpose(sum_tmidlmk / sum_tvalid,[1,0])

    return frames[half_patch:-half_patch] , lmks[half_patch:-half_patch], flmk[half_patch:-half_patch] , fblmk[half_patch:-half_patch], midlmk, valid[half_patch:-half_patch]

            
if __name__ == '__main__':

    det = FaceDetector('models/face_detection_front.tflite',
                       'data/anchors.csv')
    # lmkDet = LmkDetector('lmk-model.pb')
    # lmkDet = LmkDetector('models/pfld/model-refine-eye-epoch799.pb')
    # lmkDet = LmkDetector('models/pfld/model-only-jd-refine-eye-0-1152.pb')
    # lmkDet = LmkDetector('models/pfld/model-only-jd-refine-eye-cls-model.ckpt-703.pb')
    # lmkDet = LmkDetector('models/pfld/model-bazelface-celeba-model.ckpt-665.pb')
    lmkDet = LmkDetector('models/pfld/model-bazelface-refine-celeba-jd-2-model.ckpt-408.pb')

    lmk1 = np.zeros((20,213),dtype=int)

    # interpreter = det.interp_face
    # num_layer = 10000
    # for i in range(num_layer):
    #     try:
    #         detail = interpreter._get_tensor_details(i)
    #         print (i, detail['name'], detail['shape'])
    #     except:
    #         break
    #
    # exit()

    def parseGtLine(line):
        line = line.strip().split()
        line = np.asarray(line, dtype=str)
        name = line[0]
        gtPts = line[1:1 + 212].astype(np.float32)
        return name, gtPts
    
    # cap = cv2.VideoCapture('/Users/pengliu/Documents/work/fastface/smooth/scripts/video.mp4')
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("/Users/pengliu/eyeclose/test/video-v2/video2.mp4")
    # cap = cv2.VideoCapture('/Users/pengliu/Downloads/smooth-test/rocvideo/1572577233229988.mp4')
    # cap = cv2.VideoCapture('/Users/pengliu/Downloads/300VW_Dataset_2015_12_14/003/vid.avi')
    # cap = cv2.VideoCapture('/Users/pengliu/Downloads/ytcelebrity/1756_02_008_steven_spielberg.avi')
    frameIndex = 0
    frames = []
    lmks = []
    box = None
    outfile = open('/Users/pengliu/eyeclose/test/rocface.anno','w')
    # f1.close()
    while(1 and not os.path.exists('../smooth-test/frames-src.npy') or True):
        frameIndex += 1
        ret,frame = cap.read()
        if frame is None:
            break
        # if k == 113:
        #     exit()
        scale = 600.0 / max(frame.shape)
        frame = cv2.resize(frame,(0,0),fx=scale,fy=scale)
        # box,pts = detect(frame,det,1)
        # if frameIndex % 100 == 0:
        #     box = None
        box, pts = detectAll(frame,det,lmkDet)
        print box
        if len(box) <= 0:
            frameIndex -= 1
            continue
        # frames.append(np.copy(frame))
        # lmks.append(pts.reshape([106,2]))
        # if frameIndex >= 500:
        #     break
        
        # if len(pts) <= 0 and frameIndex < 50:
        #     continue

        # if lmk1[0,212] == 0:
        #     lmk1[0,:212] = pts
        #     lmk1[0,212] = 1
        #     img0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     p0 = lmk1[0,:212].reshape([-1,1,2]).astype(np.float32)
        # else:
        #     # diff = np.sqrt(np.sum(np.square(lmk1[0,:212].reshape([-1,2]) - pts.reshape([-1,2])),axis=1))
        #     # print diff[0]
        #     # nch = diff <= 4
        #     # lmk2 = lmk1[0,:212].reshape([-1,2])
        #     # pts = pts.reshape([-1,2])
        #     # # lmk2[ch] = pts[ch]
        #     # pts[nch] = lmk2[nch]
        #     # ch = np.where(nch,False,True)
        #     # lmk2[ch] = pts[ch]
        #     # lmk1[0,:212] = lmk2.reshape([212])
        #     # p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            
        #     img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     pts, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        #     # pts = pts.reshape([212])
        #     img0 = img1
        #     p0 = pts
            
            # if diff <= 4:
            #     pts = lmk1[0,:212]
            # else:
            #     lmk1[0,:212] = pts

        # lmk1[frameIndex%len(lmk1),:212] = pts.reshape([212]).astype(float)
        # lmk1[frameIndex%len(lmk1),212] = 1
        # lmk1sum = np.sum(lmk1,axis=0)
        # lmk_mean = lmk1sum[:212] / lmk1sum[212]
        # pts = lmk_mean
        
        if len(box) == 1 and False:
            # with open('rocface.anno','a') as outf:
            outfile.write("%06d.png None %s\n" %(frameIndex, ' '.join(box.reshape([-1]).astype(str))))
            cv2.imwrite("/Users/pengliu/eyeclose/test/rocface-3v/%06d.png" % (frameIndex),frame)
        for b in box:
            cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),(0,0,255),1)
        for p in pts.reshape([-1,106,2]).astype(int):
            for tp in p:
                cv2.circle(frame,tuple(tp),2,(0,255,0),-1)
        cv2.imshow('cap',frame)
        # cv2.imwrite('../smooth-test/videos/img-%d.jpg'%frameIndex, frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cap.release()
    outfile.close()
    exit()

    print ('star process ...')
   
    if not os.path.exists('../smooth-test/frames-src.npy'):
        print 'saving .npy'
        frames = np.array(frames,dtype=np.uint8)
        lmks = np.array(lmks,dtype=np.float32)
        
        # np.savetxt('frame-shape',frames.shape)
        np.save('../smooth-test/frames-src.npy',frames)
        # np.savetxt('lmk-shape',lmks.shape)
        np.save('../smooth-test/lmk-src.npy',lmks)
    else:
        # frame_shape = np.loadtxt('frame.shape').astype(int)
        frames = np.load('../smooth-test/frames-src.npy')
        # lmk_shape = np.loadtxt('lmk.shape').astype(int)
        lmks = np.load('../smooth-test/lmk-src.npy')

    raw_lmks = lmks.copy()
    frames , lmks, fpts, fbpts, midpts, valid = testSmoothByP_v3(frames,lmks, 5)
    print len(lmks), len(raw_lmks)
    # exit()
    # lmks[0:len(lmks)] = raw_lmks[5//2:len(lmks)+5//2].copy()
    frames , lmks, fpts, fbpts, midpts, valid = testSmoothByP(frames,lmks, 5)
    # for i in range(1):
        # frames , lmks, fpts, fbpts, midpts, valid = testSmoothByP(frames,lmks, 5)
        # frames , lmks, fpts, fbpts, midpts, valid = testSmoothByP(frames,lmks, 5)
        # frames , lmks, fpts, fbpts, midpts, valid = testSmoothByP(frames,lmks, 5)
        # frames , lmks, fpts, fbpts, midpts, valid = testSmoothByP(frames,lmks, 5)
    # print midpts.shape
    # input()
    for fi , (frame, lmk, fps, fbps, v , mps) in enumerate(zip(frames, lmks.astype(int),fpts, fbpts, valid, midpts)):
        frame2 = frame.copy()
        for pi, (fp , fbp, mp) in enumerate(zip(fps, fbps, mps)):
            if pi != 2: 
                continue
            frame3 = frame2.copy()
            for p, p2,p3 in zip(fp.astype(int),fbp.astype(int),mp.astype(int)):
                cv2.circle(frame3,tuple(p),1,(0,255,255),-1)
                cv2.circle(frame3,tuple(p2),1,(255,0,0),-1)
                cv2.circle(frame3,tuple(p3),2,(0,255,0),1)
            cv2.imshow('frame2',frame3)
            # k = cv2.waitKey()
            # if k == 113:
                # exit()
        # frame2 = frame.copy()
        # for fbp in fbps:
        #     for p in fbp.astype(int):
        #         cv2.circle(frame2,tuple(p),2,(0,255,255),-1)
        #     cv2.imshow('frame2',frame2)
        for p in lmk:
            cv2.circle(frame, tuple(p),2,(0,255,255),-1)
        print v
        cv2.imshow('img',frame)
        cv2.imwrite('../smooth-test/videos/frame-%d.jpg' % (fi) , frame)
        cv2.waitKey(100)
    exit()

    ious = []

    args = parse_arguments()

    args.dataset = '/lp/bazelface/new-dataset-test-sample10000.txt'
    args.imgset_dir = '/lp/fastface.6.20/dataset/imgset'
    args.new_dataset = '/lp/bazelface/new-new-dataset-test-sample10000.txt'
    args.img_size = 450
    args.lmk_model = '/lp/pfld/output_dir/model-bazelface-refine-celeba-jd-2/model-408.pb'
    args.predicted_file = 'predicted_file-by-py.txt'
    lmkDet = LmkDetector(args.lmk_model)


    lmk_file = args.dataset
    img_dir = args.imgset_dir
    drew_dir = 'drew-dir'
    drew_dir_expanded = "drew_dir_expanded"

    facenum0 = 'facenum0'
    if not os.path.exists(facenum0):
        os.mkdir(facenum0)
    ioult0_5 = 'ioult0.5'
    if not os.path.exists(ioult0_5):
        os.mkdir(ioult0_5)
    if not os.path.exists(drew_dir):
        os.mkdir(drew_dir)
    if not os.path.exists(drew_dir_expanded):
        os.mkdir(drew_dir_expanded)

    print args.new_dataset
    # print args.new_dataset
    new_dataset_file = open(args.new_dataset, 'w')

    predicted_file = open(args.predicted_file, 'w')

    for line in open(lmk_file).readlines():
        name, pts = parseGtLine(copy.copy(line))

        imgpath = img_dir + '/' + name
        # if os.path.exists(imgpath):
        #     continue
        # print ('no sunch path %s' % name)
        # break
        img = cv2.imread(imgpath)
        img_for_cropping = img.copy()
        resize_factor = float(args.img_size) / 450.0
        box, kpts = detect(img, det, resize_factor)
        # lmks = np.zeros([len(box.reshape(-1, 4)), 212], dtype=float)

        predicted_file.write(name)
        for i, b in enumerate(box.reshape([-1, 4])):

            b = expandBox(b, 0.3)
            croppedimg = cropImg(img, b)
            factor = (b[2:] - b[:2] + 1)/112
            resized_img = cv2.resize(croppedimg, (112, 112))
            lmk = lmkDet.predict(resized_img)
            wh = b[2:] - b[:2] + 1
            lmk = lmk.reshape([-1, 2]) * wh + b[:2]
            # lmks[i] = lmk.reshape([212])
            predicted_file.write(
                ' '+" ".join(b.astype(str))+' ' + ' '.join(lmk.reshape(212).astype(str)))
        predicted_file.write('\n')

        print ('face num %d : %s' % (len(box), name))

        if len(box) == 0:
            if not os.path.islink(facenum0 + '/' + name) and not os.path.isfile(facenum0 + '/' + name):
                os.symlink(os.path.abspath(imgpath), facenum0 + '/' + name)
            continue
        gtBox = getBox(pts)
        bestIou, bestIndex = getBestBox(gtBox, box)

        if bestIou < 0.5:
            print ("iou < 0.5 %s" % name)

        bestBox = expandBox(box[bestIndex], 0.3)
        cv2.rectangle(img, tuple(bestBox[:2]),
                      tuple(bestBox[2:]), (0, 0, 255), 3)
        gtBox = expandBox(gtBox)
        cv2.rectangle(img, tuple(gtBox[:2]), tuple(gtBox[2:]), (255, 0, 0), 1)

        for p in pts.reshape([-1, 2]).astype(int):
            cv2.circle(img, tuple(p), 1, (0, 255, 0), 1, -1)

        cv2.imwrite(drew_dir_expanded + '/' + name, img)

        if bestIou > 0.5:
            line = line.strip().split()
            line[1 + 212:1 + 212 + 4] = bestBox.astype(str)
            line = ' '.join(line) + '\n'
            new_dataset_file.write(line)

    new_dataset_file.close()
    predicted_file.close()
