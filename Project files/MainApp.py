import os
import time
from tqdm import tqdm

import subprocess
import numpy as np
import cv2
import filterpy

import torch
import super_gradients as sg
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)
from super_gradients.training import models

model = models.get("yolo_nas_s", pretrained_weights="coco").cuda()
model.eval();
video_path = './test1.mp4'

cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video file")

while(cap.isOpened()):

  # read each video frame
  ret, frame = cap.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  break

cap.release()
del cap


from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.pipelines.pipelines import DetectionPipeline

# make sure to set IOU and confidence in the pipeline constructor
pipeline = DetectionPipeline(
            model=model,
            image_processor=model._image_processor,
            post_prediction_callback=model.get_post_prediction_callback(iou=0.25, conf=0.30),
            class_names=model._class_names,
        )


def get_prediction(image_in, pipeline):
  # Preprocess
  preprocessed_image, processing_metadata = pipeline.image_processor.preprocess_image(image=image_in.copy())

  # Predict
  with torch.no_grad():
      torch_input = torch.Tensor(preprocessed_image).unsqueeze(0).to('cuda')
      model_output = model(torch_input)
      prediction = pipeline._decode_model_output(model_output, model_input=torch_input)

  # Postprocess
  return pipeline.image_processor.postprocess_predictions(predictions=prediction[0], metadata=processing_metadata)

from ocsort import ocsort

tracker = ocsort.OCSort(det_thresh=0.25)

import colorsys    

def get_color(number):
    """ Converts an integer number to a color """
    # change these however you want to
    hue = number*30 % 180
    saturation = number*103 % 256
    value = number*50 % 256

    # expects normalized values
    color = colorsys.hsv_to_rgb(hue/179, saturation/255, value/255)

    return [int(c*255) for c in color]


# get frame info for tracker and video saving 
h, w = (720, 1280)
h2, w2 = h//2, w//2
# h2, w2 = 640, 640 # this degrades performance

# OCSORT automatically rescales bboxes if we inference with a diff img size
img_info = (h, w)
img_size = (h2, w2) 


tracker = ocsort.OCSort(det_thresh=0.60, max_age=10, min_hits=10)

cap = cv2.VideoCapture(video_path)

if (cap.isOpened() == False):
    print("Error opening video file")

frames = []
i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()
track_ids_op = []
track_bbox_op = []
track_hits_op = []
while(cap.isOpened()):

  # read each video frame (read time is about 0.006 sec)
  ret, frame = cap.read()

  if ret == True:

    # read image and resize by half for inference
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(og_frame,
                       (w2, h2), interpolation=cv2.INTER_LINEAR)

    # perform inference on small frame and get (x1, y1, x2, y2, confidence)
    pred = get_prediction(frame, pipeline)
    xyxyc = np.hstack((pred.bboxes_xyxy,
                      np.c_[pred.confidence]))

    # update tracker
    tracks = tracker.update(xyxyc, img_info, img_size)

    # draw tracks on frame
    for track in tracker.trackers:

        track_id = track.id
        hits = track.hits
        color = get_color(track_id*15)
        x1,y1,x2,y2 = np.round(track.get_state()).astype(int).squeeze()
        if (track_id not in track_ids_op) and (hits>10):

            crop_img = og_frame[y1:y2, x1:x2].copy()
            if(crop_img.shape[0]<32 or crop_img.shape[1]<32):
                print(';dibg')
            else:
                track_ids_op.append(track_id)
                track_hits_op.append(hits)
                print(crop_img.shape)
                filename = './images/' + str(track_id) + '.jpg'
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename , crop_img)
        else:
            if(hits>10):
                track_hits_op[track_ids_op.index(track_id)] = hits

        cv2.rectangle(og_frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(og_frame,
                  f"{track_id}-{hits}",
                  (x1+10,y1-5),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,
                  color,
                  1,
                  cv2.LINE_AA)


    # update FPS and place on frame
    current_time = time.perf_counter()
    elapsed = (current_time - start_time)
    counter += 1
    if elapsed > 1:
      fps = counter / elapsed;
      counter = 0;
      start_time = current_time;

    cv2.putText(og_frame,
                f"FPS: {np.round(fps, 2)}",
                (10,h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2,
                cv2.LINE_AA)

    # append to list
    frames.append(og_frame)

  # Break the loop
  else:
    break

# release video capture object
cap.release()
del cap




video_savepath = 'output_video.mp4'
from moviepy.editor import VideoFileClip

videoclip = VideoFileClip(video_path)
video_fps = videoclip.fps


out = cv2.VideoWriter(video_savepath,
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      video_fps,
                      (w, h))
 
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
del out

with open('track_ids.txt', 'w') as f:
    for tio in track_ids_op:
        f.write(str(tio)+"\n")
with open('track_hits.txt', 'w') as f:
    for tho in track_hits_op:
        f.write(str(tho)+"\n")


#
p1=subprocess.Popen(["python","gui.py"],shell=True)
p1.wait()
