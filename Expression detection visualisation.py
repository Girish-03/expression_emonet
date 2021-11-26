# -*- coding: utf-8 -*-

"""# **FAN for landmarks and Emonet for emotions on video**"""

import cv2 # Image processing
import matplotlib.pyplot as plt # Plotting
from emonet2.models import EmoNet # Model for emotion detection
from emonet2.data_augmentation import DataAugmentor
import face_alignment # Facial landmark detection
import torch # PyTorch deep learning lib
import numpy as np # arrays and manipulations
from torchvision import transforms # image transformations
import time
from google.colab.patches import cv2_imshow

def load_emonet(n_expression=5,model_path="/content/emonet/pretrained/emonet_5.pth",device='cuda:0'):

    '''
    Loads EmoNet model
    
    PARAMETERS

    n_expression: Number of expressions. (5,8 for respective models)
    model_path: Path of 5 or 8 expression model or state dict. Default set for 5 expression model
    device: Cuda device. Default set to cuda:0

    RETURNS
    net: loaded model

    '''
    state_dict_path = model_path
    print(f'Loading the model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval() # Put model in evaluation mode when using on real time or test data to avoid effect of regularization measures taken during training
    return net

def draw_status(face_coord,img,metrics):

    '''
    Annotate frames with expression and VA values 

    PARAMETERS

    face_coord: coordinates of face rectangle 
    img: image frame to be annotated
    metrics: metrics to be annotated on image (valence, arousal, discrete expression)

    RETURNS

    img: Annotate image
    '''
    
    valence,arousal,expression = metrics
    # only one face is getting annotated at a time

    x,y,w,h = face_coord

    # Vertical bar location for valence with a dividing line location in middle
    vbar_xy =(x-40,y) 
    vbar_x1y1 =(x-40+10,y+h) 

    vline_start = (x-40,y+int(h/2))
    vline_end = (x-40+10,y+int(h/2))
    
    # Horizontal bar location for arousal with a dividing line location in middle
    hbar_xy =(x,y+h+30) 
    hbar_x1y1 =(x+w,y+h+30+10) 

    hline_start = (x+int(w/2),y+h+30)
    hline_end = (x+int(w/2),y+h+30+10)

    # Location of emotion text
    emotext = (int(x+w/2)-40, y-20)

    # Draw rectangle on face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255)) # face

    # Annotate text for valence, arousal and expression 
    cv2.putText(img,expression,emotext,font,1,(0,0,255),2,cv2.LINE_4);
    cv2.putText(img,'Valence',(x+int(w/2)-40,y+h+80),font,0.75,(255,0,0),2,cv2.LINE_4);
    cv2.putText(img,'Arousal',(x-120,y+int(h/2)),font,0.75,(0,255,0),2,cv2.LINE_4);

    # Draw rectangles for valence arousal bars
    cv2.rectangle(img,vbar_xy,vbar_x1y1,(0,255,0)) # vertical bar
    cv2.rectangle(img,hbar_xy,hbar_x1y1,(255,0,0)) # horizontal bar

    # Dynamic bar fill with VA strength
    vbar_fill = np.interp(arousal,[-1,1],[y+h,y])
    hbar_fill = np.interp(valence,[-1,1],[x,x+w])

    # y+int(h/2) when arousal is 0, y when arousal is 1, y+h when arousal is -1
    cv2.rectangle(img,(x-40,int(vbar_fill)),(x-40+10,y+int(h/2)),(0,255,0),cv2.FILLED) # vertical bar filling
    cv2.rectangle(img,(int(hbar_fill),y+h+30) ,(x+int(w/2),y+h+30+10),(255,0,0),cv2.FILLED) # horizontal bar filling

    # Draw lines in bars
    cv2.line(img,vline_start,vline_end,(0,0,0),2)
    cv2.line(img,hline_start,hline_end,(0,0,0),2)
          
    return img

exp_dict = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}

# creating object for face alignment network
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cuda',face_detector='dlib')

vid_path = r"./input.mp4"  # Set location of input video
vid = cv2.VideoCapture(vid_path) # Read video using opencv
write_FPS = int(vid.get(cv2.CAP_PROP_FPS)) # Fetch FPS of input video

# Preapre output video config
fourcc = cv2.VideoWriter_fourcc(*'XVID')
write_vid = cv2.VideoWriter('output.avi',fourcc, write_FPS, (int(vid.get(3)), int(vid.get(4)))) 

font = cv2.FONT_HERSHEY_SIMPLEX
device = 'cuda:0'

# Load Emonet
net = load_emonet(n_expression=5,device=device,model_path='/content/emonet/pretrained/emonet_5.pth')

# iterating on each frame
valence = []
arousal = []
expression = []

frames_read = 0
frames_wrote = 0

transform_image = transforms.ToTensor() # convert image to tensor

while (vid.isOpened()):
  ret,frame = vid.read() # Reading all frames of video and processing them further
  if ret is True:
      frames_read +=1
      
      landmarks = fa.get_landmarks(frame) # extract landmarks

      if landmarks is not None:

          frame_annotated = frame
          for i in range(len(landmarks)):
              bounding_box = np.array([landmarks[i].min(axis=0)[0], landmarks[i].min(axis=0)[1],
                                      landmarks[i].max(axis=0)[0], landmarks[i].max(axis=0)[1]] , dtype=int)
              
              (x, y, x1, y1) = bounding_box
              w = x1-x
              h = y1-y

              # Estimating emotions now
              transform_image_shape_no_flip = DataAugmentor(256, 256)
              face,land = transform_image_shape_no_flip(frame,bb=bounding_box)
              face = np.ascontiguousarray(face)

              tensor = transform_image(face).reshape(1,3,256,256)
              tensor = tensor.to(device)

              with torch.no_grad(): # context manager used in form of 'with' & turns off gradient computation as only needed when learning
                  out = net(tensor)

              metrics = (out['valence'].item(),out['arousal'].item(),exp_dict[torch.argmax(out['expression']).item()])

              valence.append(metrics[0])  # .item() coverts a single value tensor into numerical value
              arousal.append(metrics[1])
              expression.append(torch.argmax(out['expression']).item())

              # Annotating emotions on face now
              frame_annotated = draw_status(face_coord = (x,y,w,h),img = frame_annotated ,metrics = metrics)
              
          write_vid.write(frame_annotated)
          frames_wrote +=1
          
      else:
          write_vid.write(frame)
          frames_wrote +=1

  else:
    break

print(f"Frames read {frames_read} and frames wrote {frames_wrote}")

# Release objects and memory
vid.release()
write_vid.release()
cv2.destroyAllWindows()