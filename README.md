# Continuous and Discrete Expressions detection and visualisation


![Emma.gif](Emma.gif)

This project uses https://github.com/face-analysis/emonet [(Paper)](https://www.nature.com/articles/s42256-020-00280-0) to estimate the continuous and discrete emotions in a video. It utilises [Facial Alignment Network (FAN)](https://github.com/1adrianb/face-alignment) for facial landmark detection, which is used to crop the face image and further fed to EmoNet for expression detection.
The results are annotated and on every frame of the video to generate annotated videos with discrete label and changing values of valence and arousal or the continuous values of emotions.

## Installation and Usage

Clone the EmoNet repo and install FAN
```bash
!git clone https://github.com/face-analysis/emonet
!cp -a /content/emonet/emonet /content/emonet2
!pip install face-alignment

Run Expression detection visualisation.py to generate results.
Input video path can be edited in Line 109 
The 5 or 8 expression model selection can be edited in Line 121
```
