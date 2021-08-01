import numpy as np
import cv2
import sys

SECONDS = 5
FPS = 30.0

args = sys.argv
file = args[1]
frame = cv2.imread(file)

if len(args) > 2:
    new_W, new_H = int(args[2]), int(args[3])
    frame = cv2.resize(frame, (new_W, new_H))

H, W , C = frame.shape

### Define the codec and create VideoWriter object

### XVID
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 30.0, (2048,1365))

### MP4
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# new_file = 'output_{}_{}.mp4'.format(W, H)
# out = cv2.VideoWriter(new_file,fourcc, FPS, (W,H))

### H264
fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
new_file = 'output_{}_{}.264'.format(W, H)
out = cv2.VideoWriter(new_file, fourcc, FPS, (W,H))

### Save video
frames = int(FPS * SECONDS)
for i in range(frames):
    out.write(frame)

### Save image
# new_image = 'output_{}_{}.jpg'.format(W, H)
# cv2.imwrite(new_image, frame) 
        
### Release everything if job is finished
out.release()
print("Done")

# Create .264 video steam out of .mp4
# ffmpeg -i output.mp4 -codec:v libx264 -aud 1 output.264

# Cut .mp4 video
# ffmpeg -i tel_aviv_1080.mp4 -ss 00:00:00 -t 00:00:10 -async 1 tel_aviv_1080_cut.mp4
