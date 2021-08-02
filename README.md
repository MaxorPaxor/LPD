# r6

## Overview
Licsense plate detection project based on Nvidia's Deepstream


<p align="center">
<img src="images/lpr1.png" width="860" height="480">
</p>
<h5 align="center">Testing on Israeli license plates</h5>

## Requirements:

Nvidia Deepstream

Quickstart guide for Jetson: 
https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html

## References:

Deepstream Python Apps Examples:
https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

Deepstream LPR Cpp:
https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app

Deepstream LPR Python:
https://github.com/preronamajumder/deepstream-lpr-python-version

## Usage:

Run the following script:
```
python deepstream_kvish_shesh.py <input_video.264>
```
Currently the pipeline only supports .264 streams, in order to create .264 stream file from an .mp4 container file using ffmpeg:

```
ffmpeg -i output.mp4 -codec:v libx264 -aud 1 output.264
```

Using live video from an USB camera requires changing the following flag in `deepstream_kvish_shesh.py`:

```python
LIVE_VIDEO = True
```

and then: 
```
python deepstream_kvish_shesh.py /dev/video0
```

Saving the video to an .mp4 file (Currently outputs to output.mp4 at the main directory)

```python
SAVE_VIDEO = True
```

## Pipeline Overview:

The script `deepstream_kvish_shesh.py` contains two core functions:

`main` -  is the function in which the GStreamer pipeline is created.
For a live feed from an USB camera the pipeline contains the following elements:


#| Element        | Properties     | Comments  |
-| -------------- |----------------|-----------|
1| v4l2src        | ('device', args[1]) | Source |
2| capsfilter     | ('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1")) | v4l2src capsfilter |
3| videoconvert   |                | videoconvert to make sure a superset of raw formats are supported |
4| nvvideoconvert |  | nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API) |
5| capsfilter     | ('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)")) | nvvideoconvert capsfilter |
6| nvstreammux    | ('width', 1920), ('height', 1080), ('batch-size', 1), ('batched-push-timeout', 400000) | Create nvstreammux instance to form batches from one or more sources. |
7| nvinfer        | ('config-file-path', "./configs/trafficamnet_config.txt") | pgie - Deepstream inference element for primary detector (Cars, Bikes, Pedestrians, Signs) |
8| nvtracker      | Tracker properties are parsed from './configs/lpr_sample_tracker_config.txt' | Tracker - Deepstream element to track detected objects |
9| nvinfer        | ('config-file-path', "./configs/lpd_us_config.txt"), ('process-mode', 2) | sgie1 - Deepstream inference element for secondary detector (Licsense Plate Detection) |
10| nvinfer       | ('config-file-path', "./configs/lpr_config_sgie_us.txt"), ('process-mode', 2) | sgie2 - Deepstream inference element for OCR (Licsense Plate Recognition) |
11| nvvideoconvert|                | Use convertor to convert from NV12 to RGBA as required by nvosd |
12| nvdsosd       | ('display-text', 1), ('display-bbox', 1) | Create OSD to draw on the converted RGBA buffer |
13| nvegltransform|                | Deepstream element required in aarch64 for nveglglessink |
14| nveglglessink | ('sync', False) | Sink - render the osd output |

`osd_sink_pad_buffer_probe` - This function acts as a probe that is added to the sink pad of the osd element. The main purpose of this probe function is to handle metadata recieved from the pipeline (Number of cars, Number of licsense plates, LP text etc').

This metadata has a standard structure that can be seen in the following diagram:
<p align="center">
<img src="images/DS_plugin_metadata.png">
</p>

**Information Extraction:**

Our prob recieves BatchMeta and loops through all FrameMeta objects, in every FrameMeta objects it loops through ObjectMeta objects, and in every ObjectMeta object it loops through ClassifierMeta objects and extracts information in which we are interested.

For example, if we are intersted in a specific LP number and probability, we will extract it through label_info attribute in NvDsClassifierMeta object:

```python
label = label_info.result_label
prob = label_info.result_result_prob
```

**Metadata Visualization:**

The visualization is done via NvDsDisplayMeta object, both custom and out of the pipeline data.

For more information:

https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html

## Neural Network Review:

### inference

The input frame goes through 3 neural networks in the pipeline. 
The inference is done via nvinfer elements (Element number 7, 9, 10).
Each nvinfer element has a 'config-file-path' property which points to a config file.

Every config file points to a NN file, for example, our main detector's config file `./configs/trafficamnet_config.txt` has the following lines:

```
tlt-encoded-model=../models/trafficcamnet/resnet18_trafficcamnet_pruned.etlt
labelfile-path=../models/trafficcamnet/labels.txt
int8-calib-file=../models/trafficcamnet/trafficnet_int8.bin
model-engine-file=../models/trafficcamnet/resnet18_trafficcamnet_pruned.etlt_b1_gpu0_int8.engine
```

`tlt-encoded-model` is the original model downloaded from Nvidia's catalog, this model is pruned.

`labelfile-path` contains the labels of the network. Downloaded from the catalog.

`int8-calib-file` is a calibration file required for creating an INT8 TRT engine file. Also downloaded from the catalog.

`model-engine-file` is a TRT engine file, created automaticly on the first run.

More properties can be defined in the config file line `infer-dims`, `batch-size`, `network-mode` (FP32, INT8, FP16) and more.

### Models

**Main Detector**

The main detector is TrafficCamNet Model which is NVIDIA DetectNet_v2 detector with ResNet18 as a feature extractor. The model is pretrained to detect and classify 4 catogeries: car, persons, road signs and two-wheelers.

https://ngc.nvidia.com/catalog/models/nvidia:tlt_trafficcamnet

**Licsense Plate Detector**

The Secondary detector LPDNet, which is has the same architecture as the main detector, is pretrained to detect a LP object from a car images.

https://ngc.nvidia.com/catalog/models/nvidia:tlt_lpdnet

**Licsense Plate Recognition**

The last NN Element in the pipeline is the LPRNet, which aims to recognize characters in license plates from cropped RGB license plate images.

https://ngc.nvidia.com/catalog/models/nvidia:tlt_lprnet

DS5.0.1 gst-nvinfer cannot generate TRT engine for LPR model, so generate it with tlt-converter

```
./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
           ../models/LP/LPR/us_lprnet_baseline18_deployable.etlt -t fp16 -e ../models/LP/LPR/lpr_us_onnx_b16.engine
```
In addition, custom parser for lpr must be built:
```
cd nvinfer_custom_lpr_parser
make
cd ..
```

## Transfer Learning