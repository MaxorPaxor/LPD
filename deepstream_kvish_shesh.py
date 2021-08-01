#!/usr/bin/env python3

import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import configparser
import pyds

fps_streams={}
LP_dict = {}

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

LIVE_VIDEO = False
SAVE_VIDEO = False


def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list

        frame_width = frame_meta.source_frame_width
        frame_height = frame_meta.source_frame_height

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 0.5)
            
            # print(obj_meta.obj_label)  # car, roadsign, lpd
            l_class = obj_meta.classifier_meta_list

            while l_class is not None:

                try:
                    class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                except StopIteration:
                    break

                l_label = class_meta.label_info_list

                while l_label is not None:

                    try:
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                    except StopIteration:
                        break

                    # LP: label_info.result_label
                    # LP Unique ID: obj_meta.object_id
                    # PROB: label_info.result_prob

                    ### Israeli car
                    if label_info.result_prob > 0.8 and \
                       label_info.result_label.isnumeric() and \
                       (len(label_info.result_label) == 7 or len(label_info.result_label) == 8):
                        obj_meta.text_params.text_bg_clr.set(0.0, 0.7, 0.0, 0.5)
                        obj_meta.text_params.font_params.font_color.set(0.0, 0.0, 0.0, 1.0)

                        if obj_meta.object_id in LP_dict:
                            if LP_dict[obj_meta.object_id][1] < label_info.result_prob:
                                LP_dict[obj_meta.object_id] = [label_info.result_label, label_info.result_prob]
                                print(LP_dict)

                    ### Any car
                    # print(label_info.result_prob)
                    # print(label_info.result_label)
                    # if label_info.result_prob >= 0.0:
                    #     obj_meta.text_params.text_bg_clr.set(0.0, 0.7, 0.0, 0.5)
                    #     obj_meta.text_params.font_params.font_color.set(0.0, 0.0, 0.0, 1.0)

                    #     if obj_meta.object_id in LP_dict:
                    #         if LP_dict[obj_meta.object_id][1] < label_info.result_prob:
                    #             LP_dict[obj_meta.object_id] = [label_info.result_label, label_info.result_prob]
                    #             print(LP_dict)

                        else:
                            LP_dict[obj_meta.object_id] = [label_info.result_label, label_info.result_prob]
                            print(LP_dict)

                    try:
                        l_label=l_label.next
                    except StopIteration:
                        break

                try:
                    l_class=l_class.next
                except StopIteration:
                    break

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        text = "Frame Number={} Number of Objects={} Vehicle_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE])
        text = text + "\n{}x{}, LP Size: {}x{}".format(frame_width, frame_height, int(frame_width*145/1920), int(frame_height*80/1080))
        for id in LP_dict:
            text = text + "\nLP: {:10s} PROB: {}".format(LP_dict[id][0], round(LP_dict[id][1]*100, 2))

        py_nvosd_text_params.display_text = text
                                                 
        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    if LIVE_VIDEO:
        print("Creating Source \n ")
        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")

        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        if not caps_v4l2src:
            sys.stderr.write(" Unable to create v4l2src capsfilter \n")


        print("Creating Video Converter \n")

        # Adding videoconvert -> nvvideoconvert as not all
        # raw formats are supported by nvvideoconvert;
        # Say YUYV is unsupported - which is the common
        # raw format for many logi usb cams
        # In case we have a camera with raw format supported in
        # nvvideoconvert, GStreamer plugins' capability negotiation
        # shall be intelligent enough to reduce compute by
        # videoconvert doing passthrough (TODO we need to confirm this)


        # videoconvert to make sure a superset of raw formats are supported
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        if not vidconvsrc:
            sys.stderr.write(" Unable to create videoconvert \n")

        # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
        nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        if not nvvidconvsrc:
            sys.stderr.write(" Unable to create Nvvideoconvert \n")

        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        if not caps_vidconvsrc:
            sys.stderr.write(" Unable to create capsfilter \n")

    else:
        # Source element for reading from the file
        print("Creating Source \n ")
        source = Gst.ElementFactory.make("filesrc", "file-source")
        # source = Gst.ElementFactory.make("uridecodebin", "file-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")

        mp4demux = Gst.ElementFactory.make("qtdemux", "mp4-demux")
        if not mp4demux:
            sys.stderr.write(" Unable to create mp4 demux \n")

        # Since the data format in the input file is elementary h264 stream,
        # we need a h264parser
        print("Creating H264Parser \n")
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parser:
            sys.stderr.write(" Unable to create h264 parser \n")

        # Use nvdec_h264 for hardware accelerated decode on GPU
        print("Creating Decoder \n")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not decoder:
            sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")


    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 \n")
    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie2:
        sys.stderr.write(" Unable to make sgie2 \n")
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    ### TEST: Trying to import to mp4
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    # Make the encoder
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    print("Creating H264 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")

    encoder.set_property('bitrate', 40000000)
    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        encoder.set_property('bufapi-version', 1)
    
    ### END TEST

    # Finally render the osd output
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    if SAVE_VIDEO:
        print("Creating FileSink \n")
        sink = Gst.ElementFactory.make("filesink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create file sink \n")
        # sink.set_property("location", "output.264")
        sink.set_property("location", "output.mp4")
        sink.set_property("async", 'false')

    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")


    if LIVE_VIDEO:
        print("Playing cam %s " %args[1])
        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        source.set_property('device', args[1])

    else:
        print("Playing file %s " %args[1])
        source.set_property('location', args[1])
        # source.set_property("uri", "file://"+args[1])
    
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    # streammux.set_property('width', 860)
    # streammux.set_property('height', 640)   
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)

    pgie.set_property('config-file-path', "./configs/trafficamnet_config.txt")
    sgie1.set_property('config-file-path', "./configs/lpd_us_config.txt")
    sgie2.set_property('config-file-path', "./configs/lpr_config_sgie_us.txt")
    sgie1.set_property('process-mode', 2)
    sgie2.set_property('process-mode', 2)

    # nvosd.set_property('display-mask', 0)
    nvosd.set_property('display-text', 1)
    nvosd.set_property('display-bbox', 1)

    if LIVE_VIDEO:
        # Set sync = false to avoid late frame drops at the display-sink
        sink.set_property('sync', False)

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read('./configs/lpr_sample_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process',
                                 tracker_enable_batch_process)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    if LIVE_VIDEO:
        pipeline.add(caps_v4l2src)
        pipeline.add(vidconvsrc)
        pipeline.add(nvvidconvsrc)
        pipeline.add(caps_vidconvsrc)
    else:
        pipeline.add(mp4demux)
        pipeline.add(h264parser)
        pipeline.add(decoder)
    
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(sgie2)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    if SAVE_VIDEO:
        pipeline.add(nvvidconv_postosd)
        pipeline.add(caps)
        pipeline.add(encoder)

    if is_aarch64():
        pipeline.add(transform)
    
    pipeline.add(sink)

    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    if LIVE_VIDEO:
        source.link(caps_v4l2src)
        caps_v4l2src.link(vidconvsrc)
        vidconvsrc.link(nvvidconvsrc)
        nvvidconvsrc.link(caps_vidconvsrc)

        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")

        srcpad = caps_vidconvsrc.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")

        srcpad.link(sinkpad)

    else:
        source.link(h264parser)
        # mp4demux.link(h264parser)
        h264parser.link(decoder)
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")

        srcpad = decoder.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")

        srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(sgie1)
    sgie1.link(sgie2)
    sgie2.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)

    if SAVE_VIDEO:
        nvosd.link(nvvidconv_postosd)
        nvvidconv_postosd.link(caps)
        caps.link(encoder)
        encoder.link(sink)

    else:
        if is_aarch64():
            nvosd.link(transform)
            transform.link(sink)
        else:
            nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

