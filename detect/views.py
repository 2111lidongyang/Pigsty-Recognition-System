import random
import re
import paho.mqtt.client as mqtt
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from .models.common import DetectMultiBackend
from .utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from .utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from .utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path
import torch


def home(request):
    """
    # auther: 李东阳
    用户申请访客视图函数
    :param request:
    :return:
    """
    return render(request, 'detect/home1.html')


class MqttSubscriber:
    def __init__(self, mqtt_broker, mqtt_port, mqtt_keepalive, client_id, topic):
        self.client = mqtt.Client(client_id)
        self.topic = topic
        self.received_messages = []
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(mqtt_broker, mqtt_port, mqtt_keepalive)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT successfully!")
            self.client.subscribe(self.topic)
        else:
            print("Failed to connect, return code {0}".format(rc))

    def on_message(self, client, userdata, message):
        print(f"Received message: {message.payload.decode()}")
        self.received_messages.append(message.payload.decode())

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()


def getdetectdata(request):
    client_id = f'python-mqtt-subscribe-{random.randint(0, 1000)}'  # 可自定义，但要注意客户端id不能重复
    mqtt_broker = 'broker.emqx.io'
    mqtt_port = 1883
    mqtt_keepalive = 600
    topic = '###'

    subscriber = MqttSubscriber(mqtt_broker, mqtt_port, mqtt_keepalive, client_id, topic)
    try:
        a = 0
        while True:
            if a > 1000:
                break
            if len(subscriber.received_messages) != 0:
                a += 1
                break
        # 在这里添加处理接收到的消息的逻辑，例如：
        # message = input("Enter your message: ")
        # subscriber.publish_message('收到')

    except BaseException as e:
        print('error:', str(e))
    finally:
        data = subscriber.received_messages
        print('data', data)
        subscriber.disconnect()

    # 去除大括号，并编写一个正则表达式来匹配键和值，包括最后一个没有分号的键值对
    # 注意：这个正则表达式假设键和值之间只有一个冒号，且值不包含分号
    matches = re.findall(r'(\w+):([^;]+)(?=;)', str(data[0]).strip('{}'))

    # 由于最后一个键值对后面没有分号，我们需要特别处理它
    # 使用一个非贪婪匹配来找到最后一个键值对
    last_match = re.search(r'(\w+):([^;]+)$', str(data[0]).strip('{}'))
    if last_match:
        matches.append(last_match.groups())

        # 将捕获的键和值转换为字典，并根据需要转换值的类型
    dictionary = {key: float(value) if '.' in value else int(value) for key, value in matches}
    print(dictionary)
    res = []
    try:
        res.append(dictionary['temperature'])
        res.append(dictionary['humidity'])
        res.append(dictionary['stat'])
        res.append(dictionary['people'])
        res.append(dictionary['ldrValue'])
        res.append(dictionary['MQtemp'])
        ress = []
        ress.append(res)
        print('resresrse', ress)
    except:
        return JsonResponse({"msg": False, "data": 'error'})
    return JsonResponse({"msg": True, "data": ress})


@smart_inference_mode()
def run(
        weights=r"detect\best.pt",  # model path or triton URL
        source=r"detect\test.mp4",  # file/dir/URL/glob/screen/0(webcam)
        data=r"detect\coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=r'detect/runs',  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            s += "%gx%g " % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                # Stream results
            im0 = annotator.result()

            pig_num = len(det)
            cv2.putText(im0, f"Number of Pigs: {pig_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2)
            yield im0


from django.http import StreamingHttpResponse


def main():
    for frame in run():
        ret, res_image = cv2.imencode('.jpeg', frame)
        if ret:
            # 转换为byte类型的，存储在迭代器中
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + res_image.tobytes() + b'\r\n')


def pig(request):
    # opt = parse_opt()
    return StreamingHttpResponse(main(), content_type='multipart/x-mixed-replace; boundary=frame')
