import argparse
from collections import defaultdict, deque, OrderedDict

import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams
from ultralytics.utils.torch_utils import time_sync
from api.keyclipwriter import KeyClipWriter as kcw_custom
import supervision as sv
import sys
import numpy as np
import pandas as pd
import logging
import math
from datetime import datetime
import shutil
from opcua import Client
import gc


def set_logging():
    # Sets level and returns logger
    # if is_kaggle() or is_colab():
    #     for h in logging.root.handlers:
    #         logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    # rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    # level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    name = 'Road Safety v2'
    level = logging.INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)

# logging_tree.printout()
logging.getLogger('ultralytics').setLevel(logging.WARNING)
# logging.getLogger().removeHandler(logging.getLogger('ultralytics'))
# LOGGER = logging.getLogger('Yolov8 Test').setLevel(logging.INFO)
# # LOGGER = logging.basicConfig(level=logging.WARNING,force=True)
# LOGGER.setLevel(level=logging.INFO)
set_logging()
LOGGER = logging.getLogger('Road Safety v2')
# LOGGER_handler.setLevel(logging.INFO)
# LOGGER = LOGGER_handler.addHandler(LOGGER_handler)

CLASS_NAME = {
    0 : 'Person',
    2 : 'Car',
    3 : 'Motorbike',
    5 : 'Bus',
    7 : 'Truck',
}

L1 = ['58','07','28','75','71','25','15','19','35','65','34']
L2 = ['167','166']
B1 = ['10']

TRANSFROM = {
    #Cam Id : SOURCE np.array([[Top Left], [Top Right], [Bottom Right], [Bottom Left]]) , TARGET (Width (cm), Length (cm))
    #Top Side (Smoking Lane)
    '58' : [np.array([[442,169], [769,166], [1572, 564], [348, 556]]),(600,6327)], # Done Calibration
    '58-1' : [np.array([[442,169], [769,166], [1572, 564], [348, 556]]),(600,6094)], # Done Calibration
    '71' : [np.array([[107,234], [754,202], [2160,650], [232,991]]),(600,2780)], # Done Calibration
    '71-1' : [np.array([[107,234], [754,202], [2160,650], [232,991]]),(600,2493)], # Done Calibration
    '75' : [np.array([[1366,118], [1820,145], [1327,898], [-150,710]]),(600,4040)], # Done Calibration
    '75-1' : [np.array([[1366,118], [1820,145], [1327,898], [-150,710]]),(600,3878)], # Done Calibration

    #Bottom Side (Staff Entrance Lane)
    '15' : [np.array([[1486,221], [1704,435], [569,876], [399,162]]),(600,1939)], # Done Calibration 
    '07' : [np.array([[278,288], [629,306], [1900,880], [573,910]]),(600,3878)], # Done Calibration
    '25' : [np.array([[1056,332], [1312,345], [1089,904], [-240,740]]),(600,6925)], # Done Calibration
    '35' : [np.array([[137,331], [717,245], [2100,635], [1106,1026]]),(600,1662)], # Camera Moved

    #Security Entrance
    '65' : [np.array([[1290,311], [1664,353], [1152,1029], [196,800]]),(600,3047)], # Done Calibration 


    '28' : [np.array([[105,322], [693,219], [1920,620], [531,1048]]),(600,2216)], # Done Calibration

    '19' : [np.array([[0,0], [0,0], [0,0], [0,0]]),(6,19)], #Blocked View

    # B1 Ramp
    '10' : [np.array([[779,272], [1370,272], [1700,850], [241,834]]),(600,2000)], # Ramp

    # Loading bay
    '34' : [np.array([[688,280], [1419,295], [1898,855], [103,837]]),(100,3000)], # Done Calibration

    # L2 Corridor
    '166' : [np.array([[1146,149], [1311,157], [1433,970], [698,970]]),(177,2650)],
    '167' : [np.array([[1146,149], [1311,157], [1433,970], [698,970]]),(177,2650)],

    #PIE Side
}

RTSP = {
    # Cam Id to RTSP Url
    # Cam Id : RTSP Link
    '58':'rtsp://admin:Admin4321@172.30.40.161/', #L1 Driveway from Power Station
    '07':'rtsp://admin:Admin4321@172.30.40.80/', #L1 Driveway outside Lobby
    '28':'rtsp://admin:Admin4321@172.30.40.101/', #L1 Driveway outside Hazard Storage
    '75':'rtsp://admin:Admin4321@172.30.40.148/', #L1 Driveway at Smoking Area
    '71':'rtsp://admin:Admin4321@172.30.40.144/', #L1 Driveway between Smoking Area and Power Station
    '25':'rtsp://admin:Admin4321@172.30.40.98/', #L1 Driveway Outside Loading Area
    '15':'rtsp://admin:Admin4321@172.30.40.87/', #L1 Driveway outside Staff Entrance
    '19':'rtsp://admin:Admin4321@172.30.40.92/', #L1 MS Chip Comp (Obstructed)
    '35':'rtsp://admin:Admin4321@172.30.40.108/', #L1 Outside Loading Bay Left View
    '65':'rtsp://admin:Admin4321@172.30.40.138/', #L1 Guard House A Driveway
    '10':'rtsp://admin:Admin4321@172.30.40.50/', #B1 Carpark Ramp
    '137':'rtsp://admin:Admin4321@172.30.40.137/', #Lift
    '20':'rtsp://admin:Admin4321@172.30.40.93/', #L1 Road Outside Chip Comp
    '34':'rtsp://admin:Admin4321@172.30.40.107/', #L1 Loading Space
    '140':'rtsp://admin:Admin4321@172.30.40.140/', #L1 Guard House A
    '166':'rtsp://admin:Admin4321@172.30.40.166/', #L2 Corridor
    '167':'rtsp://admin:Admin4321@172.30.40.167/', #L2 Corridor 
}
CLASS_COLOR = {
    'Person':(128,0,128),
}
SHOW_DEBUG = True
IFMS = False #laptop dir or IFMS pc dir
speed_limit = 3
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=False,
        default = 'rtsp://admin:Admin4321@172.30.40.161/',
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=False,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.6,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.9, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--cam",required=True, default=0, type=str, help="Camera ID Number"
    )

    return parser.parse_args()

mouse_coordinate = []

def click_event(event,x,y,flags,params):
    global mouse_coordinate
    global cv_scale
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f'Mouse Coordinate ({x},{y})')
        # mouse_coordinate = [x, y]
        mouse_coordinate = [int(x*cv_scale), int(y*cv_scale)]

def draw_map_rect(image,source,distance=0):
    # Draw the rectangle on the left side
    # x1 = video_info.width - 120
    x1 = 20
    # x2 = video_info.width - 20
    x2 = 120
    # y1 = rectangle_top
    # y2 = rectangle_bottom
    if source[0][1]<source[1][1]:
        y1 = source[0][1]
    else:
        y1 = source [1][1]
    if source[3][1]>source[2][1]:
        y2 = source[3][1]
    else:
        y2 = source [2][1]
    distance = f'{(distance[1]/100):.1f}m' #distance in cm
    cv2.rectangle(image, (x1, y1), (x2,y2), (0,0,0), 3)
    cv2.rectangle(image, (x1, y1), (x2,y2), (255,255,255), 1)
    cv2.putText(image,'0m',(x1+15,y1+30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 3,fontScale = 1,color=(0,0,0))
    cv2.putText(image,'0m',(x1+15,y1+30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))
    cv2.putText(image, distance,(x1+5,y2-20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 3,fontScale = 1,color=(0,0,0))
    cv2.putText(image, distance,(x1+5,y2-20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))
    return image

def draw_map_point(image,coordinate,source,target,id,class_name):

    # print(coordinate)
    if source[0][1]<source[1][1]:
        y_top = source[0][1]
    else:
        y_top = source [1][1]
    if source[3][1]>source[2][1]:
        y_btm = source[3][1]
    else:
        y_btm = source [2][1]
    scaling_factor_y = (y_btm - y_top)/target[1]
    y_shift = y_top
    # y_shift = 0

    line1 = ((source[0][0],source[0][1]),(source[3][0],source[3][0]))
    line2 = ((source[1][0],source[1][1]),(source[2][0],source[2][0]))

    m1 = (source[3][1]-source[0][1])/(source[3][0]-source[0][0])
    m2 = (source[2][1]-source[1][1])/(source[2][0]-source[1][0])

    b1 = line1[0][1] - m1 * line1[0][0]
    b2 = line2[0][1] - m2 * line2[0][0]

    y_line1 = m1 * coordinate[0] + b1
    y_line2 = m2 * coordinate[0] + b2

    x_line1 = (coordinate[1]-b1)/m1
    x_line2 = (coordinate[1]-b2)/m2

    vertical_distance = abs(y_line1-y_line2)
    horizontal_distance = abs(x_line1-x_line2)
    
    scaling_factor_x = 100/target[0]

    x_shift = 20
    scaled_coordinate = (int(coordinate[0]*scaling_factor_x+x_shift),int(coordinate[1]*scaling_factor_y+y_shift))
    # scale_info = f' #{id} {coordinate} {scaled_coordinate}'
    scale_info = f' #{id}'
    cv2.putText(image, scale_info, scaled_coordinate,fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 4,fontScale = 1,color=CLASS_COLOR.get(class_name))
    cv2.putText(image, scale_info, scaled_coordinate,fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))
    cv2.circle(image,scaled_coordinate,5,(255,255,255),3,3)
    cv2.circle(image,scaled_coordinate,3,CLASS_COLOR.get(class_name),3,3)
    return image


class DefaultDequeOrderedDict(OrderedDict):
    def __init__(self, max_length, *args, **kwargs):
        self.max_length = max_length
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        self[key] = deque(maxlen=self.max_length)
        return self[key]
    
def activate_tag_2(source_split,offence):
    tag_object = f"Cam{source_split}"
    if offence==0:
        tag_object += " Speeding"
    else:
        tag_object += " Overtime Parking"
    # tag_object = " ".join(tag_object.split()[0:3])
    # print(f" Tag Object: {tag_object}")
    # LOGGER.debug(f'{tag_object} Tag Activated')
    return tag_object


##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################

if __name__ == "__main__":

    args = parse_arguments()

    if np.isin(args.cam,L1):
        floor = 'L1'
        unit = 'kmh'
    elif np.isin(args.cam,L2):
        floor = 'L2'
        unit = 'm/s'
    elif np.isin(args.cam,B1):
        floor = 'B1'
        unit = 'kmh'
    else:
        LOGGER.warning(f'Cam ID-{args.cam} not in Floor List ')
        floor = 'NA'
        unit = 'NA'
    LOGGER.info(f'Road Safety C-{floor}-{args.cam} Started')

    # Initialize Clip Recording 
    kcw = kcw_custom(bufSize=64)
    consecFrames = 0
    record_video_flag = False

    # Initialize Yolo Model and RTSP
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    LOGGER.info(f'Resolution-{video_info.width}x{video_info.height} @ {video_info.fps}fps ')
    model = YOLO("yolov8n.pt")
    smoother = sv.DetectionsSmoother(length=5)

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    scale = 0.5

    wh = [int(video_info.resolution_wh[0]*scale),int(video_info.resolution_wh[1]*scale)]

    thickness = sv.calculate_optimal_line_thickness(
        # resolution_wh=video_info.resolution_wh
        resolution_wh=wh
    )
    # text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_LEFT,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    selected_classes = [0, 2, 3, 5, 7] # [person, car, motorbike, bus, truck]

    source,target = TRANSFROM.get(str(args.cam))
    TARGET = np.array(
        [
            [0, 0],
            [target[0] - 1, 0],
            [target[0] - 1, target[1] - 1],
            [0, target[1] - 1],
        ]
    )

    polygon_zone = sv.PolygonZone(
        polygon=source, #frame_resolution_wh=video_info.resolution_wh
    )

    ZONE_COLORS = sv.Color(r=255,g=0,b=0)
    # zone = sv.PolygonZone(polygon=source, frame_resolution_wh=wh)
    zone_annotator = sv.PolygonZoneAnnotator(
            zone=polygon_zone,
            color=ZONE_COLORS,
            thickness=thickness,
            text_thickness=thickness * 2,
            text_scale=text_scale * 2,
        )
    
    view_transformer = ViewTransformer(source=source, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    speed_dict = DefaultDequeOrderedDict(max_length=video_info.fps*4)

    cam = args.cam
    rtsp = RTSP.get(str(args.cam))
    cap = cv2.VideoCapture(rtsp)
    rt_fps = video_info.fps

    scaled_mouse_point = []

    # # IFMS Python Client L2 PC
    # client = Client("opc.tcp://172.30.32.221:3290/borderlessSecurity/server/")

    # # IFMS Python Client L5 PC
    client = Client("opc.tcp://172.30.32.231:3290/borderlessSecurity/server/")

    # # Initialize OPCUA connection
    client.connect()
    client.load_type_definitions()
    LOGGER.info("OPC Client Connection Success")

    uri = "BorderlessSecurity"
    idx = client.get_namespace_index(uri)
    root = client.get_root_node()

    try:
        cam_tag_speeding = root.get_child(["0:Objects", "2:Cameras", "2:Cam"+str(args.cam)+" Speeding"])
        cam_tag_speeding.set_value('0')
        cam_tag_overtime_parking = root.get_child(["0:Objects", "2:Cameras", "2:Cam"+str(args.cam)+" Overtime Parking"])
        cam_tag_overtime_parking.set_value('0')
        cam_tag_speeding_count = root.get_child(["0:Objects", "2:Cameras", "2:Cam"+str(args.cam)+" Speeding Count"])
        cam_tag_speeding_count.set_value('0')
        cam_tag_overtime_parking_count = root.get_child(["0:Objects", "2:Cameras", "2:Cam"+str(args.cam)+" Overtime Parking Count"])
        cam_tag_overtime_parking_count.set_value('0')
    except Exception as e:
        # print(f"Tag Alert Comms Error {e}")
        LOGGER.exception(f"{datetime.now()}: Cam{str(args.cam)} Tag Error")

    #Create a new DF
    df = pd.DataFrame(columns=["Date","Time","Offence","Class","Number Plate","Speed","Camera Location","Duration"])

    while True:
        t0 = time_sync()
        success, frame = cap.read()
        if success:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections[np.isin(detections.class_id, selected_classes)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)
            detections = smoother.update_with_detections(detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            # print(f'Points {points}')
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [x, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append([x,y])

            labels = []
            annotated_frame = frame.copy()

            # Annotate Map
            MAP = True
            if MAP: 
                annotated_frame = draw_map_rect(annotated_frame,source,target)

            # Refresh ID list after each frame 
            tracker_id_list = [] #create a list for currently detected id on frame
            # tracker_id_list = DefaultDequeOrderedDict(max_length=5) #create a list for currently detected id on frame

            for detect in detections: #for each detected object in one frame
                _, _, confidence, class_id, tracker_id, _ = detect
                tracker_id_list.append(tracker_id)
                # tracker_id_list.append([tracker_id,class_id,time_sync()])
                if len(coordinates[tracker_id]) < video_info.fps / 2: #if object is tracked for less than 1/2 of fps, label object with ID
                    labels.append(f"#{tracker_id}")
                else: 
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    # distance = abs(coordinate_start - coordinate_end)
                    distance = math.dist(coordinate_start,coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps 
                    speed = (distance/100) / time
                    speed_dict[tracker_id].append(speed) # append the speed of object to speed_dict with tracker_id as key
                    # tracking_dict.append(tracker_id)
                    class_name = CLASS_NAME.get(int(class_id))
                    conf = round(confidence*100)
                    # labels.append(f"#{tracker_id} {class_name} {speed:.2f}{unit} {conf}% {coordinate_start} {coordinate_end} {distance:.1f} {time}")
                    labels.append(f"#{tracker_id} {class_name} {speed:.2f}{unit} {conf}% {coordinate_start} {coordinate_end}")
                    annotated_frame = draw_map_point(annotated_frame,coordinates[tracker_id][0],source,target,tracker_id,class_name)
            
            for index, (key, value) in enumerate(speed_dict.items()):
            # print(f"Index: {index}, Key: {key}, Value: {value}")
                value_arr = np.array(value)
                mean = np.mean(value_arr)
                max = np.max(value_arr)
                min = np.min(value_arr)
                std = np.std(value_arr)
                count = 0
                if max > speed_limit: #if max speed of vehicle crosses speed limit
                    # speeding_dict[key].append([max, class_name])
                    # speeding_dict[key].append([max])
                    count = sum(1 for num in value if num > speed_limit) #count the sum of number of frames vehicle exceeding max speed
                    if count > int(video_info.fps*2):
                        LOGGER.info(f'Speeding Detected ID {key} {class_name} Speed {max:.2f}{unit}')
                        pop_item = speed_dict.pop(key=key)
                        record_video_flag = True #Recording Flag (Set to True by default)
                        last_alert_vehicle = class_name
                        tag_object = activate_tag_2(args.cam,0)
                        tag_activate_speeding = root.get_child(["0:Objects", "2:Cameras", "2:"+tag_object+""])
                        tag_activate_speeding.set_value('1')

                        DMY_alert=f"{datetime.now().day}-{datetime.now().month}-{datetime.now().year}"
                        hmsec_alert=f"{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}" 

                        LOGGER.info(f'Pop item Alert:{key}')
                        break
                        # #append vehicle to df
                        # #Issue save df without video source as video not saved yet if speeding detected
                        # alert = pd.DataFrame([[DMY,hmsec,tag_offense,last_alert_vehicle,'NA',speed_value,tag_cam,vid_dst]],
                        #         columns=["Date","Time","Offence","Class","Number Plate","Speed","Camera Location","Link"])
                        # df = df.append(alert,ignore_index=True)
                if key not in tracker_id_list:
                    pop_item = speed_dict.pop(key=key)
                    # LOGGER.info(f'Pop item:{pop_item}')
                    # print(f"Pop Item: {pop_item}")
                    LOGGER.info(f'Pop item Dismissed:ID {key} Max: {max:.2f}{unit}')
                    # print(f'Pop item dismissed ID {key}')
                    break
                    # start_time = time_sync()
                    # while True:
                    #     current_time = time_sync()
                    #     elapsed_time = current_time-start_time
                    #     if elapsed_time>3:
                    #         pop_item = speed_dict.pop(key=key)
                    #         LOGGER.info(f'Pop item Dismissed:{key} Max: {max:.2f}')
                    #         break

                # for sublist in tracker_id_list:
                #     cv2.putText(annotated_frame,'[DEBUG] Tracker ID List '+str(tracker_id_list),(150,annotated_frame.shape[0]-200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))
                #     if key not in sublist:
                #         pop_item = speed_dict.pop(key=key)
                #         # LOGGER.info(f'Pop item:{pop_item}')
                #         # print(f"Pop Item: {pop_item}")
                #         LOGGER.info(f'Pop item Dismissed:{key} Max: {max:.2f} {sublist}')
                #         # print(f'Pop item dismissed ID {key}')
                #         break

                if SHOW_DEBUG:
                    str_speed = f'ID:{key} Mean:{mean:.2f} Max:{max:.2f} Min:{min:.2f} Std:{std:.2f} Frame Count {count}'
                    cv2.putText(annotated_frame,str_speed,(150,annotated_frame.shape[0]-500+index*30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))
            
            # Annotate Image
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            annotated_frame = zone_annotator.annotate(
                scene=annotated_frame
            )
            
            # Get delta time for real time fps calculation
            t1 = time_sync()
            dt = t1-t0
            fps = int(1/dt)

        else: #Retry Video Capture if frame drop or corrupted
            # fps = 0
            cap.release()
            cap = cv2.VideoCapture(rtsp)
            success, frame = cap.read()
            LOGGER.info(f'Retry Frame {success}')    

        # Draw FPS on Frame
        cv2.putText(annotated_frame,str(fps)+' FPS',(annotated_frame.shape[1]-150,40),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))

        

        if SHOW_DEBUG:
            cv2.putText(annotated_frame,'[DEBUG] Speed Dict',(150,annotated_frame.shape[0]-530),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))
            cv2.putText(annotated_frame,'[DEBUG] ID List '+str(tracker_id_list),(150,annotated_frame.shape[0]-200+30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,fontScale = 1,color=(255,255,255))

        
        # Pop item (FIFO) if len of speed_dict is more than 15
        if len(speed_dict)>5:
            speed_dict.popitem(last = False)
            LOGGER.debug('Pop Speed Dict >5')

        # Draw Mouse Marker on Map
        try:
            x_shift = 20
            cv2.circle(annotated_frame,mouse_coordinate,3,(200,200,200),3,3)
            cv2.drawMarker(annotated_frame, mouse_coordinate, color=[0, 0, 0], thickness=2, 
            markerType= cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,markerSize=10)
            cv2.putText(annotated_frame, ' #0 '+str(mouse_coordinate),mouse_coordinate,fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 5,fontScale = 1,color=(0,0,0))
            cv2.putText(annotated_frame, ' #0 '+str(mouse_coordinate),mouse_coordinate,fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness = 3,fontScale = 1,color=(255,255,0))
            transpose_mouse_coordinate = np.transpose(mouse_coordinate)
            scaled_mouse_point = view_transformer.transform_points(points=transpose_mouse_coordinate).astype(int)
            annotated_frame = draw_map_point(annotated_frame,scaled_mouse_point[0],source,target,0)
        except Exception as e:
            pass

        # Draw OpenCV Frame
        cv_scale = 0.5
        resize_frame = cv2.resize(annotated_frame,(int(int(video_info.width)*cv_scale),int(int(video_info.height)*cv_scale)),interpolation=cv2.INTER_AREA)
        title = f"Road Safety V2 ID-{args.cam} RTSP-{rtsp.split('.')[-1][:-1]}"
        cv2.imshow(title, resize_frame)
        
        # Mouse Event to get coordinate
        cv2.setMouseCallback(title,click_event)

        if record_video_flag and not kcw.recording:
            # print(f"Recording Video...")
            LOGGER.info(f'Recording Video...')
            consecFrames = 0
            # save_path = f"clip_{datetime.now().strftime('%Y-%d-%m--%H-%M-%S')}_cam_{args.cam}.mp4"
            save_path = f"clip_1.mp4"
            kcw.start(save_path,cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps)

            Y = datetime.now().year
            M = datetime.now().month
            D = datetime.now().day
            h = datetime.now().hour
            m = datetime.now().minute
            sec = datetime.now().second

            DMY=f"{D}-{M}-{Y}"
            hmsec=f"{h}:{m}:{sec}" 

            #Copy clip to shared folder
            timestamp = str(D).zfill(2)+str(M).zfill(2)+str(Y)+'_'+str(h).zfill(2)+str(m).zfill(2)+str(sec).zfill(2)
            # timestamp = timestamp.replace(":","_")  
            vid_src = f"clip_{str(args.cam)}.mp4" #Video Source
            #replace detect[5] with last_alert_vehicle
            vid_filename = f"{timestamp}_{tag_object}_{last_alert_vehicle}.mp4" #Video Destination with timestamp and info
            tag_offense = tag_object.split(" ")[1]
            tag_cam = tag_object.split(" ")[0]
       
        consecFrames += 1
 
        kcw.update(resize_frame)
 
        if kcw.recording and consecFrames == 128:
            kcw.finish()
            record_video_flag = False
            LOGGER.info('Recording Finised')
            # print(f"Recording Finished")
            # vid_dst = rf"\\shimanoace.local\\spl-common\\PE\\SSIP Smart Vision\\Supervision Speeding"
            vid_dst = rf"\\shimanoace.local\\spl-common\\PE\\SSIP Smart Vision\\Supervision Speeding\\{str(tag_offense)}\\{str(vid_filename)}"
            # shutil.copy(save_path,vid_dst)
            LOGGER.info(f'Video Copied')
            # print(f"Video Copied")

        # for i in range(len(tracking_dict)):
        #     # print(tracking_dict)
        #     for 

        # Alert DF (Trigger and Non Trigger) (only append to df after vehicle left the frame)
        # loop thru speed dict
        # if id not in detect
        # update alert dataframe and append
        # alert = pd.DataFrame([[DMY_alert,hmsec_alert,last_alert_vehicle,max,args.cam]],
        #                     columns=["Date","Time","Class","Speed","Camera Location"])
        # df = df.append(alert,ignore_index=True)
        # for index, (key, value) in enumerate(speed_dict.items()):
        #     if int(key) not in tracker_id:
        #         print(f'{key} is not detected. Appeding df')

        #create a new list with tracker id
        #if tracker id out of detect for number of fps
        #append df with tracker id 

        time_now = datetime.now()
        MINUTE_ = 0
        SECOND_ = 0
        Y = datetime.now().year
        M = datetime.now().month
        D = datetime.now().day
        if time_now.minute%10 == MINUTE_ and time_now.second%10 == SECOND_:
            try:
                # df.to_csv(f"Alerts_{D}_{M}_{Y}_Cam{stream_id}.csv",index=None) #Save to local drive
                if IFMS:
                    # df.to_csv(rf"\\shimanoace.local\\spl-common\\PE\\SSIP Smart Vision\\Alert Data\\Alerts_{D}_{M}_{Y}_Cam{args.cam}.csv",index=None) #Save a copy to share drive
                    df.to_csv(f"Alerts_{D}_{M}_{Y}_Cam{args.cam}.csv",index=None) #Save to local drive
                else:
                    # df.to_csv(rf"X:\\PE\\SSIP Smart Vision\\Alert Data\\Alerts_{D}_{M}_{Y}_Cam{args.cam}.csv",index=None) #Save a copy to share drive
                    df.to_csv(f"Alerts_{D}_{M}_{Y}_Cam{args.cam}.csv",index=None) #Save to local drive
                LOGGER.debug(f'Save Dataframe Successful')
            except Exception as e:
                LOGGER.exception(f"Write DataFrame to file Error \n{e}")

        if cv2.waitKey(1) & 0xFF == ord('x') or cv2.waitKey(1) & 0xFF == ord('q'): #Manual Termination by pressing 'x' or 'q'
            cv2.destroyWindow(title)
            LOGGER.info(f'Road Safety C-{floor}-{args.cam} Terminated')
            break

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    gc.collect()
    print('Exiting...') #Terminal not able to terminate properly

    exit()

##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
'''
<To Run>
python rs11.py --cam 167

<Todo>
Fix Frame Drop: (Solved)
-ultralytic>engine>model -- add try catch for missing source #deprecated (solved)

Terminal Stuck on exit: Solved (Client disconnect)

Perspective Transfrom: Done

Speed Monitoring: 

Class Label: Done

Polygon Zone: Done

Scaled Mouse Point 

OPCUA:
-initiate/reset tag to 0
-activate tag
-deactivate tag

Video Clip Speeding:
-save video clip
-copy video clip to share drive
-delete video clip after 1 week
#append vehicle to df
#Issue save df without video source as video not saved yet if speeding detected
alert = pd.DataFrame([[DMY,hmsec,tag_offense,last_alert_vehicle,'NA',speed_value,tag_cam,vid_dst]],
        columns=["Date","Time","Offence","Class","Number Plate","Speed","Camera Location","Link"])
df = df.append(alert,ignore_index=True)

MES Email:
-send mail without attachemet
-send mail with link
-send mail with image attachment
-mail formatting

Alert Excel Update
-initiate df
-append alert
-save to file
-Reset df at midnight


'''