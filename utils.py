import os
import cv2
import shutil
import subprocess
from pathlib import Path
from scenedetect import detect, AdaptiveDetector


detector = None

def split_video_to_scenes(video_path):
    global detector
    if detector is None:
        detector = AdaptiveDetector()
    scene_list = detect(video_path, detector)
    if len(scene_list) == 0:
        return None
    stop_frames = []
    for scene in scene_list[:-1]:
        stop_frames.append(scene[1].get_frames())
    return stop_frames

def get_sr_list():
    return ["BasicVSR", "BasicVSR++", "COMISR", "DBVSR", "LGFN", "RBPN",
            "Real-ESRGAN", "RealSR", "SOF-VSR-BI", "SwinIR", "VRT", "ahq-11",
            "comisr", "only-codec"]

def convert_video_to_frames(video_path, tmp_path, frame_step, scene_ends=None):
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", video_path, os.path.join(tmp_path, "frame%05d.png")])
    if scene_ends is None:
        images = [cv2.imread(os.path.join(tmp_path, x)) for x in os.listdir(tmp_path)[::frame_step]]
        return [images]
    else:
        img_list = []
        all_files = os.listdir(tmp_path)
        prev_scene_end = 0
        for end_point in scene_ends:
            current_scene = all_files[prev_scene_end : end_point]
            images = [cv2.imread(os.path.join(tmp_path, x)) for x in current_scene[::frame_step]]
            img_list.append(images)
        return img_list

def remove_tmp_folder(tmp_path):
    shutil.rmtree(tmp_path)
    Path(tmp_path).mkdir(parents=True, exist_ok=True)

def get_bitrate(video_path, tmp_path):
    duration_path = os.path.join(tmp_path, "duration.txt")
    os.system(f"ffmpeg -i {video_path} 2> {duration_path}")
    
    if not os.path.exists(duration_path):
        bitrate = 0
    else:
        try:
            with open(duration_path, 'r') as f:
                lines = list(f.readlines())
            
            sample = "Duration: "
            for line in lines:
                if sample not in line:
                    continue
                
                for i in range(len(line)):
                    if line[i] in sample:
                        i += 1
                    else:
                        break
                
                total = 0
                hours = line[i:i+2]
                total += int(hours) * 60 * 60
                minutes = line[i+3:i+5]
                total += int(minutes) * 60
                seconds = line[i+6:i+8]
                total += int(seconds)
                mseconds = line[i+9:i+11]
                total += int(mseconds) / 100

                bitrate = int(os.stat(video_path).st_size * 8 / 1024 / total)

                os.remove(duration_path)
        except:
            bitrate = 0

    return bitrate / 1000