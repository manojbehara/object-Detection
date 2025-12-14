from Detector import *
import os

def main():
    videoPath = 0
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    # Validate file paths
    if not os.path.exists(videoPath):
        raise FileNotFoundError(f"Video file not found: {videoPath}")
    if not os.path.exists(configPath):
        raise FileNotFoundError(f"Config file not found: {configPath}")
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model file not found: {modelPath}")
    if not os.path.exists(classesPath):
        raise FileNotFoundError(f"Classes file not found: {classesPath}")

    # Create Detector instance and process video
    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()


if __name__ == '__main__':
    main()
