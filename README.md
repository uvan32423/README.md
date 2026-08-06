# README

# sidewalk_safety

 
This project uses a YOLOv8 model to detect sidewalk objects like people, vehicles, bicycles, benches, chairs, bollards, street lights, and traffic signs. The model was trained in Google Colab, exported as an ONNX file, and tested on a Jetson Orin Nano.

This is an image of a crowded new york street, the model identifies a traffic sign and a person
<img width="433" height="695" alt="sidewalk_model demo" src="https://github.com/user-attachments/assets/75d27832-b534-4fef-86da-3d12c59a0225" />

## The Algorithm

The project uses YOLOv8 object detection. The model looks at an image or camera frame and predicts bounding boxes around objects it recognizes. Each detection includes a class name and a confidence score.

The Python script loads `best.onnx`, runs it with the Ultralytics library, and uses OpenCV to read images or camera frames. If an object is detected, the script prints results like: "bench detected: 0.82"

## Running this project

    Put best.onnx in the project folder.

    Install the required libraries:

pip3 install ultralytics opencv-python

    Run the live detection script:

python3 sidewalk_detect.py

    To test a single image, run:

python3 test_image.py

If the camera does not work, change CAMERA_INDEX in the script. If detections are too low, lower the confidence value.

## Notes 
The model works best on clear sidewalk objects like benches, chairs, bollards, bicycles, street lights, and traffic signs. It may miss people in crowded scenes or objects that are far away or blurry.

## Future Improvements

In the future, I would like to add text-to-speech so the Jetson can say detected objects out loud. I would also improve the model by collecting more real sidewalk images from the Jetson camera. If I added a depth camera and some datasets then maybe it would be able to tell how far away an object is 



[View a video explanation here](video link)











Here is a demo

