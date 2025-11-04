import time
from picamera2 import Picamera2
import io

from google import genai
from google.genai import types
import asyncio

import torch
from PIL import Image

from ultralytics import YOLO
model_path = 'best.pt'
model = YOLO(model_path)

# getting the gemeni api key
with open(".env", "r") as f:
    api_key = f.read().strip()

# this func uses lightweight egde ai model to detect front or back bumpers of a car
def check_bumpers_present(image_buffer, confidence_threshold=0.75):
    global model
    try:
        image = Image.open(image_buffer)
		
        results = model(image, verbose=False)
        detections = results[0].boxes
        classes = detections.cls.cpu().numpy()
        confidences = detections.conf.cpu().numpy()

        # just checking for front and back bumpers
        front_bumper_detected = any(int(c) == 8 and conf > confidence_threshold for c, conf in zip(classes, confidences))
        back_bumper_detected  = any(int(c) == 0 and conf > confidence_threshold for c, conf in zip(classes, confidences))

        if(front_bumper_detected or back_bumper_detected):
            return True
        
        return False

    except Exception as e:
        print(f"Error during detection: {e}")
        return False

# this func uses gemeni api to convert image to string to extract cars reg number
async def print_reg_no(image_bytes):
    client = genai.Client(api_key=api_key)
    start_time = time.time()
    response = await asyncio.to_thread(
        client.models.generate_content,
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
            'I want the registration number of the car in text format, without any spaces. The answer should be 1 word. If the picture is unclear or you are unable to detect a registration plate return me a empty string, dont return any other response.'
        ]
    )
    end_time = time.time()
    print(response.text)
    print('Time taken Gemeni API : ' , end_time - start_time)

camera = Picamera2()
camera.start()
time.sleep(2)

# these vars check if the car wasent there before and now it is
car_in = False
prev_car_in = False

# this func is the main loop that uses edge ai model to detect for cars bumpers (front or back)
async def driver():
    global car_in, prev_car_in

    while True:
        buffer = io.BytesIO()
        camera.capture_file(buffer, format="jpeg")
        image_binary = buffer.getvalue()

        start_time = time.time()
        bumper_detected = check_bumpers_present(buffer) # checking if a bumper is detected
        end_time = time.time()

        prev_car_in = car_in
        car_in = bumper_detected

        # if the car is detected and previously it wasent, that means we need to take a picture of it to send it to gemeni api
        if car_in and not prev_car_in:
            with open("reg_no.jpg", "wb") as f:
                f.write(image_binary)

            asyncio.create_task(print_reg_no(image_binary))
            print('time taken EDGE AI : ' , end_time - start_time)

        await asyncio.sleep(3) # this loops runs every 3 seconds

try:
    asyncio.run(driver())
except KeyboardInterrupt:
    camera.stop()
    camera.close()
    print("Exit")
