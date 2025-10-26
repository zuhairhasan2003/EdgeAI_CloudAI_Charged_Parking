# 🚗 Smart Parking Detection : Edge AI + Cloud AI (Gemini)

This project combines **Edge AI (YOLO model on Raspberry Pi)** with **Cloud AI (Google Gemini API)** to automatically detect cars entering a parking area and read their **registration numbers**.

---

## 🧠 How It Works

1. **Edge AI (On-device)**
   - A lightweight **YOLO model (`best.pt`)** runs locally using the Raspberry Pi camera (`Picamera2`).
   - It continuously monitors for **front or back bumpers**.
   - When a bumper is detected, it assumes a car has arrived.

2. **Cloud AI (Gemini API)**
   - Once a car is detected, the Pi captures an image.
   - That image is sent to **Google Gemini (via `google-genai`)**.
   - Gemini reads the **car’s registration number** and returns it as text.

This approach minimizes network usage and latency, the **edge device** does the car detection, while the **cloud AI** handles text extraction and interpretation.

---

## 🧰 Requirements

All required libraries are listed in `installs.txt`.

If you are setting up for the first time:

```bash
# create a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# install all dependencies
pip install -r installs.txt
```

---

## ⚙️ Setup

1. **Clone this repo**

   ```bash
   clone this repo
   cd EdgeAI_CloudAI_Charged_Parking
   ```

2. **Add your Gemini API key**

   * Create a `.env` file in the project directory:

     ```
     YOUR_GEMINI_API_KEY_HERE
     ```
   * (The script automatically reads it.)

3. **Connect your camera**

   * Make sure your Raspberry Pi camera is enabled and working.

---

## ▶️ Run the project

```bash
python3 init.py
```

* The script continuously monitors for car bumpers.
* When a car enters, it captures an image and sends it to Gemini for **license plate recognition**.
* The detected registration number will be printed in the terminal.

Press `Ctrl + C` to stop.

---

## 🧠 Tech Stack

| Component    | Description                                                             |
| ------------ | ----------------------------------------------------------------------- |
| **Edge AI**  | YOLO model (Ultralytics) running locally for real time object detection |
| **Cloud AI** | Google Gemini API for extracting license plate text                     |
| **Hardware** | Raspberry Pi + PiCamera2                                                |
| **Language** | Python 3                                                                |