# ⚙️ OmniCount: Smart Object Counting System

OmniCount is a modular, GUI-based desktop application built in Python. It allows users to upload an image, interactively define a custom Region of Interest (ROI) template, and automatically count repeating objects using Multi-Scale Normalized Cross-Correlation (MS-NCC) and Non-Maximum Suppression (NMS).

## 🚀 Current Status: 
- [x] Object-Oriented Application Architecture
- [x] Responsive Tkinter UI Shell
- [x] Dynamic Image Resizing and Coordinate Mapping
- [x] Interactive UI ROI Cropping Tool 
- [x] Matcher Engine (MS-NCC and NMS Algorithms)
- [x] History
- [x] PDF Report Generation

## 📁 Project Architecture
```text
OmniCount_Project/
├── main.py                  # Application entry point
├── requirements.txt         # Project dependencies
├── ui/                      
│   ├── app_window.py        # Main Tkinter window manager
│   └── tabs.py              # Front-end UI and event listeners
├── core/                    
│   ├── image_processor.py   # Preprocessing and coordinate mapping
|   ├── database.py
│   └── matcher.py           # Core mathematics and computer vision engine
└── utils/                   
    └── pdf_generator.py        # PDF exporting logic

```

### Step 1: Download the Project


### Step 2: Create the Virtual Environment
Create an isolated Python environment named ip_venv inside the project folder:

```
python -m venv ip_venv
```
### Step 3: Activate the Virtual Environment
You must activate the environment every time you want to run the app.

On Windows (Command Prompt):
```
ip_venv\Scripts\activate
```

On macOS / Linux:
```
source ip_venv/bin/activate
```

### Step 4: Install Dependencies
With the environment active, install the required packages:
```
pip install -r requirements.txt
```

### Step 5: Run the Application
Finally, launch the UI:
```
python main.py
```