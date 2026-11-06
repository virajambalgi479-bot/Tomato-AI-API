CROP MONITOR + TOMATO AI INTEGRATION

FILES
- crop-monitor.html       Your existing Crop Monitor website (ThingSpeak and dashboard retained)
- app.py                  Python API that loads Tomato_AI_Model.keras and predicts the leaf disease
- requirements.txt        Python packages
- Tomato_AI_Model.keras   YOU must copy your trained model into this same folder

IMPORTANT CLASS ORDER
The app.py class order is configured as:
1. Tomato Bacterial spot -> bacterial
2. Tomato early blight   -> leafspot
3. Tomato late blight    -> lateblight
4. Tomato healthy        -> healthy
This order must be exactly the same as the class order used during training.

WINDOWS SETUP (LAPTOP)
1. Copy Tomato_AI_Model.keras into this folder, beside app.py.
2. Install Python 3.11 (recommended for this TensorFlow setup) if it is not installed.
3. Open Command Prompt in this folder.
4. Run:
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python app.py
5. Keep this Command Prompt window open. The API runs on port 5000.
6. Open crop-monitor.html in Chrome.
7. In the website click Settings. In "Your image model URL" enter:
   http://127.0.0.1:5000/predict
8. Save Settings. Select/capture a tomato leaf photo and click "Analyse with my model".
9. The website will display the diagnosis and its existing advisory section.

MOBILE TESTING ON THE SAME WI-FI
- Start app.py on the laptop.
- Find the laptop's local IPv4 address using `ipconfig` in Command Prompt.
- In website Settings, use http://LAPTOP-IP:5000/predict instead of 127.0.0.1.
- The phone and laptop must be on the same Wi-Fi/hotspot, and Windows Firewall may ask permission.
- 127.0.0.1 on a phone means the phone itself, not the laptop.

NOTES
- ThingSpeak readings continue to come from the existing website settings and channel mapping.
- The AI image model runs in the Python backend; a browser HTML file cannot directly load a .keras model.
- Do not expose a development server publicly. For public mobile access, deploy the Python API with HTTPS and put that HTTPS /predict URL in website Settings.
- The advisory text is decision support, not a chemical prescription. Verify crop/disease registration, dose and pre-harvest interval on the current Indian product label and with a local agriculture officer.
- If the model was trained with a different class order or without an internal Rescaling layer, update app.py preprocessing/class order to match the training notebook.
