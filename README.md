# 🧠 **AI Motion Detection System**

An advanced AI-powered motion detection system using **Python**, **OpenCV**, **PyTorch**, and **MediaPipe**. It detects movement, identifies objects (e.g., person, bird), analyzes facial expressions, and overlays skeleton tracking on detected entities.

---

## 🚀 **Features**

- 📹 **Real-Time Motion Detection:** Detects and logs movement with sensitivity control.
- 🦾 **Skeleton Tracking:** Draws a skeleton overlay on detected entities (e.g., people or birds).
- 🧠 **AI Object Identification:** Identifies objects and displays confidence levels.
- 🙂 **Facial Expression Detection:** Detects emotions (e.g., smiling, frowning, neutral).
- ⚙️ **Adjustable Sensitivity:** Fine-tune sensitivity using `+` and `-` keys.
- 🛑 **Immediate Exit:** Press `q` to quit the program instantly.

---

## 🛠️ **Requirements**

Install dependencies via `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

**Dependencies:**
- opencv-python
- torch
- torchvision
- mediapipe
- numpy

---

## 📷 **Camera Selection**

When running the program, you can select your preferred camera from the listed options.

---

## ▶️ **How to Run**

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd ai_motion_detection
    ```

2. Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```

4. Run the program:
    ```bash
    python3 main.py
    ```

---

## ⚡ **Controls**

- `+` → Increase sensitivity  
- `-` → Decrease sensitivity  
- `q` → Quit program immediately  

---

## 📚 **Contributing**

Feel free to open issues or submit pull requests to improve the project!

---

## 🤝 **Support**

If you encounter issues, feel free to open a GitHub issue or contact us directly.

---

**Happy Coding! 🐍🚀**
