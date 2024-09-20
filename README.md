# CareReminder - Fetin 2024

## Requirements:
- Python 3.9
- YOLOv8
- Firestore Credentials

## How to Install:
- Create a Python project and a virtual enviroment using 3.9 version;
- Uses the "pip" command to install dependencies;
```shell
pip install opencv-python mediapipe ultralytics firebase-admin
```
- Get the credentials of the project on firestore website. Just follow the [firebase documentation](https://firebase.google.com/docs/firestore?hl=pt-br).

## How to Run:
- Just run this command on your shell:
```shell
python ./path-to-your-repository/main.py
```
- The `main.py` file is the complete project content. To run only the Mediapipe and YOLOv8 logic, just run:
```shell
python ./path-to-your-repository/ai_test.py
```
