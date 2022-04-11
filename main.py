import PySimpleGUI as sg    
from cctv import SmartCCTV
from face_detection import FaceDetection

sg.theme('DarkGrey13')    # Keep things interesting for your users

layout = [[sg.Text('Welcome to Nighthawk CCTV - Smart CCTV Camera')],      
          [sg.Button('Start Monitoring'), sg.Button('Register Faces'), sg.Button("Identify")],
          [sg.Button('Exit')]      
        ]      

window = sg.Window('Nighthawk CCTV - Smart CCTV Camera', layout)
 
while True:                             # The Event Loop
    event, values = window.read() 
    print(event, values)       
    if event == 'Start Monitoring':
        print("Monitoring")
        cctv = SmartCCTV()
        cctv.run()

    elif event == "Register Faces":
        register_face_layout = [
            [sg.Text("Full Name"), sg.InputText()],
            [sg.Text("ID"), sg.Text("        "), sg.InputText()],
            [sg.Submit()]
            ]
        register_face_window = sg.Window('Register a person', register_face_layout)
        event, values = register_face_window.read()
        register_face_window.close()
        print(event, values)
        name = values[0]
        userId = values[1]
        face_detector = FaceDetection(name, userId)
        face_detector.collect_dataset()

    elif event == "Identify":
        face_detector = FaceDetection()
        face_detector.run()

    elif event == sg.WIN_CLOSED or event == 'Exit':
        break      

window.close()