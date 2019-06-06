import cv2
import face_recognition
import csv

def face_location_video():
    
    arq = csv.writer(open('aparicao.csv', 'w'))
    arq.writerow(['Personagem', 'Frame Inicial', 'Ultimo Frame'])

    image_recognation = face_recognition.load_image_file('jaden.jpg')
    face_encoding = face_recognition.face_encodings(image_recognation)[0]

    find_faces = [face_encoding]

    faces_locations = []
    face_encodings = []
    face_names = []


    video_capture = cv2.VideoCapture('jaden_smith.mp4')
    size = (
        int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_format = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter('jaden_rosto.avi', video_format, 23.0, size)
    
    number_frame = 0
    n_aparicoes = 0
    inicial_frame = 0

    while True:
        source_frame, frame = video_capture.read()
        number_frame += 1
        if frame is None:
            break

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        faces_encondings = face_recognition.face_encodings(rgb_frame, face_locations)

        
        face_names = []

        for face_encoding in faces_encondings:
            match = face_recognition.compare_faces(find_faces, face_encoding, tolerance=0.50)
    
            name = None
            if match[0]:
                inicial_frame = number_frame if inicial_frame == 0 else 0
                name = 'Jaden Smith'
            elif not match[0] and inicial_frame > 0:
                n_aparicoes += 1
                arq.writerow(['Jaden Smith', inicial_frame, number_frame])
                inicial_frame = 0
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            cv2.rectangle(frame, (left, top), (right + 5, bottom + 5), (0, 127, 255), 2)

            cv2.rectangle(frame, (left - 10, bottom + 30), (right + (len(name) * 4), bottom + 10), (0, 127, 255), cv2.FILLED)

            cv2.putText(frame, name, (left - 10, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    print('Fim do processamento')

if __name__ == '__main__':
    face_location_video()
