# Importamos las librerías numpy y openCv
import numpy as np
import cv2
import time

# Inicialiazamos el HOG descriptor o el detector de personas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

# Abrimos la WebCam de nuestra PC
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capturamos frame por frame
    ret, frame = cap.read()

    # Redimencionamos las dimensiones de nuestra captura para una detección más rápida
    frame = cv2.resize(frame, (640, 480))
    # usamos el método cvtColor para volver la transmisión en la escala de gris y así detectarlo más rápido.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Se detectan personas en la imagen
    # Nos retorna los Marcos delimitantes 
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    # Cada uno con diferente tamaño 

    '''Tiempo de ejecución'''
    formato="%c"
    ahora=time.strftime(formato)
    inicio=time.time()
    # Almacenamos estos Marcos en la variables boxes, Coordenadas
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # Ubicamos los Marcos con un color Verde (0,255,0) RGB
        cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
    # Escribimos la salida del video
    out.write(frame.astype('uint8'))
    # Mostramos los Marcos
    cv2.imshow('frame',frame)
    fin=time.time()
    print(f'{ahora} | {fin-inicio}')

    # Para finalizar el programa usamos la tecla "q" de quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Rompemos el bucle while
        break

# Cuando todo esté listo, liberamos la captura y el output
cap.release()
out.release()
# Finalmente ceramos la pantalla
cv2.destroyAllWindows()
cv2.waitKey(1)