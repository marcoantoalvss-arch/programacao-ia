#import das bibliotecas
import cv2 #opencv -> lib responsável pelo gerenciamento de dispositivos vls. comp.
from ultralytics import YOLO #lib responsavel pelo reconhecimento facial / objetos

#Passo 1: carregamento do modelo
print ("Carregando modelo...")
model = YOLO('yolov8n.pt') #yolov8n é uma versão nano, mais leve/rápido

#2. Abrir uma conexão com webcam
cap = cv2.VideoCapture(0)
#numero 0 representa uma webcam integrada ao computador
#numero 1 representa uma webcam conectada via USB (via física)
#caso a via seja remota, o endereço de ip deve ser informado

#verifica se a camera abriu corretamente
if not cap.isOpened():
     print("Erro ao acessar câmera")
     exit()
     
print("Iniciando detecção. Pressione 'q' para sair.")

#passo 3:iniciar a leitura das detecções
while True:
    sucesso, frame = cap.read()#ler os frames (Imagens) da camera 
    
    if sucesso: #realizar a detecção 
        results = model(frame, conf=0.5)#queremos detecções com 50% ou mais de certeza
        annotated_frame = results[0].plot() #criar caixa visual na imagem
        cv2.imshow("Visão Computacional - YOLOv8", annotated_frame)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        
    else:
        break
    
#limpeza
cap.release()
cv2.destroyAllWindows()        