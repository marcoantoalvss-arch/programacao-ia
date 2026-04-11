#import das libs
import cv2
from deepface import DeepFace
import time

#Passo 1: carregar a identidade (Cadastramento)
imagem_referencia= "face_id.jpg"
print("Carregamento identidade do morador")

#pré-analise da imagem, pra garrantir que a foto de referencia é valida
try:
    DeepFace.represent(img_path=imagem_referencia, model_name="VGG-Face")
    print("identidade carregadancom sucesso!.")
except:
    print("Erro! Não encontrei o arquivo ou não há rosto nele.")
    exit()
    
#iniciar a câmera
cap = cv2.VideoCapture(0) #o numero que a camera esta integrado ao computador
print("Sistema de portaria ativo.")       

while True:
    ret, frame = cap.read() #ret retorna true se a foto foi tirada, frame recebe a imagem
    if not ret: break
    
    frame_small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    
    #desenahr retangulo para indicar area de leitura
    height, width, _ = frame.shape
    cv2.rectangle(frame, (100,100), (width-100, height-100), (255,0,0), 2)
    #tamanho, cor e espessura da linha
    
    #verificação da imagem com o rosto detectado
    cv2.putText(frame, "Pressione 'v' para verificar o acesso", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    key = cv2.waitKey(1)
    

    if key & 0xFF == ord('v'):
        print("Verificar identidade...")
        try:
            resultado = DeepFace.verify(
                img1_path = frame, #quem está na camera
                img2_path = imagem_referencia, #foto capturada
                model_name="VGG-Face",
                enforce_detection = False
            )
            
            #se resultado é verdadeiro (acesso liberado)
            if resultado ['verified']:
                print(">>> ACESSO LIBERADO! Bem vindo(a).")
                cv2.rectangle(frame, (0,0), (width, height), (0,255,0), 10)
                cv2.imshow("portaria", frame)
                cv2.waitKey(2000) #pausa por 2s para mostrar a borda verde
            else:
                print(">>>>ACESSO NEGADO>>>>")
                cv2.rectangle(frame,(0,0), (width, height), (0,0,255), 10)
                cv2.imshow("Portaria", frame)
                cv2.waitKey(2000)
                
        except Exception as e:
            print(f"Erro na leitura: {e}")
            
    cv2.imshow("Portaria", frame)
    
    if key & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()    
            