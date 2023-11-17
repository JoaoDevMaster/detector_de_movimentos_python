import cv2
import pyttsx3

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Inicializar o objeto que fará a comparação de frames
fgbg = cv2.createBackgroundSubtractorMOG2()

# Inicializar o módulo de conversão de texto para fala
engine = pyttsx3.init()

# Tamanho mínimo do contorno para considerar como movimento
min_contour_area = 1000

# Fator de redimensionamento
resize_factor = 0.5

# Inicializar a variável de contagem
contador_movimentos = 0

while True:
    # Ler o frame da câmera
    ret, frame = cap.read()

    # Redimensionar a imagem para melhorar o desempenho
    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    # Aplicar a subtração de fundo para destacar as regiões em movimento
    fgmask = fgbg.apply(frame)

    # Aplicar um pouco de suavização para reduzir o ruído
    fgmask = cv2.medianBlur(fgmask, 5)

    # Encontrar contornos na imagem resultante
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre os contornos
    for contour in contours:
        # Calcular a área do contorno
        area = cv2.contourArea(contour)

        # Se a área do contorno for maior que o limite mínimo
        if area > min_contour_area:
            # Incrementar a contagem de movimentos
            contador_movimentos += 1

            # Falar a mensagem com o número de movimentos
            mensagem = f"Movimento detectado! Movimento #{contador_movimentos}"
            engine.say(mensagem)

            # Exibir a mensagem no console
            print(mensagem)

            # Aguardar até que a fala seja concluída antes de prosseguir
            engine.runAndWait()

    # Desenhar os contornos na imagem original
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Exibir a imagem resultante e a imagem original em janelas separadas
    cv2.imshow('Detecção de Movimento', frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
