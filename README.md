# pongplay
Implementação de um modelo de machine learning para jogar o jogo pong, usando neural networks e tensorflow.

https://medium.com/pilotorobo

Dependencias:

* Python 3.6
* numpy==1.13.1
* scikit\-learn==0.19.0
* opencv-python==3.3.0.10
* pywin32==220
* tensorflow==1.2.1


## Como rodar:

O projeto possui três arquivos principais:

* get_train_data.py - Captura seus movimentos durante o jogo e salva junto com os dados da tela em um arquivo.
* neural_net.py - Treina uma neural network com os dados de treino capturados.
* infer_action.py - Utiliza a neural network treinada para simular os botões a ser pressionados durante o jogo, tentando copiar os movimentos do jogador.

O projeto já acompanha dados de treino e um modelo treinado, então você pode pular as etapas de coletar dados e treinar a rede se quiser.

1. Abra o jogo pong em https://pong-2.com/ 
2. Inicie o arquivo get_train_data.py, uma nova janela mostrando parte da sua tela irá se abrir. 
3. Ajuste o tamanho e a localização da tela do browser com o jogo para enquadrar exatamente dentro da outra janela, de formar que nenhum pedaço do browser apareça, apenas a tela do jogo, o score e as barras que se movimentam.
4. Inicie o jogo e quando apertar a seta pra cima ou pra baixo, o arquivo começará a gravar os dados. Aperte a letra 'Q' para parar de gravar e salvar os dados. Aproximadamente 20000 datapoints geram um bom resultado.
5. Inicie o arquivo neural_net.py e aguarde a rede ser treinada. Você pode mudar os parâmetros da rede dentro do arquivo.
6. Inicie o arquivo infer_action.py e repita o passo 3. Desative a tecla numlock (pois o programa usa o 8 e 2 do teclado numérico) e aperte a seta para cima ou para baixo para o jogo começar a jogar sozinho. 
7. Aperte 'Q' para parar quando quiser.

**Importante**: o computador agora está apertando frequentemente as teclas 8 e 2 do teclado númerico, então se você trocar a janela, ele continuará apertando. Lembre-se de apertar 'Q' para parar, ou volte para a tela do jogo.





