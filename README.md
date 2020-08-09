# Aprendizado de transformações de imagens via classificação de microrregiões (Iniciação Científica)
<p> Várias transformações de imagens podem ser realizadas por operadores de imagens que são modelados por funções que processam os pixels individualmente. Esta caracterı́stica permite que a definição dessas funções seja inserida no contexto de aprendizado de máquina como um problema de aprendizado de classificadores de pixels. Dentre os processamentos de imagens comumente utilizados destaca-se a segmentação, que produz uma partição dos pontos da imagem tal que determinadas regiões correspondem aos componentes de interesse e outras a partes a serem ignoradas em uma posterior análise. Devido ao alto custo computacional de se processar cada pixel individualmente, pixels “similares” podem ser agrupados em microrregiões de forma a reduzir o número de elementos atômicos. O problema de classificar pixels pode então ser trocado pelo problema de classificar essas microrregiões. Em problemas de segmentação, contanto que a borda dessas microrregiões tenha boa aderência ao contorno das regiões de interesse, não há perda de precisão e poderá haver um ganho significativo em termos de custo computacional. O objetivo deste projeto de pesquisa é estender os métodos de aprendizado de operadores já estabelecidos, adicionando métodos de aprendizado de operadores que atuam sobre microrregiões. Os novos métodos deverão ser integrados à bilioteca TRIOSlib, mantida pelo grupo, aplicados em problemas de segmentação de imagens, e comparados aos operadores que atuam sobre pixels. </p>

## Produção preliminar 

* **ImageTransf**: programa que implementa transformações básicas sobre imagens, tais como: erosão, dilatação, closing e opening. Execução:

```
$ make 
$ java ImageTransf
```
   <p>Em seguida, o prompt perguntará qual a operação que o usuário deseja, o nome do arquivo contendo o elemento estruturante, o nome do arquivo da imagem e se deve exibir as matrizes que representam a imagem, possibilitando uma visualização matemática das operações. </p>
  <p> Um arquivo que contém o elemento estruturante (ES) consiste basicamente de um arquivo texto. Na primeira linha, deve conter dois inteiros que indicam o número de linhas e colunas, respectivamente. Em seguida, há uma sequência de inteiros 0 ou 1, onde os números 1 dão o formato do ES. Exemplo de ES em formato de cruz com 3 linhas e 3 colunas: </p>
  
```
3 3
0 1 0
1 1 1
0 1 0
```
  

## Desenvolvimento

 * **LocalSLIC**: contém o código que aplica o algoritmo SLIC em regiões de interesse, que são indicadas por uma imagem binária que serve como máscara. Execução: 

```
image = io.imread('filename.jpg')
processor = SLIC(image = image, binaryImg = binaryImg, K = 10000, M = 1)
labels = processor.execute(iterations = 3, labWeight = 0.2)
``` 

* **ImageSegmentation**: contém imagens produzidas por cada um dos métodos de segmentações abordados neste trabalho. 

* **MAC0215**: contém material produzido durante a disciplina "Atividade Curricular em Pesquisa". Um documento com o acompanhamento das atividades pode ser conferido no _README_. 

 * **Network**: contém os códigos que criam uma rede convolucional, treinam e a testam em diferentes granularidades. As imagens de treinamento estão na pasta _train_, as de teste, na pasta _test_. Para reproduzir os experimentos, basta fazer: 

 ``` 
python3 conv.py
 ``` 

* **Results**: contém o relatório final relatando os resultados obtidos apresentado a FAPESP. Além disso, contém o pôster apresentado ao final da disciplina MAC0215.  

## Financiamento 

Este projeto foi financiado pela Fundação de Amparo à Pesquisa do Estado de São Paulo, sob processo n.º  2018/11899-8. 