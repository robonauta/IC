# Aprendizado de transformações de imagens via classificação de microrregiões (Iniciação Científica)
Várias transformações de imagens podem ser realizadas por operadores de imagens que são modelados por funções que processam os pixels individualmente. Esta caracterı́stica permite que a definição dessas funções seja inserida no contexto de aprendizado de máquina como um problema de aprendizado de classificadores de pixels. Dentre os processamentos de imagens comumente utilizados destaca-se a segmentação, que produz uma partição dos pontos da imagem tal que determinadas regiões correspondem aos componentes de interesse e outras a partes a serem ignoradas em uma posterior análise. Devido ao alto custo computacional de se processar cada pixel individualmente, pixels “similares” podem ser agrupados em microrregiões de forma a reduzir o número de elementos atômicos. O problema de classificar pixels pode então ser trocado pelo problema de classificar essas microrregiões. Em problemas de segmentação, contanto que a borda dessas microrregiões tenha boa aderência ao contorno das regiões de interesse, não há perda de precisão e poderá haver um ganho significativo em termos de custo computacional. O objetivo deste projeto de pesquisa é estender os métodos de aprendizado de operadores já estabelecidos, adicionando métodos de aprendizado de operadores que atuam sobre microrregiões. Os novos métodos deverão ser integrados à bilioteca TRIOSlib, mantida pelo grupo, aplicados em problemas de segmentação de imagens, e comparados aos operadores que atuam sobre pixels. 

* Produção preliminar 

  * **ImageTransf**: programa que implementa transformações básicas sobre imagens, tais como: erosão, dilatação, closing e opening. Execução:
  
*$ make*

*$ java ImageTransf arg1 arg2 arg3*

arg1: -d ou -e

arg2: nome do arquivo texto contendo o elemento estruturante*

arg3: nome do arquivo da imagem
