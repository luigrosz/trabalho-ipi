# Algoritmos de Processamento de Imagens

## Ampliação e Redução

### Vizinho Mais Próximo

Pra cada pixel na imagem nova, calcula qual pixel da imagem original é mais perto e copia o valor dele.

```
pixel_novo[y][x] = pixel_original[round(y * escala_y)][round(x * escala_x)]
```

Rápido, mas gera efeito "serrilhado" (pixelado).

### Interpolação Bilinear

Pra cada pixel na imagem nova, pega os 4 vizinhos mais próximos na imagem original e faz uma média ponderada pela distância.

```
valor = A*(1-dx)*(1-dy) + B*dx*(1-dy) + C*(1-dx)*dy + D*dx*dy
```

Onde A, B, C, D são os 4 vizinhos e dx, dy são as distâncias fracionárias. Resultado mais suave que vizinho mais próximo.

---

## Conversão RGB ↔ HSI

### RGB → HSI

Separa cor em 3 componentes independentes:
- **H (Matiz):** ângulo da cor (0 a 2π). Vermelho=0, Verde=2π/3, Azul=4π/3.
- **S (Saturação):** pureza da cor (0=cinza, 1=cor pura). `S = 1 - min(R,G,B)/I`
- **I (Intensidade):** brilho médio. `I = (R+G+B)/3`

O matiz é calculado com arccos usando as diferenças entre canais.

### HSI → RGB

Converte de volta dividindo em 3 setores de 120° cada (RG, GB, BR) e aplicando fórmulas diferentes em cada setor.

---

## Transformações de Intensidade

### Saturação

Converte RGB→HSI, multiplica o canal S pelo fator, converte HSI→RGB.

- Fator > 1: cores mais vivas
- Fator < 1: cores mais lavadas
- Fator = 0: imagem em tons de cinza

### Negativo

Inverte cada pixel: `pixel_novo = 255 - pixel_original`

Pixels claros viram escuros e vice-versa.

### Correção Gamma

```
pixel_novo = 255 * (pixel_original / 255) ^ gamma
```

- Gamma < 1: clareia (expande tons escuros)
- Gamma > 1: escurece (comprime tons escuros)
- Gamma = 1: sem mudança

### Brilho

Soma um valor constante a todos os pixels: `pixel_novo = pixel + valor`

Valores positivos clareiam, negativos escurecem. Resultado cortado em [0, 255].

### Contraste

Multiplica a distância de cada pixel ao meio (128): `pixel_novo = 128 + fator * (pixel - 128)`

- Fator > 1: aumenta diferença entre claros e escuros
- Fator < 1: reduz diferença (imagem mais "lavada")

### Equalização de Histograma

Redistribui as intensidades pra usar toda a faixa [0, 255] uniformemente.

1. Calcula histograma (conta quantos pixels tem de cada intensidade)
2. Calcula CDF (distribuição acumulada)
3. Mapeia cada intensidade: `novo = 255 * (CDF[i] - CDF_min) / (1 - CDF_min)`

Resultado: melhora contraste em imagens com intensidades concentradas numa faixa pequena.

### Especificação de Histograma

Transforma histograma da imagem pra ficar parecido com o de uma imagem de referência.

1. Calcula CDF da imagem fonte
2. Calcula CDF da imagem referência
3. Pra cada intensidade i, acha o j onde `CDF_fonte[i] ≈ CDF_referência[j]`
4. Mapeia i → j

### Limiarização

Converte pra preto e branco com um limiar:

```
se pixel >= limiar: branco (255)
se pixel < limiar:  preto (0)
```

Converte pra escala de cinza primeiro (média dos canais RGB) se for colorida.

---

## Filtragem Espacial

### Convolução

Base de todos os filtros espaciais. Pra cada pixel, coloca o kernel centrado nele, multiplica cada vizinho pelo peso correspondente do kernel e soma tudo.

```
resultado[y][x] = Σ vizinhos * kernel
```

Bordas da imagem tratadas com padding (replica pixels da borda).

### Filtro Box (Média)

Kernel com todos os valores iguais: `1/(n*n)` onde n é o tamanho.

```
1/9 * [1 1 1]
      [1 1 1]
      [1 1 1]
```

Cada pixel vira a média dos vizinhos. Suaviza/borra a imagem.

### Filtro Gaussiano

Kernel com pesos que seguem curva gaussiana — centro tem peso maior, bordas peso menor.

```
peso = e^(-(x² + y²) / (2σ²))
```

Suavização mais natural que box — preserva melhor as bordas.

### Filtro Mediana

Não usa convolução. Pra cada pixel, ordena os vizinhos e pega o valor do meio.

Bom pra remover ruído "sal e pimenta" sem borrar tanto as bordas.

### Sobel (Detecção de Bordas)

Dois kernels que detectam gradientes em direções opostas:

```
Gx (horizontal):     Gy (vertical):
[-1  0  1]           [-1 -2 -1]
[-2  0  2]           [ 0  0  0]
[-1  0  1]           [ 1  2  1]
```

- Gx detecta bordas verticais
- Gy detecta bordas horizontais
- Magnitude combinada: `√(Gx² + Gy²)` — detecta bordas em qualquer direção

### Laplaciano (Aguçamento)

Detecta bordas em todas as direções com um único kernel:

```
[ 0 -1  0]
[-1  4 -1]
[ 0 -1  0]
```

Soma as bordas detectadas à imagem original pra aguçar: `resultado = original + bordas`

### High-Boost (Aguçamento)

1. Suaviza a imagem com filtro box
2. Calcula máscara: `máscara = original - suavizada`
3. Soma máscara amplificada: `resultado = original + fator * máscara`

Fator controla intensidade do aguçamento. Mais flexível que laplaciano.

### Aguçamento por Gradiente

Procedimento em 4 passos:

1. **Suaviza** com filtro gaussiano pra eliminar ruído
2. **Calcula gradiente** da imagem suavizada usando Sobel (Gx e Gy)
3. **Combina** os gradientes calculando magnitude: `M(x,y) = √(Gx² + Gy²)`
4. **Soma** à imagem original: `g(x,y) = f(x,y) + c * M(x,y)`

O fator `c` controla intensidade do aguçamento. Diferente do high-boost que usa máscara (original - suavizada), esse usa magnitude do gradiente (Sobel) como máscara de bordas.
