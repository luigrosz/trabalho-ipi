import numpy as np

# === Ampliação e Redução ===


def redimensionar_vizinho(imagem, novo_largura, novo_altura):
    """Redimensiona imagem usando interpolação por vizinho mais próximo."""
    altura, largura = imagem.shape[:2]
    resultado = np.zeros(
        (novo_altura, novo_largura, *imagem.shape[2:]), dtype=imagem.dtype
    )

    # Calcula fator de escala entre imagem original e nova
    escala_y = altura / novo_altura
    escala_x = largura / novo_largura

    # Pra cada pixel na imagem nova, acha o pixel mais próximo na original
    for y in range(novo_altura):
        for x in range(novo_largura):
            orig_y = int(y * escala_y)
            orig_x = int(x * escala_x)
            # Garante que não ultrapassa os limites da imagem
            orig_y = min(orig_y, altura - 1)
            orig_x = min(orig_x, largura - 1)
            resultado[y, x] = imagem[orig_y, orig_x]

    return resultado


def redimensionar_bilinear(imagem, novo_largura, novo_altura):
    """Redimensiona imagem usando interpolação bilinear."""
    altura, largura = imagem.shape[:2]
    resultado = np.zeros((novo_altura, novo_largura, *imagem.shape[2:]), dtype=np.uint8)

    escala_y = altura / novo_altura
    escala_x = largura / novo_largura

    for y in range(novo_altura):
        for x in range(novo_largura):
            # Coordenadas fracionárias na imagem original
            orig_y = y * escala_y
            orig_x = x * escala_x

            # 4 vizinhos mais próximos
            y0 = int(orig_y)
            x0 = int(orig_x)
            y1 = min(y0 + 1, altura - 1)
            x1 = min(x0 + 1, largura - 1)

            # Distância fracionária até o vizinho superior esquerdo
            dy = orig_y - y0
            dx = orig_x - x0

            # Média ponderada dos 4 vizinhos pela distância
            valor = (
                imagem[y0, x0] * (1 - dx) * (1 - dy)
                + imagem[y0, x1] * dx * (1 - dy)
                + imagem[y1, x0] * (1 - dx) * dy
                + imagem[y1, x1] * dx * dy
            )
            resultado[y, x] = np.clip(valor, 0, 255).astype(np.uint8)

    return resultado


# === Conversão RGB ↔ HSI ===


def rgb_para_hsi(imagem):
    """Converte imagem RGB para HSI."""
    # Normaliza pra [0, 1]
    rgb = imagem.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Intensidade = média dos 3 canais
    intensidade = (r + g + b) / 3.0

    # Saturação = 1 - 3*min(R,G,B) / (R+G+B)
    minimo = np.minimum(np.minimum(r, g), b)
    soma = r + g + b
    saturacao = np.where(soma == 0, 0, 1 - 3 * minimo / soma)

    # Matiz = ângulo calculado com arccos usando diferenças entre canais
    numerador = 0.5 * ((r - g) + (r - b))
    denominador = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    denominador = np.where(denominador == 0, 1e-10, denominador)
    theta = np.arccos(np.clip(numerador / denominador, -1, 1))

    # Se B > G, matiz = 2π - θ (espelha pro outro lado do círculo)
    matiz = np.where(b <= g, theta, 2 * np.pi - theta)

    return np.stack([matiz, saturacao, intensidade], axis=2)


def hsi_para_rgb(imagem_hsi):
    """Converte imagem HSI para RGB. Divide em 3 setores de 120° cada."""
    h, s, i = imagem_hsi[:, :, 0], imagem_hsi[:, :, 1], imagem_hsi[:, :, 2]
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # H < 120°: B = I(1-S), R = I[1 + S·cosH/cos(60°-H)], G = 3I - (R+B)
    mask1 = h < 2 * np.pi / 3
    b[mask1] = i[mask1] * (1 - s[mask1])
    r[mask1] = i[mask1] * (
        1 + s[mask1] * np.cos(h[mask1]) / np.cos(np.pi / 3 - h[mask1])
    )
    g[mask1] = 3 * i[mask1] - (r[mask1] + b[mask1])

    # 120° <= H < 240°: subtrai 120° do H e rotaciona os canais (R↔G↔B)
    mask2 = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    h2 = h[mask2] - 2 * np.pi / 3
    r[mask2] = i[mask2] * (1 - s[mask2])
    g[mask2] = i[mask2] * (1 + s[mask2] * np.cos(h2) / np.cos(np.pi / 3 - h2))
    b[mask2] = 3 * i[mask2] - (r[mask2] + g[mask2])

    # 240° <= H < 360°: subtrai 240° do H e rotaciona os canais novamente
    mask3 = h >= 4 * np.pi / 3
    h3 = h[mask3] - 4 * np.pi / 3
    g[mask3] = i[mask3] * (1 - s[mask3])
    b[mask3] = i[mask3] * (1 + s[mask3] * np.cos(h3) / np.cos(np.pi / 3 - h3))
    r[mask3] = 3 * i[mask3] - (g[mask3] + b[mask3])

    resultado = np.stack([r, g, b], axis=2)
    return (resultado * 255).clip(0, 255).astype(np.uint8)


# === Transformações de Intensidade ===


def ajuste_saturacao(imagem, fator):
    """Ajusta saturação. Converte RGB→HSI, multiplica canal S."""
    hsi = rgb_para_hsi(imagem)
    hsi[:, :, 1] = (hsi[:, :, 1] * fator).clip(0, 1)
    return hsi_para_rgb(hsi)


def negativo(imagem):
    """Inverte cada pixel: novo = 255 - original."""
    return 255 - imagem


def ajuste_gamma(imagem, gamma):
    """Aplica curva gamma via HSI — I_novo = I^gamma."""
    hsi = rgb_para_hsi(imagem)
    # I já está em [0, 1], aplica potência direto
    # gamma < 1: curva sobe rápido → tons escuros ficam mais claros
    # gamma > 1: curva sobe devagar → tons claros ficam mais escuros
    # gamma = 1: sem mudança (x^1 = x)
    hsi[:, :, 2] = np.power(hsi[:, :, 2], gamma)
    return hsi_para_rgb(hsi)


def ajuste_brilho(imagem, valor):
    """Ajusta brilho via HSI — altera intensidade."""
    hsi = rgb_para_hsi(imagem)
    # Valor vem em [-255, 255], divide por 255 pra converter pra escala de I [0, 1]
    # Ex: valor=127 → soma 0.5 na intensidade, valor=-255 → subtrai 1.0 (tudo preto)
    hsi[:, :, 2] = (hsi[:, :, 2] + valor / 255.0).clip(0, 1)
    return hsi_para_rgb(hsi)


def ajuste_contraste(imagem, fator):
    """Ajusta contraste via HSI — altera intensidade."""
    hsi = rgb_para_hsi(imagem)
    # Multiplica distância de cada pixel ao meio (0.5) pelo fator
    # fator > 1 aumenta diferença entre claros e escuros, < 1 reduz
    hsi[:, :, 2] = (0.5 + fator * (hsi[:, :, 2] - 0.5)).clip(0, 1)
    return hsi_para_rgb(hsi)


def equalizar_histograma(imagem):
    """Equaliza histograma via HSI — aplica só no canal I, preserva cor."""
    hsi = rgb_para_hsi(imagem)
    # Converte canal I de [0, 1] pra [0, 255] pra equalizar
    canal_i = (hsi[:, :, 2] * 255).astype(np.uint8)
    canal_i_equalizado = _equalizar_canal(canal_i)
    # Converte de volta pra [0, 1]
    hsi[:, :, 2] = canal_i_equalizado.astype(np.float64) / 255.0
    return hsi_para_rgb(hsi)


def _calcular_histograma(canal):
    """Conta quantos pixels tem de cada intensidade (0 a 255)."""
    histograma = np.zeros(256, dtype=np.int64)
    for valor in canal.ravel():
        histograma[valor] += 1
    return histograma


def _calcular_acumulada(histograma):
    """Calcula distribuição acumulada normalizada do histograma."""
    acumulada = np.zeros(256, dtype=np.float64)
    acumulada[0] = histograma[0]
    # Soma acumulada: cada posição = soma de todas as anteriores
    for i in range(1, 256):
        acumulada[i] = acumulada[i - 1] + histograma[i]
    # Normaliza pra [0, 1]
    total = acumulada[255]
    if total > 0:
        acumulada = acumulada / total
    return acumulada


def _equalizar_canal(canal):
    """Equaliza histograma de um canal usando distribuição acumulada como função de mapeamento."""
    histograma = _calcular_histograma(canal)
    acumulada = _calcular_acumulada(histograma)

    # Mapeia cada intensidade usando a distribuição acumulada normalizada
    acumulada_min = acumulada[acumulada > 0].min()
    mapeamento = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapeamento[i] = np.clip(
            ((acumulada[i] - acumulada_min) / (1.0 - acumulada_min)) * 255, 0, 255
        )

    # Aplica mapeamento a todos os pixels do canal
    return mapeamento[canal]


def especificar_histograma(imagem, imagem_referencia):
    """Especificação de histograma via HSI — aplica só no canal I, preserva cor."""
    hsi = rgb_para_hsi(imagem)
    hsi_ref = rgb_para_hsi(imagem_referencia)
    # Converte canais I pra [0, 255] pra especificar
    canal_i = (hsi[:, :, 2] * 255).astype(np.uint8)
    canal_i_ref = (hsi_ref[:, :, 2] * 255).astype(np.uint8)
    canal_i_especificado = _especificar_canal(canal_i, canal_i_ref)
    # Converte de volta pra [0, 1]
    hsi[:, :, 2] = canal_i_especificado.astype(np.float64) / 255.0
    return hsi_para_rgb(hsi)


def _especificar_canal(canal_fonte, canal_referencia):
    """Especificação de histograma: mapeia acumulada da fonte pra acumulada da referência."""
    # Calcula distribuição acumulada de ambas as imagens
    hist_fonte = _calcular_histograma(canal_fonte)
    acumulada_fonte = _calcular_acumulada(hist_fonte)

    hist_ref = _calcular_histograma(canal_referencia)
    acumulada_ref = _calcular_acumulada(hist_ref)

    # Pra cada intensidade i, acha j onde acumulada_fonte[i] ≈ acumulada_ref[j]
    mapeamento = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diferenca_minima = 1.0
        melhor_j = 0
        for j in range(256):
            diferenca = abs(acumulada_fonte[i] - acumulada_ref[j])
            if diferenca < diferenca_minima:
                diferenca_minima = diferenca
                melhor_j = j
        mapeamento[i] = melhor_j

    return mapeamento[canal_fonte]


def limiarizar(imagem, limiar):
    """Converte pra preto e branco: pixel >= limiar vira 255, senão vira 0."""
    # Converte pra escala de cinza (média dos canais) se for colorida
    if len(imagem.shape) == 3:
        cinza = np.mean(imagem.astype(np.float64), axis=2)
    else:
        cinza = imagem.astype(np.float64)

    resultado = np.zeros_like(cinza, dtype=np.uint8)
    # Converte tudo acima do limiar para branco, resto preto
    resultado[cinza >= limiar] = 255
    return resultado


# === Filtragem Espacial ===


def _aplicar_filtro(imagem, kernel):
    """Aplica convolução com kernel. Se colorida, processa cada canal separado."""
    kh, kw = kernel.shape
    pad_y = kh // 2
    pad_x = kw // 2

    if len(imagem.shape) == 3:
        resultado = np.zeros_like(imagem, dtype=np.float64)
        for c in range(imagem.shape[2]):
            resultado[:, :, c] = _convolver_canal(imagem[:, :, c], kernel, pad_y, pad_x)
        return resultado.clip(0, 255).astype(np.uint8)

    resultado = _convolver_canal(imagem, kernel, pad_y, pad_x)
    return resultado.clip(0, 255).astype(np.uint8)


def _convolver_canal(canal, kernel, pad_y, pad_x):
    """Convolução: pra cada pixel, multiplica vizinhos pelo kernel e soma."""
    altura, largura = canal.shape
    kh, kw = kernel.shape

    # Padding com réplica das bordas
    padded = np.pad(
        canal.astype(np.float64), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge"
    )
    resultado = np.zeros_like(canal, dtype=np.float64)

    # Percorre cada pixel e aplica o kernel na vizinhança
    for y in range(altura):
        for x in range(largura):
            regiao = padded[y : y + kh, x : x + kw]
            resultado[y, x] = np.sum(regiao * kernel)

    return resultado


def filtro_media(imagem, tamanho=3):
    """Filtro box: kernel com todos os pesos iguais (1/n²). Suaviza a imagem."""
    kernel = np.ones((tamanho, tamanho), dtype=np.float64) / (tamanho * tamanho)
    return _aplicar_filtro(imagem, kernel)


def filtro_mediana(imagem, tamanho=3):
    """Filtro de mediana: ordena vizinhos e pega valor do meio. Bom pra ruído sal e pimenta."""
    pad = tamanho // 2

    if len(imagem.shape) == 3:
        resultado = np.zeros_like(imagem)
        for c in range(imagem.shape[2]):
            resultado[:, :, c] = _mediana_canal(imagem[:, :, c], tamanho, pad)
        return resultado

    return _mediana_canal(imagem, tamanho, pad)


def _mediana_canal(canal, tamanho, pad):
    """Aplica filtro de mediana em um canal."""
    altura, largura = canal.shape
    padded = np.pad(canal, ((pad, pad), (pad, pad)), mode="edge")
    resultado = np.zeros_like(canal)

    for y in range(altura):
        for x in range(largura):
            # Pega todos os vizinhos, ordena e pega o do meio
            vizinhos = padded[y : y + tamanho, x : x + tamanho].ravel()
            resultado[y, x] = np.sort(vizinhos)[len(vizinhos) // 2]

    return resultado


def filtro_gaussiano(imagem, tamanho=3, sigma=1.0):
    """Filtro gaussiano: pesos seguem curva gaussiana. Centro pesa mais, bordas menos."""
    centro = tamanho // 2
    kernel = np.zeros((tamanho, tamanho), dtype=np.float64)

    # Gera pesos com fórmula gaussiana: e^(-(x² + y²) / 2σ²)
    for y in range(tamanho):
        for x in range(tamanho):
            dy = y - centro
            dx = x - centro
            kernel[y, x] = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))

    # Normaliza pra soma dos pesos = 1
    kernel = kernel / kernel.sum()
    return _aplicar_filtro(imagem, kernel)


def filtro_sobel(imagem):
    """Sobel combinado: calcula magnitude √(Gx² + Gy²) das bordas horizontal e vertical."""
    gx = filtro_sobel_horizontal(imagem).astype(np.float64)
    gy = filtro_sobel_vertical(imagem).astype(np.float64)
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude.clip(0, 255).astype(np.uint8)


def filtro_sobel_horizontal(imagem):
    """Sobel horizontal: detecta bordas horizontais com kernel Gy."""
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    return _aplicar_filtro(imagem, kernel)


def filtro_sobel_vertical(imagem):
    """Sobel vertical: detecta bordas verticais com kernel Gx."""
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    return _aplicar_filtro(imagem, kernel)


def filtro_laplaciano(imagem):
    """Laplaciano: detecta bordas em todas as direções e soma à original pra aguçar."""
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

    # Detecta bordas
    bordas = _aplicar_filtro(imagem, kernel)
    # Soma bordas à imagem original pra aguçar
    resultado = imagem.astype(np.float64) + bordas.astype(np.float64)
    return resultado.clip(0, 255).astype(np.uint8)


def filtro_high_boost(imagem, fator=1.5, tamanho=3):
    """High-boost: máscara = original - suavizada, resultado = original + fator * máscara."""
    # 1. Suaviza com filtro box
    suavizada = filtro_media(imagem, tamanho)
    # 2. Máscara = detalhes perdidos na suavização
    mascara = imagem.astype(np.float64) - suavizada.astype(np.float64)
    # 3. Soma máscara amplificada à original
    resultado = imagem.astype(np.float64) + fator * mascara
    return resultado.clip(0, 255).astype(np.uint8)


def agucamento_gradiente(imagem, fator=1.0):
    """Aguçamento por gradiente: g(x,y) = f(x,y) + c * M(x,y)."""
    # 1. Suaviza com gaussiano pra eliminar ruído
    suavizada = filtro_gaussiano(imagem)
    # 2. Calcula gradiente com Sobel (horizontal e vertical)
    gx = filtro_sobel_horizontal(suavizada).astype(np.float64)
    gy = filtro_sobel_vertical(suavizada).astype(np.float64)
    # 3. Magnitude do gradiente = força da borda em qualquer direção
    magnitude = np.sqrt(gx**2 + gy**2)
    # 4. Soma magnitude ponderada à imagem original
    resultado = imagem.astype(np.float64) + fator * magnitude
    return resultado.clip(0, 255).astype(np.uint8)
