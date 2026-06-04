import numpy as np

# === Ampliação e Redução ===


def redimensionar_vizinho(imagem, novo_largura, novo_altura):
    """Redimensiona imagem usando interpolação por vizinho mais próximo."""
    altura, largura = imagem.shape[:2]
    resultado = np.zeros(
        (novo_altura, novo_largura, *imagem.shape[2:]), dtype=imagem.dtype
    )

    escala_y = altura / novo_altura
    escala_x = largura / novo_largura

    for y in range(novo_altura):
        for x in range(novo_largura):
            orig_y = int(y * escala_y)
            orig_x = int(x * escala_x)
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
            # Coordenadas na imagem original
            orig_y = y * escala_y
            orig_x = x * escala_x

            y0 = int(orig_y)
            x0 = int(orig_x)
            y1 = min(y0 + 1, altura - 1)
            x1 = min(x0 + 1, largura - 1)

            # Pesos da interpolação
            dy = orig_y - y0
            dx = orig_x - x0

            # Interpolação bilinear
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
    rgb = imagem.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    intensidade = (r + g + b) / 3.0

    minimo = np.minimum(np.minimum(r, g), b)
    saturacao = np.where(intensidade == 0, 0, 1 - minimo / intensidade)

    numerador = 0.5 * ((r - g) + (r - b))
    denominador = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    denominador = np.where(denominador == 0, 1e-10, denominador)
    theta = np.arccos(np.clip(numerador / denominador, -1, 1))

    matiz = np.where(b <= g, theta, 2 * np.pi - theta)

    return np.stack([matiz, saturacao, intensidade], axis=2)


def hsi_para_rgb(imagem_hsi):
    """Converte imagem HSI para RGB."""
    h, s, i = imagem_hsi[:, :, 0], imagem_hsi[:, :, 1], imagem_hsi[:, :, 2]
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Setor RG (0 <= H < 2pi/3)
    mask1 = h < 2 * np.pi / 3
    b[mask1] = i[mask1] * (1 - s[mask1])
    r[mask1] = i[mask1] * (1 + s[mask1] * np.cos(h[mask1]) / np.cos(np.pi / 3 - h[mask1]))
    g[mask1] = 3 * i[mask1] - (r[mask1] + b[mask1])

    # Setor GB (2pi/3 <= H < 4pi/3)
    mask2 = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    h2 = h[mask2] - 2 * np.pi / 3
    r[mask2] = i[mask2] * (1 - s[mask2])
    g[mask2] = i[mask2] * (1 + s[mask2] * np.cos(h2) / np.cos(np.pi / 3 - h2))
    b[mask2] = 3 * i[mask2] - (r[mask2] + g[mask2])

    # Setor BR (4pi/3 <= H < 2pi)
    mask3 = h >= 4 * np.pi / 3
    h3 = h[mask3] - 4 * np.pi / 3
    g[mask3] = i[mask3] * (1 - s[mask3])
    b[mask3] = i[mask3] * (1 + s[mask3] * np.cos(h3) / np.cos(np.pi / 3 - h3))
    r[mask3] = 3 * i[mask3] - (g[mask3] + b[mask3])

    resultado = np.stack([r, g, b], axis=2)
    return (resultado * 255).clip(0, 255).astype(np.uint8)


# === Transformações de Intensidade ===


def ajuste_saturacao(imagem, fator):
    """Ajusta saturação da imagem. fator > 1 aumenta, < 1 reduz."""
    hsi = rgb_para_hsi(imagem)
    hsi[:, :, 1] = (hsi[:, :, 1] * fator).clip(0, 1)
    return hsi_para_rgb(hsi)


def negativo(imagem):
    """Inverte intensidades da imagem (negativo)."""
    return 255 - imagem


def ajuste_gamma(imagem, gamma):
    """Aplica correção gamma. gamma < 1 clareia, gamma > 1 escurece."""
    normalizada = imagem.astype(np.float64) / 255.0
    corrigida = np.power(normalizada, gamma)
    return (corrigida * 255).clip(0, 255).astype(np.uint8)


def ajuste_brilho(imagem, valor):
    """Ajusta brilho somando valor a todos os pixels."""
    resultado = imagem.astype(np.float64) + valor
    return resultado.clip(0, 255).astype(np.uint8)


def ajuste_contraste(imagem, fator):
    """Ajusta contraste multiplicando pelo fator em torno do meio (128)."""
    resultado = 128 + fator * (imagem.astype(np.float64) - 128)
    return resultado.clip(0, 255).astype(np.uint8)


def equalizar_histograma(imagem):
    """Equaliza histograma da imagem para melhorar contraste."""
    if len(imagem.shape) == 3:
        resultado = np.zeros_like(imagem)
        for c in range(imagem.shape[2]):
            resultado[:, :, c] = _equalizar_canal(imagem[:, :, c])
        return resultado
    return _equalizar_canal(imagem)


def _calcular_histograma(canal):
    """Calcula histograma de um canal."""
    histograma = np.zeros(256, dtype=np.int64)
    for valor in canal.ravel():
        histograma[valor] += 1
    return histograma


def _calcular_cdf(histograma):
    """Calcula distribuição acumulada normalizada."""
    cdf = np.zeros(256, dtype=np.float64)
    cdf[0] = histograma[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histograma[i]
    total = cdf[255]
    if total > 0:
        cdf = cdf / total
    return cdf


def _equalizar_canal(canal):
    """Equaliza histograma de um canal individual."""
    histograma = _calcular_histograma(canal)
    cdf = _calcular_cdf(histograma)

    cdf_min = cdf[cdf > 0].min()
    mapeamento = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapeamento[i] = np.clip(((cdf[i] - cdf_min) / (1.0 - cdf_min)) * 255, 0, 255)

    return mapeamento[canal]


def especificar_histograma(imagem, imagem_referencia):
    """Especificação de histograma — transforma imagem pra ter histograma similar à referência."""
    if len(imagem.shape) == 3:
        resultado = np.zeros_like(imagem)
        for c in range(imagem.shape[2]):
            resultado[:, :, c] = _especificar_canal(
                imagem[:, :, c], imagem_referencia[:, :, c]
            )
        return resultado
    return _especificar_canal(imagem, imagem_referencia)


def _especificar_canal(canal_fonte, canal_referencia):
    """Especificação de histograma para um canal individual."""
    # CDF da fonte e referência
    hist_fonte = _calcular_histograma(canal_fonte)
    cdf_fonte = _calcular_cdf(hist_fonte)

    hist_ref = _calcular_histograma(canal_referencia)
    cdf_ref = _calcular_cdf(hist_ref)

    # Mapeia cada nível da fonte pro nível mais próximo na CDF da referência
    mapeamento = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diferenca_minima = 1.0
        melhor_j = 0
        for j in range(256):
            diferenca = abs(cdf_fonte[i] - cdf_ref[j])
            if diferenca < diferenca_minima:
                diferenca_minima = diferenca
                melhor_j = j
        mapeamento[i] = melhor_j

    return mapeamento[canal_fonte]


def limiarizar(imagem, limiar):
    """Limiarização simples — pixels acima do limiar viram 255, abaixo viram 0."""
    # Converte pra escala de cinza se for colorida
    if len(imagem.shape) == 3:
        cinza = np.mean(imagem.astype(np.float64), axis=2)
    else:
        cinza = imagem.astype(np.float64)

    resultado = np.zeros_like(cinza, dtype=np.uint8)
    resultado[cinza >= limiar] = 255
    return resultado


# === Filtragem Espacial ===


def _aplicar_filtro(imagem, kernel):
    """Aplica convolução com kernel na imagem."""
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
    """Aplica convolução em um canal individual."""
    altura, largura = canal.shape
    kh, kw = kernel.shape

    padded = np.pad(
        canal.astype(np.float64), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge"
    )
    resultado = np.zeros_like(canal, dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            regiao = padded[y : y + kh, x : x + kw]
            resultado[y, x] = np.sum(regiao * kernel)

    return resultado


def filtro_media(imagem, tamanho=3):
    """Filtro de média (suavização) com kernel de tamanho x tamanho."""
    kernel = np.ones((tamanho, tamanho), dtype=np.float64) / (tamanho * tamanho)
    return _aplicar_filtro(imagem, kernel)


def filtro_mediana(imagem, tamanho=3):
    """Filtro de mediana (suavização) — bom pra ruído sal e pimenta."""
    altura, largura = imagem.shape[:2]
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
            vizinhos = padded[y:y + tamanho, x:x + tamanho].ravel()
            resultado[y, x] = np.sort(vizinhos)[len(vizinhos) // 2]

    return resultado


def filtro_gaussiano(imagem, tamanho=3, sigma=1.0):
    """Filtro gaussiano para suavização."""
    centro = tamanho // 2
    kernel = np.zeros((tamanho, tamanho), dtype=np.float64)

    for y in range(tamanho):
        for x in range(tamanho):
            dy = y - centro
            dx = x - centro
            kernel[y, x] = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))

    kernel = kernel / kernel.sum()
    return _aplicar_filtro(imagem, kernel)


def filtro_sobel(imagem):
    """Filtro Sobel — magnitude combinada das bordas horizontais e verticais."""
    gx = filtro_sobel_horizontal(imagem).astype(np.float64)
    gy = filtro_sobel_vertical(imagem).astype(np.float64)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return magnitude.clip(0, 255).astype(np.uint8)


def filtro_sobel_horizontal(imagem):
    """Filtro Sobel horizontal — detecta bordas horizontais."""
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    return _aplicar_filtro(imagem, kernel)


def filtro_sobel_vertical(imagem):
    """Filtro Sobel vertical — detecta bordas verticais."""
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    return _aplicar_filtro(imagem, kernel)


def filtro_laplaciano(imagem):
    """Filtro laplaciano para aguçamento."""
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

    bordas = _aplicar_filtro(imagem, kernel)
    resultado = imagem.astype(np.float64) + bordas.astype(np.float64)
    return resultado.clip(0, 255).astype(np.uint8)


def filtro_high_boost(imagem, fator=1.5, tamanho=3):
    """Filtro high-boost para aguçamento. fator > 1 aumenta nitidez."""
    suavizada = filtro_media(imagem, tamanho)
    mascara = imagem.astype(np.float64) - suavizada.astype(np.float64)
    resultado = imagem.astype(np.float64) + fator * mascara
    return resultado.clip(0, 255).astype(np.uint8)


def agucamento_gradiente(imagem, fator=1.0):
    """Aguçamento usando gradientes (Sobel). g(x,y) = f(x,y) + c*M(x,y)"""
    # 1. Suaviza com gaussiano pra eliminar ruído
    suavizada = filtro_gaussiano(imagem)
    # 2. Calcula gradiente com Sobel
    gx = filtro_sobel_horizontal(suavizada).astype(np.float64)
    gy = filtro_sobel_vertical(suavizada).astype(np.float64)
    # 3. Magnitude do gradiente
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # 4. Soma magnitude à imagem original
    resultado = imagem.astype(np.float64) + fator * magnitude
    return resultado.clip(0, 255).astype(np.uint8)
