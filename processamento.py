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


# === Transformações de Intensidade ===


def negativo(imagem):
    """Inverte intensidades da imagem (negativo)."""
    return 255 - imagem


def ajuste_gamma(imagem, gamma):
    """Aplica correção gamma. gamma < 1 clareia, gamma > 1 escurece."""
    normalizada = imagem.astype(np.float64) / 255.0
    corrigida = np.power(normalizada, gamma)
    return (corrigida * 255).clip(0, 255).astype(np.uint8)


def equalizar_histograma(imagem):
    """Equaliza histograma da imagem para melhorar contraste."""
    if len(imagem.shape) == 3:
        resultado = np.zeros_like(imagem)
        for c in range(imagem.shape[2]):
            resultado[:, :, c] = _equalizar_canal(imagem[:, :, c])
        return resultado
    return _equalizar_canal(imagem)


def _equalizar_canal(canal):
    """Equaliza histograma de um canal individual."""
    histograma = np.zeros(256, dtype=np.int64)
    for valor in canal.ravel():
        histograma[valor] += 1

    # Distribuição acumulada
    cdf = np.zeros(256, dtype=np.float64)
    cdf[0] = histograma[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histograma[i]

    # Normaliza CDF
    cdf_min = cdf[cdf > 0].min()
    total = canal.size
    mapeamento = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapeamento[i] = np.clip(((cdf[i] - cdf_min) / (total - cdf_min)) * 255, 0, 255)

    return mapeamento[canal]


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
