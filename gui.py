import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import processamento as proc

DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))


def carregar_estilo():
    """Carrega stylesheet do arquivo estilo.qss."""
    caminho = os.path.join(DIRETORIO_BASE, "estilo.qss")
    with open(caminho, "r") as f:
        return f.read()


def carregar_imagem(caminho):
    """Carrega imagem com OpenCV e retorna array uint8 RGB."""
    # cv2.imread lê em BGR por padrão
    imagem = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar: {caminho}")
    # Converte BGR → RGB
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def array_para_qimage(array):
    """Converte numpy array para QImage pra exibição."""
    if len(array.shape) == 2:
        array = np.stack([array, array, array], axis=2)
    array = np.ascontiguousarray(array)
    altura, largura, canais = array.shape
    bytes_por_linha = largura * canais
    return QImage(
        array.data, largura, altura, bytes_por_linha, QImage.Format_RGB888
    ).copy()


def atualizar_info(estado, label_info):
    """Atualiza label de informações da imagem."""
    if estado["imagem_atual"] is None:
        label_info.setText("")
        return

    atual = estado["imagem_atual"]
    texto = f"Atual: {atual.shape[1]} x {atual.shape[0]} px"

    if estado["imagem_original"] is not None:
        orig = estado["imagem_original"]
        texto += f"    |    Original: {orig.shape[1]} x {orig.shape[0]} px"

    label_info.setText(texto)


def exibir_imagem(estado, widgets):
    """Atualiza exibição da imagem na tela."""
    if estado["imagem_atual"] is None:
        return
    qimage = array_para_qimage(estado["imagem_atual"])
    pixmap = QPixmap.fromImage(qimage)
    widgets["label_imagem"].setPixmap(pixmap)
    widgets["label_imagem"].adjustSize()
    atualizar_info(estado, widgets["label_info"])


def aplicar_processamento(estado, janela, widgets, funcao, *args):
    """Aplica função de processamento na imagem atual."""
    if estado["imagem_atual"] is None:
        QMessageBox.information(janela, "Aviso", "Nenhuma imagem carregada.")
        return

    estado["historico"].append(estado["imagem_atual"].copy())
    resultado = funcao(estado["imagem_atual"], *args)
    estado["imagem_atual"] = resultado
    exibir_imagem(estado, widgets)


def abrir_imagem(estado, janela, widgets):
    """Abre diálogo para carregar arquivo de imagem."""
    filtros = "Imagens (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Todos os arquivos (*)"
    caminho, _ = QFileDialog.getOpenFileName(janela, "Abrir Imagem", "", filtros)

    if not caminho:
        return

    try:
        imagem = carregar_imagem(caminho)
    except Exception as e:
        QMessageBox.warning(janela, "Erro", f"Não foi possível abrir: {caminho}\n{e}")
        return

    estado["imagem_original"] = imagem.copy()
    estado["imagem_atual"] = imagem
    exibir_imagem(estado, widgets)
    janela.setWindowTitle(f"Editor de Imagens — {caminho}")


def salvar_imagem(estado, janela):
    """Abre diálogo para salvar imagem modificada."""
    if estado["imagem_atual"] is None:
        QMessageBox.information(janela, "Aviso", "Nenhuma imagem para salvar.")
        return

    filtros = "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;TIFF (*.tiff)"
    caminho, filtro_usado = QFileDialog.getSaveFileName(
        janela, "Salvar Imagem", "", filtros
    )

    if not caminho:
        return

    # Adiciona extensão se usuário não digitou
    extensoes = {
        "PNG (*.png)": ".png",
        "JPEG (*.jpg)": ".jpg",
        "BMP (*.bmp)": ".bmp",
        "TIFF (*.tiff)": ".tiff",
    }
    if "." not in os.path.basename(caminho):
        caminho += extensoes.get(filtro_usado, ".png")

    try:
        # Converte RGB → BGR pra OpenCV salvar
        bgr = cv2.cvtColor(estado["imagem_atual"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(caminho, bgr)
    except Exception as e:
        QMessageBox.warning(janela, "Erro", f"Não foi possível salvar: {caminho}\n{e}")


def desfazer(estado, janela, widgets):
    """Desfaz última operação."""
    if not estado["historico"]:
        QMessageBox.information(janela, "Aviso", "Nada para desfazer.")
        return

    estado["imagem_atual"] = estado["historico"].pop()
    exibir_imagem(estado, widgets)


def restaurar_original(estado, janela, widgets):
    """Restaura imagem ao estado original."""
    if estado["imagem_original"] is None:
        QMessageBox.information(
            janela, "Aviso", "Nenhuma imagem original para restaurar."
        )
        return

    estado["historico"].append(estado["imagem_atual"].copy())
    estado["imagem_atual"] = estado["imagem_original"].copy()
    exibir_imagem(estado, widgets)


def pedir_redimensionar(estado, janela, widgets, metodo):
    """Pede dimensões e redimensiona imagem."""
    if estado["imagem_atual"] is None:
        QMessageBox.information(janela, "Aviso", "Nenhuma imagem carregada.")
        return

    altura, largura = estado["imagem_atual"].shape[:2]
    proporcao = largura / altura

    dialogo = QDialog(janela)
    dialogo.setWindowTitle("Redimensionar")
    layout = QFormLayout(dialogo)

    spin_largura = QSpinBox()
    spin_largura.setRange(1, 10000)
    spin_largura.setValue(largura)
    layout.addRow("Largura:", spin_largura)

    spin_altura = QSpinBox()
    spin_altura.setRange(1, 10000)
    spin_altura.setValue(altura)
    layout.addRow("Altura:", spin_altura)

    check_proporcao = QCheckBox("Manter proporção")
    check_proporcao.setChecked(True)
    layout.addRow(check_proporcao)

    atualizando = {"flag": False}

    def largura_mudou(valor):
        if check_proporcao.isChecked() and not atualizando["flag"]:
            atualizando["flag"] = True
            spin_altura.setValue(int(valor / proporcao))
            atualizando["flag"] = False

    def altura_mudou(valor):
        if check_proporcao.isChecked() and not atualizando["flag"]:
            atualizando["flag"] = True
            spin_largura.setValue(int(valor * proporcao))
            atualizando["flag"] = False

    spin_largura.valueChanged.connect(largura_mudou)
    spin_altura.valueChanged.connect(altura_mudou)

    botoes = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    botoes.accepted.connect(dialogo.accept)
    botoes.rejected.connect(dialogo.reject)
    layout.addRow(botoes)

    if dialogo.exec_() != QDialog.Accepted:
        return

    estado["historico"].append(estado["imagem_atual"].copy())
    resultado = metodo(
        estado["imagem_atual"], spin_largura.value(), spin_altura.value()
    )
    estado["imagem_atual"] = resultado
    exibir_imagem(estado, widgets)


def pedir_saturacao(estado, janela, widgets):
    """Pede fator de saturação e aplica."""
    fator, ok = QInputDialog.getDouble(
        janela, "Saturação", "Fator (< 1 reduz, > 1 aumenta):", 1.0, 0.0, 10.0, 2
    )
    if not ok:
        return
    aplicar_processamento(estado, janela, widgets, proc.ajuste_saturacao, fator)


def pedir_brilho(estado, janela, widgets):
    """Pede valor de brilho e aplica."""
    valor, ok = QInputDialog.getInt(
        janela, "Brilho", "Valor (-255 a 255):", 0, -255, 255
    )
    if not ok:
        return
    aplicar_processamento(estado, janela, widgets, proc.ajuste_brilho, valor)


def pedir_contraste(estado, janela, widgets):
    """Pede fator de contraste e aplica."""
    fator, ok = QInputDialog.getDouble(
        janela, "Contraste", "Fator (< 1 reduz, > 1 aumenta):", 1.0, 0.0, 10.0, 2
    )
    if not ok:
        return
    aplicar_processamento(estado, janela, widgets, proc.ajuste_contraste, fator)


def pedir_gamma(estado, janela, widgets):
    """Pede valor gamma e aplica correção."""
    gamma, ok = QInputDialog.getDouble(
        janela, "Gamma", "Valor gamma (< 1 clareia, > 1 escurece):", 1.0, 0.1, 10.0, 2
    )
    if not ok:
        return
    aplicar_processamento(estado, janela, widgets, proc.ajuste_gamma, gamma)


def pedir_especificacao_histograma(estado, janela, widgets):
    """Pede imagem de referência e aplica especificação de histograma."""
    if estado["imagem_atual"] is None:
        QMessageBox.information(janela, "Aviso", "Nenhuma imagem carregada.")
        return

    filtros = "Imagens (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Todos os arquivos (*)"
    caminho, _ = QFileDialog.getOpenFileName(
        janela, "Imagem de Referência", "", filtros
    )

    if not caminho:
        return

    try:
        array_ref = carregar_imagem(caminho)
    except Exception as e:
        QMessageBox.warning(janela, "Erro", f"Não foi possível abrir: {caminho}\n{e}")
        return

    aplicar_processamento(
        estado, janela, widgets, proc.especificar_histograma, array_ref
    )


def pedir_limiar(estado, janela, widgets):
    """Pede valor de limiar e aplica limiarização."""
    limiar, ok = QInputDialog.getInt(
        janela, "Limiarização", "Valor do limiar (0 a 255):", 128, 0, 255
    )
    if not ok:
        return
    aplicar_processamento(estado, janela, widgets, proc.limiarizar, limiar)


def pedir_high_boost(estado, janela, widgets):
    """Pede fator e aplica high-boost."""
    fator, ok = QInputDialog.getDouble(
        janela, "High-Boost", "Fator de aguçamento:", 1.5, 0.1, 10.0, 2
    )
    if not ok:
        return
    aplicar_processamento(estado, janela, widgets, proc.filtro_high_boost, fator)


def pedir_agucamento_gradiente(estado, janela, widgets):
    """Pede fator e aplica aguçamento por gradiente."""
    fator, ok = QInputDialog.getDouble(
        janela, "Aguçamento (Gradiente)", "Fator c:", 1.0, 0.1, 10.0, 2
    )
    if not ok:
        return
    aplicar_processamento(estado, janela, widgets, proc.agucamento_gradiente, fator)


def criar_menu(estado, janela, widgets):
    """Cria barra de menu com ações."""
    barra_menu = janela.menuBar()
    barra_menu.setNativeMenuBar(False)

    # Menu Arquivo
    menu_arquivo = barra_menu.addMenu("  Arquivo  ")

    acao_abrir = QAction("Abrir Imagem", janela)
    acao_abrir.setShortcut("Ctrl+O")
    acao_abrir.triggered.connect(lambda: abrir_imagem(estado, janela, widgets))
    menu_arquivo.addAction(acao_abrir)

    acao_salvar = QAction("Salvar Imagem", janela)
    acao_salvar.setShortcut("Ctrl+S")
    acao_salvar.triggered.connect(lambda: salvar_imagem(estado, janela))
    menu_arquivo.addAction(acao_salvar)

    acao_desfazer = QAction("Desfazer", janela)
    acao_desfazer.setShortcut("Ctrl+Z")
    acao_desfazer.triggered.connect(lambda: desfazer(estado, janela, widgets))
    menu_arquivo.addAction(acao_desfazer)

    acao_restaurar = QAction("Restaurar Original", janela)
    acao_restaurar.setShortcut("Ctrl+Shift+Z")
    acao_restaurar.triggered.connect(
        lambda: restaurar_original(estado, janela, widgets)
    )
    menu_arquivo.addAction(acao_restaurar)

    menu_arquivo.addSeparator()

    acao_sair = QAction("Sair", janela)
    acao_sair.setShortcut("Ctrl+Q")
    acao_sair.triggered.connect(janela.close)
    menu_arquivo.addAction(acao_sair)

    # Menu Redimensionar
    menu_redimensionar = barra_menu.addMenu("  Redimensionar  ")

    acao_vizinho = QAction("Vizinho Mais Próximo", janela)
    acao_vizinho.triggered.connect(
        lambda: pedir_redimensionar(estado, janela, widgets, proc.redimensionar_vizinho)
    )
    menu_redimensionar.addAction(acao_vizinho)

    acao_bilinear = QAction("Interpolação Bilinear", janela)
    acao_bilinear.triggered.connect(
        lambda: pedir_redimensionar(
            estado, janela, widgets, proc.redimensionar_bilinear
        )
    )
    menu_redimensionar.addAction(acao_bilinear)

    # Menu Intensidade
    menu_intensidade = barra_menu.addMenu("  Intensidade  ")

    acao_brilho = QAction("Brilho", janela)
    acao_brilho.triggered.connect(lambda: pedir_brilho(estado, janela, widgets))
    menu_intensidade.addAction(acao_brilho)

    acao_contraste = QAction("Contraste", janela)
    acao_contraste.triggered.connect(lambda: pedir_contraste(estado, janela, widgets))
    menu_intensidade.addAction(acao_contraste)

    acao_saturacao = QAction("Saturação", janela)
    acao_saturacao.triggered.connect(lambda: pedir_saturacao(estado, janela, widgets))
    menu_intensidade.addAction(acao_saturacao)

    acao_negativo = QAction("Negativo", janela)
    acao_negativo.triggered.connect(
        lambda: aplicar_processamento(estado, janela, widgets, proc.negativo)
    )
    menu_intensidade.addAction(acao_negativo)

    acao_gamma = QAction("Correção Gamma", janela)
    acao_gamma.triggered.connect(lambda: pedir_gamma(estado, janela, widgets))
    menu_intensidade.addAction(acao_gamma)

    menu_intensidade.addSeparator()

    acao_equalizar = QAction("Equalizar Histograma", janela)
    acao_equalizar.triggered.connect(
        lambda: aplicar_processamento(
            estado, janela, widgets, proc.equalizar_histograma
        )
    )
    menu_intensidade.addAction(acao_equalizar)

    acao_especificar = QAction("Especificação de Histograma", janela)
    acao_especificar.triggered.connect(
        lambda: pedir_especificacao_histograma(estado, janela, widgets)
    )
    menu_intensidade.addAction(acao_especificar)

    acao_limiar = QAction("Limiarização", janela)
    acao_limiar.triggered.connect(lambda: pedir_limiar(estado, janela, widgets))
    menu_intensidade.addAction(acao_limiar)

    # Menu Filtros
    menu_filtros = barra_menu.addMenu("  Filtros  ")

    acao_media = QAction("Box (Suavização)", janela)
    acao_media.triggered.connect(
        lambda: aplicar_processamento(estado, janela, widgets, proc.filtro_media)
    )
    menu_filtros.addAction(acao_media)

    acao_gaussiano = QAction("Gaussiano (Suavização)", janela)
    acao_gaussiano.triggered.connect(
        lambda: aplicar_processamento(estado, janela, widgets, proc.filtro_gaussiano)
    )
    menu_filtros.addAction(acao_gaussiano)

    acao_mediana = QAction("Mediana (Suavização)", janela)
    acao_mediana.triggered.connect(
        lambda: aplicar_processamento(estado, janela, widgets, proc.filtro_mediana)
    )
    menu_filtros.addAction(acao_mediana)

    menu_filtros.addSeparator()

    acao_sobel = QAction("Sobel (Bordas)", janela)
    acao_sobel.triggered.connect(
        lambda: aplicar_processamento(estado, janela, widgets, proc.filtro_sobel)
    )
    menu_filtros.addAction(acao_sobel)

    acao_sobel_h = QAction("Sobel Horizontal (Bordas)", janela)
    acao_sobel_h.triggered.connect(
        lambda: aplicar_processamento(
            estado, janela, widgets, proc.filtro_sobel_horizontal
        )
    )
    menu_filtros.addAction(acao_sobel_h)

    acao_sobel_v = QAction("Sobel Vertical (Bordas)", janela)
    acao_sobel_v.triggered.connect(
        lambda: aplicar_processamento(
            estado, janela, widgets, proc.filtro_sobel_vertical
        )
    )
    menu_filtros.addAction(acao_sobel_v)

    acao_laplaciano = QAction("Laplaciano (Aguçamento)", janela)
    acao_laplaciano.triggered.connect(
        lambda: aplicar_processamento(estado, janela, widgets, proc.filtro_laplaciano)
    )
    menu_filtros.addAction(acao_laplaciano)

    acao_highboost = QAction("High-Boost (Aguçamento)", janela)
    acao_highboost.triggered.connect(lambda: pedir_high_boost(estado, janela, widgets))
    menu_filtros.addAction(acao_highboost)

    acao_grad = QAction("Gradiente (Aguçamento)", janela)
    acao_grad.triggered.connect(
        lambda: pedir_agucamento_gradiente(estado, janela, widgets)
    )
    menu_filtros.addAction(acao_grad)


def criar_janela(estado):
    """Cria e configura janela principal."""
    janela = QMainWindow()
    janela.setWindowTitle("Editor de Imagens — IPI")
    janela.setMinimumSize(800, 600)
    janela.setStyleSheet(carregar_estilo())

    # Layout central com imagem e barra de info
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    label_imagem = QLabel()
    label_imagem.setAlignment(Qt.AlignCenter)
    label_imagem.setText("Nenhuma imagem carregada")
    label_imagem.setFont(QFont("Segoe UI", 16))

    scroll_area = QScrollArea()
    scroll_area.setWidget(label_imagem)
    scroll_area.setWidgetResizable(True)
    scroll_area.setObjectName("scroll_area")
    layout.addWidget(scroll_area, 1)

    label_info = QLabel()
    label_info.setAlignment(Qt.AlignCenter)
    label_info.setObjectName("label_info")
    layout.addWidget(label_info)

    janela.setCentralWidget(container)

    widgets = {
        "label_imagem": label_imagem,
        "label_info": label_info,
    }

    criar_menu(estado, janela, widgets)

    return janela


def iniciar_app(estado):
    """Cria e executa aplicação."""
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 11))
    janela = criar_janela(estado)
    janela.show()
    app.exec_()
