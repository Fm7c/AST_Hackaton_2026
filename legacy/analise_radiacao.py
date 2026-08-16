import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# CONFIGURAÇÃO
#NOME_PASTA = "DataFernando"

UNIDADE_TEMPO_ENTRADA = "min"

LIMITE_EVENTO_EM_CADEIA = 2.0
MIN_DETECOES_POR_CADEIA = 2

#if NOME_PASTA:
#    CAMINHO_PADRAO = Path.home() / "Desktop" / NOME_PASTA / "Processed" / "Sardal"
#else:
#    CAMINHO_PADRAO = Path.home() / "Desktop"

CAMINHO_PADRAO = Path.home() / "Desktop"

NUM_BINS_HISTOGRAMA = None
LARGURA_BIN_HISTOGRAMA_MIN = 0.05


BQ_POR_CI = 3.7e10
RAD_POR_GY = 100.0
REM_POR_SV = 100.0

COEFICIENTES_NUCLIDEOS_SV_POR_BQ = {
    "Tritium": 1e-11,
    "Iodine-131": 1e-8,
    "Plutonium-239": 1e-5,
}


CURVA_CONTA_PARA_SV = np.array([
    [0.0001, 0.42999999999999999],
    [0.00011783082199683598, 0.48186660501522499],
    [0.00013435877428667281, 0.57029236976621744],
    [0.0001583160481668601, 0.67494485741192234],
    [0.00018052280227817075, 0.79880178080157016],
    [0.0002058444644096744, 0.94538728313079201],
    [0.00023471795802734155, 1.0577675602831738],
    [0.00026764149319497362, 1.1835067295194666],
    [0.00030518316315316133, 1.4006881788178693],
    [0.00035959982973931469, 1.5671910803783082],
    [0.00041004035734188403, 1.7534865500829593],
    [0.00045246183004857027, 2.0752631321450554],
    [0.00051592797085352746, 2.3219542502889787],
    [0.00058829641183315718, 2.5979700872256446],
    [0.00069319449784090062, 3.252333870135681],
    [0.00081679677484277182, 3.849159021949677],
    [0.0009945456062566793, 4.5555056054680199],
    [0.001097438135091823, 5.3914715404350648],
    [0.0013362592862884309, 6.3808428501175971],
    [0.0015236943678430656, 7.139347751502866],
    [0.0016813309464905979, 7.9880178080156847],
    [0.0019171687111441365, 9.4538728313079403],
    [0.0021860870845575812, 10.577675602831759],
    [0.0024122527093101598, 11.835067295194689],
    [0.0027506157709849609, 13.241927918860453],
    [0.0031364405107269329, 15.671910803783049],
    [0.0036956936352313015, 18.547811900706108],
    [0.0044999392507366143, 21.951460202346972],
    [0.0053023154085012156, 25.979700872256444],
    [0.0062477618307018676, 30.747150813218319],
    [0.0073617891215205632, 34.402132630397588],
    [0.0086744566355613402, 43.067170676405141],
    [0.010221183557441, 50.970286316888924],
    [0.011654893611569161, 60.323676861572267],
    [0.013289708019978464, 67.494485741192364],
    [0.015153835388163679, 79.880178080156853],
    [0.017855888801920666, 94.538728313079403],
    [0.021039740550144129, 111.88722115874199],
    [0.024791299236236426, 132.41927918860398],
    [0.029211791673752669, 156.71910803783049],
    [0.034420494249186029, 175.34865500829557],
    [0.039248605240030397, 207.52631321450554],
    [0.044753948102420692, 232.1954250288974],
    [0.05449316625481266, 290.67965371321191],
    [0.064209745730148135, 344.02132630397517],
    [0.075658871195911762, 407.15155477896775],
    [0.086271427335616185, 455.55056054680108],
    [0.10165433197795951, 570.29236976621632],
    [0.11978013496502206, 638.08428501175899],
    [0.13658151445489153, 755.17704531203481],
    [0.15071180403235152, 798.8017808015685],
    [0.17758495753758058, 945.38728313079105],
    [0.20249453260319963, 1118.8722115874189],
    [0.2465607772879348, 1324.1927918860413],
    [0.29052459060016134, 1567.1910803783035],
    [0.35374766893090331, 1854.781190070609],
    [0.41682378609592979, 2195.1460202346952],
    [0.49114689343516693, 2597.9700872256417],
    [0.54195939069738619, 2906.796537132122],
    [0.63859520494777333, 3440.2132630397518],
    [0.77756435238085431, 4306.7170676405058],
    [0.88663213533134355, 4555.5056054680108],
    [1.0109986922563736, 5391.4715404350536],
    [1.1912680694629461, 6380.842850117584],
    [1.4505082057728624, 7551.7704531203481],
    [1.7661633925627953, 8937.5711510542224],
    [2.0810848432639406, 10577.675602831738],
    [2.4521593772694694, 12518.750258625241],
    [2.7961200872751597, 14006.881788178636],
    [3.294691282854965, 16577.238855893109],
    [4.0116719853189045, 20752.63132145051],
    [4.7269860761180595, 23219.542502889741],
    [5.390033307876319, 27480.48480853334],
    [6.3511205525739101, 30747.150813218257],
    [7.241982686106021, 36389.459900993817],
    [8.5332877279072576, 38491.590219496691],
    [9.7302391711050848, 43067.170676405054],
    [11.465220797571254, 50970.286316888814],
    [13.509563909530415, 60323.676861572145],
    [15.918430202787532, 67494.485741192228],
    [18.756817156937174, 79880.178080156678],
    [22.10131183646261, 94538.728313079017],
    [25.201429627253113, 105776.75602831716],
    [27.808689473655498, 111887.22115874176],
    [31.709372546848417, 118350.67295194641],
    [36.157198571596389, 132419.27918860398],
    [45.494326977278433, 148160.25176256566],
    [53.606339439255429, 165772.3885589311],
    [63.164790403688556, 165773.3885589311]
], dtype=float)


CPM_MINIMO_EXTRAPOLACAO = 0.1


def cpm_para_usv_h(cpm, permitir_extrapolacao_baixa=True):
    escalar = np.isscalar(cpm)
    cpm_array = np.asarray(cpm, dtype=float)

    cpm_minimo_tabela = CURVA_CONTA_PARA_SV[0, 1]

    usv_h = 1e3 * np.interp(
        cpm_array,
        CURVA_CONTA_PARA_SV[:, 1],
        CURVA_CONTA_PARA_SV[:, 0],
        left=np.nan,
        right=np.nan,
    )

    if permitir_extrapolacao_baixa:
        log_cpm_tab = np.log10(CURVA_CONTA_PARA_SV[:10, 1])
        log_dose_tab = np.log10(CURVA_CONTA_PARA_SV[:10, 0])
        declive, ordenada = np.polyfit(log_cpm_tab, log_dose_tab, 1)

        na_zona_extrapolada = (cpm_array >= CPM_MINIMO_EXTRAPOLACAO) & (cpm_array < cpm_minimo_tabela)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_dose_extra = ordenada + declive * np.log10(cpm_array)
        dose_extra_usv_h = 1e3 * (10 ** log_dose_extra)
        usv_h = np.where(na_zona_extrapolada, dose_extra_usv_h, usv_h)

    return float(usv_h) if escalar else usv_h


def bq_para_ci(bq):
    return np.asarray(bq, dtype=float) / BQ_POR_CI


def ci_para_bq(ci):
    return np.asarray(ci, dtype=float) * BQ_POR_CI


def gy_para_rad(gy):
    return np.asarray(gy, dtype=float) * RAD_POR_GY


def rad_para_gy(rad):
    return np.asarray(rad, dtype=float) / RAD_POR_GY


def sv_para_rem(sv):
    return np.asarray(sv, dtype=float) * REM_POR_SV


def rem_para_sv(rem):
    return np.asarray(rem, dtype=float) / REM_POR_SV


def usv_h_para_sv_h(usv_h):
    return np.asarray(usv_h, dtype=float) * 1e-6


def usv_h_para_rem_h(usv_h):
    return sv_para_rem(usv_h_para_sv_h(usv_h))


def atividade_bq_equivalente_de_cpm(cpm):
    return np.asarray(cpm, dtype=float) / 60.0


def dose_sv_por_bq(atividade_bq, nuclideo):
    try:
        coef = COEFICIENTES_NUCLIDEOS_SV_POR_BQ[nuclideo]
    except KeyError as exc:
        disponiveis = ", ".join(COEFICIENTES_NUCLIDEOS_SV_POR_BQ)
        raise ValueError(f"Nuclídeo desconhecido. Usa um destes: {disponiveis}") from exc
    return np.asarray(atividade_bq, dtype=float) * coef


def fator_tempo_para_minutos(unidade):
    unidade = unidade.lower().strip()
    fatores = {
        "s": 1.0 / 60.0,
        "segundo": 1.0 / 60.0,
        "segundos": 1.0 / 60.0,
        "min": 1.0,
        "mins": 1.0,
        "minuto": 1.0,
        "minutos": 1.0,
        "h": 60.0,
        "hora": 60.0,
        "horas": 60.0,
    }
    if unidade not in fatores:
        raise ValueError('UNIDADE_TEMPO_ENTRADA deve ser "s", "min" ou "h".')
    return fatores[unidade]


def carregar_timestamps(caminho):
    valores = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                valores.append(float(linha.replace(",", ".")))
            except ValueError:
                continue
    return np.array(sorted(valores), dtype=float)


def converter_timestamps_para_minutos(timestamps, unidade=UNIDADE_TEMPO_ENTRADA):
    return np.asarray(timestamps, dtype=float) * fator_tempo_para_minutos(unidade)

def detetar_cadeias_de_decaimento(
    timestamps_min,
    intervalos_s,
    limite_intervalo_s,
    min_detecoes_por_cadeia=MIN_DETECOES_POR_CADEIA,
):
    """
    Agrupa deteções consecutivas próximas numa única cadeia.

    Exemplo: se os intervalos entre eventos forem [1 s, 2 s, 20 s]
    e o limite for 3 s, os 3 primeiros eventos contam como 1 cadeia,
    não como 2 pares separados.
    """

    timestamps_min = np.asarray(timestamps_min, dtype=float)
    intervalos_s = np.asarray(intervalos_s, dtype=float)

    cadeias = []
    inicio_cadeia = None
    intervalos_da_cadeia = []

    for i, intervalo_s in enumerate(intervalos_s):
        if intervalo_s <= limite_intervalo_s:
            # O intervalo i liga o evento i ao evento i+1.
            # Se ainda não havia cadeia aberta, esta começa no evento i.
            if inicio_cadeia is None:
                inicio_cadeia = i
                intervalos_da_cadeia = []

            intervalos_da_cadeia.append(float(intervalo_s))
        else:
            # Encontrámos um intervalo grande: se havia cadeia aberta, fechamo-la.
            if inicio_cadeia is not None:
                fim_cadeia = i
                n_detecoes = fim_cadeia - inicio_cadeia + 1

                if n_detecoes >= min_detecoes_por_cadeia:
                    cadeias.append({
                        "indice_primeiro_evento": int(inicio_cadeia),
                        "indice_ultimo_evento": int(fim_cadeia),
                        "indices_eventos": list(range(int(inicio_cadeia), int(fim_cadeia) + 1)),
                        "n_detecoes": int(n_detecoes),
                        "tempo_inicio_min": float(timestamps_min[inicio_cadeia]),
                        "tempo_fim_min": float(timestamps_min[fim_cadeia]),
                        "duracao_cadeia_s": float((timestamps_min[fim_cadeia] - timestamps_min[inicio_cadeia]) * 60.0),
                        "intervalos_s": intervalos_da_cadeia.copy(),
                        "intervalo_medio_s": float(np.mean(intervalos_da_cadeia)),
                        "intervalo_max_s": float(np.max(intervalos_da_cadeia)),
                    })

                inicio_cadeia = None
                intervalos_da_cadeia = []

    # Se o ficheiro acaba enquanto uma cadeia está aberta, também temos de a fechar.
    if inicio_cadeia is not None:
        fim_cadeia = len(timestamps_min) - 1
        n_detecoes = fim_cadeia - inicio_cadeia + 1

        if n_detecoes >= min_detecoes_por_cadeia:
            cadeias.append({
                "indice_primeiro_evento": int(inicio_cadeia),
                "indice_ultimo_evento": int(fim_cadeia),
                "indices_eventos": list(range(int(inicio_cadeia), int(fim_cadeia) + 1)),
                "n_detecoes": int(n_detecoes),
                "tempo_inicio_min": float(timestamps_min[inicio_cadeia]),
                "tempo_fim_min": float(timestamps_min[fim_cadeia]),
                "duracao_cadeia_s": float((timestamps_min[fim_cadeia] - timestamps_min[inicio_cadeia]) * 60.0),
                "intervalos_s": intervalos_da_cadeia.copy(),
                "intervalo_medio_s": float(np.mean(intervalos_da_cadeia)),
                "intervalo_max_s": float(np.max(intervalos_da_cadeia)),
            })

    return cadeias


def calcular_estatisticas(timestamps):
    timestamps_min = converter_timestamps_para_minutos(timestamps)

    n = len(timestamps_min)
    duracao_min = timestamps_min[-1] - timestamps_min[0]
    duracao_h = (duracao_min - (duracao_min % 60.0)) / 60.0
    duracao_hm = duracao_min % 60.0
    intervalos_min = np.diff(timestamps_min)
    intervalos_s = intervalos_min * 60.0
    cadeias_de_decaimento = detetar_cadeias_de_decaimento(
        timestamps_min,
        intervalos_s,
        LIMITE_EVENTO_EM_CADEIA,
    )
    n_deteccoes_em_cadeias = sum(c["n_detecoes"] for c in cadeias_de_decaimento)

    media_intervalo_min = intervalos_min.mean()
    std_intervalo_min = intervalos_min.std(ddof=1)
    mediana_intervalo_min = np.median(intervalos_min)

    media_intervalo_s = media_intervalo_min * 60.0
    std_intervalo_s = std_intervalo_min * 60.0
    mediana_intervalo_s = mediana_intervalo_min * 60.0

    # Como o tempo está em minutos, a taxa natural é CPM = 1 / intervalo médio em minutos.
    taxa_cpm = 1.0 / media_intervalo_min
    taxa_hz = taxa_cpm / 60.0
    atividade_bq = atividade_bq_equivalente_de_cpm(taxa_cpm)

    taxa_dose_usv_h = cpm_para_usv_h(taxa_cpm)
    taxa_dose_sv_h = usv_h_para_sv_h(taxa_dose_usv_h)
    taxa_dose_rem_h = usv_h_para_rem_h(taxa_dose_usv_h)

    return {
        "n_eventos": n,
        "timestamps_min": timestamps_min,
        "duracao_min": duracao_min,
        "duracao_h": duracao_h,
        "duracao_hm": duracao_hm,
        "intervalo_medio_min": media_intervalo_min,
        "intervalo_std_min": std_intervalo_min,
        "intervalo_mediana_min": mediana_intervalo_min,
        "intervalo_medio_s": media_intervalo_s,
        "intervalo_std_s": std_intervalo_s,
        "intervalo_mediana_s": mediana_intervalo_s,
        "razao_std_media": std_intervalo_min / media_intervalo_min,
        "taxa_cpm": taxa_cpm,
        "taxa_hz": taxa_hz,
        "atividade_bq_equivalente": atividade_bq,
        "atividade_ci_equivalente": bq_para_ci(atividade_bq),
        "taxa_dose_usv_h": taxa_dose_usv_h,
        "taxa_dose_sv_h": taxa_dose_sv_h,
        "taxa_dose_rem_h": taxa_dose_rem_h,
        "lambda_por_min": taxa_cpm,
        "lambda_por_s": taxa_hz,
        "intervalos_min": intervalos_min,
        "intervalos_s": intervalos_s,
        "cadeias_de_decaimento": cadeias_de_decaimento,
        "n_cadeias_de_decaimento": len(cadeias_de_decaimento),
        "n_deteccoes_em_cadeias": n_deteccoes_em_cadeias,
        "n_eventos_isolados": n - n_deteccoes_em_cadeias,
        "limite_evento_em_cadeia_s": LIMITE_EVENTO_EM_CADEIA,
        "min_detecoes_por_cadeia": MIN_DETECOES_POR_CADEIA,
    }


def formatar_valor(valor, formato=".4g"):
    valor = float(valor)
    if np.isnan(valor):
        return "fora da curva"
    return f"{valor:{formato}}"


def imprimir_estatisticas(nome, stats):
    print(f"=== {nome} ===")
    print(f"Nº de eventos detetados:        {stats['n_eventos']}")
    print(f"Duração da medição:             {stats['duracao_min']:.4f} min  ({stats['duracao_h']:.0f}:{stats['duracao_hm']:.0f} h)")
    print(f"Taxa média de contagem:         {stats['taxa_cpm']:.4f} CPM  ({stats['taxa_hz']:.6f} Hz)")
    print(f"Atividade equivalente:          {stats['atividade_bq_equivalente']:.6g} Bq  ({stats['atividade_ci_equivalente']:.3e} Ci)")
    print(f"Taxa de dose estimada:          {formatar_valor(stats['taxa_dose_usv_h'], '.6g')} µSv/h")
    print(f"                               {formatar_valor(stats['taxa_dose_sv_h'], '.6g')} Sv/h")
    print(f"                               {formatar_valor(stats['taxa_dose_rem_h'], '.6g')} rem/h")
    print(f"Intervalo médio entre deteções: {stats['intervalo_medio_min']:.6f} min  ({stats['intervalo_medio_s']:.4f} s)")
    print(f"Desvio padrão do intervalo:     {stats['intervalo_std_min']:.6f} min  ({stats['intervalo_std_s']:.4f} s)")
    print(f"Mediana do intervalo:           {stats['intervalo_mediana_min']:.6f} min  ({stats['intervalo_mediana_s']:.4f} s)")
    print(f"Desvio/Média (≈1.0 esperado p/ Poisson puro): {stats['razao_std_media']:.3f}")
    print()
    print(f"Nº de possíveis decaimentos em cadeia:    {stats['n_cadeias_de_decaimento']}")
    print(f"Deteções dentro de cadeias:     {stats['n_deteccoes_em_cadeias']}")
    print(f"Eventos isolados:               {stats['n_eventos_isolados']}")
    print(f"Percentagem de decaimentos em cadeias: {stats['n_cadeias_de_decaimento'] / stats['n_eventos'] * 100:.2f}%")

    #if stats["cadeias_de_decaimento"]:
    #    print()
    #    print("Cadeias detetadas:")
    #    for n_cadeia, cadeia in enumerate(stats["cadeias_de_decaimento"], start=1):
    #        print(
    #            f"  Cadeia {n_cadeia}: "
    #            f"{cadeia['n_detecoes']} deteções próximas "
    #            f"(eventos {cadeia['indice_primeiro_evento']}–{cadeia['indice_ultimo_evento']}), "
    #            f"t={cadeia['tempo_inicio_min']:.2f}–{cadeia['tempo_fim_min']:.2f} min, "
    #            f"duração={cadeia['duracao_cadeia_s']:.2f} s, "
    #            f"intervalo máximo={cadeia['intervalo_max_s']:.2f} s"
    #        )
    if np.isnan(stats["taxa_dose_usv_h"]):
        print("Nota: a CPM média está fora do intervalo da curva Matlab; a conversão para µSv/h ficou indefinida.")
    print()


def grafico_histograma(nome, stats, num_bins=NUM_BINS_HISTOGRAMA, largura_bin_min=LARGURA_BIN_HISTOGRAMA_MIN):
    intervalos = stats["intervalos_min"]
    lam = stats["lambda_por_min"]

    if largura_bin_min is not None:
        bins = np.arange(0, intervalos.max() + largura_bin_min, largura_bin_min)
    else:
        bins = num_bins

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        intervalos, bins=bins, density=True, alpha=0.6,
        color="tab:blue", edgecolor="black", label="Dados observados"
    )

    x = np.linspace(0, intervalos.max(), 30000)
    y = lam * np.exp(-lam * x)
    ax.plot(x, y, "r-", linewidth=2, label=f"Exponencial ajustada (λ={lam:.3f}/min)")

    ax.set_xlabel("Intervalo entre deteções (min)")
    ax.set_ylabel("Densidade de probabilidade (1/min)")
    ax.set_title(f"Distribuição dos intervalos entre deteções — {nome}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def grafico_contagem_acumulada(nome, stats):
    timestamps_min = stats["timestamps_min"]
    fig, ax = plt.subplots(figsize=(9, 5))
    n_acumulado = np.arange(1, len(timestamps_min) + 1)
    ax.step(timestamps_min, n_acumulado, where="post", color="tab:purple")
    ax.set_xlabel("Tempo (min)")
    ax.set_ylabel("Nº de eventos acumulados")
    ax.set_title(f"Processo de contagem acumulada — {nome}")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def calcular_taxa_por_janela(stats, largura_janela_min=None):
    timestamps_min = stats["timestamps_min"]
    duracao_min = stats["duracao_min"]
    if largura_janela_min is None:
        # ~20 janelas ao longo da medição, com mínimo de 1 segundo = 1/60 min.
        largura_janela_min = max(1.0 / 60.0, duracao_min / 20.0)

    bordas = np.arange(timestamps_min[0], timestamps_min[-1] + largura_janela_min, largura_janela_min)
    contagens, _ = np.histogram(timestamps_min, bins=bordas)
    taxa_cpm = contagens / largura_janela_min
    centros = (bordas[:-1] + bordas[1:]) / 2.0
    return centros, taxa_cpm, largura_janela_min


def grafico_taxa_por_janela(nome, stats, largura_janela_min=None):
    centros, taxa_cpm, largura_janela_min = calcular_taxa_por_janela(stats, largura_janela_min)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(centros, taxa_cpm, width=largura_janela_min * 0.9, color="tab:orange", alpha=0.8)
    ax.axhline(stats["taxa_cpm"], color="black", linestyle="--",
               linewidth=1, label="Taxa média global")
    ax.set_xlabel("Tempo (min)")
    ax.set_ylabel("Taxa de contagem (CPM)")
    ax.set_title(f"Taxa de contagem ao longo do tempo — {nome}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def grafico_taxa_dose_por_janela(nome, stats, largura_janela_min=None):
    centros, taxa_cpm, largura_janela_min = calcular_taxa_por_janela(stats, largura_janela_min)
    taxa_dose_usv_h = cpm_para_usv_h(taxa_cpm)

    if np.all(np.isnan(taxa_dose_usv_h)):
        print(f"{nome}: não foi possível desenhar taxa de dose por janela; valores fora da curva Matlab.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centros, taxa_dose_usv_h, marker="o", linewidth=1.5, label="Taxa de dose por janela")
    if not np.isnan(stats["taxa_dose_usv_h"]):
        ax.axhline(stats["taxa_dose_usv_h"], color="black", linestyle="--",
                   linewidth=1, label="Taxa de dose média global")
    ax.set_xlabel("Tempo (min)")
    ax.set_ylabel("Taxa de dose estimada (µSv/h)")
    ax.set_title(f"Taxa de dose estimada ao longo do tempo — {nome}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def analisar_ficheiro(caminho):
    nome = caminho.stem
    timestamps = carregar_timestamps(caminho)

    if len(timestamps) < 2:
        print(f"{nome}: menos de 2 eventos válidos, não é possível analisar.")
        return None

    stats = calcular_estatisticas(timestamps)
    imprimir_estatisticas(nome, stats)

    grafico_histograma(nome, stats)
    grafico_contagem_acumulada(nome, stats)
    grafico_taxa_por_janela(nome, stats, largura_janela_min=70.0)
    grafico_taxa_dose_por_janela(nome, stats, largura_janela_min=70.0)

    return stats


def main():
    if len(sys.argv) >= 2:
        caminho = Path(sys.argv[1])
    else:
        caminho = CAMINHO_PADRAO
        print(f"Nenhum caminho indicado na linha de comandos — a usar o valor "
              f"pré-definido no script:\n  {caminho}\n")

    if not caminho.exists():
        print(f"Caminho não encontrado: {caminho}")
        sys.exit(1)

    resumo = []

    if caminho.is_dir():
        ficheiros = sorted(caminho.glob("*.txt"))
        if not ficheiros:
            print(f"Não há ficheiros .txt em {caminho}")
            sys.exit(1)
        for f in ficheiros:
            stats = analisar_ficheiro(f)
            if stats:
                resumo.append((f.stem, stats))
    else:
        stats = analisar_ficheiro(caminho)
        if stats:
            resumo.append((caminho.stem, stats))

    if len(resumo) > 1:
        print("=== Resumo comparativo ===")
        print(f"{'Ficheiro':<40}{'Nº eventos':>12}{'Cadeias':>10}{'Razão de cadeias':>20}{'Duração(min)':>15}{'Taxa(CPM)':>12}{'Taxa(Hz)':>12}{'Dose(µSv/h)':>15}")
        for nome, s in resumo:
            print(f"{nome:<40}{s['n_eventos']:>12}{s['n_cadeias_de_decaimento']:>10}{s['n_cadeias_de_decaimento'] / s['n_eventos'] * 100:>19.2f}%"
                  f"{s['duracao_min']:>15.2f}{s['taxa_cpm']:>12.4f}{s['taxa_hz']:>12.6f}{s['taxa_dose_usv_h']:>15.6g}")

    print("A abrir plots")
    plt.show()


if __name__ == "__main__":
    main()