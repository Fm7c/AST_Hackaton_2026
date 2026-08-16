import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FICHEIRO_1 = "medicoes_uv.txt"

ALFA_CROSSTALK_UVC = 0.008

RE_CABECALHO = re.compile(
    r"Hora_python\s*-\s*(\d{2}):(\d{2});\s*Hora_API\s*-\s*(\d{2}):(\d{2});"
    r"\s*Índice_UV_API\s*-\s*([\d.,]+|None)"
)

def parse_valor(valor_str):
    valor_str = valor_str.strip()
    try:
        return float(valor_str.replace(",", "."))
    except ValueError:
        return None


def carregar_dados(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        linhas = [linha.rstrip("\n") for linha in f if linha.strip()]

    registos = []
    dia_offset = 0
    minutos_anteriores = None
    data_base = datetime(2000, 1, 1)

    i = 0
    while i < len(linhas):
        m = RE_CABECALHO.match(linhas[i])
        if not m:
            i += 1
            continue

        hp_h, hp_m, ha_h, ha_m, uvi_api_str = m.groups()
        hp_h, hp_m = int(hp_h), int(hp_m)

        minutos_atuais = hp_h * 60 + hp_m
        if minutos_anteriores is not None and minutos_atuais < minutos_anteriores:
            dia_offset += 1
        minutos_anteriores = minutos_atuais

        timestamp = data_base + timedelta(days=dia_offset, hours=hp_h, minutes=hp_m)

        registo = {
            "timestamp": timestamp,
            "hora_api_str": f"{ha_h}:{ha_m}",
            "uvi_api": parse_valor(uvi_api_str) if uvi_api_str != "None" else None,
        }

        # Linha de dados imediatamente a seguir ao cabeçalho
        if i + 1 < len(linhas) and "=" in linhas[i + 1]:
            for par in linhas[i + 1].split(","):
                if "=" not in par:
                    continue
                chave, valor = par.split("=", 1)
                registo[chave.strip()] = parse_valor(valor)
            i += 2
        else:
            i += 1

        registos.append(registo)

    if not registos:
        raise ValueError("Não foi possível interpretar nenhum registo no ficheiro.")

    df = pd.DataFrame(registos).sort_values("timestamp").reset_index(drop=True)

    if "uva" in df.columns and "uvc" in df.columns:
        df["uvc_corrigido"] = df["uvc"] - ALFA_CROSSTALK_UVC * df["uva"]

    return df


def inserir_lacunas(df, tolerancia=timedelta(minutes=1, seconds=30)):
    """
    Sempre que o intervalo entre duas medições consecutivas for maior do
    que a tolerância (ou seja, faltam um ou mais minutos), insere uma
    linha de valores NaN a meio desse intervalo. Como o matplotlib não
    desenha uma linha através de valores NaN, isto faz com que o plot
    fique com um espaço em branco nesses períodos, em vez de ligar os
    pontos antes e depois da falha como se fossem contínuos.
    """
    timestamps = df["timestamp"].tolist()
    linhas_novas = []

    for i in range(len(timestamps) - 1):
        delta = timestamps[i + 1] - timestamps[i]
        if delta > tolerancia:
            lacuna = {coluna: float("nan") for coluna in df.columns}
            lacuna["timestamp"] = timestamps[i] + delta / 2
            linhas_novas.append(lacuna)

    if not linhas_novas:
        return df

    df_lacunas = pd.DataFrame(linhas_novas)
    df_final = pd.concat([df, df_lacunas], ignore_index=True)
    df_final = df_final.sort_values("timestamp").reset_index(drop=True)
    return df_final


def formatar_eixo_x(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("dia %d  %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Hora da medição")
    ax.grid(True, alpha=0.3)


def grafico_uv(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(df["timestamp"], df["uva"], label="UVA (µW/cm²)", color="tab:purple")
    ax1.plot(df["timestamp"], df["uvb"], label="UVB (µW/cm²)", color="tab:blue")
    ax1.plot(df["timestamp"], df["uvc"], label="UVC corrigido (µW/cm²)", color="tab:cyan")
#    if "uvc_corrigido" in df.columns:
#        ax1.plot(df["timestamp"], df["uvc_corrigido"], label="UVC corrigido (µW/cm²)",
#                  color="tab:cyan", linestyle=":", linewidth=1.5)
#    ax1.set_ylabel("Irradiância (µW/cm²)")

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["uvi"], label="UVI sensor (aprox.)",
              color="tab:orange", linestyle="--")
    ax2.plot(df["timestamp"], df["uvi_api"], label="UVI referência (API)",
              color="tab:red", linestyle="-", marker="o", markersize=3)
    ax2.set_ylabel("Índice UV")

    linhas1, labels1 = ax1.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("Irradiância UVA/UVB/UVC e Índice UV (sensor vs. referência)")
    formatar_eixo_x(ax1)

    fig.tight_layout()


def grafico_opt3001(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["timestamp"], df["opt3001_lux"], color="tab:green")
    ax.set_ylabel("Luminosidade (lux)")
    ax.set_title("OPT3001 (Ambient 2 Click) — Luminosidade ambiente")
    formatar_eixo_x(ax)
    fig.tight_layout()


def grafico_veml_canais(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    canais = [
        ("veml_C", "Clear", "black"),
        ("veml_R", "Vermelho", "tab:red"),
        ("veml_G", "Verde", "tab:green"),
        ("veml_B", "Azul", "tab:blue"),
        ("veml_I", "Infravermelho", "tab:brown"),
    ]
    for coluna, nome, cor in canais:
        if coluna in df.columns:
            ax.plot(df["timestamp"], df[coluna], label=nome, color=cor)

    ax.set_ylabel("Contagem bruta")
    ax.set_title("VEML3328 (Color 10 Click) — Canais RGB + Clear + IV")
    ax.legend(loc="upper left", fontsize=9)
    formatar_eixo_x(ax)
    fig.tight_layout()


def grafico_veml_lux_cct(df):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["timestamp"], df["veml_lux"], color="tab:green", label="Lux (VEML)")
    ax1.set_ylabel("Lux (VEML3328)")

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["veml_cct"], color="tab:orange", label="CCT (K)")
    ax2.set_ylabel("Temperatura de cor correlacionada (K)")

    linhas1, labels1 = ax1.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("VEML3328 — Lux estimado e Temperatura de Cor (CCT)")
    formatar_eixo_x(ax1)
    fig.tight_layout()


def grafico_uv_temp(df):
    if "uvTemp" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["uvTemp"], color="tab:red")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title("AS7331 — Temperatura interna do sensor")
    formatar_eixo_x(ax)
    fig.tight_layout()


def main():
    caminho = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(FICHEIRO_1)
    if not caminho.exists():
        print(f"Ficheiro não encontrado: {caminho}")
        sys.exit(1)

    print(f"A ler {caminho} ...")
    df = carregar_dados(caminho)
    print(f"{len(df)} registos lidos, de {df['timestamp'].min()} a {df['timestamp'].max()}.")
    df = inserir_lacunas(df)

    grafico_uv(df)
    grafico_opt3001(df)
    grafico_veml_canais(df)
    #grafico_veml_lux_cct(df)
    grafico_uv_temp(df)
    plt.show()


if __name__ == "__main__":
    main()