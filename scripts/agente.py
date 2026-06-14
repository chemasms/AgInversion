#!/usr/bin/env python3
# o
import yfinance as yf
import pandas as pd
import talib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging

# Configuración de logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")  # Ruta absoluta a la carpeta logs/
os.makedirs(LOG_DIR, exist_ok=True)  # Crea la carpeta si no existe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"agente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

# Configuración del agente
TICKER = "AAPL"  # Símbolo de la acción (Apple)
PERIOD = "1y"   # Período de datos (1 año)
INTERVAL = "1d" # Intervalo (diario)
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

def descargar_datos(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Descarga datos de Yahoo Finance usando yf.Ticker.
    - Garantiza columnas estándar: ['Open', 'High', 'Low', 'Close', 'Volume'].
    - Evita el problema de MultiIndex.
    """
    logging.info(f"Descargando datos de {ticker} para el período {period}...")

    # Usar yf.Ticker en lugar de yf.download
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period=period, interval=interval)

    # Verificar que las columnas sean estándar
    columnas_esperadas = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in columnas_esperadas:
        if col not in data.columns:
            logging.error(f"Columna '{col}' no encontrada. Columnas disponibles: {data.columns.tolist()}")
            raise ValueError(f"Falta la columna estándar: {col}")

    logging.info(f"Datos descargados: {len(data)} registros. Columnas: {data.columns.tolist()}")
    return data

def calcular_indicadores(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos (SMA_20, RSI_14, MACD) usando TA-Lib.
    - Maneja columnas con MultiIndex (ej: ('Close', 'AAPL')).
    - Renombra la columna 'Close' a un nombre simple antes de calcular.
    """
    logging.info("Calculando indicadores técnicos con TA-Lib...")

    # --- 1. Encontrar la columna que contiene 'Close' (aunque sea MultiIndex) ---
    close_col = None
    for col in data.columns:
        if isinstance(col, tuple) and 'Close' in col:
            close_col = col
            break
        elif 'Close' in str(col):
            close_col = col
            break

    if close_col is None:
        logging.error(f"No se encontró la columna 'Close'. Columnas disponibles: {data.columns.tolist()}")
        raise KeyError("No se encontró la columna 'Close' en el DataFrame.")

    # --- 2. Renombrar la columna a 'Close' (si no lo es) ---
    if close_col != 'Close':
        data = data.rename(columns={close_col: 'Close'})

    # --- 3. Convertir 'Close' a array 1D de float64 ---
    close_prices = data['Close'].astype('float64').to_numpy().flatten()

    # --- 4. Calcular indicadores con TA-Lib ---
    sma_20 = talib.SMA(close_prices, timeperiod=20)
    rsi_14 = talib.RSI(close_prices, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        close_prices, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # --- 5. Crear un DataFrame con los indicadores y el mismo índice ---
    indicadores = pd.DataFrame({
        'SMA_20': sma_20,
        'RSI_14': rsi_14,
        'MACD': macd,
        'MACD_Signal': macd_signal,
        'MACD_Hist': macd_hist
    }, index=data.index)

    # --- 6. Rellenar NaN ---
    indicadores = indicadores.bfill()
    indicadores['MACD_Hist'] = indicadores['MACD_Hist'].fillna(0)

    # --- 7. Unir con el DataFrame original ---
    data = pd.concat([data, indicadores], axis=1)

    logging.info("Indicadores calculados: SMA_20, RSI_14, MACD.")
    return data

def generar_senales(data: pd.DataFrame) -> pd.DataFrame:
    """Genera señales de compra (1) y venta (-1)."""
    logging.info("Generando señales de compra/venta...")

    # --- Verificar que las columnas existan ---
    columnas_requeridas = ['Close', 'SMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist']
    for columna in columnas_requeridas:
        if columna not in data.columns:
            logging.error(f"Falta la columna: {columna}. Columnas disponibles: {data.columns.tolist()}")
            raise KeyError(f"Falta la columna: {columna}")

    data['Signal'] = 0  # 0: mantener

    # --- Usar directamente las columnas del DataFrame (ya están alineadas) ---
    cond_compra_1 = (data['Close'] > data['SMA_20']) & (data['RSI_14'] < 30)
    cond_compra_2 = (data['MACD'] > data['MACD_Signal']) & (data['MACD_Hist'] > 0)
    cond_venta_1 = (data['Close'] < data['SMA_20']) & (data['RSI_14'] > 70)
    cond_venta_2 = (data['MACD'] < data['MACD_Signal']) & (data['MACD_Hist'] < 0)

    # Aplicar señales
    data.loc[cond_compra_1 | cond_compra_2, 'Signal'] = 1
    data.loc[cond_venta_1 | cond_venta_2, 'Signal'] = -1

    logging.info(f"Señales generadas: {data['Signal'].value_counts().to_dict()}")
    return data
def guardar_datos(data: pd.DataFrame, ticker: str) -> str:
    """Guarda los datos en un archivo CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"{ticker}_{timestamp}.csv")
    data.to_csv(filename)
    logging.info(f"Datos guardados en: {filename}")
    return filename

def graficar_datos(data: pd.DataFrame, ticker: str) -> None:
    """Grafica los datos y las señales."""
    logging.info("Generando gráficos...")

    plt.figure(figsize=(14, 10))

    # Gráfico 1: Precio y SMA_20
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Precio de cierre', color='blue', alpha=0.5)
    plt.plot(data['SMA_20'], label='SMA 20 días', color='orange')
    plt.scatter(
        data[data['Signal'] == 1].index,
        data[data['Signal'] == 1]['Close'],
        label='Compra', marker='^', color='green', s=100
    )
    plt.scatter(
        data[data['Signal'] == -1].index,
        data[data['Signal'] == -1]['Close'],
        label='Vende', marker='v', color='red', s=100
    )
    plt.title(f"Análisis técnico de {ticker} - Precio y SMA")
    plt.legend()
    plt.grid()

    # Gráfico 2: RSI
    plt.subplot(3, 1, 2)
    plt.plot(data['RSI_14'], label='RSI 14 días', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Sobrecomprado (70)')
    plt.axhline(30, color='green', linestyle='--', label='Sobrevendido (30)')
    plt.title("RSI")
    plt.legend()
    plt.grid()

    # Gráfico 3: MACD
    plt.subplot(3, 1, 3)
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['MACD_Signal'], label='Señal MACD', color='orange')
    plt.bar(data.index, data['MACD_Hist'], label='Histograma MACD', color='gray', alpha=0.5)
    plt.title("MACD")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"{ticker}_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.show()
    logging.info(f"Gráfico guardado en: data/{ticker}_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

def main():
    """Función principal del agente."""
    try:
        # --- 1. Descargar datos ---
        logging.info("Iniciando la ejecución del agente...")
        data = descargar_datos(TICKER, PERIOD, INTERVAL)

        # Verificar que los datos se descargaron correctamente
        if data.empty:
            logging.error("No se descargaron datos. Verifica el símbolo o la conexión a Internet.")
            raise ValueError("Datos vacíos: no se pudo descargar información de Yahoo Finance.")

        # --- 2. Calcular indicadores ---
        data = calcular_indicadores(data)

        # --- 3. Verificar que las columnas de indicadores existan ---
        columnas_requeridas = ['Close', 'SMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist']
        for columna in columnas_requeridas:
            if columna not in data.columns:
                logging.error(f"La columna '{columna}' no existe en el DataFrame después de calcular indicadores.")
                logging.error(f"Columnas disponibles: {data.columns.tolist()}")
                raise KeyError(f"Falta la columna: {columna}")

        # --- 4. Generar señales ---
        data = generar_senales(data)

        # --- 5. Verificar que la columna 'Signal' exista ---
        if 'Signal' not in data.columns:
            logging.error("La columna 'Signal' no se generó correctamente.")
            raise KeyError("Falta la columna 'Signal'")

        # --- 6. Guardar datos ---
        guardar_datos(data, TICKER)

        # --- 7. Graficar datos ---
        graficar_datos(data, TICKER)

        # --- 8. Mostrar resultados en la consola ---
        print("\n=== Últimos 5 días con señales ===")
        print(data[['Close', 'SMA_20', 'RSI_14', 'Signal']].tail())

        logging.info("✅ Ejecución del agente completada con éxito.")
        return data  # Opcional: devolver el DataFrame para uso posterior

    except Exception as e:
        logging.error(f"❌ Error en la ejecución del agente: {e}")
        # Mostrar el traceback completo en los logs
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()