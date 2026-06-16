#!/usr/bin/env python3

# --- Configuración inicial para usar el entorno virtual ---
import sys
import os

# Ruta al entorno virtual (ajustar según tu estructura)
venv_path = os.path.join(os.path.dirname(__file__), "../venv")

# Añadir el entorno virtual al PATH y a sys.path
if os.path.exists(venv_path):
    # Añadir el directorio bin/ del entorno virtual al PATH
    os.environ["PATH"] = os.path.join(venv_path, "bin") + os.pathsep + os.environ.get("PATH", "")
    # Añadir el directorio site-packages al sys.path
    site_packages = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    if os.path.exists(site_packages):
        sys.path.insert(0, site_packages)

# --- Importaciones ---
import yfinance as yf
import pandas as pd
import talib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa el backend 'Agg' para evitar problemas de renderizado
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# --- Configuración de logging ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"agente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

# --- Configuración del agente ---
TICKER = "AAPL"
PERIOD = "1y"
INTERVAL = "1d"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)


def descargar_datos(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga datos de Yahoo Finance."""
    logging.info(f"Descargando datos de {ticker} para el período {period}...")
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)
        
        # Verificar columnas estándar
        columnas_esperadas = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in columnas_esperadas:
            if col not in data.columns:
                logging.error(f"Columna '{col}' no encontrada. Columnas disponibles: {data.columns.tolist()}")
                raise ValueError(f"Falta la columna estándar: {col}")
        
        logging.info(f"Datos descargados: {len(data)} registros. Columnas: {data.columns.tolist()}")
        return data
    except Exception as e:
        logging.error(f"Error al descargar datos: {e}")
        raise


def calcular_indicadores(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos (SMA_20, RSI_14, MACD)."""
    logging.info("Calculando indicadores técnicos con TA-Lib...")
    
    # Encontrar columna 'Close'
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
        raise KeyError("No se encontró la columna 'Close'")
    
    if close_col != 'Close':
        data = data.rename(columns={close_col: 'Close'})
    
    # Calcular indicadores
    close_prices = data['Close'].astype('float64').to_numpy()
    
    sma_20 = talib.SMA(close_prices, timeperiod=20)
    rsi_14 = talib.RSI(close_prices, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Crear DataFrame con indicadores
    indicadores = pd.DataFrame({
        'SMA_20': sma_20,
        'RSI_14': rsi_14,
        'MACD': macd,
        'MACD_Signal': macd_signal,
        'MACD_Hist': macd_hist
    }, index=data.index)
    
    # Rellenar NaN
    indicadores = indicadores.bfill()
    indicadores['MACD_Hist'] = indicadores['MACD_Hist'].fillna(0)
    
    # Unir con datos originales
    data = pd.concat([data, indicadores], axis=1)
    logging.info("Indicadores calculados: SMA_20, RSI_14, MACD.")
    return data


def generar_senales(data: pd.DataFrame) -> pd.DataFrame:
    """Genera señales de compra (1) y venta (-1)."""
    logging.info("Generando señales de compra/venta...")
    
    # Verificar columnas requeridas
    columnas_requeridas = ['Close', 'SMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist']
    for columna in columnas_requeridas:
        if columna not in data.columns:
            logging.error(f"Falta la columna: {columna}. Columnas disponibles: {data.columns.tolist()}")
            raise KeyError(f"Falta la columna: {columna}")
    
    data['Signal'] = 0
    
    # Condiciones de compra y venta
    cond_compra_1 = (data['Close'] > data['SMA_20']) & (data['RSI_14'] < 30)
    cond_compra_2 = (data['MACD'] > data['MACD_Signal']) & (data['MACD_Hist'] > 0)
    cond_venta_1 = (data['Close'] < data['SMA_20']) & (data['RSI_14'] > 70)
    cond_venta_2 = (data['MACD'] < data['MACD_Signal']) & (data['MACD_Hist'] < 0)
    
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
    plt.scatter(data[data['Signal'] == 1].index, data[data['Signal'] == 1]['Close'],
                label='Compra', marker='^', color='green', s=100)
    plt.scatter(data[data['Signal'] == -1].index, data[data['Signal'] == -1]['Close'],
                label='Vende', marker='v', color='red', s=100)
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
    
    # Guardar gráfico
    graph_filename = os.path.join(DATA_DIR, f"{ticker}_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(graph_filename)
    plt.close()
    logging.info(f"Gráfico guardado en: {graph_filename}")


def main():
    """Función principal del agente."""
    try:
        logging.info("Iniciando la ejecución del agente...")
        
        # Descargar datos
        data = descargar_datos(TICKER, PERIOD, INTERVAL)
        if data.empty:
            logging.error("No se descargaron datos. Verifica el símbolo o la conexión a Internet.")
            raise ValueError("Datos vacíos")
        
        # Calcular indicadores
        data = calcular_indicadores(data)
        
        # Generar señales
        data = generar_senales(data)
        
        # Guardar datos
        guardar_datos(data, TICKER)
        
        # Graficar datos
        graficar_datos(data, TICKER)
        
        # Mostrar resultados
        print("\n=== Últimos 5 días con señales ===")
        print(data[['Close', 'SMA_20', 'RSI_14', 'Signal']].tail())
        
        logging.info("✅ Ejecución del agente completada con éxito.")
        return data
        
    except Exception as e:
        logging.error(f"❌ Error en la ejecución del agente: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()