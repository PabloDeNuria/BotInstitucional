import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def analyze_trading_data(csv_file_path):
    """
    Analiza los datos de trading recolectados y genera informes con conclusiones.
    
    Args:
        csv_file_path: Ruta al archivo CSV con los datos recolectados
    """
    print(f"Analizando datos de trading desde {csv_file_path}...")
    
    # Cargar los datos - cambiando a separador de tabulación
    df = pd.read_csv(csv_file_path, sep='\t')
    
    # Mostrar primeras filas para depuración
    print("\nPrimeras filas del archivo:")
    print(df.head())
    
    # Mostrar nombres de columnas
    print("\nColumnas encontradas en el archivo:")
    print(df.columns.tolist())
    
    # Verificar si hay datos
    if df.empty:
        print("ERROR: El archivo CSV está vacío.")
        return
    
    # Identificar la columna de fecha (buscar una columna que contenga 'date' o 'time' en el nombre)
    date_column = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_column = col
            break
    
    if date_column is None:
        # Si no se encontró una columna con 'date' o 'time', usar la primera columna como fecha
        date_column = df.columns[0]
        print(f"No se encontró columna de fecha explícita. Usando la primera columna: '{date_column}'")
    
    # Intentar convertir a datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column], format='%Y.%m.%d %H:%M')
        print(f"Columna '{date_column}' convertida exitosamente a datetime.")
    except Exception as e:
        print(f"Error al convertir la columna '{date_column}' a datetime: {e}. Continuando sin conversión.")
    
    # Columnas ya están bien nombradas según los errores, no hace falta mapeo
    
    # Convertir columnas numéricas a tipos apropiados
    numeric_columns = ['Price', 'EntryPrice', 'StopLoss', 'TakeProfit', 'ATR', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"Columna '{col}' convertida a numérico exitosamente.")
            except Exception as e:
                print(f"Error al convertir columna '{col}' a numérico: {e}. Continuando sin conversión.")
    
    # Añadir columnas adicionales para análisis
    if date_column in df.columns:
        try:
            df['Hour'] = df[date_column].dt.hour
            df['Minute'] = df[date_column].dt.minute
            df['Day'] = df[date_column].dt.day
            df['Month'] = df[date_column].dt.month
            df['Year'] = df[date_column].dt.year
            print("Columnas temporales añadidas exitosamente.")
        except Exception as e:
            print(f"Error al extraer componentes de fecha/hora: {e}. Algunas análisis temporales no estarán disponibles.")
    
    # Calcular el Risk-Reward Ratio (RRR)
    if all(col in df.columns for col in ['EntryPrice', 'StopLoss', 'TakeProfit']):
        try:
            df['RRR'] = abs((df['TakeProfit'] - df['EntryPrice']) / (df['EntryPrice'] - df['StopLoss']))
            print("RRR calculado exitosamente.")
        except Exception as e:
            print(f"Error al calcular RRR: {e}. Continuando sin esta métrica.")
    
    # Simular resultados hipotéticos
    simulate_trade_outcomes(df)
    
    # Generar análisis y conclusiones
    pattern_stats = generate_pattern_analysis(df)
    session_stats = generate_session_analysis(df)
    day_stats = generate_day_analysis(df)
    hour_stats = generate_hour_analysis(df)
    
    # Generar archivos CSV para el bot
    generate_output_files(df, pattern_stats, session_stats, day_stats)
    
    # Generar visualizaciones
    generate_visualizations(df)
    
    print("\nAnálisis completado con éxito!")


def simulate_trade_outcomes(df):
    """Simula resultados de trading para análisis"""
    print("Simulando resultados de trading...")
    
    # Verificar columnas necesarias
    required_columns = ['EntryPrice', 'StopLoss', 'TakeProfit', 'EventType']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Advertencia: Faltan columnas para simular resultados: {missing_columns}")
        print("Usando un método simplificado para simulación...")
        
        # Método simplificado: asignar probabilidades basadas solo en el tipo de evento
        pattern_probabilities = {
            'BullishEngulfing': 0.58,
            'BearishEngulfing': 0.56,
            'BullishSweep': 0.55,
            'BearishSweep': 0.54,
            'BullishBreakout': 0.60,
            'BearishBreakout': 0.59,
            'SupportRetest': 0.62,
            'ResistanceRetest': 0.60,
            'BullishEMACross': 0.52,
            'BearishEMACross': 0.51,
            'Consolidation': 0.48,
            'HighVolumeBullish': 0.57,
            'HighVolumeBearish': 0.56,
            'LowVolume': 0.45
        }
        
        # Asignar probabilidad basada en el tipo de evento
        if 'EventType' in df.columns:
            df['SuccessProbability'] = df['EventType'].map(
                lambda x: pattern_probabilities.get(x, 0.5)
            )
        else:
            # Si no hay columna de tipo de evento, usar probabilidad constante
            df['SuccessProbability'] = 0.5
            
        # Generar resultados aleatorios basados en probabilidades
        np.random.seed(42)  # Para reproducibilidad
        df['Random'] = np.random.random(len(df))
        df['Outcome'] = np.where(df['Random'] < df['SuccessProbability'], 'Win', 'Loss')
        
        # Asignar valores simulados de pips
        df['ResultPips'] = np.where(
            df['Outcome'] == 'Win',
            np.random.normal(loc=20, scale=10, size=len(df)),  # Ganancia media 20 pips
            np.random.normal(loc=-15, scale=5, size=len(df))   # Pérdida media 15 pips
        )
        
        # Limpiar datos de simulación
        df.drop(['Random', 'SuccessProbability'], axis=1, inplace=True, errors='ignore')
        print("Simulación simplificada completada.")
        return# Si tenemos todas las columnas requeridas, procedemos con simulación completa
    print("Realizando simulación completa de resultados.")
    
    # Calcular probabilidad de éxito basada en RRR
    df['SuccessProbability'] = 0.5 - (df['RRR'] - 2) * 0.05
    df['SuccessProbability'] = df['SuccessProbability'].clip(0.3, 0.7)
    
    # Ajustar probabilidad según tipo de patrón
    pattern_adjustments = {
        'BullishEngulfing': 0.05,
        'BearishEngulfing': 0.05,
        'BullishSweep': 0.03,
        'BearishSweep': 0.03,
        'BullishBreakout': 0.07,
        'BearishBreakout': 0.07,
        'SupportRetest': 0.08,
        'ResistanceRetest': 0.08,
        'BullishEMACross': 0.02,
        'BearishEMACross': 0.02,
        'Consolidation': 0.0,
        'HighVolumeBullish': 0.06,
        'HighVolumeBearish': 0.06,
        'LowVolume': -0.05
    }
    
    if 'EventType' in df.columns:
        for pattern, adjustment in pattern_adjustments.items():
            df.loc[df['EventType'] == pattern, 'SuccessProbability'] += adjustment
    
    # Ajustar probabilidad según sesión
    if 'Session' in df.columns:
        session_adjustments = {
            'Asian': -0.02,
            'European': 0.03,
            'American': 0.02
        }
        
        for session, adjustment in session_adjustments.items():
            df.loc[df['Session'] == session, 'SuccessProbability'] += adjustment
    
    # Limitar probabilidades entre 0.25 y 0.75
    df['SuccessProbability'] = df['SuccessProbability'].clip(0.25, 0.75)
    
    # Generar resultados aleatorios basados en probabilidades
    np.random.seed(42)  # Para reproducibilidad
    df['Random'] = np.random.random(len(df))
    df['Outcome'] = np.where(df['Random'] < df['SuccessProbability'], 'Win', 'Loss')
    
    # Calcular result en pips
    if all(col in df.columns for col in ['TakeProfit', 'EntryPrice', 'StopLoss']):
        df['ResultPips'] = np.where(
            df['Outcome'] == 'Win',
            (df['TakeProfit'] - df['EntryPrice']) * 10000,
            (df['EntryPrice'] - df['StopLoss']) * -10000
        )
        
        # Para pares que no son USD, ajustar escala de pips
        if 'Symbol' in df.columns:
            non_usd_pairs = df['Symbol'].str.contains('JPY', na=False)
            if any(non_usd_pairs):
                df.loc[non_usd_pairs, 'ResultPips'] = df.loc[non_usd_pairs, 'ResultPips'] / 100
    else:
        # Si faltan columnas para calcular pips, asignar valores simulados
        df['ResultPips'] = np.where(
            df['Outcome'] == 'Win',
            np.random.normal(loc=20, scale=10, size=len(df)),  # Ganancia media 20 pips
            np.random.normal(loc=-15, scale=5, size=len(df))   # Pérdida media 15 pips
        )
    
    # Limpiar datos de simulación
    df.drop(['Random', 'SuccessProbability'], axis=1, inplace=True, errors='ignore')
    print("Simulación completa finalizada.")


def generate_pattern_analysis(df):
    """
    Analiza el rendimiento de diferentes patrones de trading
    
    Args:
        df: DataFrame con los datos de trading
        
    Returns:
        dict: Estadísticas de rendimiento por patrón
    """
    print("Generando análisis de patrones...")
    
    # Verificar si tenemos la columna EventType
    if 'EventType' not in df.columns:
        print("No se encontró la columna EventType. No se puede generar análisis de patrones.")
        return {}
    
    # Verificar si tenemos resultados
    if 'Outcome' not in df.columns or 'ResultPips' not in df.columns:
        print("No se encontraron columnas de resultados. No se puede generar análisis de patrones.")
        return {}
    
    # Inicializar diccionario de estadísticas
    pattern_stats = {}
    
    # Agrupar por tipo de evento y calcular estadísticas
    pattern_groups = df.groupby('EventType')
    
    for pattern, group in pattern_groups:
        total_trades = len(group)
        if total_trades < 1:
            continue
            
        wins = (group['Outcome'] == 'Win').sum()
        win_rate = wins / total_trades
        
        avg_win = group.loc[group['Outcome'] == 'Win', 'ResultPips'].mean() if wins > 0 else 0
        avg_loss = group.loc[group['Outcome'] == 'Loss', 'ResultPips'].mean() if (total_trades - wins) > 0 else 0
        
        pattern_stats[pattern] = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': win_rate,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'expectancy': (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        }
    
    # Imprimir resultados
    print("\nEstadísticas por patrón:")
    for pattern, stats in pattern_stats.items():
        print(f"- {pattern}:")
        print(f"  Operaciones: {stats['total_trades']}")
        print(f"  Tasa de éxito: {stats['win_rate']:.2%}")
        print(f"  Expectativa: {stats['expectancy']:.2f} pips")
        print()
    
    return pattern_stats


def generate_session_analysis(df):
    """
    Analiza el rendimiento de trading por sesión de mercado
    
    Args:
        df: DataFrame con los datos de trading
        
    Returns:
        dict: Estadísticas de rendimiento por sesión
    """
    print("Generando análisis por sesión...")
    
    # Verificar si tenemos la columna Session
    if 'Session' not in df.columns:
        print("No se encontró la columna Session. No se puede generar análisis por sesión.")
        return {}
    
    # Verificar si tenemos resultados
    if 'Outcome' not in df.columns or 'ResultPips' not in df.columns:
        print("No se encontraron columnas de resultados. No se puede generar análisis por sesión.")
        return {}
    
    # Inicializar diccionario de estadísticas
    session_stats = {}
    
    # Agrupar por sesión y calcular estadísticas
    session_groups = df.groupby('Session')
    
    for session, group in session_groups:
        total_trades = len(group)
        if total_trades < 1:
            continue
            
        wins = (group['Outcome'] == 'Win').sum()
        win_rate = wins / total_trades
        
        avg_win = group.loc[group['Outcome'] == 'Win', 'ResultPips'].mean() if wins > 0 else 0
        avg_loss = group.loc[group['Outcome'] == 'Loss', 'ResultPips'].mean() if (total_trades - wins) > 0 else 0
        
        session_stats[session] = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': win_rate,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'expectancy': (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        }# Imprimir resultados
    print("\nEstadísticas por sesión:")
    for session, stats in session_stats.items():
        print(f"- {session}:")
        print(f"  Operaciones: {stats['total_trades']}")
        print(f"  Tasa de éxito: {stats['win_rate']:.2%}")
        print(f"  Expectativa: {stats['expectancy']:.2f} pips")
        print()
    
    return session_stats


def generate_day_analysis(df):
    """
    Analiza el rendimiento de trading por día de la semana
    
    Args:
        df: DataFrame con los datos de trading
        
    Returns:
        dict: Estadísticas de rendimiento por día
    """
    print("Generando análisis por día de la semana...")
    
    # Verificar si tenemos la columna DayOfWeek o si podemos generarla desde DateTime
    day_column = None
    
    if 'DayOfWeek' in df.columns:
        day_column = 'DayOfWeek'
    elif 'DateTime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['DateTime']):
        # Crear columna de día de semana si tenemos DateTime
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        day_column = 'DayOfWeek'
    
    if day_column is None:
        print("No se encontró la columna de día de la semana. No se puede generar análisis por día.")
        return {}
    
    # Verificar si tenemos resultados
    if 'Outcome' not in df.columns or 'ResultPips' not in df.columns:
        print("No se encontraron columnas de resultados. No se puede generar análisis por día.")
        return {}
    
    # Inicializar diccionario de estadísticas
    day_stats = {}
    
    # Agrupar por día y calcular estadísticas
    day_groups = df.groupby(day_column)
    
    for day, group in day_groups:
        total_trades = len(group)
        if total_trades < 1:
            continue
            
        wins = (group['Outcome'] == 'Win').sum()
        win_rate = wins / total_trades
        
        avg_win = group.loc[group['Outcome'] == 'Win', 'ResultPips'].mean() if wins > 0 else 0
        avg_loss = group.loc[group['Outcome'] == 'Loss', 'ResultPips'].mean() if (total_trades - wins) > 0 else 0
        
        day_stats[day] = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': win_rate,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'expectancy': (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        }
    
    # Imprimir resultados
    print("\nEstadísticas por día de la semana:")
    for day, stats in day_stats.items():
        print(f"- {day}:")
        print(f"  Operaciones: {stats['total_trades']}")
        print(f"  Tasa de éxito: {stats['win_rate']:.2%}")
        print(f"  Expectativa: {stats['expectancy']:.2f} pips")
        print()
    
    return day_stats


def generate_hour_analysis(df):
    """
    Analiza el rendimiento de trading por hora del día
    
    Args:
        df: DataFrame con los datos de trading
        
    Returns:
        dict: Estadísticas de rendimiento por hora
    """
    print("Generando análisis por hora del día...")
    
    # Verificar si tenemos la columna Hour o si podemos generarla desde DateTime
    if 'Hour' not in df.columns:
        if 'DateTime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['DateTime']):
            # Crear columna de hora si tenemos DateTime
            df['Hour'] = df['DateTime'].dt.hour
        else:
            print("No se encontró la columna Hour. No se puede generar análisis por hora.")
            return {}
    
    # Verificar si tenemos resultados
    if 'Outcome' not in df.columns or 'ResultPips' not in df.columns:
        print("No se encontraron columnas de resultados. No se puede generar análisis por hora.")
        return {}
    
    # Inicializar diccionario de estadísticas
    hour_stats = {}
    
    # Agrupar por hora y calcular estadísticas
    hour_groups = df.groupby('Hour')
    
    for hour, group in hour_groups:
        total_trades = len(group)
        if total_trades < 1:
            continue
            
        wins = (group['Outcome'] == 'Win').sum()
        win_rate = wins / total_trades
        
        avg_win = group.loc[group['Outcome'] == 'Win', 'ResultPips'].mean() if wins > 0 else 0
        avg_loss = group.loc[group['Outcome'] == 'Loss', 'ResultPips'].mean() if (total_trades - wins) > 0 else 0
        
        hour_stats[hour] = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': win_rate,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'expectancy': (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        }
    
    # Imprimir resultados
    print("\nEstadísticas por hora del día:")
    for hour, stats in sorted(hour_stats.items()):
        print(f"- Hora {hour}:")
        print(f"  Operaciones: {stats['total_trades']}")
        print(f"  Tasa de éxito: {stats['win_rate']:.2%}")
        print(f"  Expectativa: {stats['expectancy']:.2f} pips")
        print()
    
    return hour_stats


def generate_visualizations(df):
    """
    Genera visualizaciones de los datos de trading
    
    Args:
        df: DataFrame con los datos de trading
    """
    print("Generando visualizaciones...")
    
    # Configurar estilo de las gráficas
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Verificar si tenemos las columnas necesarias
    if 'EventType' not in df.columns or 'Outcome' not in df.columns:
        print("Faltan columnas necesarias para las visualizaciones.")
        return
    
    # Crear directorio para gráficas si no existe
    output_dir = "trading_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Gráfica de barras de tasa de éxito por patrón
    try:
        plt.figure(figsize=(12, 8))
        
        # Calcular tasa de éxito por patrón
        pattern_win_rates = df.groupby('EventType')['Outcome'].apply(
            lambda x: (x == 'Win').mean()
        ).reset_index()
        pattern_win_rates.columns = ['EventType', 'WinRate']
        
        # Ordenar de mayor a menor tasa de éxito
        pattern_win_rates = pattern_win_rates.sort_values('WinRate', ascending=False)
        
        # Crear gráfica de barras
        ax = sns.barplot(x='EventType', y='WinRate', data=pattern_win_rates)
        plt.title('Tasa de Éxito por Patrón de Trading', fontsize=16)
        plt.ylabel('Tasa de Éxito (%)', fontsize=12)
        plt.xlabel('Patrón', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Añadir etiquetas de porcentaje
        for i, v in enumerate(pattern_win_rates['WinRate']):
            ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/win_rate_by_pattern.png")
        plt.close()
        print("Gráfica de tasa de éxito por patrón generada.")
    except Exception as e:
        print(f"Error al generar gráfica de tasa de éxito por patrón: {e}")
    
    # 2. Gráfica de resultados por sesión
    if 'Session' in df.columns and 'ResultPips' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            
            # Gráfica de violín de resultados por sesión
            ax = sns.violinplot(x='Session', y='ResultPips', data=df)
            plt.title('Distribución de Resultados por Sesión', fontsize=16)
            plt.ylabel('Resultado (pips)', fontsize=12)
            plt.xlabel('Sesión', fontsize=12)
            
            # Añadir línea de 0
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/results_by_session.png")
            plt.close()
            print("Gráfica de resultados por sesión generada.")
        except Exception as e:
            print(f"Error al generar gráfica de resultados por sesión: {e}")
    
    # 3. Gráfica de resultados por día de la semana
    if 'DayOfWeek' in df.columns and 'ResultPips' in df.columns:
        try:
            plt.figure(figsize=(10, 6))
            
            # Ordenar días de la semana
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            if 'Saturday' in df['DayOfWeek'].values:
                day_order.append('Saturday')
            if 'Sunday' in df['DayOfWeek'].values:
                day_order.append('Sunday')
            
            # Gráfica de barras de resultados medios por día
            day_results = df.groupby('DayOfWeek')['ResultPips'].mean().reindex(day_order).reset_index()
            ax = sns.barplot(x='DayOfWeek', y='ResultPips', data=day_results)
            
            plt.title('Resultados Medios por Día de la Semana', fontsize=16)
            plt.ylabel('Resultado Medio (pips)', fontsize=12)
            plt.xlabel('Día', fontsize=12)
            
            # Añadir etiquetas de valores
            for i, v in enumerate(day_results['ResultPips']):
                ax.text(i, v + 0.5 if v >= 0 else v - 2, f'{v:.1f}', ha='center', fontsize=10)
            
            # Añadir línea de 0
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/results_by_day.png")
            plt.close()
            print("Gráfica de resultados por día generada.")
        except Exception as e:
            print(f"Error al generar gráfica de resultados por día: {e}")
    
    # 4. Gráfica de resultados por hora del día
    if 'Hour' in df.columns and 'ResultPips' in df.columns:
        try:
            plt.figure(figsize=(14, 7))
            
            # Gráfica de barras de resultados medios por hora
            hour_results = df.groupby('Hour')['ResultPips'].mean().reset_index()
            hour_results = hour_results.sort_values('Hour')
            
            ax = sns.barplot(x='Hour', y='ResultPips', data=hour_results)
            
            plt.title('Resultados Medios por Hora del Día', fontsize=16)
            plt.ylabel('Resultado Medio (pips)', fontsize=12)
            plt.xlabel('Hora (GMT)', fontsize=12)
            
            # Añadir línea de 0
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/results_by_hour.png")
            plt.close()
            print("Gráfica de resultados por hora generada.")
        except Exception as e:
            print(f"Error al generar gráfica de resultados por hora: {e}")
    
    print(f"Visualizaciones guardadas en el directorio '{output_dir}'")


def generate_output_files(df, pattern_stats, session_stats, day_stats):
    """
    Genera archivos CSV con los resultados del análisis
    
    Args:
        df: DataFrame con los datos de trading
        pattern_stats: Estadísticas por patrón
        session_stats: Estadísticas por sesión
        day_stats: Estadísticas por día
    """
    print("Generando archivos de salida...")
    
    # Crear directorio si no existe
    output_dir = "trading_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar datos procesados
    try:
        df.to_csv(f"{output_dir}/processed_data.csv", index=False)
        print(f"Datos procesados guardados en {output_dir}/processed_data.csv")
    except Exception as e:
        print(f"Error al guardar datos procesados: {e}")
    
    # Guardar estadísticas de patrones
    if pattern_stats:
        try:
            pattern_df = pd.DataFrame.from_dict(pattern_stats, orient='index')
            pattern_df.reset_index(inplace=True)
            pattern_df.rename(columns={'index': 'Pattern'}, inplace=True)
            pattern_df.to_csv(f"{output_dir}/pattern_stats.csv", index=False)
            print(f"Estadísticas por patrón guardadas en {output_dir}/pattern_stats.csv")
        except Exception as e:
            print(f"Error al guardar estadísticas por patrón: {e}")
    
   # Guardar estadísticas de sesiones
    if session_stats:
        try:
            session_df = pd.DataFrame.from_dict(session_stats, orient='index')
            session_df.reset_index(inplace=True)
            session_df.rename(columns={'index': 'Session'}, inplace=True)
            session_df.to_csv(f"{output_dir}/session_stats.csv", index=False)
            print(f"Estadísticas por sesión guardadas en {output_dir}/session_stats.csv")
        except Exception as e:
            print(f"Error al guardar estadísticas por sesión: {e}")
    
    # Guardar estadísticas por día
    if day_stats:
        try:
            day_df = pd.DataFrame.from_dict(day_stats, orient='index')
            day_df.reset_index(inplace=True)
            day_df.rename(columns={'index': 'DayOfWeek'}, inplace=True)
            day_df.to_csv(f"{output_dir}/day_stats.csv", index=False)
            print(f"Estadísticas por día guardadas en {output_dir}/day_stats.csv")
        except Exception as e:
            print(f"Error al guardar estadísticas por día: {e}")
    
    print("Archivos de salida generados con éxito.")


# Ejecutar análisis
if __name__ == "__main__":
    # Ruta al archivo CSV generado por DataCollector.mq5
    csv_file_path = "market_data.csv"
    
    # Verificar si el archivo existe
    if os.path.exists(csv_file_path):
        analyze_trading_data(csv_file_path)
    else:
        print(f"Error: No se pudo encontrar el archivo {csv_file_path}")
        print("Asegúrate de que el archivo CSV generado por DataCollector.mq5 esté en la misma carpeta que este script.")
