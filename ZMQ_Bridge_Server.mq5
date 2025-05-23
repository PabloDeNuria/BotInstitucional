import MetaTrader5 as mt5
import time
import pandas as pd
from datetime import datetime

# Clase para manejar operaciones con MetaTrader 5
class MT5Handler:
    def __init__(self):
        """Inicializa la conexión con MetaTrader 5"""
        # Intentar inicializar la conexión con MetaTrader 5
        if not mt5.initialize():
            print(f"Inicialización fallida. Error: {mt5.last_error()}")
            raise Exception("No se pudo inicializar MetaTrader 5")
        
        # Mostrar información de la versión de MetaTrader 5
        print(f"MetaTrader 5 versión {mt5.version()} inicializado correctamente")
        
        # Verificar que podemos obtener datos del mercado
        if not mt5.symbol_info_tick("EURUSD"):
            print("Error al obtener datos de mercado. Verificar conexión a internet y broker.")
            mt5.shutdown()
            raise Exception("No se pudieron obtener datos de mercado")
        
        print("Conexión establecida correctamente. Sistema listo para operar.")
    
    def __del__(self):
        """Cierra la conexión con MetaTrader 5 al destruir el objeto"""
        mt5.shutdown()
        print("Conexión con MetaTrader 5 cerrada")
    
    def send_market_order(self, symbol, order_type, volume, 
                          sl=0.0, tp=0.0, deviation=10, comment="Python Order"):
        """
        Envía una orden de mercado
        
        Args:
            symbol: Símbolo a operar (ej. "EURUSD")
            order_type: Tipo de orden (mt5.ORDER_TYPE_BUY o mt5.ORDER_TYPE_SELL)
            volume: Volumen en lotes (ej. 0.1)
            sl: Stop Loss en precio (0.0 para no usar)
            tp: Take Profit en precio (0.0 para no usar)
            deviation: Desviación máxima del precio
            comment: Comentario para la orden
            
        Returns:
            Ticket de la orden o None si hay error
        """
        # Verificar que el símbolo existe
        if not mt5.symbol_info(symbol):
            print(f"El símbolo {symbol} no existe")
            return None
        
        # Obtener el precio actual
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"Error al obtener el precio para {symbol}")
            return None
        
        # Determinar precio según tipo de orden
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Crear solicitud de orden
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": 123456,  # Número mágico para identificar órdenes de este sistema
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Enviar la orden
        result = mt5.order_send(order)
        
        # Verificar el resultado
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Orden fallida: {result.retcode}, {result.comment}")
            return None
        
        print(f"Orden ejecutada: {result.order} - {self.order_type_to_string(order_type)} {volume} {symbol} a {price}")
        return result.order
    
    def close_position(self, ticket):
        """
        Cierra una posición abierta
        
        Args:
            ticket: Número de ticket de la posición
            
        Returns:
            True si se cerró correctamente, False si hubo error
        """
        # Seleccionar la posición por ticket
        if not mt5.positions_get(ticket=ticket):
            print(f"No se encontró la posición con ticket {ticket}")
            return False
        
        # Obtener información de la posición
        position = mt5.positions_get(ticket=ticket)[0]
        
        # Preparar la solicitud de cierre
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,  # Tipo opuesto
            "price": mt5.symbol_info_tick(position.symbol).ask if position.type == 1 else mt5.symbol_info_tick(position.symbol).bid,
            "deviation": 10,
            "magic": 123456,
            "comment": "Python Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Enviar la solicitud
        result = mt5.order_send(request)
        
        # Verificar el resultado
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Error al cerrar posición: {result.retcode}, {result.comment}")
            return False
        
        print(f"Posición {ticket} cerrada exitosamente")
        return True
    
    def get_market_data(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, bars=50):
        """
        Obtiene datos históricos del mercado
        
        Args:
            symbol: Símbolo (ej. "EURUSD")
            timeframe: Marco temporal (ej. mt5.TIMEFRAME_M5)
            bars: Número de barras a obtener
            
        Returns:
            DataFrame con los datos o None si hay error
        """
        # Obtener barras históricas
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            print(f"Error al obtener datos para {symbol}")
            return None
        
        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def get_account_info(self):
        """
        Obtiene información de la cuenta
        
        Returns:
            Dict con información de la cuenta o None si hay error
        """
        account_info = mt5.account_info()
        if not account_info:
            print("Error al obtener información de la cuenta")
            return None
        
        # Convertir a diccionario
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage,
            'profit': account_info.profit
        }
    
    def get_open_positions(self):
        """
        Obtiene todas las posiciones abiertas
        
        Returns:
            DataFrame con las posiciones o None si hay error
        """
        positions = mt5.positions_get()
        if positions is None:
            print("No hay posiciones abiertas o error al obtenerlas")
            return None
        
        # Convertir a DataFrame
        if len(positions) > 0:
            df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            # Convertir tiempo a formato datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['time_update'] = pd.to_datetime(df['time_update'], unit='s')
            return df
        
        return pd.DataFrame()
    
    def order_type_to_string(self, order_type):
        """Convierte el tipo de orden a string"""
        types = {
            mt5.ORDER_TYPE_BUY: "BUY",
            mt5.ORDER_TYPE_SELL: "SELL",
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL STOP"
        }
        return types.get(order_type, "UNKNOWN")

# Ejemplo de uso
if __name__ == "__main__":
    try:
        # Inicializar handler
        mt5_handler = MT5Handler()
        
        # Mostrar información de la cuenta
        account_info = mt5_handler.get_account_info()
        print(f"Balance: {account_info['balance']}, Equity: {account_info['equity']}")
        
        # Obtener datos de mercado
        df = mt5_handler.get_market_data("EURUSD", mt5.TIMEFRAME_M5, 10)
        print("\nÚltimos datos de mercado:")
        print(df.tail())
        
        # Enviar una orden de compra
        ticket = mt5_handler.send_market_order(
            symbol="EURUSD",
            order_type=mt5.ORDER_TYPE_BUY,
            volume=0.1,
            sl=mt5.symbol_info_tick("EURUSD").bid - 0.0050,  # Stop Loss 50 pips abajo
            tp=mt5.symbol_info_tick("EURUSD").ask + 0.0100,  # Take Profit 100 pips arriba
            comment="Ejemplo Python"
        )
        
        if ticket:
            print(f"Orden ejecutada con éxito. Ticket: {ticket}")
            
            # Esperar 5 segundos
            print("Esperando 5 segundos...")
            time.sleep(5)
            
            # Verificar posiciones abiertas
            positions = mt5_handler.get_open_positions()
            print("\nPosiciones abiertas:")
            print(positions)
            
            # Cerrar la posición
            if mt5_handler.close_position(ticket):
                print("Posición cerrada exitosamente")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Asegurarse de que MT5 se cierra correctamente incluso si hay errores
        if 'mt5_handler' in locals():
            del mt5_handler
        elif mt5.initialize():  # Si no se creó el handler pero MT5 está inicializado
            mt5.shutdown()