import zmq
import time
import pandas as pd
import json

class MT5Bridge:
    def __init__(self, server_address="tcp://localhost:5555", push_address="tcp://localhost:5556"):
        """Inicializa el bridge a MetaTrader 5"""
        self.context = zmq.Context()
        
        # Socket para solicitar datos (request-reply)
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(server_address)
        
        # Socket para enviar comandos (push)
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(push_address)
        
        # Establecer timeout
        self.req_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 segundos
        
        print(f"Conectado a MetaTrader5 en {server_address} y {push_address}")
        
        # Verificar conexión
        self.ping()
    
    def ping(self):
        """Verifica la conexión con el servidor"""
        try:
            self.req_socket.send_string("PING")
            response = self.req_socket.recv_string()
            if response == "PONG":
                print("Conexión exitosa con MetaTrader")
                return True
            else:
                print(f"Respuesta inesperada: {response}")
                return False
        except zmq.error.Again:
            print("Timeout al esperar respuesta de MetaTrader")
            return False
    
    def get_market_data(self):
        """Obtiene datos actuales del mercado"""
        try:
            self.req_socket.send_string("DATA")
            response = self.req_socket.recv_string()
            
            parts = response.split('|')
            if parts[0] != "DATA":
                print(f"Error: {response}")
                return None
            
            # Procesar los datos recibidos
            data_lines = response[5:].strip().split('\n')  # Omitir 'DATA|'
            
            market_data = {
                'price': {},
                'bars': []
            }
            
            for line in data_lines:
                items = line.split('|')
                if items[0] == "PRICE":
                    market_data['price'] = {
                        'symbol': items[1],
                        'bid': float(items[2]),
                        'ask': float(items[3])
                    }
                elif items[0] == "BAR":
                    market_data['bars'].append({
                        'time': items[1],
                        'open': float(items[2]),
                        'high': float(items[3]),
                        'low': float(items[4]),
                        'close': float(items[5]),
                        'volume': int(items[6])
                    })
            
            return market_data
        except zmq.error.Again:
            print("Timeout al esperar datos de mercado")
            return None
    
    def send_trade(self, trade_type, symbol, volume, price=0, wait_response=True):
        """
        Envía una orden de trading
        
        Args:
            trade_type: Tipo de orden ('BUY', 'SELL', 'BUYLIMIT', 'SELLLIMIT')
            symbol: Símbolo a operar
            volume: Volumen (lotes)
            price: Precio (solo para órdenes pending)
            wait_response: Si es True, espera respuesta (para socket REQ)
        
        Returns:
            ticket de la orden o None si hay error
        """
        command = f"TRADE|{trade_type}|{symbol}|{volume}|{price}"
        
        if wait_response:
            try:
                self.req_socket.send_string(command)
                response = self.req_socket.recv_string()
                
                parts = response.split('|')
                if parts[0] == "SUCCESS":
                    return int(parts[1])
                else:
                    print(f"Error en la operación: {response}")
                    return None
            except zmq.error.Again:
                print("Timeout al esperar respuesta de trading")
                return None
        else:
            self.push_socket.send_string(command)
            return True
    
    def close_position(self, ticket, wait_response=True):
        """
        Cierra una posición específica
        
        Args:
            ticket: Número de ticket de la posición
            wait_response: Si es True, espera respuesta (para socket REQ)
        
        Returns:
            True si se cerró correctamente, False en caso contrario
        """
        command = f"CLOSE|{ticket}"
        
        if wait_response:
            try:
                self.req_socket.send_string(command)
                response = self.req_socket.recv_string()
                
                parts = response.split('|')
                if parts[0] == "SUCCESS":
                    return True
                else:
                    print(f"Error al cerrar posición: {response}")
                    return False
            except zmq.error.Again:
                print("Timeout al esperar respuesta de cierre")
                return False
        else:
            self.push_socket.send_string(command)
            return True
    
    def close(self):
        """Cierra la conexión con el servidor"""
        self.req_socket.close()
        self.push_socket.close()
        self.context.term()
        print("Conexión cerrada con MetaTrader")
