//+------------------------------------------------------------------+
//|                                                     Logger.mqh |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Tu Nombre"
#property link      "https://www.ejemplo.com"

// Función auxiliar para obtener descripciones de error (versión simplificada)
string GetErrorDescription(int errorCode) {
    switch(errorCode) {
        case 0:     return "Operación completada exitosamente";
        case 4001:  return "Error interno inesperado";
        case 4002:  return "Parámetro interno incorrecto";
        case 4003:  return "Parámetro inválido";
        case 4004:  return "Memoria insuficiente";
        case 4005:  return "Estructura contiene objetos o clases dinámicas";
        case 4006:  return "Array inválido";
        case 4007:  return "Error al cambiar tamaño del array";
        case 4008:  return "Error al cambiar tamaño de cadena";
        case 4009:  return "Cadena no inicializada";
        case 4010:  return "Fecha y/o hora inválida";
        case 4011:  return "Tamaño de array solicitado supera 2GB";
        // Errores de gráficos
        case 4101:  return "ID de gráfico incorrecto o no responde";
        case 4103:  return "Gráfico no encontrado";
        case 4104:  return "No hay experto en el gráfico";
        case 4105:  return "Error al abrir el gráfico";
        // Errores de archivos
        case 5001:  return "No se puede abrir el archivo";
        case 5002:  return "No se puede reescribir el archivo";
        case 5003:  return "Nombre de directorio inválido";
        case 5004:  return "El archivo no existe";
        // Errores de trading
        case 10004: return "Trading deshabilitado";
        case 10005: return "Orden modificada";
        case 10006: return "Petición aceptada";
        case 10007: return "Petición ejecutada";
        case 10008: return "Petición rechazada";
        case 10009: return "Petición cancelada";
        case 10010: return "Petición en proceso";
        case 10011: return "Petición rechazada por timeout";
        case 10012: return "Función inválida";
        case 10013: return "Demasiadas operaciones";
        case 10014: return "Dealer concurridos";
        case 10015: return "Precio no válido";
        case 10016: return "SL inválido";
        case 10017: return "TP inválido";
        case 10018: return "Volumen insuficiente";
        case 10019: return "Precio cambiado";
        // Errores genéricos de trading
        case 10021: return "No hay conexión con el servidor";
        case 10022: return "Usuario no autorizado";
        case 10023: return "Cotización no encontrada";
        case 10024: return "Método no permitido";
        case 10025: return "Demasiadas peticiones";
        case 10026: return "Servidor ocupado";
        case 10027: return "Tiempo expirado";
        default:    return "Error desconocido #" + IntegerToString(errorCode);
    }
}

class CLogger {
private:
    bool m_enabled;
    string m_logFileName;
    int m_fileHandle;
    
public:
    CLogger();
    ~CLogger();
    
    // Configuración
    void Configure(bool enabled, string logFileName = "trading_bot_log.txt");
    
    // Registrar mensaje
    void Log(string message, bool showTime = true);
    
    // Registrar detalle de operación
    void LogTrade(string action, string symbol, double volume, double price, 
                 double stopLoss, double takeProfit, string comment);
    
    // Registrar error
    void LogError(string message, int errorCode = 0);
};

// Implementación de la clase
CLogger::CLogger() {
    m_enabled = true;
    m_logFileName = "trading_bot_log.txt";
    m_fileHandle = INVALID_HANDLE;
}

CLogger::~CLogger() {
    if(m_fileHandle != INVALID_HANDLE) {
        FileClose(m_fileHandle);
    }
}

void CLogger::Configure(bool enabled, string logFileName) {
    m_enabled = enabled;
    m_logFileName = logFileName;
    
    if(m_enabled) {
        // Abrir archivo para escritura
        m_fileHandle = FileOpen(m_logFileName, FILE_WRITE|FILE_TXT|FILE_ANSI);
        
        if(m_fileHandle == INVALID_HANDLE) {
            Print("Error al abrir archivo de log: ", GetLastError());
        } else {
            // Escribir encabezado
            FileWrite(m_fileHandle, "=== Trading Bot Log ===");
            FileWrite(m_fileHandle, "Iniciado: ", TimeToString(TimeCurrent()));
            FileWrite(m_fileHandle, "===========================");
            FileFlush(m_fileHandle);
        }
    }
}

void CLogger::Log(string message, bool showTime) {
    if(!m_enabled) return;
    
    string logMessage = "";
    
    if(showTime) {
        logMessage = TimeToString(TimeCurrent()) + " - " + message;
    } else {
        logMessage = message;
    }
    
    // Escribir en el archivo de log
    if(m_fileHandle != INVALID_HANDLE) {
        FileWrite(m_fileHandle, logMessage);
        FileFlush(m_fileHandle);
    }
    
    // También imprimir en el log de MT5
    Print(logMessage);
}

void CLogger::LogTrade(string action, string symbol, double volume, double price, 
                      double stopLoss, double takeProfit, string comment) {
    if(!m_enabled) return;
    
    string message = StringFormat(
        "%s: %s, Vol: %.2f, Precio: %.5f, SL: %.5f, TP: %.5f, %s",
        action, symbol, volume, price, stopLoss, takeProfit, comment
    );
    
    Log(message);
}

void CLogger::LogError(string message, int errorCode) {
    if(!m_enabled) return;
    
    string errorMsg = message;
    
    if(errorCode != 0) {
        errorMsg += " - Error #" + IntegerToString(errorCode) + ": " + GetErrorDescription(errorCode);
    }
    
    Log("ERROR: " + errorMsg);
}