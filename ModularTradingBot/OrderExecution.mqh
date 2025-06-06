//+------------------------------------------------------------------+
//|                                           OrderExecution.mqh |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Tu Nombre"
#property link      "https://www.ejemplo.com"

class COrderExecution {
private:
    bool m_useTrailingStop;
    double m_trailingStopPoints;
    bool m_useBreakEven;
    double m_breakEvenPoints;
    
    // Método para aplicar trailing stop
    void ApplyTrailingStop(ulong ticket);
    
    // Método para mover a punto de equilibrio
    void MoveToBreakEven(ulong ticket);
    
public:
    COrderExecution();
    ~COrderExecution();
    
    // Configuración
    void Configure(bool useTrailingStop = true, double trailingStopPoints = 50, 
                  bool useBreakEven = true, double breakEvenPoints = 30);
    
    // Colocar orden
    ulong PlaceOrder(string symbol, ENUM_ORDER_TYPE orderType, double volume, 
                    double price, double stopLoss, double takeProfit, string comment);
    
    // Modificar orden
    bool ModifyOrder(ulong ticket, double price, double stopLoss, double takeProfit);
    
    // Cerrar orden
    bool CloseOrder(ulong ticket, double volume = 0.0);
    
    // Cerrar todas las órdenes
    void CloseAllOrders();
    
    // Gestionar órdenes abiertas
    void ManageOpenOrders();
};

// Implementación de la clase
COrderExecution::COrderExecution() {
    m_useTrailingStop = true;
    m_trailingStopPoints = 50;
    m_useBreakEven = true;
    m_breakEvenPoints = 30;
}

COrderExecution::~COrderExecution() {
    // Limpieza si es necesario
}

void COrderExecution::Configure(bool useTrailingStop, double trailingStopPoints, 
                               bool useBreakEven, double breakEvenPoints) {
    m_useTrailingStop = useTrailingStop;
    m_trailingStopPoints = trailingStopPoints;
    m_useBreakEven = useBreakEven;
    m_breakEvenPoints = breakEvenPoints;
}

ulong COrderExecution::PlaceOrder(string symbol, ENUM_ORDER_TYPE orderType, double volume, 
                                 double price, double stopLoss, double takeProfit, string comment) {
    // Verificar si es una orden de mercado o pendiente
    bool isMarketOrder = (orderType == ORDER_TYPE_BUY || orderType == ORDER_TYPE_SELL);
    
    // Preparar la solicitud de trading
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = isMarketOrder ? TRADE_ACTION_DEAL : TRADE_ACTION_PENDING;
    request.symbol = symbol;
    request.volume = volume;
    request.type = orderType;
    request.price = isMarketOrder ? 0 : price;  // Para órdenes de mercado, el precio se ignora
    request.sl = stopLoss;
    request.tp = takeProfit;
    request.deviation = 5;  // Slippage permitido en puntos
    request.magic = 123456;  // Número mágico
    request.comment = comment;
    request.type_filling = ORDER_FILLING_FOK;
    
    // Enviar la orden
    bool success = OrderSend(request, result);
    
    // Verificar resultado
    if(success && result.retcode == TRADE_RETCODE_DONE) {
        return result.order;
    } else {
        Print("Error al colocar orden: ", result.retcode, " - ", GetLastError());
        return 0;
    }
}

bool COrderExecution::ModifyOrder(ulong ticket, double price, double stopLoss, double takeProfit) {
    if(!OrderSelect(ticket)) {
        Print("Error al seleccionar orden para modificar: ", GetLastError());
        return false;
    }
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_MODIFY;
    request.order = ticket;
    request.price = price;
    request.sl = stopLoss;
    request.tp = takeProfit;
    
    bool success = OrderSend(request, result);
    
    return (success && result.retcode == TRADE_RETCODE_DONE);
}

bool COrderExecution::CloseOrder(ulong ticket, double volume) {
    if(!PositionSelectByTicket(ticket)) {
        Print("Error al seleccionar posición para cerrar: ", GetLastError());
        return false;
    }
    
    double posVolume = PositionGetDouble(POSITION_VOLUME);
    if(volume <= 0 || volume > posVolume) {
        volume = posVolume;
    }
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.position = ticket;
    request.volume = volume;
    request.symbol = PositionGetString(POSITION_SYMBOL);
    request.type = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
    request.price = 0;  // Precio de mercado
    request.deviation = 5;
    
    bool success = OrderSend(request, result);
    
    return (success && result.retcode == TRADE_RETCODE_DONE);
}

void COrderExecution::CloseAllOrders() {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0) {
            CloseOrder(ticket);
        }
    }
}

void COrderExecution::ManageOpenOrders() {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0) {
            // Gestionar trailing stop si está habilitado
            if(m_useTrailingStop) {
                ApplyTrailingStop(ticket);
            }
            
            // Gestionar break even si está habilitado
            if(m_useBreakEven) {
                MoveToBreakEven(ticket);
            }
        }
    }
}

void COrderExecution::ApplyTrailingStop(ulong ticket) {
    if(!PositionSelectByTicket(ticket)) return;
    
    double currentSL = PositionGetDouble(POSITION_SL);
    double currentTP = PositionGetDouble(POSITION_TP);
    string symbol = PositionGetString(POSITION_SYMBOL);
    double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
    double currentPrice = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 
                          SymbolInfoDouble(symbol, SYMBOL_BID) : 
                          SymbolInfoDouble(symbol, SYMBOL_ASK);
    
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double newSL = 0;
    bool needModify = false;
    
    // Calcular nuevo SL para posiciones largas
    if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
        double minSL = currentPrice - m_trailingStopPoints * point;
        if(currentSL < minSL || currentSL == 0) {
            newSL = minSL;
            needModify = true;
        }
    }
    // Calcular nuevo SL para posiciones cortas
    else {
        double maxSL = currentPrice + m_trailingStopPoints * point;
        if(currentSL > maxSL || currentSL == 0) {
            newSL = maxSL;
            needModify = true;
        }
    }
    
    // Modificar la orden si es necesario
    if(needModify) {
        ModifyOrder(ticket, 0, newSL, currentTP);
    }
}

void COrderExecution::MoveToBreakEven(ulong ticket) {
    if(!PositionSelectByTicket(ticket)) return;
    
    double currentSL = PositionGetDouble(POSITION_SL);
    double currentTP = PositionGetDouble(POSITION_TP);
    string symbol = PositionGetString(POSITION_SYMBOL);
    double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
    double currentPrice = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 
                          SymbolInfoDouble(symbol, SYMBOL_BID) : 
                          SymbolInfoDouble(symbol, SYMBOL_ASK);
    
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double newSL = 0;
    bool needModify = false;
    
    // Para posiciones largas
    if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
        // Si el precio ha subido lo suficiente y el SL actual es menor que el precio de entrada
        if(currentPrice >= entryPrice + m_breakEvenPoints * point && (currentSL < entryPrice || currentSL == 0)) {
            newSL = entryPrice + 2 * point;  // SL ligeramente por encima del precio de entrada
            needModify = true;
        }
    }
    // Para posiciones cortas
    else {
        // Si el precio ha bajado lo suficiente y el SL actual es mayor que el precio de entrada
        if(currentPrice <= entryPrice - m_breakEvenPoints * point && (currentSL > entryPrice || currentSL == 0)) {
            newSL = entryPrice - 2 * point;  // SL ligeramente por debajo del precio de entrada
            needModify = true;
        }
    }
    
    // Modificar la orden si es necesario
    if(needModify) {
        ModifyOrder(ticket, 0, newSL, currentTP);
    }
}