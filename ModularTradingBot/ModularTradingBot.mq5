//+------------------------------------------------------------------+
//|                                         SimpleTradingBot.mq5      |
//|                           Bot de trading basado en análisis de datos|
//+------------------------------------------------------------------+
#property copyright "Tu Nombre"
#property link      "https://www.example.com"
#property version   "1.01"

// Enumeración de patrones
enum PATTERN_TYPE {
   PATTERN_NONE,
   PATTERN_BULLISH_BREAKOUT,     // 67.2% tasa éxito
   PATTERN_BEARISH_BREAKOUT,     // 64.3% tasa éxito
   PATTERN_HIGH_VOLUME_BULLISH,  // 63.0% tasa éxito
   PATTERN_BULLISH_SWEEP,        // 61.2% tasa éxito
   PATTERN_LOW_VOLUME            // 0.0% tasa éxito
};

// Inputs para configuración
input group "=== Configuración de Patrones ==="
input bool EnableBullishBreakout = true;    // Activar BullishBreakout
input bool EnableBearishBreakout = true;    // Activar BearishBreakout
input bool EnableHighVolumeBullish = true;  // Activar VolumenAlto Alcista
input bool EnableBullishSweep = true;       // Activar Barrido Alcista
input bool DisableLowVolume = true;         // Evitar Volumen Bajo

input group "=== Filtro de Tiempo ==="
input bool EnableTimeFilter = true;          // Activar filtro horario
input int OptimalHour1 = 19;                // Hora óptima 1 (GMT)
input int OptimalHour2 = 18;                // Hora óptima 2 (GMT)
input int OptimalHour3 = 13;                // Hora óptima 3 (GMT)
input int OptimalHour4 = 14;                // Hora óptima 4 (GMT)
input int AvoidHour1 = 9;                   // Hora a evitar 1 (GMT)
input int AvoidHour2 = 16;                  // Hora a evitar 2 (GMT)
input int AvoidHour3 = 23;                  // Hora a evitar 3 (GMT)

input group "=== Gestión de Riesgo ==="
input double RiskPercent = 1.0;             // Riesgo por operación (%)
input double RewardRatio = 2.0;             // Ratio Riesgo:Beneficio
input int MaxSpread = 5;                    // Spread máximo en puntos
input int MaxTrades = 3;                    // Máximo de operaciones

input group "=== Análisis Técnico ==="
input int ATRPeriod = 14;                   // Período ATR
input double ATRMultiplier = 1.5;           // Multiplicador ATR para SL
input int FastMA = 8;                       // MA Rápida
input int SlowMA = 21;                      // MA Lenta

// Variables globales
int magicNumber = 123456;
datetime lastTradeTime = 0;

//+------------------------------------------------------------------+
//| Función para verificar si es hora óptima para operar             |
//+------------------------------------------------------------------+
bool IsOptimalTimeToTrade() {
   if(!EnableTimeFilter) return true; // Si el filtro está desactivado, siempre es hora de operar
   
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int currentHour = dt.hour;
   
   // Verificar si es hora a evitar
   if(currentHour == AvoidHour1 || currentHour == AvoidHour2 || currentHour == AvoidHour3)
      return false;
   
   // Verificar si es hora óptima
   if(currentHour == OptimalHour1 || currentHour == OptimalHour2 || 
      currentHour == OptimalHour3 || currentHour == OptimalHour4)
      return true;
   
   // RELAJADO: Por defecto, permitir trading fuera de horas óptimas
   return true; // Cambiado para permitir más oportunidades
}

//+------------------------------------------------------------------+
//| Función para calcular el ajuste de expectativa basado en la hora |
//+------------------------------------------------------------------+
double GetHourlyMultiplier() {
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int currentHour = dt.hour;
   
   if(currentHour == 19) return 2.5;  // Mejor hora
   if(currentHour == 18) return 2.0;
   if(currentHour == 13 || currentHour == 14) return 1.5;
   if(currentHour == 9 || currentHour == 16) return 0.5;  // Peores horas
   
   return 1.0; // Valor por defecto
}

//+------------------------------------------------------------------+
//| Función para detectar patrones                                   |
//+------------------------------------------------------------------+
PATTERN_TYPE DetectPattern() {
   // Implementación simplificada - en un bot real necesitarías una lógica más compleja
   
   // Verificar patrón BullishBreakout
   if(EnableBullishBreakout && IsBullishBreakout()) {
      return PATTERN_BULLISH_BREAKOUT;
   }
   
   // Verificar patrón BearishBreakout
   if(EnableBearishBreakout && IsBearishBreakout()) {
      return PATTERN_BEARISH_BREAKOUT;
   }
   
   // Verificar patrón HighVolumeBullish
   if(EnableHighVolumeBullish && IsHighVolumeBullish()) {
      return PATTERN_HIGH_VOLUME_BULLISH;
   }
   
   // Verificar patrón BullishSweep
   if(EnableBullishSweep && IsBullishSweep()) {
      return PATTERN_BULLISH_SWEEP;
   }
   
   // Verificar patrón LowVolume
   if(IsLowVolume()) {
      return PATTERN_LOW_VOLUME;
   }
   
   return PATTERN_NONE;
}

//+------------------------------------------------------------------+
//| Función para detectar patrón BullishBreakout                     |
//+------------------------------------------------------------------+
bool IsBullishBreakout() {
   double highs[], closes[];
   long volumes[];  // Cambiado a long para funcionar con CopyTickVolume
   
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(volumes, true);
   
   CopyHigh(_Symbol, PERIOD_CURRENT, 0, 20, highs);
   CopyClose(_Symbol, PERIOD_CURRENT, 0, 20, closes);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 20, volumes);
   
   // Calcular resistencia reciente (máximo de últimos 10 períodos)
   double resistance = 0;
   for(int i = 1; i < 10; i++) {
      if(highs[i] > resistance) resistance = highs[i];
   }
   
   // Calcular volumen promedio
   double avgVolume = 0;
   for(int i = 1; i < 10; i++) {
      avgVolume += (double)volumes[i];  // Convertir a double para cálculos
   }
   avgVolume /= 9;
   
   // RELAJADO: Ruptura con umbral más bajo
   double breakoutThreshold = 0.0001; // Pequeña ruptura es suficiente
   
   // RELAJADO: Requisito de volumen menos estricto
   double volumeMultiplier = 1.2; // Solo 20% sobre el promedio
   
   // Patrón: ruptura alcista con alto volumen
   bool priceBreakout = closes[0] > resistance + breakoutThreshold;
   bool volumeIncrease = (double)volumes[0] > avgVolume * volumeMultiplier;
   
   // Agregar mensaje de depuración
   if(priceBreakout || volumeIncrease) {
      Print("BullishBreakout - Precio: ", priceBreakout ? "SÍ" : "NO", 
            ", Volumen: ", volumeIncrease ? "SÍ" : "NO",
            ", Cierre: ", closes[0],
            ", Resistencia: ", resistance,
            ", Vol Actual: ", volumes[0],
            ", Vol Promedio: ", avgVolume);
   }
   
   return priceBreakout && volumeIncrease;
}

//+------------------------------------------------------------------+
//| Función para detectar patrón BearishBreakout                     |
//+------------------------------------------------------------------+
bool IsBearishBreakout() {
   double lows[], closes[];
   long volumes[];  // Cambiado a long para funcionar con CopyTickVolume
   
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(volumes, true);
   
   CopyLow(_Symbol, PERIOD_CURRENT, 0, 20, lows);
   CopyClose(_Symbol, PERIOD_CURRENT, 0, 20, closes);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 20, volumes);
   
   // Calcular soporte reciente (mínimo de últimos 10 períodos)
   double support = DBL_MAX;
   for(int i = 1; i < 10; i++) {
      if(lows[i] < support) support = lows[i];
   }
   
   // Calcular volumen promedio
   double avgVolume = 0;
   for(int i = 1; i < 10; i++) {
      avgVolume += (double)volumes[i];  // Convertir a double para cálculos
   }
   avgVolume /= 9;
   
   // RELAJADO: Ruptura con umbral más bajo
   double breakoutThreshold = 0.0001; // Pequeña ruptura es suficiente
   
   // RELAJADO: Requisito de volumen menos estricto
   double volumeMultiplier = 1.2; // Solo 20% sobre el promedio
   
   // Patrón: ruptura bajista con alto volumen
   bool priceBreakout = closes[0] < support - breakoutThreshold;
   bool volumeIncrease = (double)volumes[0] > avgVolume * volumeMultiplier;
   
   // Agregar mensaje de depuración
   if(priceBreakout || volumeIncrease) {
      Print("BearishBreakout - Precio: ", priceBreakout ? "SÍ" : "NO", 
            ", Volumen: ", volumeIncrease ? "SÍ" : "NO",
            ", Cierre: ", closes[0],
            ", Soporte: ", support,
            ", Vol Actual: ", volumes[0],
            ", Vol Promedio: ", avgVolume);
   }
   
   return priceBreakout && volumeIncrease;
}

//+------------------------------------------------------------------+
//| Función para detectar patrón HighVolumeBullish                   |
//+------------------------------------------------------------------+
bool IsHighVolumeBullish() {
   double opens[], closes[];
   long volumes[];  // Cambiado a long para funcionar con CopyTickVolume
   
   ArraySetAsSeries(opens, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(volumes, true);
   
   CopyOpen(_Symbol, PERIOD_CURRENT, 0, 10, opens);
   CopyClose(_Symbol, PERIOD_CURRENT, 0, 10, closes);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 10, volumes);
   
   // Calcular volumen promedio
   double avgVolume = 0;
   for(int i = 1; i < 10; i++) {
      avgVolume += (double)volumes[i];  // Convertir a double para cálculos
   }
   avgVolume /= 9;
   
   // RELAJADO: Requisito de volumen menos estricto
   double volumeMultiplier = 1.5; // 50% sobre el promedio
   
   // Patrón: alto volumen con cierre alcista
   bool highVolume = (double)volumes[0] > avgVolume * volumeMultiplier;
   bool bullishClose = closes[0] > opens[0];
   
   // Agregar mensaje de depuración
   if(highVolume || bullishClose) {
      Print("HighVolumeBullish - Volumen: ", highVolume ? "SÍ" : "NO", 
            ", Alcista: ", bullishClose ? "SÍ" : "NO",
            ", Vol Actual: ", volumes[0],
            ", Vol Promedio: ", avgVolume,
            ", Apertura: ", opens[0],
            ", Cierre: ", closes[0]);
   }
   
   return highVolume && bullishClose;
}

//+------------------------------------------------------------------+
//| Función para detectar patrón BullishSweep                        |
//+------------------------------------------------------------------+
bool IsBullishSweep() {
   double highs[], lows[], closes[];
   
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   
   CopyHigh(_Symbol, PERIOD_CURRENT, 0, 10, highs);
   CopyLow(_Symbol, PERIOD_CURRENT, 0, 10, lows);
   CopyClose(_Symbol, PERIOD_CURRENT, 0, 10, closes);
   
   // Encontrar mínimo reciente
   double recentLow = lows[1];
   for(int i = 2; i < 5; i++) {
      if(lows[i] < recentLow) recentLow = lows[i];
   }
   
   // RELAJADO: Criterios más flexibles
   // Patrón: precio se acerca a mínimos recientes pero cierra al alza
   bool sweptLows = lows[0] < recentLow + (10 * _Point); // Cerca del mínimo
   bool bullishRecovery = closes[0] > (lows[0] + (highs[0] - lows[0]) * 0.5); // Cierra en la mitad superior
   
   // Agregar mensaje de depuración
   if(sweptLows || bullishRecovery) {
      Print("BullishSweep - Barrido: ", sweptLows ? "SÍ" : "NO", 
            ", Recuperación: ", bullishRecovery ? "SÍ" : "NO",
            ", Mínimo actual: ", lows[0],
            ", Mínimo reciente: ", recentLow,
            ", Cierre: ", closes[0]);
   }
   
   return sweptLows && bullishRecovery;
}

//+------------------------------------------------------------------+
//| Función para detectar patrón LowVolume                           |
//+------------------------------------------------------------------+
bool IsLowVolume() {
   long volumes[];  // Cambiado a long para funcionar con CopyTickVolume
   
   ArraySetAsSeries(volumes, true);
   CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, 10, volumes);
   
   // Calcular volumen promedio
   double avgVolume = 0;
   for(int i = 1; i < 10; i++) {
      avgVolume += (double)volumes[i];  // Convertir a double para cálculos
   }
   avgVolume /= 9;
   
   // Patrón: volumen muy bajo (menos del 30% del promedio)
   bool lowVolume = (double)volumes[0] < avgVolume * 0.3;
   
   return lowVolume;
}

//+------------------------------------------------------------------+
//| Función para calcular el stop loss basado en ATR                 |
//+------------------------------------------------------------------+
double CalculateStopLoss(ENUM_ORDER_TYPE orderType) {
   double atr[];
   ArraySetAsSeries(atr, true);
   
   int atrHandle = iATR(_Symbol, PERIOD_CURRENT, ATRPeriod);
   CopyBuffer(atrHandle, 0, 0, 1, atr);
   
   double atrValue = atr[0];
   
   if(orderType == ORDER_TYPE_BUY) {
      return SymbolInfoDouble(_Symbol, SYMBOL_ASK) - (atrValue * ATRMultiplier);
   } else {
      return SymbolInfoDouble(_Symbol, SYMBOL_BID) + (atrValue * ATRMultiplier);
   }
}

//+------------------------------------------------------------------+
//| Función para calcular el take profit basado en risk:reward       |
//+------------------------------------------------------------------+
double CalculateTakeProfit(ENUM_ORDER_TYPE orderType, double stopLoss) {
   double entryPrice = (orderType == ORDER_TYPE_BUY) ? 
      SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double stopDistance = MathAbs(entryPrice - stopLoss);
   
   if(orderType == ORDER_TYPE_BUY) {
      return entryPrice + (stopDistance * RewardRatio);
   } else {
      return entryPrice - (stopDistance * RewardRatio);
   }
}

//+------------------------------------------------------------------+
//| Función para calcular el tamaño del lote basado en riesgo        |
//+------------------------------------------------------------------+
double CalculateLotSize(double stopLoss) {
   double accountEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskAmount = accountEquity * RiskPercent / 100.0;
   
   double entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stopDistance = MathAbs(entryPrice - stopLoss);
   
   // Convertir distancia a pips
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   
   // Ajustar para 5 dígitos en lugar de 4
   double pointsPerPip = 10.0;
   if(SymbolInfoInteger(_Symbol, SYMBOL_DIGITS) == 3 || SymbolInfoInteger(_Symbol, SYMBOL_DIGITS) == 5) {
      pointsPerPip = 10.0;
   } else {
      pointsPerPip = 1.0;
   }
   
   double pipsDistance = stopDistance / (tickSize * pointsPerPip);
   
   // Calcular valor del pip
   double tickValue = pipValue / pointsPerPip;
   double valuePerPip = tickValue / tickSize;
   
   // Calcular lote basado en riesgo
   double lotSize = riskAmount / (pipsDistance * valuePerPip);
   
   // Ajustar a los límites del broker
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| Función para determinar el modo de llenado adecuado              |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode(string symbol) {
   // Obtener los modos de llenado permitidos para el símbolo
   uint filling = (uint)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   
   // Verificar modos disponibles en orden de preferencia
   if((filling & SYMBOL_FILLING_FOK) != 0) {
      return ORDER_FILLING_FOK;
   }
   if((filling & SYMBOL_FILLING_IOC) != 0) {
      return ORDER_FILLING_IOC;
   }
   // Si nada más está disponible, usar RETURN
   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Función para obtener descripción de error                        |
//+------------------------------------------------------------------+
string GetErrorDescription(int errorCode) {
   switch(errorCode) {
      case 10004: return "Requiere negociador en cola";
      case 10006: return "La cuenta está deshabilitada para negociar";
      case 10007: return "Solo está permitida la lectura";
      case 10008: return "No hay suficiente memoria para ejecutar la función";
      case 10009: return "Estructura no válida del pedido";
      case 10010: return "Símbolo de comercio no válido";
      case 10011: return "No hay negociador para ejecutar la petición";
      case 10013: return "Demasiados órdenes";
      case 10014: return "Demasiadas posiciones";
      case 10015: return "Órdenes muy cercanas";
      case 10016: return "Muchas órdenes";
      case 10017: return "Intentos de cierre";
      case 10018: return "Modo de llenado no soportado";
      case 10019: return "No hay suficiente liquidez para esta orden";
      case 10020: return "Las opciones de mercado están deshabilitadas";
      case 10021: return "Petición rechazada";
      case 10022: return "Fuera del mercado";
      case 10023: return "Comercio deshabilitado";
      case 10024: return "No hay suficiente dinero";
      case 10025: return "Error al enviar orden";
      case 10026: return "Error al modificar orden";
      case 10027: return "Error al eliminar orden";
      case 10028: return "Error al procesar orden";
      default: return "Error desconocido: " + IntegerToString(errorCode);
   }
}

//+------------------------------------------------------------------+
//| Función para abrir orden de compra                               |
//+------------------------------------------------------------------+
bool OpenBuyOrder(double lotSize, double stopLoss, double takeProfit, string comment) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = ORDER_TYPE_BUY;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.sl = NormalizeDouble(stopLoss, _Digits);
   request.tp = NormalizeDouble(takeProfit, _Digits);
   request.deviation = 10;
   request.magic = magicNumber;
   request.comment = comment;
   // CORREGIDO: Usar el modo de llenado adecuado
   request.type_filling = GetFillingMode(_Symbol);
   
   bool success = OrderSend(request, result);
   
   if(!success) {
      Print("Error al abrir orden BUY: ", GetLastError(), " - ", GetErrorDescription(GetLastError()));
      
      // Intento alternativo con orden pendiente si falla
      if(GetLastError() == 10018) { // Modo de llenado no soportado
         return OpenBuyPendingOrder(lotSize, stopLoss, takeProfit, comment);
      }
      
      return false;
   }
   
   Print("Orden BUY ejecutada con éxito. Ticket: ", result.order);
   lastTradeTime = TimeCurrent();
   return true;
}

//+------------------------------------------------------------------+
//| Función para abrir orden de compra pendiente                     |
//+------------------------------------------------------------------+
bool OpenBuyPendingOrder(double lotSize, double stopLoss, double takeProfit, string comment) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   request.action = TRADE_ACTION_PENDING;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = ORDER_TYPE_BUY_LIMIT;
   request.price = NormalizeDouble(currentPrice - 5 * _Point, _Digits); // Ligeramente por debajo
   request.sl = NormalizeDouble(stopLoss, _Digits);
   request.tp = NormalizeDouble(takeProfit, _Digits);
   request.deviation = 10;
   request.magic = magicNumber;
   request.comment = comment + " (Pendiente)";
   request.type_filling = GetFillingMode(_Symbol);
   request.type_time = ORDER_TIME_GTC;
   
   bool success = OrderSend(request, result);
   
   if(!success) {
      Print("Error al abrir orden pendiente BUY: ", GetLastError(), " - ", GetErrorDescription(GetLastError()));
      return false;
   }
   
   Print("Orden pendiente BUY colocada con éxito. Ticket: ", result.order);
   lastTradeTime = TimeCurrent();
   return true;
}

//+------------------------------------------------------------------+
//| Función para abrir orden de venta                                |
//+------------------------------------------------------------------+
bool OpenSellOrder(double lotSize, double stopLoss, double takeProfit, string comment) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = ORDER_TYPE_SELL;
   request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl = NormalizeDouble(stopLoss, _Digits);
   request.tp = NormalizeDouble(takeProfit, _Digits);
   request.deviation = 10;
   request.magic = magicNumber;
   request.comment = comment;
   // CORREGIDO: Usar el modo de llenado adecuado
   request.type_filling = GetFillingMode(_Symbol);
   
   bool success = OrderSend(request, result);
   
   if(!success) {
      Print("Error al abrir orden SELL: ", GetLastError(), " - ", GetErrorDescription(GetLastError()));
      
      // Intento alternativo con orden pendiente si falla
      if(GetLastError() == 10018) { // Modo de llenado no soportado
         return OpenSellPendingOrder(lotSize, stopLoss, takeProfit, comment);
      }
      
      return false;
   }
   
   Print("Orden SELL ejecutada con éxito. Ticket: ", result.order);
   lastTradeTime = TimeCurrent();
   return true;
}

//+------------------------------------------------------------------+
//| Función para abrir orden de venta pendiente                      |
//+------------------------------------------------------------------+
bool OpenSellPendingOrder(double lotSize, double stopLoss, double takeProfit, string comment) {
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   request.action = TRADE_ACTION_PENDING;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = ORDER_TYPE_SELL_LIMIT;
   request.price = NormalizeDouble(currentPrice + 5 * _Point, _Digits); // Ligeramente por encima
   request.sl = NormalizeDouble(stopLoss, _Digits);
   request.tp = NormalizeDouble(takeProfit, _Digits);
   request.deviation = 10;
   request.magic = magicNumber;
   request.comment = comment + " (Pendiente)";
   request.type_filling = GetFillingMode(_Symbol);
   request.type_time = ORDER_TIME_GTC;
   
   bool success = OrderSend(request, result);
   
   if(!success) {
      Print("Error al abrir orden pendiente SELL: ", GetLastError(), " - ", GetErrorDescription(GetLastError()));
      return false;
   }
   
   Print("Orden pendiente SELL colocada con éxito. Ticket: ", result.order);
   lastTradeTime = TimeCurrent();
   return true;
}

//+------------------------------------------------------------------+
//| Función para verificar spread                                    |
//+------------------------------------------------------------------+
bool IsSpreadAcceptable() {
   long currentSpread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return currentSpread <= MaxSpread;
}

//+------------------------------------------------------------------+
//| Función para verificar número de operaciones abiertas            |
//+------------------------------------------------------------------+
bool CanOpenNewTrade() {
   int openPositions = 0;
   
   for(int i = 0; i < PositionsTotal(); i++) {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == magicNumber) {
         openPositions++;
      }
   }
   
   return openPositions < MaxTrades;
}

//+------------------------------------------------------------------+
//| Función para convertir tipo de patrón a texto                    |
//+------------------------------------------------------------------+
string PatternToString(PATTERN_TYPE pattern) {
   switch(pattern) {
      case PATTERN_BULLISH_BREAKOUT: return "BULLISH BREAKOUT";
      case PATTERN_BEARISH_BREAKOUT: return "BEARISH BREAKOUT";
      case PATTERN_HIGH_VOLUME_BULLISH: return "HIGH VOLUME BULLISH";
      case PATTERN_BULLISH_SWEEP: return "BULLISH SWEEP";
      case PATTERN_LOW_VOLUME: return "LOW VOLUME";
      case PATTERN_NONE: return "NINGUNO";
      default: return "DESCONOCIDO";
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   Print("Bot de Trading inicializado con configuración personalizada");
   Print("Horarios óptimos: ", OptimalHour1, ", ", OptimalHour2, ", ", OptimalHour3, ", ", OptimalHour4);
   Print("Modos de llenado disponibles para ", _Symbol, ": ", SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE));
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   Print("Bot de Trading desactivado, razón: ", reason);
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   // Mensaje cada hora para verificar que el bot sigue funcionando
   static datetime lastCheckTime = 0;
   if(TimeCurrent() - lastCheckTime > 3600) { // 1 hora
      Print("Bot en ejecución, hora actual: ", TimeToString(TimeCurrent()), ", GMT: ", TimeToString(TimeGMT()));
      lastCheckTime = TimeCurrent();
   }
   
   // Verificar si es hora óptima para operar
   if(!IsOptimalTimeToTrade()) {
      return;
   }
   
   // Verificar spread
   if(!IsSpreadAcceptable()) {
      return;
   }
   
   // Verificar número de operaciones abiertas
   if(!CanOpenNewTrade()) {
      return;
   }
   
   // Detectar patrón
   PATTERN_TYPE currentPattern = DetectPattern();
   
   // Si se detectó algún patrón que no sea NONE, registrarlo para depuración
   if(currentPattern != PATTERN_NONE) {
      Print("Patrón detectado: ", PatternToString(currentPattern));
   }
   
   // Procesar patrón detectado
   switch(currentPattern) {
      case PATTERN_BULLISH_BREAKOUT:
         if(EnableBullishBreakout) {
            double expectancyMultiplier = GetHourlyMultiplier();
            double adjustedExpectancy = 2.5 * expectancyMultiplier;
            
            Print("Evaluando BullishBreakout - Expectativa ajustada: ", adjustedExpectancy);
            
            if(adjustedExpectancy > 1.0) {
               double stopLoss = CalculateStopLoss(ORDER_TYPE_BUY);
               double takeProfit = CalculateTakeProfit(ORDER_TYPE_BUY, stopLoss);
               double lotSize = CalculateLotSize(stopLoss);
               
               Print("Intentando abrir orden BUY: Lote=", lotSize, ", SL=", stopLoss, ", TP=", takeProfit);
               
               if(OpenBuyOrder(lotSize, stopLoss, takeProfit, "BullishBreakout")) {
                  Print("Patrón BullishBreakout detectado, orden BUY ejecutada (Exp: ", adjustedExpectancy, ")");
               }
            }
         }
         break;
         
      case PATTERN_BEARISH_BREAKOUT:
         if(EnableBearishBreakout) {
            double expectancyMultiplier = GetHourlyMultiplier();
            double adjustedExpectancy = 2.3 * expectancyMultiplier;
            
            Print("Evaluando BearishBreakout - Expectativa ajustada: ", adjustedExpectancy);
            
            if(adjustedExpectancy > 1.0) {
               double stopLoss = CalculateStopLoss(ORDER_TYPE_SELL);
               double takeProfit = CalculateTakeProfit(ORDER_TYPE_SELL, stopLoss);
               double lotSize = CalculateLotSize(stopLoss);
               
               Print("Intentando abrir orden SELL: Lote=", lotSize, ", SL=", stopLoss, ", TP=", takeProfit);
               
               if(OpenSellOrder(lotSize, stopLoss, takeProfit, "BearishBreakout")) {
                  Print("Patrón BearishBreakout detectado, orden SELL ejecutada (Exp: ", adjustedExpectancy, ")");
               }
            }
         }
         break;
         
      case PATTERN_HIGH_VOLUME_BULLISH:
         if(EnableHighVolumeBullish) {
            double expectancyMultiplier = GetHourlyMultiplier();
            double adjustedExpectancy = 2.1 * expectancyMultiplier;
            
            Print("Evaluando HighVolumeBullish - Expectativa ajustada: ", adjustedExpectancy);
            
            if(adjustedExpectancy > 1.0) {
               double stopLoss = CalculateStopLoss(ORDER_TYPE_BUY);
               double takeProfit = CalculateTakeProfit(ORDER_TYPE_BUY, stopLoss);
               double lotSize = CalculateLotSize(stopLoss);
               
               Print("Intentando abrir orden BUY: Lote=", lotSize, ", SL=", stopLoss, ", TP=", takeProfit);
               
               if(OpenBuyOrder(lotSize, stopLoss, takeProfit, "HighVolumeBullish")) {
                  Print("Patrón HighVolumeBullish detectado, orden BUY ejecutada (Exp: ", adjustedExpectancy, ")");
               }
            }
         }
         break;
         
      case PATTERN_BULLISH_SWEEP:
         if(EnableBullishSweep) {
            double expectancyMultiplier = GetHourlyMultiplier();
            double adjustedExpectancy = 2.0 * expectancyMultiplier;
            
            Print("Evaluando BullishSweep - Expectativa ajustada: ", adjustedExpectancy);
            
            if(adjustedExpectancy > 1.0) {
               double stopLoss = CalculateStopLoss(ORDER_TYPE_BUY);
               double takeProfit = CalculateTakeProfit(ORDER_TYPE_BUY, stopLoss);
               double lotSize = CalculateLotSize(stopLoss);
               
               Print("Intentando abrir orden BUY: Lote=", lotSize, ", SL=", stopLoss, ", TP=", takeProfit);
               
               if(OpenBuyOrder(lotSize, stopLoss, takeProfit, "BullishSweep")) {
                  Print("Patrón BullishSweep detectado, orden BUY ejecutada (Exp: ", adjustedExpectancy, ")");
               }
            }
         }
         break;
         
      case PATTERN_LOW_VOLUME:
         if(DisableLowVolume) {
            Print("Patrón LowVolume detectado, evitando operar (tasa de éxito: 0%)");
         }
         break;
         
      case PATTERN_NONE:
         // No se detectó ningún patrón, no hacer nada
         break;
   }
   
   // NUEVO: Manejo de órdenes pendientes existentes
   ManagePendingOrders();
}

//+------------------------------------------------------------------+
//| Función para gestionar órdenes pendientes                        |
//+------------------------------------------------------------------+
void ManagePendingOrders() {
   // Verificar órdenes pendientes que puedan haber expirado
   for(int i = OrdersTotal() - 1; i >= 0; i--) {
      ulong ticket = OrderGetTicket(i);
      
      if(OrderSelect(ticket) && OrderGetString(ORDER_SYMBOL) == _Symbol && OrderGetInteger(ORDER_MAGIC) == magicNumber) {
         // Verificar si la orden tiene más de 10 minutos sin ejecutarse
         datetime orderTime = (datetime)OrderGetInteger(ORDER_TIME_SETUP);
         if(TimeCurrent() - orderTime > 600) { // 10 minutos
            // Eliminar órdenes pendientes antiguas
            MqlTradeRequest request = {};
            MqlTradeResult result = {};
            
            request.action = TRADE_ACTION_REMOVE;
            request.order = ticket;
            
            if(!OrderSend(request, result)) {
               Print("Error al eliminar orden pendiente antigua #", ticket, ": ", GetLastError());
            } else {
               Print("Orden pendiente antigua #", ticket, " eliminada con éxito");
            }
         }
      }
   }
}