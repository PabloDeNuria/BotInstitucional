//+------------------------------------------------------------------+
//|                                               DataCollector.mq5 |
//|                      Copyright 2025, Tu Nombre                   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Tu Nombre"
#property link      "https://www.ejemplo.com"
#property version   "1.00"
#property script_show_inputs

// Parámetros de entrada
input string   OutputFileName = "market_data.csv";  // Nombre del archivo CSV
input datetime StartDate = D'2020.01.01';           // Fecha de inicio para recolección
input int      MaxBars = 10000;                     // Máximo número de barras a procesar
input double   SensitivityMultiplier = 1.5;         // Multiplicador de sensibilidad (mayor = más patrones)

// Estructuras para almacenar eventos de mercado
struct MarketEvent
  {
   datetime          time;
   string            symbol;
   string            eventType;
   double            price;
   double            entryPrice;
   double            stopLoss;
   double            takeProfit;
   double            atr;
   int               volume;
   string            session;
   string            dayOfWeek;
  };

// Variables globales
int fileHandle;
int totalEventsFound = 0;

// Prototipos de funciones
void SaveEvent(datetime time, string symbol, string eventType, double price,
               double entryPrice, double stopLoss, double takeProfit,
               double atr, int volume, string session, string dayOfWeek);
void DetectSweeps(int index, datetime time, string session, string dayOfWeek, double atr);
void DetectEngulfingPatterns(int index, datetime time, string session, string dayOfWeek, double atr);
void DetectBreakouts(int index, datetime time, string session, string dayOfWeek, double atr);
void DetectRetests(int index, datetime time, string session, string dayOfWeek, double atr);
void DetectEMACrosses(int index, datetime time, string session, string dayOfWeek, double atr);
void DetectConsolidations(int index, datetime time, string session, string dayOfWeek, double atr);
void DetectVolumePatterns(int index, datetime time, int volume, string session, string dayOfWeek, double atr);
string DetermineSession(int hour);

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   Print("Iniciando script DataCollector...");

// Preparar la ruta y nombre de archivo
   string filePath = OutputFileName;

// Inicializar recolección de datos
   if(!InitializeDataCollection(filePath))
     {
      return; // Salir si hay errores
     }

// Recolectar datos históricos
   CollectHistoricalData();

// Finalizar recolección
   if(fileHandle != INVALID_HANDLE)
     {
      FileClose(fileHandle);
      Print("Recolección de datos finalizada. Total de eventos encontrados: ", totalEventsFound);

      // Mostrar ubicación del archivo para facilitar su localización
      if(totalEventsFound > 0)
        {
         string filePath = "\\Files\\" + OutputFileName;
         if(FileIsExist(OutputFileName, FILE_COMMON))
           {
            filePath = "\\Common\\Files\\" + OutputFileName;
            Print("Archivo CSV generado en la carpeta común: ", filePath);
            Print("Ruta completa: ", TerminalInfoString(TERMINAL_DATA_PATH), filePath);
           }
         else
           {
            Print("Archivo CSV generado en la carpeta de datos de terminal: ", filePath);
            Print("Ruta completa: ", TerminalInfoString(TERMINAL_DATA_PATH), filePath);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Inicializa la recolección de datos                               |
//+------------------------------------------------------------------+
bool InitializeDataCollection(string filePath)
  {
// Abrir archivo para escritura en la carpeta común (garantiza permisos)
   fileHandle = FileOpen(filePath, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON);

   if(fileHandle == INVALID_HANDLE)
     {
      // Si falla, intentar en la carpeta estándar
      fileHandle = FileOpen(filePath, FILE_WRITE|FILE_CSV|FILE_ANSI);

      if(fileHandle == INVALID_HANDLE)
        {
         int errorCode = GetLastError();
         Print("Error al abrir el archivo: ", errorCode, " - ",
               (errorCode == 5004 ? "No se pudo crear el archivo. Verifique permisos." :
                errorCode == 4103 ? "Ruta inválida." :
                "Error desconocido."));
         return false;
        }
     }

// Escribir encabezados del CSV
   FileWrite(fileHandle,
             "DateTime", "Symbol", "EventType", "Price",
             "EntryPrice", "StopLoss", "TakeProfit",
             "ATR", "Volume", "Session", "DayOfWeek"
            );

   Print("Iniciando recolección de datos desde ", TimeToString(StartDate));
   return true;
  }

//+------------------------------------------------------------------+
//| Función principal para recolectar datos históricos               |
//+------------------------------------------------------------------+
void CollectHistoricalData()
  {
// Obtener número de barras disponibles
   int barsCount = MathMin(Bars(_Symbol, PERIOD_CURRENT), MaxBars);
   Print("Procesando ", barsCount, " barras en ", _Symbol, " ", EnumToString(PERIOD_CURRENT));

// Handle del indicador ATR
   int atrHandle = iATR(_Symbol, PERIOD_CURRENT, 14);
   if(atrHandle == INVALID_HANDLE)
     {
      Print("Error al obtener handle de ATR: ", GetLastError());
      return;
     }

// Buffer para valores ATR
   double atrValues[];
   ArraySetAsSeries(atrValues, true);

// Contadores para depuración
   int processedBars = 0;
   int skippedBars = 0;

// Procesar barras desde la más antigua a la más reciente
   for(int i = barsCount-1; i >= 0; i--)
     {
      datetime barTime = iTime(_Symbol, PERIOD_CURRENT, i);

      // Saltamos barras antes de la fecha de inicio
      if(barTime < StartDate)
        {
         skippedBars++;
         continue;
        }

      // Obtener datos de la barra actual
      double open = iOpen(_Symbol, PERIOD_CURRENT, i);
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      double close = iClose(_Symbol, PERIOD_CURRENT, i);
      long volume = iVolume(_Symbol, PERIOD_CURRENT, i);

      // Obtener valor ATR para la barra actual
      if(CopyBuffer(atrHandle, 0, i, 1, atrValues) <= 0)
        {
         Print("Error al copiar valores de ATR: ", GetLastError());
         continue;
        }
      double atr = atrValues[0];

      // Mostrar progreso cada 1000 barras
      processedBars++;
      if(processedBars % 1000 == 0)
        {
         Print("Procesadas ", processedBars, " barras...");
        }

      // Obtener información de tiempo
      MqlDateTime dt;
      TimeToStruct(barTime, dt);
      string dayOfWeek = EnumToString((ENUM_DAY_OF_WEEK)dt.day_of_week);
      string session = DetermineSession(dt.hour);

      // Detectar eventos de mercado
      DetectPriceEvents(i, barTime, open, high, low, close, volume, atr * SensitivityMultiplier, session, dayOfWeek);
     }

// Liberar handle del indicador
   IndicatorRelease(atrHandle);

   Print("Procesamiento completado: ", processedBars, " barras procesadas, ", skippedBars, " barras omitidas");
  }

//+------------------------------------------------------------------+
//| Detecta varios eventos de precio en las barras                   |
//+------------------------------------------------------------------+
void DetectPriceEvents(int index, datetime time, double open, double high,
                       double low, double close, long volume, double atr,
                       string session, string dayOfWeek)
  {

// Depuración (descomentar si necesitas ver los valores que se están procesando)
// if(index < 10) Print("Procesando barra ", index, ": O=", open, " H=", high, " L=", low, " C=", close, " ATR=", atr);

// ===== Detección de Sweep de Máximos/Mínimos =====
   DetectSweeps(index, time, session, dayOfWeek, atr);

// ===== Detección de Velas Envolventes =====
   DetectEngulfingPatterns(index, time, session, dayOfWeek, atr);

// ===== Detección de Breakouts =====
   DetectBreakouts(index, time, session, dayOfWeek, atr);

// ===== Detección de Retests =====
   DetectRetests(index, time, session, dayOfWeek, atr);

// ===== Detección de Cruces de EMA =====
   DetectEMACrosses(index, time, session, dayOfWeek, atr);

// ===== Detección de Consolidaciones =====
   DetectConsolidations(index, time, session, dayOfWeek, atr);

// ===== Detección de Volumen Alto/Bajo =====
   DetectVolumePatterns(index, time, (int)volume, session, dayOfWeek, atr);
  }

//+------------------------------------------------------------------+
//| Determina la sesión de trading basada en la hora GMT             |
//+------------------------------------------------------------------+
string DetermineSession(int hour)
  {
// Ajusta estos rangos según tu zona horaria y las sesiones que te interesen
   if(hour >= 0 && hour < 8)
      return "Asian";
   if(hour >= 8 && hour < 16)
      return "European";
   return "American";
  }

//+------------------------------------------------------------------+
//| Detecta patrones de velas envolventes                            |
//+------------------------------------------------------------------+
void DetectEngulfingPatterns(int index, datetime time, string session, string dayOfWeek, double atr)
  {
   if(index < 1)
      return; // Necesitamos al menos 2 velas

   double currOpen = iOpen(_Symbol, PERIOD_CURRENT, index);
   double currClose = iClose(_Symbol, PERIOD_CURRENT, index);
   double currHigh = iHigh(_Symbol, PERIOD_CURRENT, index);
   double currLow = iLow(_Symbol, PERIOD_CURRENT, index);
   double prevOpen = iOpen(_Symbol, PERIOD_CURRENT, index+1);
   double prevClose = iClose(_Symbol, PERIOD_CURRENT, index+1);
   double prevHigh = iHigh(_Symbol, PERIOD_CURRENT, index+1);
   double prevLow = iLow(_Symbol, PERIOD_CURRENT, index+1);

// Vela envolvente alcista (versión más permisiva)
   if(currClose > currOpen && // Vela actual alcista
      prevClose < prevOpen && // Vela previa bajista
      currClose > prevOpen && // Cierre actual por encima de apertura previa
      currOpen < prevClose)   // Apertura actual por debajo de cierre previo
     {

      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = currClose;
      double stopLoss = MathMin(currLow, prevLow) - atr * 0.5;
      double takeProfit = currClose + (currClose - stopLoss);

      // Registramos el evento
      SaveEvent(time, _Symbol, "BullishEngulfing", currClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }

// Vela envolvente bajista (versión más permisiva)
   if(currClose < currOpen && // Vela actual bajista
      prevClose > prevOpen && // Vela previa alcista
      currClose < prevOpen && // Cierre actual por debajo de apertura previa
      currOpen > prevClose)   // Apertura actual por encima de cierre previo
     {

      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = currClose;
      double stopLoss = MathMax(currHigh, prevHigh) + atr * 0.5;
      double takeProfit = currClose - (stopLoss - currClose);

      // Registramos el evento
      SaveEvent(time, _Symbol, "BearishEngulfing", currClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }
  }

//+------------------------------------------------------------------+
//| Detecta sweep de máximos/mínimos                                 |
//+------------------------------------------------------------------+
void DetectSweeps(int index, datetime time, string session, string dayOfWeek, double atr)
  {
   if(index < 10)
      return; // Necesitamos suficientes barras para detectar un sweep

// Buscar máximos locales recientes
   double highestHigh = 0;
   int highestIndex = -1;

// Encontrar el máximo local en las últimas 5 barras
   for(int i = index + 1; i <= index + 5; i++)
     {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      if(high > highestHigh)
        {
         highestHigh = high;
         highestIndex = i;
        }
     }

// Buscar mínimos locales recientes
   double lowestLow = 999999;
   int lowestIndex = -1;

// Encontrar el mínimo local en las últimas 5 barras
   for(int i = index + 1; i <= index + 5; i++)
     {
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      if(low < lowestLow)
        {
         lowestLow = low;
         lowestIndex = i;
        }
     }

   double currentHigh = iHigh(_Symbol, PERIOD_CURRENT, index);
   double currentLow = iLow(_Symbol, PERIOD_CURRENT, index);
   double currentClose = iClose(_Symbol, PERIOD_CURRENT, index);

// Sweep de máximo (criteria relajado)
   if(highestIndex > 0 &&
      currentHigh > highestHigh && // La vela actual supera el máximo local previo
      (currentClose < currentHigh - (atr * 0.3)))   // Y cierra por debajo del máximo (wick superior)
     {

      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = highestHigh - (atr * 0.3);
      double stopLoss = currentHigh + atr * 0.7;
      double takeProfit = entryPrice - (stopLoss - entryPrice);

      // Registramos el evento
      SaveEvent(time, _Symbol, "BearishSweep", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }

// Sweep de mínimo (criteria relajado)
   if(lowestIndex > 0 &&
      currentLow < lowestLow && // La vela actual supera el mínimo local previo
      (currentClose > currentLow + (atr * 0.3)))   // Y cierra por encima del mínimo (wick inferior)
     {

      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = lowestLow + (atr * 0.3);
      double stopLoss = currentLow - atr * 0.7;
      double takeProfit = entryPrice + (entryPrice - stopLoss);

      // Registramos el evento
      SaveEvent(time, _Symbol, "BullishSweep", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }
  }

//+------------------------------------------------------------------+
//| Detecta breakouts                                                |
//+------------------------------------------------------------------+
void DetectBreakouts(int index, datetime time, string session, string dayOfWeek, double atr)
  {
   if(index < 20)
      return; // Necesitamos suficientes barras para detectar una consolidación

// Encontrar rango de consolidación
   double highestHigh = 0;
   double lowestLow = 999999;

// Calcular rango en las últimas 10 barras
   for(int i = index + 1; i <= index + 10; i++)
     {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);

      if(high > highestHigh)
         highestHigh = high;
      if(low < lowestLow)
         lowestLow = low;
     }

   double range = highestHigh - lowestLow;

// Si el rango es pequeño (consolidación), buscar breakout
// Criterio más permisivo: 3.5x ATR en lugar de 2x
   if(range < atr * 3.5)
     {
      double currentHigh = iHigh(_Symbol, PERIOD_CURRENT, index);
      double currentLow = iLow(_Symbol, PERIOD_CURRENT, index);
      double currentClose = iClose(_Symbol, PERIOD_CURRENT, index);
      double previousClose = iClose(_Symbol, PERIOD_CURRENT, index+1);

      // Breakout alcista más permisivo
      if(currentClose > highestHigh * 0.9999 && currentClose > previousClose)
        {
         // Calculamos posibles niveles de entrada, SL y TP
         double entryPrice = currentClose;
         double stopLoss = highestHigh - (atr * 0.7);
         double takeProfit = entryPrice + (entryPrice - stopLoss);

         // Registramos el evento
         SaveEvent(time, _Symbol, "BullishBreakout", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
        }

      // Breakout bajista más permisivo
      if(currentClose < lowestLow * 1.0001 && currentClose < previousClose)
        {
         // Calculamos posibles niveles de entrada, SL y TP
         double entryPrice = currentClose;
         double stopLoss = lowestLow + (atr * 0.7);
         double takeProfit = entryPrice - (stopLoss - entryPrice);

         // Registramos el evento
         SaveEvent(time, _Symbol, "BearishBreakout", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
        }
     }
  }

//+------------------------------------------------------------------+
//| Detecta retests                                                  |
//+------------------------------------------------------------------+
void DetectRetests(int index, datetime time, string session, string dayOfWeek, double atr)
  {
   if(index < 20)
      return; // Necesitamos suficientes barras

   double currentClose = iClose(_Symbol, PERIOD_CURRENT, index);
   double previousClose = iClose(_Symbol, PERIOD_CURRENT, index + 1);

// Buscar niveles importantes (soportes/resistencias) recientes
   double significantLevel = 0;
   string levelType = "";

// Simplificación: usar máximos/mínimos recientes como niveles
   double recentHigh = 0;
   double recentLow = 999999;
   int highIndex = -1;
   int lowIndex = -1;

// Buscar en las últimas 20 barras
   for(int i = index + 1; i < index + 20; i++)
     {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);

      if(high > recentHigh)
        {
         recentHigh = high;
         highIndex = i;
        }

      if(low < recentLow)
        {
         recentLow = low;
         lowIndex = i;
        }
     }

// Detectar retest de resistencia - más permisivo (0.5 ATR en lugar de 0.3)
   if(MathAbs(currentClose - recentHigh) < atr * 0.5 &&
      currentClose < recentHigh &&
      index - highIndex > 3)   // Asegurar que haya pasado algo de tiempo
     {

      significantLevel = recentHigh;
      levelType = "Resistance";

      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = currentClose;
      double stopLoss = significantLevel + atr;
      double takeProfit = entryPrice - (stopLoss - entryPrice);

      // Registramos el evento
      SaveEvent(time, _Symbol, "ResistanceRetest", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }

// Detectar retest de soporte - más permisivo (0.5 ATR en lugar de 0.3)
   if(MathAbs(currentClose - recentLow) < atr * 0.5 &&
      currentClose > recentLow &&
      index - lowIndex > 3)   // Asegurar que haya pasado algo de tiempo
     {

      significantLevel = recentLow;
      levelType = "Support";

      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = currentClose;
      double stopLoss = significantLevel - atr;
      double takeProfit = entryPrice + (entryPrice - stopLoss);

      // Registramos el evento
      SaveEvent(time, _Symbol, "SupportRetest", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }
  }

//+------------------------------------------------------------------+
//| Detecta cruces de EMA                                            |
//+------------------------------------------------------------------+
void DetectEMACrosses(int index, datetime time, string session, string dayOfWeek, double atr)
  {
   if(index < 50)
      return; // Necesitamos suficientes barras para las EMAs

// Calcular EMAs
   double ema20Array[], ema50Array[];
   ArraySetAsSeries(ema20Array, true);
   ArraySetAsSeries(ema50Array, true);

   int ema20Handle = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);
   int ema50Handle = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);

   if(ema20Handle == INVALID_HANDLE || ema50Handle == INVALID_HANDLE)
     {
      Print("Error al obtener handles de EMAs: ", GetLastError());
      return;
     }

   if(CopyBuffer(ema20Handle, 0, index, 3, ema20Array) <= 0 ||
      CopyBuffer(ema50Handle, 0, index, 3, ema50Array) <= 0)
     {
      Print("Error al copiar datos de EMAs: ", GetLastError());
      IndicatorRelease(ema20Handle);
      IndicatorRelease(ema50Handle);
      return;
     }

   IndicatorRelease(ema20Handle);
   IndicatorRelease(ema50Handle);

   double currentClose = iClose(_Symbol, PERIOD_CURRENT, index);

// Cruce alcista (EMA corta cruza por encima de EMA larga)
   if(ema20Array[1] <= ema50Array[1] && ema20Array[0] > ema50Array[0])
     {
      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = currentClose;
      double stopLoss = entryPrice - atr * 1.5;
      double takeProfit = entryPrice + atr * 2;

      // Registramos el evento
      SaveEvent(time, _Symbol, "BullishEMACross", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }

// Cruce bajista (EMA corta cruza por debajo de EMA larga)
   if(ema20Array[1] >= ema50Array[1] && ema20Array[0] < ema50Array[0])
     {
      // Calculamos posibles niveles de entrada, SL y TP
      double entryPrice = currentClose;
      double stopLoss = entryPrice + atr * 1.5;
      double takeProfit = entryPrice - atr * 2;

      // Registramos el evento
      SaveEvent(time, _Symbol, "BearishEMACross", currentClose, entryPrice, stopLoss, takeProfit, atr, 0, session, dayOfWeek);
     }
  }

//+------------------------------------------------------------------+
//| Detecta consolidaciones                                          |
//+------------------------------------------------------------------+
void DetectConsolidations(int index, datetime time, string session, string dayOfWeek, double atr)
  {
   if(index < 10)
      return; // Necesitamos suficientes barras

// Calcular rango de las últimas 5 barras
   double highestHigh = 0;
   double lowestLow = 999999;

   for(int i = index; i < index + 5; i++)
     {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);

      if(high > highestHigh)
         highestHigh = high;
      if(low < lowestLow)
         lowestLow = low;
     }

   double range = highestHigh - lowestLow;

// Si el rango es pequeño comparado con ATR, es una consolidación
// Criterio más permisivo: 1.2 ATR en lugar de 0.8
   if(range < atr * 1.2)
     {
      double currentClose = iClose(_Symbol, PERIOD_CURRENT, index);

      // Calculamos posibles niveles de entrada, SL y TP para ambas direcciones
      double entryLong = highestHigh + (atr * 0.2);
      double stopLossLong = lowestLow - (atr * 0.2);
      double takeProfitLong = entryLong + (entryLong - stopLossLong);

      double entryShort = lowestLow - (atr * 0.2);
      double stopLossShort = highestHigh + (atr * 0.2);
      double takeProfitShort = entryShort - (stopLossShort - entryShort);

      // Registramos el evento (consolidación con posibles breakouts en ambas direcciones)
      SaveEvent(time, _Symbol, "Consolidation", currentClose, entryLong, stopLossLong, takeProfitLong, atr, 0, session, dayOfWeek);
     }
  }

//+------------------------------------------------------------------+
//| Detecta patrones de volumen                                      |
//+------------------------------------------------------------------+
void DetectVolumePatterns(int index, datetime time, int volumeValue, string session, string dayOfWeek, double atr)
  {
   if(index < 20)
      return; // Necesitamos suficientes barras para calcular volumen promedio

// Calcular volumen promedio de las últimas 20 barras
   long totalVolume = 0;
   for(int i = index + 1; i <= index + 20; i++)
     {
      totalVolume += iVolume(_Symbol, PERIOD_CURRENT, i);
     }

   double avgVolume = totalVolume / 20.0;

   double currentClose = iClose(_Symbol, PERIOD_CURRENT, index);
   double previousClose = iClose(_Symbol, PERIOD_CURRENT, index + 1);

// Volumen alto (1.8x promedio en lugar de 2x - más permisivo)
   if(volumeValue > avgVolume * 1.8)
     {
      // Volumen alto en vela alcista
      if(currentClose > previousClose)
        {
         // Calculamos posibles niveles de entrada, SL y TP
         double entryPrice = currentClose;
         double stopLoss = currentClose - atr;
         double takeProfit = currentClose + atr * 1.5;

         // Registramos el evento
         SaveEvent(time, _Symbol, "HighVolumeBullish", currentClose, entryPrice, stopLoss, takeProfit, atr, volumeValue, session, dayOfWeek);
        }
      // Volumen alto en vela bajista
      else
         if(currentClose < previousClose)
           {
            // Calculamos posibles niveles de entrada, SL y TP
            double entryPrice = currentClose;
            double stopLoss = currentClose + atr;
            double takeProfit = currentClose - atr * 1.5;

            // Registramos el evento
            SaveEvent(time, _Symbol, "HighVolumeBearish", currentClose, entryPrice, stopLoss, takeProfit, atr, volumeValue, session, dayOfWeek);
           }
     }

// Volumen bajo (menos de 0.6x promedio en lugar de 0.5x - más permisivo)
   if(volumeValue < avgVolume * 0.6)
     {
      // Añadir análisis para determinar posible dirección después de volumen bajo
      double currentHigh = iHigh(_Symbol, PERIOD_CURRENT, index);
      double currentLow = iLow(_Symbol, PERIOD_CURRENT, index);
      double rangeSize = currentHigh - currentLow;

      // Verificar si estamos cerca de un nivel significativo
      bool nearResistance = false;
      bool nearSupport = false;
      double significantLevel = 0;

      // Buscar niveles cercanos
      for(int i = index + 1; i <= index + 10; i++)
        {
         double high = iHigh(_Symbol, PERIOD_CURRENT, i);
         double low = iLow(_Symbol, PERIOD_CURRENT, i);

         //+------------------------------------------------------------------+
         //| Detecta patrones de volumen (continuación)                       |
         //+------------------------------------------------------------------+
         // Cerca de resistencia
         if(MathAbs(currentHigh - high) < atr * 0.5)
           {
            nearResistance = true;
            significantLevel = high;
            break;
           }

         // Cerca de soporte
         if(MathAbs(currentLow - low) < atr * 0.5)
           {
            nearSupport = true;
            significantLevel = low;
            break;
           }
        }

      // Calcular niveles basados en posición respecto a niveles significativos
      double entryPrice = currentClose;
      double stopLoss = 0;
      double takeProfit = 0;

      // Cerca de resistencia, posible señal bajista
      if(nearResistance)
        {
         stopLoss = significantLevel + atr * 0.7;
         takeProfit = currentClose - (stopLoss - currentClose) * 1.5;

         // Registramos el evento con dirección potencial
         SaveEvent(time, _Symbol, "LowVolumeNearResistance", currentClose, entryPrice, stopLoss, takeProfit, atr, volumeValue, session, dayOfWeek);
        }
      // Cerca de soporte, posible señal alcista
      else
         if(nearSupport)
           {
            stopLoss = significantLevel - atr * 0.7;
            takeProfit = currentClose + (currentClose - stopLoss) * 1.5;

            // Registramos el evento con dirección potencial
            SaveEvent(time, _Symbol, "LowVolumeNearSupport", currentClose, entryPrice, stopLoss, takeProfit, atr, volumeValue, session, dayOfWeek);
           }
         // Sin nivel cercano, registrar como indeterminado pero con datos
         else
           {
            // Calcular niveles basados en volatilidad reciente
            double highestHigh = 0;
            double lowestLow = 999999;

            for(int i = index + 1; i <= index + 5; i++)
              {
               if(iHigh(_Symbol, PERIOD_CURRENT, i) > highestHigh)
                  highestHigh = iHigh(_Symbol, PERIOD_CURRENT, i);

               if(iLow(_Symbol, PERIOD_CURRENT, i) < lowestLow)
                  lowestLow = iLow(_Symbol, PERIOD_CURRENT, i);
              }

            // Registramos el evento con posibles niveles en ambas direcciones
            SaveEvent(time, _Symbol, "LowVolume", currentClose, currentClose, lowestLow - atr * 0.3, highestHigh + atr * 0.3, atr, volumeValue, session, dayOfWeek);
           }
     }
  }

//+------------------------------------------------------------------+
//| Guarda un evento detectado en el archivo CSV                     |
//+------------------------------------------------------------------+
void SaveEvent(datetime time, string symbol, string eventType, double price,
               double entryPrice, double stopLoss, double takeProfit,
               double atr, int volume, string session, string dayOfWeek)
  {

// Asegurar que volumen nunca sea 0
   if(volume <= 0)
     {
      // Intentar obtener el volumen de la vela en cuestión
      int barIndex = iBarShift(symbol, PERIOD_CURRENT, time);
      if(barIndex >= 0)
        {
         volume = (int)iVolume(symbol, PERIOD_CURRENT, barIndex);
        }

      // Si aún es 0 o no se pudo obtener, usar un valor mínimo
      if(volume <= 0)
        {
         volume = 1; // Volumen mínimo, nunca 0
        }
     }

   if(fileHandle != INVALID_HANDLE)
     {
      FileWrite(fileHandle,
                TimeToString(time), symbol, eventType, DoubleToString(price, _Digits),
                DoubleToString(entryPrice, _Digits), DoubleToString(stopLoss, _Digits),
                DoubleToString(takeProfit, _Digits), DoubleToString(atr, _Digits),
                IntegerToString(volume), session, dayOfWeek
               );

      totalEventsFound++;
      if(totalEventsFound % 100 == 0)
        {
         Print("Eventos encontrados hasta ahora: ", totalEventsFound);
        }
      else
         if(totalEventsFound < 10)
           {
            // Mostrar los primeros eventos encontrados para depuración
            Print("Evento #", totalEventsFound, " encontrado: ", eventType, " en ", TimeToString(time));
           }
     }
  }
//+------------------------------------------------------------------+
