//+------------------------------------------------------------------+
//|                                             MarketAnalysis.mqh    |
//|                             Basado en análisis estadístico actual |
//+------------------------------------------------------------------+

// Enumeración de patrones identificados
enum TRADING_PATTERN {
   PATTERN_NONE = 0,
   PATTERN_BULLISH_BREAKOUT,    // 67.2% tasa éxito
   PATTERN_BEARISH_BREAKOUT,    // 64.3% tasa éxito
   PATTERN_HIGH_VOLUME_BULLISH, // 63.0% tasa éxito
   PATTERN_BULLISH_SWEEP,       // 61.2% tasa éxito
   PATTERN_HIGH_VOLUME_BEARISH, // 60.3% tasa éxito
   PATTERN_BULLISH_ENGULFING,   // 60.2% tasa éxito
   PATTERN_BEARISH_ENGULFING,   // 59.5% tasa éxito
   PATTERN_BEARISH_SWEEP,       // 59.1% tasa éxito
   PATTERN_BEARISH_EMA_CROSS,   // 58.6% tasa éxito
   PATTERN_CONSOLIDATION,       // 56.2% tasa éxito - baja prioridad
   PATTERN_BULLISH_EMA_CROSS,   // 55.6% tasa éxito - baja prioridad
   PATTERN_LOW_VOLUME           // 0.0% tasa éxito - evitar
};

// Estructura de patrón con información adicional
struct PatternInfo {
   TRADING_PATTERN pattern;
   double winRate;
   double expectancy;
};

class CMarketAnalysis {
private:
   int maPeriodFast;
   int maPeriodSlow;
   int atrPeriod;
   double breakoutThreshold;
   double volumeThreshold;
   
   // Declaraciones de funciones privadas de detección
   bool IsBullishBreakout(string symbol, ENUM_TIMEFRAMES timeframe);
   bool IsBearishBreakout(string symbol, ENUM_TIMEFRAMES timeframe);
   bool IsHighVolumeBullish(string symbol, ENUM_TIMEFRAMES timeframe);
   bool IsBullishSweep(string symbol, ENUM_TIMEFRAMES timeframe);
   bool IsLowVolume(string symbol, ENUM_TIMEFRAMES timeframe);
   double CalculateExpectancy(TRADING_PATTERN pattern);
   
public:
   CMarketAnalysis() {
      maPeriodFast = 8;
      maPeriodSlow = 21;
      atrPeriod = 14;
      breakoutThreshold = 1.5;
      volumeThreshold = 2.0;
   }
   
   // Configurar parámetros
   void Setup(int fastMA, int slowMA, int atr, double breakout, double volume) {
      maPeriodFast = fastMA;
      maPeriodSlow = slowMA;
      atrPeriod = atr;
      breakoutThreshold = breakout;
      volumeThreshold = volume;
   }
   
   // Analizar mercado para detectar patrones
   PatternInfo AnalyzeMarket(string symbol, ENUM_TIMEFRAMES timeframe) {
      PatternInfo result;
      result.pattern = PATTERN_NONE;
      result.winRate = 0.0;
      result.expectancy = 0.0;
      
      // Verificar patrón BullishBreakout (mayor tasa de éxito 67.2%)
      if(IsBullishBreakout(symbol, timeframe)) {
         result.pattern = PATTERN_BULLISH_BREAKOUT;
         result.winRate = 0.672;
         result.expectancy = CalculateExpectancy(PATTERN_BULLISH_BREAKOUT);
         return result;
      }
      
      // Verificar patrón BearishBreakout (64.3%)
      if(IsBearishBreakout(symbol, timeframe)) {
         result.pattern = PATTERN_BEARISH_BREAKOUT;
         result.winRate = 0.643;
         result.expectancy = CalculateExpectancy(PATTERN_BEARISH_BREAKOUT);
         return result;
      }
      
      // Verificar patrón HighVolumeBullish (63.0%)
      if(IsHighVolumeBullish(symbol, timeframe)) {
         result.pattern = PATTERN_HIGH_VOLUME_BULLISH;
         result.winRate = 0.630;
         result.expectancy = CalculateExpectancy(PATTERN_HIGH_VOLUME_BULLISH);
         return result;
      }
      
      // Verificar patrón BullishSweep (61.2%)
      if(IsBullishSweep(symbol, timeframe)) {
         result.pattern = PATTERN_BULLISH_SWEEP;
         result.winRate = 0.612;
         result.expectancy = CalculateExpectancy(PATTERN_BULLISH_SWEEP);
         return result;
      }
      
      // Continuar con el resto de patrones...
      
      // Verificar patrón LowVolume (Evitar - 0% tasa éxito)
      if(IsLowVolume(symbol, timeframe)) {
         result.pattern = PATTERN_LOW_VOLUME;
         result.winRate = 0.0;
         result.expectancy = 0.0; // Expectativa negativa
         return result;
      }
      
      return result;
   }
};

// Implementaciones de las funciones privadas
bool CMarketAnalysis::IsBullishBreakout(string symbol, ENUM_TIMEFRAMES timeframe) {
   // Obtener datos recientes
   double highs[], lows[], closes[];
   long volumes[];  // Cambiado a long[] para CopyTickVolume
   
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(volumes, true);
   
   int copied = CopyHigh(symbol, timeframe, 0, 20, highs);
   if(copied < 20) return false;
   
   copied = CopyLow(symbol, timeframe, 0, 20, lows);
   if(copied < 20) return false;
   
   copied = CopyClose(symbol, timeframe, 0, 20, closes);
   if(copied < 20) return false;
   
   // CopyTickVolume requiere un array de tipo long[]
   copied = CopyTickVolume(symbol, timeframe, 0, 20, volumes);
   if(copied < 20) return false;
   
   // Calcular resistencia reciente (máximo de los últimos N períodos)
   double resistance = 0;
   for(int i = 1; i < 10; i++) {
      if(highs[i] > resistance) resistance = highs[i];
   }
   
   // Verificar ruptura alcista con aumento de volumen
   bool priceBreakout = closes[0] > resistance && closes[0] > closes[1];
   bool volumeIncrease = volumes[0] > volumes[1] * volumeThreshold;
   
   return priceBreakout && volumeIncrease;
}

bool CMarketAnalysis::IsBearishBreakout(string symbol, ENUM_TIMEFRAMES timeframe) {
   // Obtener datos recientes
   double highs[], lows[], closes[];
   long volumes[];  // Cambiado a long[]
   
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(volumes, true);
   
   int copied = CopyHigh(symbol, timeframe, 0, 20, highs);
   if(copied < 20) return false;
   
   copied = CopyLow(symbol, timeframe, 0, 20, lows);
   if(copied < 20) return false;
   
   copied = CopyClose(symbol, timeframe, 0, 20, closes);
   if(copied < 20) return false;
   
   copied = CopyTickVolume(symbol, timeframe, 0, 20, volumes);
   if(copied < 20) return false;
   
   // Calcular soporte reciente (mínimo de los últimos N períodos)
   double support = DBL_MAX;
   for(int i = 1; i < 10; i++) {
      if(lows[i] < support) support = lows[i];
   }
   
   // Verificar ruptura bajista con aumento de volumen
   bool priceBreakout = closes[0] < support && closes[0] < closes[1];
   bool volumeIncrease = volumes[0] > volumes[1] * volumeThreshold;
   
   return priceBreakout && volumeIncrease;
}

bool CMarketAnalysis::IsHighVolumeBullish(string symbol, ENUM_TIMEFRAMES timeframe) {
   // Obtener datos recientes
   double opens[], closes[];
   long volumes[];  // Cambiado a long[]
   
   ArraySetAsSeries(opens, true);
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(volumes, true);
   
   int copied = CopyOpen(symbol, timeframe, 0, 10, opens);
   if(copied < 10) return false;
   
   copied = CopyClose(symbol, timeframe, 0, 10, closes);
   if(copied < 10) return false;
   
   copied = CopyTickVolume(symbol, timeframe, 0, 10, volumes);
   if(copied < 10) return false;
   
   // Calcular volumen promedio
   double avgVolume = 0;
   for(int i = 1; i < 10; i++) {
      avgVolume += (double)volumes[i]; // Convertir a double para el cálculo
   }
   avgVolume /= 9; // Promedio de los últimos 9 períodos (excluyendo el actual)
   
   // Verificar volumen alto y movimiento alcista
   bool highVolume = (double)volumes[0] > avgVolume * volumeThreshold;
   bool bullishMove = closes[0] > opens[0] && closes[0] > closes[1];
   
   return highVolume && bullishMove;
}

bool CMarketAnalysis::IsBullishSweep(string symbol, ENUM_TIMEFRAMES timeframe) {
   // Obtener datos recientes
   double highs[], lows[], closes[];
   
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);
   
   int copied = CopyHigh(symbol, timeframe, 0, 10, highs);
   if(copied < 10) return false;
   
   copied = CopyLow(symbol, timeframe, 0, 10, lows);
   if(copied < 10) return false;
   
   copied = CopyClose(symbol, timeframe, 0, 10, closes);
   if(copied < 10) return false;
   
   // Encontrar el mínimo reciente
   double recentLow = lows[1];
   int lowIndex = 1;
   for(int i = 2; i < 5; i++) {
      if(lows[i] < recentLow) {
         recentLow = lows[i];
         lowIndex = i;
      }
   }
   
   // Verificar barrido alcista (sweep): precio rompe mínimos recientes y luego sube
   bool sweptLows = lows[0] < recentLow; // Precio rompió los mínimos recientes
   bool bullishRecovery = closes[0] > (highs[1] + lows[1]) / 2; // Cerró por encima del punto medio de la vela anterior
   
   return sweptLows && bullishRecovery;
}

bool CMarketAnalysis::IsLowVolume(string symbol, ENUM_TIMEFRAMES timeframe) {
   // Obtener datos de volumen recientes
   long volumes[];  // Cambiado a long[]
   
   ArraySetAsSeries(volumes, true);
   
   int copied = CopyTickVolume(symbol, timeframe, 0, 10, volumes);
   if(copied < 10) return false;
   
   // Calcular volumen promedio
   double avgVolume = 0;
   for(int i = 1; i < 10; i++) {
      avgVolume += (double)volumes[i]; // Convertir a double para el cálculo
   }
   avgVolume /= 9; // Promedio de los últimos 9 períodos
   
   // Verificar si el volumen actual es significativamente menor al promedio
   bool lowVolume = (double)volumes[0] < avgVolume * 0.5; // Menos del 50% del promedio
   
   return lowVolume;
}

double CMarketAnalysis::CalculateExpectancy(TRADING_PATTERN pattern) {
   // Valores basados en el análisis estadístico
   switch(pattern) {
      case PATTERN_BULLISH_BREAKOUT: return 2.5;
      case PATTERN_BEARISH_BREAKOUT: return 2.3;
      case PATTERN_HIGH_VOLUME_BULLISH: return 2.1;
      case PATTERN_BULLISH_SWEEP: return 2.0;
      // Añadir otros patrones con su expectativa
      case PATTERN_LOW_VOLUME: return -0.5; // Expectativa negativa
      default: return 0.0;
   }
}