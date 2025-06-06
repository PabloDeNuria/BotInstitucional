//+------------------------------------------------------------------+
//|                                    MultiTimeframeAnalysis.mqh |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Tu Nombre"
#property link      "https://www.ejemplo.com"

class CMultiTimeframe {
private:
    ENUM_TIMEFRAMES m_higherTF;
    ENUM_TIMEFRAMES m_middleTF;
    ENUM_TIMEFRAMES m_lowerTF;
    
    // Detectar tendencia en timeframe
    int DetectTrend(string symbol, ENUM_TIMEFRAMES timeframe);
    
    // Evaluar la calidad del patrón en un timeframe específico
    double EvaluatePattern(string symbol, ENUM_TIMEFRAMES timeframe, string pattern);
    
public:
    CMultiTimeframe();
    ~CMultiTimeframe();
    
    // Configuración
    void Configure(ENUM_TIMEFRAMES higherTF, ENUM_TIMEFRAMES middleTF, ENUM_TIMEFRAMES lowerTF);
    
    // Análisis multitemporal
    double Analyze(string symbol, ENUM_TIMEFRAMES higherTF, ENUM_TIMEFRAMES middleTF, 
                  ENUM_TIMEFRAMES lowerTF, string pattern);
};

// Implementación de la clase
CMultiTimeframe::CMultiTimeframe() {
    m_higherTF = PERIOD_D1;
    m_middleTF = PERIOD_H4;
    m_lowerTF = PERIOD_M15;
}

CMultiTimeframe::~CMultiTimeframe() {
    // Limpieza si es necesario
}

void CMultiTimeframe::Configure(ENUM_TIMEFRAMES higherTF, ENUM_TIMEFRAMES middleTF, ENUM_TIMEFRAMES lowerTF) {
    m_higherTF = higherTF;
    m_middleTF = middleTF;
    m_lowerTF = lowerTF;
}

double CMultiTimeframe::Analyze(string symbol, ENUM_TIMEFRAMES higherTF, ENUM_TIMEFRAMES middleTF, 
                               ENUM_TIMEFRAMES lowerTF, string pattern) {
    // Detectar la tendencia en cada timeframe
    int higherTrend = DetectTrend(symbol, higherTF);
    int middleTrend = DetectTrend(symbol, middleTF);
    int lowerTrend = DetectTrend(symbol, lowerTF);
    
    // Evaluar patrón en cada timeframe
    double higherScore = EvaluatePattern(symbol, higherTF, pattern);
    double middleScore = EvaluatePattern(symbol, middleTF, pattern);
    double lowerScore = EvaluatePattern(symbol, lowerTF, pattern);
    
    // Calcular puntuación ponderada
    // Mayor peso para timeframes superiores en tendencia, menor para timeframes inferiores
    double weightedScore = (higherScore * 0.5) + (middleScore * 0.3) + (lowerScore * 0.2);
    
    // Bonificación por alineación de tendencia entre timeframes
    if(higherTrend > 0 && middleTrend > 0 && lowerTrend > 0) {
        // Todos alcistas
        if(StringFind(pattern, "Bullish") >= 0) {
            weightedScore *= 1.2;  // 20% de bonificación
        }
    }
    else if(higherTrend < 0 && middleTrend < 0 && lowerTrend < 0) {
        // Todos bajistas
        if(StringFind(pattern, "Bearish") >= 0) {
            weightedScore *= 1.2;  // 20% de bonificación
        }
    }
    
    return weightedScore;
}

int CMultiTimeframe::DetectTrend(string symbol, ENUM_TIMEFRAMES timeframe) {
    // Utilizando EMAs para detectar tendencia
    double ema20Array[], ema50Array[];
    
    // Obtener valores EMA
    ArraySetAsSeries(ema20Array, true);
    ArraySetAsSeries(ema50Array, true);
    
    int ema20Handle = iMA(symbol, timeframe, 20, 0, MODE_EMA, PRICE_CLOSE);
    int ema50Handle = iMA(symbol, timeframe, 50, 0, MODE_EMA, PRICE_CLOSE);
    
    if(ema20Handle == INVALID_HANDLE || ema50Handle == INVALID_HANDLE) {
        Print("Error al obtener handles de EMAs: ", GetLastError());
        return 0;  // Neutro
    }
    
    // Copiar datos
    if(CopyBuffer(ema20Handle, 0, 0, 3, ema20Array) <= 0 ||
       CopyBuffer(ema50Handle, 0, 0, 3, ema50Array) <= 0) {
        Print("Error al copiar datos de EMAs: ", GetLastError());
        return 0;  // Neutro
    }
    
    // Liberar handles
    IndicatorRelease(ema20Handle);
    IndicatorRelease(ema50Handle);
    
    // Comparar EMAs para detectar tendencia
    if(ema20Array[0] > ema50Array[0] && ema20Array[1] > ema50Array[1]) {
        return 1;  // Tendencia alcista
    }
    else if(ema20Array[0] < ema50Array[0] && ema20Array[1] < ema50Array[1]) {
        return -1;  // Tendencia bajista
    }
    
    return 0;  // Tendencia neutra o indecisa
}

double CMultiTimeframe::EvaluatePattern(string symbol, ENUM_TIMEFRAMES timeframe, string pattern) {
    // Implementación básica - aquí podrías hacer análisis más sofisticados
    // Por ahora, simplemente devolvemos un valor entre 0 y 1 según el tipo de patrón
    
    double score = 0.5;  // Valor predeterminado neutral
    
    // Evaluar según el tipo de patrón y el timeframe
    if(StringFind(pattern, "Engulfing") >= 0) {
        // Los patrones de envolvente suelen ser más confiables en timeframes más altos
        if(timeframe >= PERIOD_H4) score = 0.7;
        else if(timeframe >= PERIOD_M30) score = 0.6;
        else score = 0.5;
    }
    else if(StringFind(pattern, "Sweep") >= 0) {
        // Los sweeps pueden ser más confiables en timeframes medios
        if(timeframe >= PERIOD_H1 && timeframe <= PERIOD_H4) score = 0.7;
        else score = 0.5;
    }
    else if(StringFind(pattern, "Breakout") >= 0) {
        // Los breakouts suelen ser más confiables en timeframes más altos
        if(timeframe >= PERIOD_H4) score = 0.8;
        else if(timeframe >= PERIOD_H1) score = 0.6;
        else score = 0.4;
    }
    
    return score;
}