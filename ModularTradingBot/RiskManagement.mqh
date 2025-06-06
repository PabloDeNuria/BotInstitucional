//+------------------------------------------------------------------+
//|                                            RiskManagement.mqh |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Tu Nombre"
#property link      "https://www.ejemplo.com"

class CRiskManagement {
private:
    double m_riskPerTrade;
    bool m_dynamicPosition;
    double m_maxDrawdown;
    
    // Variables para gestión dinámica de riesgo
    double m_initialBalance;
    double m_currentRiskMultiplier;
    int m_consecutiveWins;
    int m_consecutiveLosses;
    
    // Correlación entre símbolos
    string m_correlatedSymbols[];
    
public:
    CRiskManagement();
    ~CRiskManagement();
    
    // Configuración
    void Configure(double riskPerTrade, bool dynamicPosition, double maxDrawdown);
    
    // Cálculo de tamaño de lote basado en riesgo
    double CalculateLotSize(double riskPercent, double entryPrice, double stopLoss);
    
    // Verificación de drawdown
    bool CheckDrawdown(double maxDrawdownPercent);
    
    // Verificación de correlación
    bool CheckCorrelation(string symbol);
    
    // Actualización de riesgo dinámico
    void UpdateDynamicRisk();
    
    // Gestión de resultados de operaciones
    void RegisterTradeResult(bool isWin, double profitLoss);
    
    // Añadir símbolo correlacionado
    void AddCorrelatedSymbol(string symbol);
};

// Implementación de la clase
CRiskManagement::CRiskManagement() {
    m_initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    m_currentRiskMultiplier = 1.0;
    m_consecutiveWins = 0;
    m_consecutiveLosses = 0;
}

CRiskManagement::~CRiskManagement() {
    // Limpieza si es necesario
}

void CRiskManagement::Configure(double riskPerTrade, bool dynamicPosition, double maxDrawdown) {
    m_riskPerTrade = riskPerTrade;
    m_dynamicPosition = dynamicPosition;
    m_maxDrawdown = maxDrawdown;
}

double CRiskManagement::CalculateLotSize(double riskPercent, double entryPrice, double stopLoss) {
    // Aplicar multiplicador dinámico si está habilitado
    double effectiveRisk = riskPercent;
    if(m_dynamicPosition) {
        effectiveRisk *= m_currentRiskMultiplier;
    }
    
    // Limitar el riesgo máximo
    if(effectiveRisk > 5.0) effectiveRisk = 5.0;
    if(effectiveRisk < 0.1) effectiveRisk = 0.1;
    
    // Calcular capital en riesgo
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * (effectiveRisk / 100.0);
    
    // Calcular diferencia en pips
    double pipSize = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 
                    (SymbolInfoInteger(_Symbol, SYMBOL_DIGITS) == 3 || SymbolInfoInteger(_Symbol, SYMBOL_DIGITS) == 5 ? 10 : 1);
    double pipDifference = MathAbs(entryPrice - stopLoss) / pipSize;
    
    // Calcular valor por pip
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double valuePerPip = tickValue * (pipSize / tickSize);
    
    // Calcular tamaño de lote
    double lotSize = riskAmount / (pipDifference * valuePerPip);
    
    // Ajustar al tamaño de lote mínimo permitido
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    
    if(lotSize < minLot) lotSize = minLot;
    if(lotSize > maxLot) lotSize = maxLot;
    
    return lotSize;
}

bool CRiskManagement::CheckDrawdown(double maxDrawdownPercent) {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Calcular drawdown actual
    double currentDrawdown = (1.0 - equity / balance) * 100.0;
    
    // Verificar si excede el máximo permitido
    return (currentDrawdown <= maxDrawdownPercent);
}

bool CRiskManagement::CheckCorrelation(string symbol) {
    int openPositions = 0;
    double totalExposure = 0.0;
    
    // Verificar posiciones abiertas en símbolos correlacionados
    for(int i = 0; i < ArraySize(m_correlatedSymbols); i++) {
        int positions = 0;
        
        // Contar posiciones abiertas en este símbolo correlacionado
        for(int pos = PositionsTotal() - 1; pos >= 0; pos--) {
            if(PositionGetSymbol(pos) == m_correlatedSymbols[i]) {
                positions++;
                
                // Sumar a la exposición total
                totalExposure += PositionGetDouble(POSITION_VOLUME);
            }
        }
        
        if(positions > 0) {
            openPositions++;
        }
    }
    
    // Límite simple: no más de 3 posiciones correlacionadas abiertas simultáneamente
    // y no más del 10% del balance en exposición total
    double maxExposure = AccountInfoDouble(ACCOUNT_BALANCE) * 0.1 / SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    
    return (openPositions < 3 && totalExposure < maxExposure);
}

void CRiskManagement::UpdateDynamicRisk() {
    if(!m_dynamicPosition) return;
    
    // Implementación anti-martingala: incrementar tamaño después de victorias, reducir después de pérdidas
    if(m_consecutiveWins > 2) {
        // Aumentar riesgo gradualmente después de varias victorias consecutivas
        m_currentRiskMultiplier = 1.0 + (m_consecutiveWins * 0.1);  // +10% por cada victoria consecutiva
    } else if(m_consecutiveLosses > 0) {
        // Reducir riesgo después de pérdidas
        m_currentRiskMultiplier = 1.0 - (m_consecutiveLosses * 0.2);  // -20% por cada pérdida consecutiva
    } else {
        // Reset a valores normales
        m_currentRiskMultiplier = 1.0;
    }
    
    // Limitar el multiplicador
    if(m_currentRiskMultiplier < 0.5) m_currentRiskMultiplier = 0.5;
    if(m_currentRiskMultiplier > 2.0) m_currentRiskMultiplier = 2.0;
}

void CRiskManagement::RegisterTradeResult(bool isWin, double profitLoss) {
    if(isWin) {
        m_consecutiveWins++;
        m_consecutiveLosses = 0;
    } else {
        m_consecutiveLosses++;
        m_consecutiveWins = 0;
    }
    
    // Actualizar el multiplicador de riesgo dinámico
    UpdateDynamicRisk();
}

void CRiskManagement::AddCorrelatedSymbol(string symbol) {
    int size = ArraySize(m_correlatedSymbols);
    ArrayResize(m_correlatedSymbols, size + 1);
    m_correlatedSymbols[size] = symbol;
}