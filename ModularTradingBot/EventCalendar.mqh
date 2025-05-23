//+------------------------------------------------------------------+
//|                                            EventCalendar.mqh |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Tu Nombre"
#property link      "https://www.ejemplo.com"

// Estructura para eventos económicos
struct EconomicEvent {
    datetime time;
    string currency;
    string event;
    int impact;  // 1-3, donde 3 es el mayor impacto
};

class CEventCalendar {
private:
    bool m_enabled;
    EconomicEvent m_events[];
    datetime m_lastUpdate;
    
    // Método para cargar eventos desde archivo o API
    bool LoadEvents();
    
public:
    CEventCalendar();
    ~CEventCalendar();
    
    // Configuración
    void Configure(bool enabled);
    
    // Actualizar calendario
    bool Update();
    
    // Verificar si hay eventos de alto impacto próximos
    bool HasImpactfulNewsIncoming(string symbol, int minutesAhead);
    
    // Obtener próximo evento para un símbolo
    bool GetNextEvent(string symbol, EconomicEvent &event);
};

// Implementación de la clase
CEventCalendar::CEventCalendar() {
    m_enabled = false;
    m_lastUpdate = 0;
}

CEventCalendar::~CEventCalendar() {
    // Limpieza si es necesario
}

void CEventCalendar::Configure(bool enabled) {
    m_enabled = enabled;
}

bool CEventCalendar::Update() {
    if(!m_enabled) return false;
    
    // Actualizar solo cada hora para evitar sobrecarga
    if(TimeCurrent() - m_lastUpdate < 3600) return true;
    
    // Cargar eventos desde archivo o API
    bool result = LoadEvents();
    
    if(result) {
        m_lastUpdate = TimeCurrent();
    }
    
    return result;
}

bool CEventCalendar::LoadEvents() {
    // En una implementación real, aquí cargarías eventos desde un archivo CSV o API
    // Para este ejemplo, simularemos algunos eventos
    
    // Limpiar eventos anteriores
    ArrayFree(m_events);
    
    // Simular algunos eventos económicos
    int numEvents = 5;
    ArrayResize(m_events, numEvents);
    
    datetime currentTime = TimeCurrent();
    
    // Evento 1: Decisión de tipo de interés Fed - alto impacto
    m_events[0].time = currentTime + 3600 * 24;  // Mañana
    m_events[0].currency = "USD";
    m_events[0].event = "Fed Interest Rate Decision";
    m_events[0].impact = 3;
    
    // Evento 2: PIB EE.UU. - alto impacto
    m_events[1].time = currentTime + 3600 * 48;  // Pasado mañana
    m_events[1].currency = "USD";
    m_events[1].event = "GDP Report";
    m_events[1].impact = 3;
    
    // Evento 3: PMI Eurozona - impacto medio
    m_events[2].time = currentTime + 3600 * 12;  // En 12 horas
    m_events[2].currency = "EUR";
    m_events[2].event = "Eurozone PMI";
    m_events[2].impact = 2;
    
    // Evento 4: Inventarios de petróleo - impacto bajo
    m_events[3].time = currentTime + 3600 * 6;  // En 6 horas
    m_events[3].currency = "OIL";
    m_events[3].event = "Crude Oil Inventories";
    m_events[3].impact = 1;
    
    // Evento 5: Tasa de desempleo Japón - impacto medio
    m_events[4].time = currentTime + 3600 * 36;  // En 36 horas
    m_events[4].currency = "JPY";
    m_events[4].event = "Unemployment Rate";
    m_events[4].impact = 2;
    
    return true;
}

bool CEventCalendar::HasImpactfulNewsIncoming(string symbol, int minutesAhead) {
    if(!m_enabled) return false;
    
    // Obtener las monedas del par
    string baseCurrency = StringSubstr(symbol, 0, 3);
    string quoteCurrency = StringSubstr(symbol, 3, 3);
    
    // Verificar eventos para ambas monedas
    datetime currentTime = TimeCurrent();
    datetime futureTime = currentTime + minutesAhead * 60;
    
    for(int i = 0; i < ArraySize(m_events); i++) {
        // Solo nos preocupan eventos de alto impacto (nivel 3)
        if(m_events[i].impact < 3) continue;
        
        // Verificar si el evento está dentro del rango de tiempo especificado
        if(m_events[i].time >= currentTime && m_events[i].time <= futureTime) {
            // Verificar si el evento afecta a alguna de las monedas del par
            if(m_events[i].currency == baseCurrency || m_events[i].currency == quoteCurrency) {
                return true;  // Hay un evento de alto impacto próximo
            }
        }
    }
    
    return false;  // No hay eventos de alto impacto próximos
}

bool CEventCalendar::GetNextEvent(string symbol, EconomicEvent &event) {
    if(!m_enabled) return false;
    
    // Obtener las monedas del par
    string baseCurrency = StringSubstr(symbol, 0, 3);
    string quoteCurrency = StringSubstr(symbol, 3, 3);
    
    // Buscar el próximo evento relevante
    datetime currentTime = TimeCurrent();
    datetime nearestTime = D'2099.12.31 23:59:59';
    int nearestIndex = -1;
    
    for(int i = 0; i < ArraySize(m_events); i++) {
        // Verificar si el evento es en el futuro y afecta a alguna de las monedas
        if(m_events[i].time > currentTime && 
           (m_events[i].currency == baseCurrency || m_events[i].currency == quoteCurrency)) {
            
            // Verificar si este evento es más cercano que el anterior más cercano
            if(m_events[i].time < nearestTime) {
                nearestTime = m_events[i].time;
                nearestIndex = i;
            }
        }
    }
    
    // Si encontramos un evento, devolverlo
    if(nearestIndex >= 0) {
        event = m_events[nearestIndex];
        return true;
    }
    
    return false;  // No se encontró ningún evento futuro relevante
}