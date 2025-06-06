//+------------------------------------------------------------------+
//|                                                 TimeFilter.mqh    |
//|                           Basado en análisis estadístico de horas |
//+------------------------------------------------------------------+

// Incluir para MqlDateTime
#include <Object.mqh>

class CTimeFilter {
private:
   bool enabled;
   int optimalHours[];
   int avoidHours[];
   
public:
   CTimeFilter() {
      enabled = true;
      ArrayResize(optimalHours, 4);
      optimalHours[0] = 19; // Hora más rentable
      optimalHours[1] = 18; 
      optimalHours[2] = 13;
      optimalHours[3] = 14;
      
      ArrayResize(avoidHours, 3);
      avoidHours[0] = 9;  // Horas a evitar
      avoidHours[1] = 16;
      avoidHours[2] = 23;
   }
   
   void SetEnabled(bool state) {
      enabled = state;
   }
   
   void ConfigureOptimalHours(int &hours[]) {
      ArrayCopy(optimalHours, hours);
   }
   
   void ConfigureAvoidHours(int &hours[]) {
      ArrayCopy(avoidHours, hours);
   }
   
   bool IsOptimalTimeToTrade() {
      if(!enabled) return true; // Si está desactivado, siempre retorna true
      
      // Obtener hora actual GMT
      MqlDateTime dt;
      TimeToStruct(TimeGMT(), dt);
      int currentHour = dt.hour;
      
      // Verificar si es hora a evitar
      for(int i = 0; i < ArraySize(avoidHours); i++) {
         if(currentHour == avoidHours[i]) {
            return false; // Hora a evitar
         }
      }
      
      // Verificar si es hora óptima
      for(int i = 0; i < ArraySize(optimalHours); i++) {
         if(currentHour == optimalHours[i]) {
            return true; // Hora óptima
         }
      }
      
      // Si no es hora óptima pero tampoco a evitar, decisión según configuración
      return false; // Por defecto, solo operar en horas óptimas
   }
   
   // Calcular valor de expectativa basado en la hora
   double GetHourlyExpectancyMultiplier() {
      // Obtener hora actual GMT
      MqlDateTime dt;
      TimeToStruct(TimeGMT(), dt);
      int currentHour = dt.hour;
      
      // Basado en los resultados del análisis, asignar multiplicadores
      switch(currentHour) {
         case 19: return 2.5;  // Hora con mejor rendimiento
         case 18: return 2.0;
         case 13: case 14: return 1.5;
         case 9: case 16: return 0.5; // Horas con rendimiento negativo
         default: return 1.0;
      }
   }
};