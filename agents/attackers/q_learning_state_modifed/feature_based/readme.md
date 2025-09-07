
# Implementación de Q-agent con feature-based state representation

Este documento describe las modificaciones realizadas al agente Q-learning original para implementar una representación basada en características (feature-based)

## Principales cambios

### Integración del feature extractor
```python
from feature_extractor import FeatureExtractor

class QAgent(BaseAgent):
    def __init__(self, ...):
        # ...existing code...
        self.feature_extractor = FeatureExtractor()
```

- Se incorporó un feature extractor para transformar estados crudos en vectores de características.
- Reemplaza el sistema anterior de mapeo de strings a ids

---

### Representación del estado

**q-agent original:**
```python
    def get_state_id(self, state:GameState) -> int:
        # Here the state has to be ordered, so different orders are not taken as two different states.
        state_str = state_as_ordered_string(state)
        if state_str not in self._str_to_id:
            self._str_to_id[state_str] = len(self._str_to_id)
        return self._str_to_id[state_str]
```
- Usa un mapeo directo string -> int
- Cada estado único recibe un ID
- No hay generalización entre estados similares

Ejemplo de state_id: 
```python
state_id:  16
```

---

### Modificación del método get_state_id

```python
def get_state_id(self, state:GameState) -> tuple:
    """
    Extrae características del estado y las usa como identificador
    """
    state_str = state_as_ordered_string(state)
    features = self.feature_extractor.extract_features(state_str)
    return tuple(np.round(features, decimals=2))
```
- Extrae características numéricas del estado
- Retorna tuple con conteos de elementos clave
- Permite generalización al agrupar estados similares

Ejemplo de state_id:
```python
state_id:  (3, 7, 3, 3, 1)
```

Ahora retorna un tuple de características numéricas en lugar de un id entero
Las características incluyen conteos de:
- Redes conocidas
- Hosts conocidos
- Hosts controlados
- Servicios totales
- Datos encontrados

De esta manera, la Q-table logra una mayor capacidad de generalización, ya que se accede mediante:

```python
self.q_values[state_id, action]
```

--- 

### Archivos modificados
- q_agent_feature_based.py: Nueva implementación del agente
- feature_extractor.py: Módulo para extracción de características