
# Implementación de Q-agent con feature-based state representation

Este documento describe las modificaciones realizadas al agente Q-learning original para implementar una representación basada en características (feature-based)

## Principales Cambios

### Integración del feature extractor
```python
from feature_extractor import FeatureExtractor

class QAgent(BaseAgent):
    def __init__(self, ...):
        # ...existing code...
        self.feature_extractor = FeatureExtractor()
```

- Se añadió el feature extractor para transformar estados crudos en vectores de características
- Reemplaza el sistema anterior de mapeo de strings a ids

### Ejemplo de transformación

**Estado de entrada:**
```python
"nets:[192.168.1.0/24,192.168.2.0/24],
 hosts:[192.168.1.2,192.168.1.3],
 controlled:[192.168.1.2],
 services:{192.168.1.2:[Service(name='ssh')]},
 data:{192.168.1.2:[Data(owner='user')]}"
```

**Vector resultante:**
```python
[2,  # 2 redes conocidas
 2,  # 2 hosts conocidos
 1,  # 1 host controlado
 1,  # 1 servicio total
 1]  # 1 dato encontrado
```

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

Ahora retorna un tuple de características numéricas en lugar de un id entero
Las características incluyen conteos de:
- Redes conocidas
- Hosts conocidos
- Hosts controlados
- Servicios totales
- Datos encontrados

### Archivos modificados
- q_agent_feature_based.py: Nueva implementación del agente
- feature_extractor.py: Módulo para extracción de características