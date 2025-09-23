
# Agente Q-Learning con representación basada en características

## Descripción general

Esta implementación mejora el agente Q-Learning original mediante una representación del estado basada en características (feature-based), diseñada para mejorar la generalización y eficiencia del aprendizaje en entornos de ciberseguridad.

### Problema abordado

El agente Q-Learning original utiliza una representación de estado que mapea cada estado único a un identificador específico, lo que puede resultar en:
- **Falta de generalización**: Estados similares se tratan como completamente independientes
- **Aprendizaje lento**: El agente no puede transferir conocimiento entre estados relacionados

### Solución propuesta

La nueva implementación transforma estados complejos en vectores de características numéricas que capturan la información esencial del entorno, permitiendo que estados similares compartan conocimiento.

## Arquitectura del sistema

### 1. Extractor de características

```python
from feature_extractor import FeatureExtractor

class QAgent(BaseAgent):
    def __init__(self, ...):
        self.feature_extractor = FeatureExtractor()
```

**Funcionalidad:**
- Convierte estados complejos en vectores numéricos de 5 dimensiones
- Extrae métricas relevantes del entorno de ciberseguridad
- Permite configuración personalizada de características

### 2. Representación del estado

#### Implementación original
```python
def get_state_id(self, state: GameState) -> int:
    state_str = state_as_ordered_string(state)
    if state_str not in self._str_to_id:
        self._str_to_id[state_str] = len(self._str_to_id)
    return self._str_to_id[state_str]
```

**Características:**
- Mapeo directo string → entero
- Cada estado único recibe un ID único
- Sin generalización entre estados

#### Nueva implementación
```python
def get_state_id(self, state: GameState) -> tuple:
    """Extrae características del estado y las usa como identificador"""
    state_str = state_as_ordered_string(state)
    features = self.feature_extractor.extract_features(state_str)
    return tuple(int(x) for x in features)
```

**Características:**
- Vector de características como identificador
- Generalización automática entre estados similares
- Espacio de estados reducido y manejable

## Vector de características

El sistema extrae 5 características principales:

```python
[num_networks, num_known_hosts, num_controlled_hosts, total_services, total_data]
```

### Descripción de características

1. **`num_networks`**: Cantidad de redes descubiertas
2. **`num_known_hosts`**: Número total de hosts identificados
3. **`num_controlled_hosts`**: Hosts bajo control del atacante
4. **`total_services`**: Servicios detectados en todos los hosts
5. **`total_data`**: Elementos de datos encontrados

### Ejemplo práctico

**Estado del entorno:**
```
np.array([num_networks, num_known_hosts, num_controlled_hosts, total_services, total_data])
```
    example_state: """
    nets:[192.168.1.0/24,192.168.2.0/24,192.168.3.0/24,213.47.23.192/26],
    hosts:[192.168.1.2,192.168.1.3,192.168.1.4,192.168.2.1,192.168.2.2,192.168.2.3,192.168.2.4,192.168.2.5,192.168.2.6,213.47.23.195],
    controlled:[192.168.2.2,192.168.2.4,192.168.2.6,213.47.23.195],
    services:{192.168.1.3:[Service(name='postgresql', type='passive', version='14.3.0', is_local=False),Service(name='ssh', type='passive', version='8.1.0', is_local=False)]192.168.2.2:[Service(name='ms-wbt-server', type='passive', version='10.0.19041', is_local=False)]192.168.2.3:[Service(name='ms-wbt-server', type='passive', version='10.0.19041', is_local=False)]192.168.2.4:[Service(name='bash', type='passive', version='5.0.0', is_local=True),Service(name='ssh', type='passive', version='8.1.0', is_local=False)]192.168.2.5:[Service(name='ssh', type='passive', version='8.1.0', is_local=False)]192.168.2.6:[Service(name='bash', type='passive', version='5.0.0', is_local=True)]213.47.23.195:[Service(name='bash', type='passive', version='5.0.0', is_local=True),Service(name='listener', type='passive', version='1.0.0', is_local=False)]},
    data:{192.168.2.2:[Data(owner='system', id='logfile', size=439, type='log')]192.168.2.4:[Data(owner='system', id='logfile', size=439, type='log')]213.47.23.195:[Data(owner='system', id='logfile', size=676, type='log')]},
    blocks:{}"""


**Vector de características resultante:**
```python
[4, 10, 4, 10, 3]
```

## Ventajas de la implementación

### Generalización mejorada
- Estados con características similares comparten conocimiento
- Reducción significativa del tiempo de entrenamiento
- Mejor transferencia de aprendizaje entre escenarios

### Eficiencia computacional
- Espacio de estados drasticamente reducido
- Menor uso de memoria para la Q-table
- Convergencia más rápida del algoritmo (entre 4000 y 6000 episodios).

### Interpretabilidad
- Características tienen significado semántico claro
- Fácil análisis del comportamiento del agente
- Debugging simplificado

