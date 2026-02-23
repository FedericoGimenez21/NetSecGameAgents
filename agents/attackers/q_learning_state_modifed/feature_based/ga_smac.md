# Documentación: Optimización de Q-Table con Algoritmos Genéticos y SMAC3

## Índice
1. [Introducción](#introducción)
2. [Problema a Resolver](#problema-a-resolver)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Componentes Principales](#componentes-principales)
   - 4.1 [QTableOptimizationProblem](#1-clase-qtableoptimizationproblem)
   - 4.2 [QTableGeneticOptimizer](#2-clase-qtablegeneticoptimizer)
   - 4.3 [SMACGAObjective](#3-clase-smacgaobjective)
   - 4.4 [optimize_with_smac()](#4-función-optimize_with_smac)
   - 4.5 [plot_smac_results()](#5-función-plot_smac_results)
5. [Modos de Evaluación Paralela](#modos-de-evaluación-paralela)
6. [Sistema de Checkpoint y Reanudación](#sistema-de-checkpoint-y-reanudación)
7. [Definición del Espacio de Configuración](#definición-del-espacio-de-configuración)
8. [Proceso de Ejecución](#proceso-de-ejecución)
9. [Modos de Operación](#modos-de-operación)
10. [Resultados y Visualizaciones](#resultados-y-visualizaciones)
11. [Recomendaciones y Mejores Prácticas](#recomendaciones-y-mejores-prácticas)
12. [Conclusión](#conclusión)

---

## Introducción

`ga_smac.py` es un sistema de optimización multinivel que combina **Algoritmos Genéticos (GA)** con **Optimización Bayesiana (SMAC3)** para encontrar la mejor inicialización de una Q-table utilizada en agentes de aprendizaje por refuerzo para el juego NetSecGame.

### Características principales:
- **Optimización de Q-tables**: Inicializa tablas Q con valores optimizados para acelerar el aprendizaje
- **Dos niveles de optimización**: GA para los valores de la Q-table y SMAC3 para los hiperparámetros del GA
- **Evaluación en entorno real**: Cada candidato es evaluado ejecutando el agente via subprocess en el juego
- **Evaluación paralela**: Soporte para N workers simultáneos mediante `ThreadPoolExecutor`
- **Modo multi-servidor**: Gestión automática de N instancias de NetSecGame (una por worker)
- **Sistema de checkpoint**: Persistencia atómica del estado de SMAC; reanudación automática si se interrumpe
- **Visualizaciones completas**: Genera 15+ gráficas detalladas (PNG 300 DPI) y un README Markdown con tabla de resultados

---

## Problema a Resolver

### Contexto
En Q-Learning, la **Q-table** es una estructura que almacena valores Q(s,a) para cada par estado-acción. Tradicionalmente, esta tabla se inicializa con valores aleatorios o ceros, lo que puede resultar en:
- **Aprendizaje lento**: El agente necesita muchos episodios para converger
- **Exploración subóptima**: Valores iniciales pobres pueden llevar a políticas subóptimas
- **Ineficiencia**: Desperdicio de recursos computacionales en exploración innecesaria

### Solución Propuesta
Este sistema resuelve el problema mediante:

1. **Optimización de Inicialización (GA)**:
   - Usa algoritmos genéticos para encontrar valores iniciales óptimos para la Q-table
   - Cada "individuo" es una Q-table completa
   - El fitness se mide ejecutando el agente con esa Q-table en episodios de prueba

2. **Meta-optimización de Hiperparámetros (SMAC3)**:
   - Optimiza los hiperparámetros del GA (tamaño de población, probabilidades de cruce/mutación, etc.)
   - Usa optimización bayesiana con Random Forest como modelo surrogate
   - Encuentra la mejor configuración del GA para optimizar Q-tables

### Beneficios
- Agentes que aprenden más rápido desde el inicio
- Mejor rendimiento inicial (mayor win rate)
- Convergencia más rápida a políticas óptimas
- Configuración automática de hiperparámetros

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     NIVEL 2: SMAC3                          │
│              (Meta-optimización Bayesiana)                  │
│                                                             │
│  Optimiza: {population_size, n_generations, sbx_prob,       │
│             sbx_eta, pm_prob_var, pm_eta}                   │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │           Cada trial de SMAC ejecuta:             │    │
│  │                                                     │    │
│  │       ┌─────────────────────────────────┐         │    │
│  │       │   NIVEL 1: Algoritmo Genético    │         │    │
│  │       │    (Optimización de Q-table)     │         │    │
│  │       │                                   │         │    │
│  │       │  Población de Q-tables           │         │    │
│  │       │         ↓                         │         │    │
│  │       │  Evaluación en NetSecGame        │         │    │
│  │       │         ↓                         │         │    │
│  │       │  Selección → Cruce → Mutación    │         │    │
│  │       │         ↓                         │         │    │
│  │       │  Mejor Q-table → win_rate        │         │    │
│  │       └─────────────────────────────────┘         │    │
│  │                                                     │    │
│  │       Retorna: win_rate como fitness               │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  SMAC aprende qué hiperparámetros producen mejores         │
│  Q-tables y sugiere nuevas configuraciones                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Componentes Principales

### 1. Clase `QTableOptimizationProblem`

**Propósito**: Define el problema de optimización para pymoo (librería de algoritmos genéticos).

**Responsabilidades**:
- Cargar estados y acciones desde archivos JSON
- Definir el espacio de búsqueda (límites de Q-values)
- Evaluar individuos (Q-tables) ejecutándolas en el agente
- Extraer métricas de rendimiento (win_rate)

#### Funciones principales:

##### `__init__(agent_script_path, host, port, test_episodes, reward_range, actions_file, states_file, n_workers=1, required_players=1, task_config_path=None)`
Inicializa el problema de optimización.

**Parámetros**:
- `agent_script_path`: Ruta al script del agente Q-learning
- `host`, `port`: Host y puerto base del servidor del juego
- `test_episodes`: Número de episodios para evaluar cada Q-table
- `reward_range`: Rango de valores permitidos (e.g., `(-1, 1)`)
- `actions_file`: JSON con las acciones disponibles (`registration_info.json`)
- `states_file`: JSON con los estados a optimizar (`estadosSMALL.json`)
- `n_workers` *(int, default 1)*: Número de workers paralelos para evaluar individuos
- `required_players` *(int, default 1)*: Jugadores requeridos por servidor.
- `task_config_path` *(str|None)*: Ruta al `netsecenv_conf.yaml`. Si se provee, activa el **modo multi-servidor**

**Proceso**:
1. Carga acciones desde `registration_info.json`
2. Carga estados desde `estadosSMALL.json`
3. Calcula tamaño de Q-table: `n_states × n_actions`
4. Define límites del espacio de búsqueda (`xl`, `xu`)
5. Si `task_config_path` es provisto: inicializa `_port_pool` (Queue) y lista de procesos de servidor

##### `_load_actions()`
Carga y convierte las acciones del juego a objetos `Action`.

**Proceso**:
1. Lee `registration_info.json` con doble parsing JSON
2. Convierte strings de tipo de acción a enums `ActionType`
3. Convierte parámetros a estructuras hashables (tuplas)
4. Crea objetos `Action` para uso en Q-table

##### `_load_states()`
Carga los estados específicos a optimizar.

**Proceso**:
1. Lee archivo JSON con lista de estados
2. Convierte strings de tuplas a tuplas Python reales
3. Cada estado es una tupla: `(n_redes, n_hosts_conocidos, n_hosts_controlados, n_servicios, n_datos)`

##### `_create_q_table_from_params(params)`
Convierte el vector de parámetros del GA en una Q-table utilizable.

**Proceso**:
1. Recibe un array 1D de tamaño `n_states × n_actions`
2. Hace reshape a matriz 2D: `[n_states, n_actions]`
3. Crea diccionario: `{(estado, acción): q_value}`
4. Retorna estructura compatible con el agente

**Ejemplo**:
```python
# Si n_states=10 y n_actions=20, params tiene 200 valores
params = [0.5, -0.3, 0.8, ...]  # 200 valores
↓
q_table_matrix = [[0.5, -0.3, 0.8, ...],  # Estado 0
                  [0.2, 0.1, -0.5, ...],  # Estado 1
                  ...]                     # ...
↓
q_table = {
    ((4,8,4,7,2), Action(SCAN_NETWORK)): 0.5,
    ((4,8,4,7,2), Action(LOCAL_EXPLOIT)): -0.3,
    ...
}
```

##### `_start_game_servers()` / `_stop_game_servers()` / `_wait_for_server()`
Métodos de gestión de servidores para el **modo multi-servidor**.

`_start_game_servers()`: Lanza `n_workers` instancias de `NSEGameCoordinator` como subprocesos independientes, asignándoles puertos consecutivos (`port`, `port+1`, ..., `port+n_workers-1`). Puebla `self._port_pool` (una `Queue`) con esos puertos para distribuirlos a los workers.

`_stop_game_servers()`: Termina todos los subprocesos de servidor lanzados (`self._server_procs`) y vacía el `_port_pool`.

`_wait_for_server(host, port, timeout=60)`: Intenta conectarse al socket TCP del servidor hasta que responde o expira el timeout. Se llama tras lanzar cada servidor para asegurar que esté listo antes de proceder.

`start_servers()` / `stop_servers()`: Wrappers públicos llamados por `QTableGeneticOptimizer.optimize()` al inicio y en el bloque `finally` para gestionar el ciclo de vida.

---

##### `_evaluate(X, out, *args, **kwargs)`
Función de evaluación del GA. Evalúa una población de Q-tables. Implementa **tres modos** según la configuración:

**Modo 1 — Multi-servidor paralelo** (`task_config_path` provisto):
- Usa `ThreadPoolExecutor(max_workers=n_workers)`
- Cada thread llama a `_evaluate_single(idx, x, total, port=pool.get())` y devuelve el puerto al pool tras finalizar
- Permite paralelismo real sin batching: cada individuo va inmediatamente al primer worker disponible

**Modo 2 — Secuencial** (`n_workers=1`, sin `task_config_path`):
- Bucle simple sobre la población
- Llama a `_evaluate_single()` uno a uno

**Común a todos los modos**:
- `_evaluate_single` crea un archivo pickle temporal, llama `_run_agent_evaluation()`, y borra el temporal en `finally`
- Fitness: `-win_rate` (negativo porque GA minimiza, pymoo espera valores a minimizar)

##### `_evaluate_single(idx, x, total, port=None)`
Evalúa un único individuo (Q-table).

**Proceso**:
1. Convierte el vector `x` en Q-table con `_create_q_table_from_params()`
2. Guarda en archivo temporal `.pickle` (prefijo `ga_qtable_`, sufijo `.pkl`)
3. Llama a `_run_agent_evaluation(q_table_path, port=port)`
4. Elimina el archivo temporal en `finally` (incluso si hay excepción)
5. Retorna fitness (escalar negativo)

##### `_run_agent_evaluation(q_table_path, port=None)`
Ejecuta el agente con una Q-table y extrae el fitness.

**Proceso**:
1. Construye comando: `python q_agent_feature_based.py --testing True --previous_model <q_table>`
2. Si `port` es provisto, añade `--port <port>` al comando
3. Ejecuta con `subprocess.run()` con `timeout=400` segundos
4. Captura stdout y stderr
5. Busca línea con `winrate=XX%` en la salida y extrae valor numérico usando `_extract_metric()`
6. Retorna **fitness = `-win_rate`** (el GA minimiza, por lo que el mejor individuo tiene fitness más negativo)

**Ejemplo de salida parseada**:
```
Testing completed: winrate=75.5%
↓
win_rate = 75.5
fitness = -75.5
```

> **Nota**: El fitness es únicamente `-win_rate`. No incluye `avg_return` ni ningún otro término.

---

### 2. Clase `QTableGeneticOptimizer`

**Propósito**: Orquesta la optimización usando algoritmos genéticos con pymoo.

**Responsabilidades**:
- Configurar y ejecutar el GA
- Gestionar el problema de optimización
- Guardar/cargar Q-tables optimizadas
- Generar visualizaciones del progreso

#### Funciones principales:

##### `__init__(agent_script_path, host, port, test_episodes, reward_range, actions_file, states_file, n_workers=1, required_players=1, task_config_path=None)`
Inicializa el optimizador.

**Proceso**:
1. Guarda configuración
2. Crea instancia de `QTableOptimizationProblem` (con todos los parámetros, incluyendo `n_workers`, `required_players` y `task_config_path`)
3. Inicializa `self.best_q_table = None` y `self.optimization_history = []`

##### `optimize(population_size, n_generations, sbx_prob, sbx_eta, pm_prob_var, pm_eta, verbose=True)`
Ejecuta la optimización con GA.

> **Nota**: Esta función **no** acepta parámetros de timeout. El GA se ejecuta durante exactamente `n_generations` generaciones.

**Parámetros del GA**:
- `population_size`: Número de Q-tables en cada generación (5-20)
- `n_generations`: Número de generaciones evolutivas (5-20)
- `sbx_prob`: Probabilidad de aplicar cruzamiento SBX (0.8-1.0)
- `sbx_eta`: Parámetro de distribución del cruzamiento (0-60)
- `pm_prob_var`: Probabilidad de mutar cada variable (0.01-0.6)
- `pm_eta`: Parámetro de distribución de la mutación (0-60)

**Parámetros fijos**:
- `sbx_prob_var = 1.0`: Probabilidad por variable en SBX
- `pm_prob = 1.0`: Probabilidad de aplicar mutación

**Proceso**:
1. **Configurar algoritmo**:
   ```python
   algorithm = GA(
       pop_size=population_size,
       sampling=FloatRandomSampling(),  # Inicialización aleatoria
       crossover=SBX(...),              # Simulated Binary Crossover
       mutation=PM(...),                # Polynomial Mutation
       eliminate_duplicates=True
   )
   ```

2. **Operadores genéticos**:
   - **SBX (Simulated Binary Crossover)**: Combina dos Q-tables padres creando hijos
   - **PM (Polynomial Mutation)**: Modifica valores individuales de forma controlada
   - **FloatRandomSampling**: Genera población inicial con valores aleatorios en el rango

3. **Configurar terminación**:
   ```python
   termination = get_termination("n_gen", n_generations)  # Simple: N generaciones exactas
   ```

4. **Gestionar servidores y ejecutar optimización**:
   ```python
   self.problem.start_servers()  # Arranca N instancias NetSecGame si task_config_path
   try:
       res = minimize(
           self.problem,
           algorithm,
           termination,
           verbose=verbose,
           save_history=True
       )
   finally:
       self.problem.stop_servers()  # Siempre detiene los servidores
   ```

5. **Procesar resultados**:
   - Extraer mejor individuo: `res.X`
   - Convertir a Q-table: `_create_q_table_from_params(res.X)`
   - Guardar historial: `[entry.opt.get("F")[0] for entry in res.history]`
   - Calcular win_rate estimado: `-res.F[0]`

**Flujo de una generación**:
```
Generación N:
  1. Evaluar población actual (ejecutar agentes con cada Q-table)
  2. Seleccionar mejores individuos
  3. Aplicar cruzamiento (SBX) → crear hijos
  4. Aplicar mutación (PM) → introducir variación
  5. Crear nueva población para Generación N+1
```

##### `save_q_table(filename)`
Guarda la Q-table optimizada en formato pickle.

**Formato guardado**:
```python
{
    'q_table': {(estado, acción): q_value, ...},
    'state_mapping': {}  # Vacío, el agente lo construye dinámicamente
}
```

##### `plot_optimization_progress(save_path)`
Genera gráficas del progreso del GA.

**Visualizaciones**:
1. **Fitness por generación**: Muestra cómo mejora el mejor fitness
2. **Win rate por generación**: Convierte fitness a win_rate estimado

##### `analyze_q_table()`
Analiza estadísticas de la Q-table optimizada.

**Métricas**:
- Valor mínimo, máximo, media, desviación estándar, mediana
- Distribución por percentiles
- Mejora del fitness durante optimización

---

### 3. Clase `SMACGAObjective`

**Propósito**: Encapsula la función objetivo para SMAC3 (meta-optimización).

**Responsabilidades**:
- Definir espacio de configuración de hiperparámetros
- Ejecutar GA con hiperparámetros sugeridos por SMAC
- Calcular costo (para minimización bayesiana)
- Registrar historial de trials

#### Funciones principales:

##### `__init__(base_config, checkpoint_file=None)`
Inicializa el objetivo de SMAC.

**Parámetros**:
- `base_config`: Configuración fija (rutas, host, puerto, archivos). Puede incluir `n_workers`, `required_players`, `task_config_path`.
- `checkpoint_file` *(str|None)*: Ruta al archivo JSON de checkpoint. Si el archivo existe, se restaura automáticamente el historial de trials anteriores.

**Variables de instancia**:
- `trial_count`: Contador de trials ejecutados
- `best_win_rate`: Mejor win_rate encontrado
- `best_trial`: Número del mejor trial
- `results_history`: Lista de todos los resultados
- `n_workers`, `required_players`, `task_config_path`: heredados de `base_config`

**Restauración desde checkpoint**:
Si `checkpoint_file` existe al inicializar, se carga el JSON y se restauran `results_history`, `best_win_rate`, `best_trial` y `trial_count`. Esto permite reanudar SMAC sin perder el historial de trials completados.

```python
if checkpoint_file and os.path.isfile(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        ckpt = json.load(f)
    self.results_history = ckpt.get('results_history', [])
    self.best_win_rate   = ckpt.get('best_win_rate', 0.0)
    # ... (trial_count, best_trial)
    print(f"[Checkpoint] Estado restaurado: {len(self.results_history)} trials previos")
```

##### `configspace` (property)
**Define el espacio de configuración para SMAC3.**

Esta es una de las funciones más importantes del sistema, ya que determina qué hiperparámetros SMAC3 puede optimizar y en qué rangos.

**Implementación**:
```python
@property
def configspace(self) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=42)
    
    # Hiperparámetros a optimizar
    population_size = Integer("population_size", (5, 20))
    n_generations = Integer("n_generations", (5, 20))
    sbx_prob = Float("sbx_prob", (0.8, 1.0))
    sbx_eta = Float("sbx_eta", (0.0, 60.0))
    pm_prob_var = Float("pm_prob_var", (0.01, 0.6))
    pm_eta = Float("pm_eta", (0.0, 60.0))
    
    cs.add([population_size, n_generations, sbx_prob, 
            sbx_eta, pm_prob_var, pm_eta])
    
    return cs
```

**Explicación de cada hiperparámetro**:

1. **`population_size` (5-20)**:
   - **Qué es**: Número de Q-tables en cada generación del GA
   - **Efecto**: 
     - Valores bajos (5-10): Convergencia rápida pero puede quedar en óptimos locales
     - Valores altos (15-20): Mayor diversidad, exploración más amplia
   - **Trade-off**: Más población = más evaluaciones por generación = más tiempo

2. **`n_generations` (5-20)**:
   - **Qué es**: Número de iteraciones evolutivas del GA
   - **Efecto**:
     - Pocas generaciones: Rápido pero puede no converger
     - Muchas generaciones: Mejor convergencia pero más costoso
   - **Trade-off**: Directo entre calidad y tiempo de ejecución

3. **`sbx_prob` (0.8-1.0)**:
   - **Qué es**: Probabilidad de aplicar cruzamiento SBX entre dos padres
   - **Efecto**:
     - Alto (~1.0): Más explotación de buenas soluciones (offspring similar a padres)
     - Bajo (~0.8): Más exploración (menos cruzamiento, más aleatorio)
   - **Nota**: Rango estrecho porque cruzamiento suele ser beneficioso

4. **`sbx_eta` (0.0-60.0)**:
   - **Qué es**: Índice de distribución del cruzamiento SBX
   - **Efecto**:
     - Bajo (0-10): Offspring muy diferentes a los padres (exploración)
     - Alto (30-60): Offspring muy similares a los padres (explotación)
   - **Uso típico**: 15-30 para balance exploración/explotación

5. **`pm_prob_var` (0.01-0.6)**:
   - **Qué es**: Probabilidad de mutar cada variable (cada Q-value) individualmente
   - **Efecto**:
     - Bajo (~0.01-0.1): Poca mutación, cambios conservadores
     - Alto (~0.3-0.6): Mucha mutación, mayor variación
   - **Trade-off**: Mutación ayuda a escapar de óptimos locales pero puede destruir buenas soluciones

6. **`pm_eta` (0.0-60.0)**:
   - **Qué es**: Índice de distribución de la mutación polinomial
   - **Efecto**:
     - Bajo (0-10): Mutaciones grandes (cambios drásticos)
     - Alto (30-60): Mutaciones pequeñas (ajustes finos)
   - **Uso típico**: 20 para mutaciones moderadas

**Valores fijos (no optimizados por SMAC)**:
- `sbx_prob_var = 1.0`: Probabilidad por variable en SBX (siempre 1.0)
- `pm_prob = 1.0`: Probabilidad de aplicar mutación (siempre 1.0)
- `test_episodes = 25`: Episodios para evaluar cada Q-table (fijo por costo computacional)

**Ejemplo de configuración sugerida por SMAC**:
```python
# Trial 5 de SMAC podría sugerir:
config = {
    'population_size': 12,
    'n_generations': 8,
    'sbx_prob': 0.95,
    'sbx_eta': 22.3,
    'pm_prob_var': 0.15,
    'pm_eta': 18.7
}
```

##### `train(config, seed)`
Función objetivo que SMAC3 llama para evaluar una configuración.

**Proceso completo**:

1. **Extraer hiperparámetros de la configuración**:
   ```python
   population_size = config["population_size"]
   n_generations = config["n_generations"]
   sbx_prob = config["sbx_prob"]
   sbx_eta = config["sbx_eta"]
   pm_prob_var = config["pm_prob_var"]
   pm_eta = config["pm_eta"]
   ```

2. **Crear optimizador con esos hiperparámetros**:
   ```python
   optimizer = QTableGeneticOptimizer(
       agent_script_path=...,
       test_episodes=25,  # Fijo
       ...
   )
   ```

3. **Ejecutar GA completo**:
   ```python
   optimizer.optimize(
       population_size=population_size,
       n_generations=n_generations,
       sbx_prob=sbx_prob,
       sbx_eta=sbx_eta,
       pm_prob_var=pm_prob_var,
       pm_eta=pm_eta
   )
   ```
   Esto ejecuta todo el GA: múltiples generaciones, evaluando cada Q-table en el juego.

4. **Extraer mejor fitness**:
   ```python
   best_fitness = optimizer.optimization_history[-1]
   best_win_rate = -best_fitness  # Convertir a porcentaje positivo
   ```

5. **Calcular costo para SMAC** (minimización):
   ```python
   cost = (100.0 - best_win_rate) / 100.0
   ```
   - win_rate = 100% → cost = 0.0 (óptimo)
   - win_rate = 0% → cost = 1.0 (peor)

6. **Registrar resultado**:
   ```python
   self.results_history.append({
       'trial': trial_num,
       'win_rate': best_win_rate,
       'cost': cost,
       'config': dict(config)
   })
   ```

7. **Guardar Q-table si es la mejor**:
   ```python
   if best_win_rate > self.best_win_rate:
       self.best_win_rate = best_win_rate
       optimizer.save_q_table(f"smac_trial_{trial_num}_wr_{best_win_rate:.1f}.pickle")
   ```

8. **Persistir checkpoint y retornar costo**:
   ```python
   self._save_checkpoint()  # Escritura atómica al disco
   return cost
   ```
   En caso de excepción: registra `cost=1.0` (penalización máxima), guarda checkpoint y retorna `1.0`.

##### `_save_checkpoint()`
Persiste el estado actual al archivo `checkpoint_file` (si está configurado).

**Escritura atómica**: Escribe primero en un archivo temporal (`.tmp`) y luego usa `os.replace()` para renombrar. Esto garantiza que el checkpoint nunca quede corrupto si el proceso se interrumpe a mitad de la escritura.

```python
tmp = self.checkpoint_file + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(ckpt, f, indent=2, default=str)
os.replace(tmp, self.checkpoint_file)  # Operación atómica
```

**Contenido del JSON de checkpoint**:
```json
{
  "trial_count": 15,
  "best_win_rate": 72.4,
  "best_trial": 11,
  "saved_at": "2026-02-23 14:35:00",
  "results_history": [
    {"trial": 1, "win_rate": 55.2, "cost": 0.448, "config": {...}},
    ...
  ]
}
```

**Flujo de SMAC**:
```
SMAC Trial 1:
  → Sugiere config = {pop_size: 10, n_gen: 5, ...}
  → Ejecuta GA completo con esos parámetros
  → GA ejecuta 5 generaciones × 10 individuos = 50 evaluaciones de Q-tables
  → Retorna: win_rate = 60% → cost = 0.40

SMAC aprende: Esta configuración dio cost=0.40

SMAC Trial 2:
  → Random Forest predice mejores configuraciones
  → Sugiere config = {pop_size: 15, n_gen: 8, ...}
  → Ejecuta GA...
  → Retorna: win_rate = 72% → cost = 0.28

SMAC aprende: Segunda config fue mejor (cost menor)

... y así sucesivamente, SMAC aprende qué hiperparámetros funcionan mejor
```

---

### 4. Función `optimize_with_smac(base_config, n_trials, output_dir, n_workers)`

**Propósito**: Ejecuta la optimización bayesiana de hiperparámetros con SMAC3. Detecta automáticamente si existe una ejecución previa para **reanudar** en lugar de empezar desde cero.

**Proceso detallado**:

#### Fase 1: Configuración y Detección de Checkpoint

1. **Setup de logging**:
   ```python
   log_file, logger = setup_smac_logging()  # Crea directorio y archivo de log
   ```

2. **Detectar si existe ejecución previa**:
   ```python
   checkpoint_file = os.path.join(output_dir, 'objective_checkpoint.json')
   smac_run_dir = Path(output_dir) / 'ga_qtable_optimization'
   resuming = smac_run_dir.exists() and os.path.isfile(checkpoint_file)
   ```
   Si `resuming=True`, tanto el historial del objetivo como el `RunHistory` de SMAC serán restaurados.

3. **Crear instancia del objetivo** (restaura historial si hay checkpoint):
   ```python
   objective = SMACGAObjective(base_config, checkpoint_file=checkpoint_file)
   ```

4. **Crear Scenario de SMAC**:
   ```python
   scenario = Scenario(
       configspace=objective.configspace,
       deterministic=False,  # El juego es estocástico
       n_trials=n_trials,
       seed=42,
       output_directory=Path(output_dir),
       name="ga_qtable_optimization"
   )
   ```

#### Fase 2: Diseño Inicial (Exploración Aleatoria)

**¿Por qué es necesario?**
El Random Forest de SMAC necesita datos iniciales para entrenar. Sin estos datos, no puede hacer predicciones.

**Implementación**:
```python
n_initial_configs = max(2, int(n_trials * 0.2))  # 20% de trials
initial_design = HyperparameterOptimizationFacade.get_initial_design(
    scenario, n_configs=n_initial_configs
)
```

**Ejemplo**:
- Si `n_trials=10` → `n_initial_configs=2`
- Si `n_trials=20` → `n_initial_configs=4`

Estos trials iniciales usan **muestreo aleatorio uniforme** dentro de los rangos definidos en `configspace`.

#### Fase 3: Optimización Bayesiana

**Crear facade SMAC**:
```python
# overwrite=False cuando se reanuda: SMAC retoma su propio RunHistory
# overwrite=True  cuando es nueva: comienza desde cero
smac = HyperparameterOptimizationFacade(
    scenario=scenario,
    target_function=objective.train,
    initial_design=initial_design,
    overwrite=not resuming,  # False = reanudar; True = inicio fresco
)
```

> **Comportamiento de reanudación**: Con `overwrite=False`, SMAC carga su propio `RunHistory` previo (almacenado en `output_dir/ga_qtable_optimization/`). El objetivo ya tiene su historial restaurado desde `checkpoint_file`. Juntos permiten continuar exactamente donde se detuvo.

**Ejecutar optimización**:
```python
incumbent = smac.optimize()
```

**¿Qué hace SMAC internamente?**

1. **Trials 1-N (diseño inicial)**:
   - Genera configuraciones aleatorias
   - Ejecuta `objective.train(config)` para cada una
   - Acumula datos: (config → cost)

2. **Trial N+1 en adelante**:
   - **Entrenar Random Forest**: Usa datos acumulados
   - **Predecir**: ¿Qué configuraciones no evaluadas podrían ser buenas?
   - **Acquisition Function (Expected Improvement)**:
     - Balancea exploración vs explotación
     - Selecciona próxima configuración a evaluar
   - **Evaluar**: Ejecuta `objective.train(config_nueva)`
   - **Actualizar modelo**: Añade nuevo dato y reentrena Random Forest
   - **Repetir**

**Modelo Surrogate (Random Forest)**:
```
Entrada: [population_size, n_generations, sbx_prob, sbx_eta, pm_prob_var, pm_eta]
↓
Random Forest (entrenado con trials previos)
↓
Salida: Predicción de cost (y incertidumbre)
```

**Expected Improvement (EI)**:
```
EI(config) = E[max(0, incumbent_cost - predicted_cost)]
```
- Alto EI → Configuración prometedora → Evaluar
- Favorece configuraciones con:
  - Bajo `predicted_cost` (explotación)
  - Alta incertidumbre (exploración)

#### Fase 4: Validación y Resultados

**Validar configuración incumbente**:
```python
incumbent_cost = smac.validate(incumbent)
incumbent_win_rate = 100.0 * (1.0 - incumbent_cost)
```

**Guardar resultados**:
```python
results_file = f"smac_results_{timestamp}.json"
results_data = {
    'best_win_rate_validated': incumbent_win_rate,
    'best_win_rate_internal': objective.best_win_rate,
    'incumbent_params': dict(incumbent),
    'n_trials': n_trials,
    'history': objective.results_history
}
json.dump(results_data, file)
```

**Retornar**:
```python
return incumbent, objective, smac
```

---

### 5. Función `plot_smac_results(objective, save_dir)`

**Propósito**: Genera visualizaciones exhaustivas de los resultados de SMAC y un README Markdown.

**Visualizaciones generadas** (8 tipos, guardadas en `save_dir/`):

1. **`smac_win_rate_per_trial.png`**: Línea de win rate por trial (con marcadores)
2. **`smac_cost_per_trial.png`**: Línea de costo por trial (lo que SMAC minimiza)
3. **`smac_win_rate_distribution.png`**: Histograma de win rates con línea vertical del mejor
4. **`smac_hyperparameters_vs_winrate.png`**: Scatter 2D `pm_prob_var` vs `sbx_prob` coloreado por win rate
5. **`smac_PARAM_vs_winrate.png` (6 archivos)**: Scatter individual de cada hiperparámetro vs win rate con línea de tendencia
6. **`smac_correlation_matrix.png`**: Heatmap de correlación (7×7: 6 hiperparámetros + win rate)
7. **`smac_hyperparams_evolution.png`**: 6 subplots — todos los trials en gris, top-5 resaltados por win rate
8. **`smac_boxplot_PARAM.png` (6 archivos)**: Box plots por rangos percentílicos de cada hiperparámetro

**Bins adaptativos en box plots**:
```python
if n_trials < 20:   n_bins = 4
elif n_trials < 50: n_bins = 5
elif n_trials < 100: n_bins = 6
else:               n_bins = 8
```
Los bins usan percentiles equiespaciados (≈ igual número de trials por bin). Con `n_trials ≥ 40` se habilitan notches; con `n_trials ≤ 200` se superpone jitter de puntos individuales.

**README Markdown** (`save_dir/README.md`):
- Generado automáticamente al final de `plot_smac_results()`
- Incluye: fecha, número de trials, mejor win rate, tabla de espacio de hiperparámetros, tabla completa de resultados ordenados por win rate, e imagen embed de cada `.png` generado

**Tabla resumen en consola** (ordenada por win rate):
```
Trial | Win Rate | Costo   | Pop | Gens | SBX_P  | SBX_eta | PM_pv  | PM_eta
------|----------|---------|-----|------|--------|---------|--------|--------
   15 |   78.50% |  0.2150 |  17 |   12 | 0.9200 |   25.30 | 0.1800 |  22.50
    8 |   75.20% |  0.2480 |  14 |    9 | 0.8900 |   18.70 | 0.2100 |  19.30
  ...
```

---

## Modos de Evaluación Paralela

`QTableOptimizationProblem._evaluate()` soporta tres modos de evaluación que se seleccionan automáticamente:

### Modo 1: Multi-servidor paralelo (`--task_config` provisto)

Este es el modo de máximo rendimiento. Se activa cuando se pasa `--task_config` en la línea de comandos.

```
Worker 1 ──[ puerto 9000 ]──> NetSecGame instancia 1
Worker 2 ──[ puerto 9001 ]──> NetSecGame instancia 2
  ...                               ...
Worker N ──[ puerto 900N-1 ]─> NetSecGame instancia N
```

- `start_servers()` lanza N instancias de `NSEGameCoordinator` y espera que cada una acepte conexiones (`_wait_for_server()`).
- Un `Queue` (`_port_pool`) distribuye puertos a los workers; el worker devuelve el puerto al terminar.
- `ThreadPoolExecutor(max_workers=n_workers)` evalúa todos los individuos de la población en paralelo.
- No hay batching: cada individuo va al primer worker libre, maximizando utilización.
- `stop_servers()` en `finally` garantiza que los servidores se detienen siempre.

**Requisito**: el YAML de configuración debe permitir crear N instancias con puertos distintos.

### Modo 2: Secuencial (`n_workers=1`, sin `--task_config`)

Modo por defecto. Un solo servidor (externo o ya en marcha) en `host:port`.

- Bucle simple: evalúa individuo por individuo.
- Útil para debugging, entornos con un solo servidor disponible o cuando se quiere traza limpia.


---

## Sistema de Checkpoint y Reanudación

SMAC puede tardar días en completar todos sus trials. El sistema de checkpoint permite interrumpir y reanudar sin perder progreso.

### Archivos involucrados

| Archivo | Quién lo escribe | Propósito |
|---|---|---|
| `output_dir/objective_checkpoint.json` | `SMACGAObjective._save_checkpoint()` | Historial de trials del objetivo |
| `output_dir/ga_qtable_optimization/` | SMAC3 internamente | RunHistory, modelos Random Forest |

### Flujo de reanudación

```
Primera ejecución:
  optimize_with_smac() → resuming=False → overwrite=True
  Trial 1 completa → _save_checkpoint() escribe JSON
  Trial 2 completa → _save_checkpoint() actualiza JSON
  ...proceso interrumpido en Trial 8...

Segunda ejecución (mismo comando):
  optimize_with_smac() → resuming=True (ambos archivos existen)
  SMACGAObjective.__init__() restaura 7 trials del JSON
  SMAC facade carga RunHistory previo (overwrite=False)
  Continúa desde Trial 8
```

### Escritura atómica del checkpoint

Para evitar que el JSON quede corrupto si el proceso se interrumpe durante la escritura:

```python
tmp = checkpoint_file + '.tmp'
with open(tmp, 'w') as f:
    json.dump(ckpt, f, indent=2, default=str)
os.replace(tmp, checkpoint_file)  # Rename atómico (OS garantiza atomicidad)
```

### Activar checkpoint

El checkpoint se activa automáticamente al usar `--smac`. No requiere configuración adicional: el archivo se guarda en `<smac_output_dir>/objective_checkpoint.json`.

---

## Definición del Espacio de Configuración

El espacio de configuración es el conjunto de hiperparámetros que SMAC3 puede optimizar y sus rangos válidos.

### Implementación

Se define usando la librería `ConfigSpace`:

```python
from ConfigSpace import ConfigurationSpace, Float, Integer

cs = ConfigurationSpace(seed=42)

# Hiperparámetros enteros
population_size = Integer("population_size", (5, 20))
n_generations = Integer("n_generations", (5, 20))

# Hiperparámetros continuos
sbx_prob = Float("sbx_prob", (0.8, 1.0))
sbx_eta = Float("sbx_eta", (0.0, 60.0))
pm_prob_var = Float("pm_prob_var", (0.01, 0.6))
pm_eta = Float("pm_eta", (0.0, 60.0))

# Añadir al espacio
cs.add([population_size, n_generations, sbx_prob, 
        sbx_eta, pm_prob_var, pm_eta])
```

### Tipos de Hiperparámetros

1. **Integer**: Valores enteros discretos
   - Ejemplo: `population_size ∈ {5, 6, 7, ..., 20}`
   - SMAC muestrea enteros dentro del rango

2. **Float**: Valores continuos
   - Ejemplo: `sbx_prob ∈ [0.8, 1.0]`
   - SMAC puede sugerir cualquier valor decimal en el rango

### Cómo SMAC Genera Valores

1. **Fase inicial (diseño inicial)**:
   ```python
   # Muestreo aleatorio uniforme
   population_size = random.randint(5, 20)
   sbx_prob = random.uniform(0.8, 1.0)
   ```

2. **Fase bayesiana**:
   ```python
   # Random Forest predice costo esperado
   for config_candidata in espacio_de_busqueda:
       predicted_cost, uncertainty = random_forest.predict(config_candidata)
       expected_improvement = calculate_EI(predicted_cost, uncertainty)
   
   # Seleccionar config con mayor EI
   next_config = argmax(expected_improvement)
   ```

### Ejemplo de Generación de Configuraciones

**Trial 1 (aleatorio)**:
```python
config = {
    'population_size': 8,
    'n_generations': 12,
    'sbx_prob': 0.87,
    'sbx_eta': 35.2,
    'pm_prob_var': 0.23,
    'pm_eta': 45.8
}
```

**Trial 5 (bayesiano)**:
SMAC ha aprendido que `population_size` alto y `pm_prob_var` bajo funciona bien:
```python
config = {
    'population_size': 17,      # ← Alto (aprendido)
    'n_generations': 9,
    'sbx_prob': 0.94,
    'sbx_eta': 22.1,
    'pm_prob_var': 0.08,        # ← Bajo (aprendido)
    'pm_eta': 28.3
}
```

### Restricciones y Condiciones

En este sistema, los hiperparámetros son **independientes** (sin restricciones condicionales).

**Ejemplo de restricción que NO tenemos**:
```python
# Esto NO está implementado en nuestro sistema:
if population_size < 10:
    n_generations must be > 15
```

Todos los valores dentro de los rangos son válidos independientemente de otros hiperparámetros.

---

## Proceso de Ejecución

### Flujo Completo del Sistema

#### Modo Estándar (GA con hiperparámetros fijos)

```
1. Usuario ejecuta:
   python ga_smac.py --agent_script q_agent.py --population 10 --generations 5

2. Crear QTableGeneticOptimizer
   ├─ Cargar acciones y estados
   ├─ Crear QTableOptimizationProblem
   └─ Definir límites de Q-values

3. Ejecutar GA:
   Generación 0:
     ├─ Crear población inicial aleatoria (10 Q-tables)
     ├─ Evaluar cada Q-table:
     │   ├─ Guardar Q-table en .pickle
     │   ├─ Ejecutar: python q_agent.py --previous_model <pickle>
     │   ├─ Extraer win_rate de salida
     │   └─ Fitness = -win_rate
     └─ Mejor fitness Generación 0: -45.0 (win_rate=45%)
   
   Generación 1:
     ├─ Seleccionar mejores individuos
     ├─ Cruzamiento (SBX): Combinar Q-tables
     ├─ Mutación (PM): Modificar valores
     ├─ Evaluar nueva población...
     └─ Mejor fitness Generación 1: -52.3 (win_rate=52.3%)
   
   ... Generación 2, 3, 4 ...
   
   Generación 5 (final):
     └─ Mejor fitness: -68.5 (win_rate=68.5%)

4. Guardar mejor Q-table:
   → optimized_q_table_ga.pickle

5. Generar visualizaciones:
   → Gráfica de progreso por generación
```

#### Modo SMAC (Meta-optimización de hiperparámetros)

```
1. Usuario ejecuta:
   python ga_smac.py --agent_script q_agent.py --smac --smac_trials 10

2. Crear SMACGAObjective
   ├─ Definir configspace (6 hiperparámetros)
   └─ Configurar base_config

3. Configurar SMAC3:
   ├─ Crear Scenario
   ├─ Diseño inicial: 2 configuraciones aleatorias (20% de 10)
   └─ Crear HyperparameterOptimizationFacade

4. SMAC Trial 1 (diseño inicial - aleatorio):
   Config: {pop:8, gen:7, sbx_prob:0.89, sbx_eta:30.1, pm_pv:0.25, pm_eta:35.2}
   ├─ objective.train(config) ejecuta:
   │   ├─ Crear QTableGeneticOptimizer con estos hiperparámetros
   │   ├─ Ejecutar GA completo (7 generaciones × 8 individuos)
   │   │   └─ 56 evaluaciones de Q-tables en total
   │   ├─ Mejor win_rate del GA: 55.2%
   │   └─ Return cost = (100-55.2)/100 = 0.448
   └─ SMAC registra: config → cost=0.448

5. SMAC Trial 2 (diseño inicial - aleatorio):
   Config: {pop:15, gen:5, sbx_prob:0.93, sbx_eta:18.5, pm_pv:0.12, pm_eta:22.7}
   ├─ Ejecutar GA completo...
   ├─ Mejor win_rate: 62.8%
   └─ Return cost = 0.372

6. SMAC Trial 3 (bayesiano):
   ├─ Random Forest entrenado con Trials 1-2
   ├─ Predice: pop_size alto podría ser mejor (Trial 2 fue mejor)
   ├─ Sugiere: {pop:17, gen:8, sbx_prob:0.91, sbx_eta:25.3, pm_pv:0.15, pm_eta:28.1}
   ├─ Ejecutar GA completo...
   ├─ Mejor win_rate: 68.5%
   └─ Return cost = 0.315 ← ¡Mejor que antes!

7. SMAC Trials 4-10:
   ├─ Random Forest se refina con más datos
   ├─ Exploración/explotación balanceada por EI
   └─ Convergencia hacia hiperparámetros óptimos

8. Trial final (Trial 9):
   Config: {pop:16, gen:10, sbx_prob:0.95, sbx_eta:22.0, pm_pv:0.11, pm_eta:20.5}
   └─ Mejor win_rate: 75.3% ← MEJOR RESULTADO

9. Validar incumbente:
   ├─ Re-evaluar mejor configuración
   └─ Win rate validado: 74.8%

10. Guardar resultados:
    ├─ Q-table del mejor trial: smac_trial_9_wr_75.3.pickle
    ├─ Historial completo: smac_results_20260210_143022.json
    └─ Visualizaciones: smac_plots/*.png (15 gráficas + README.md)

11. Entrenamiento final (opcional):
    ├─ Usar mejores hiperparámetros encontrados
    ├─ Ejecutar GA con más generaciones para producción
    └─ Guardar: optimized_q_table_ga_smac_best.pickle
```

> **Nota**: Si el proceso se interrumpe en cualquier trial, relanzar el mismo comando reanuda desde el último checkpoint automáticamente.



---

## Modos de Operación

### Modo 1: GA con Hiperparámetros Fijos

**Uso** (secuencial, un servidor externo ya en marcha):
```bash
python ga_smac.py \
  --agent_script q_agent_feature_based.py \
  --actions registration_info.json \
  --states estadosSMALL.json \
  --host 127.0.0.1 --port 9000 \
  --population 15 \
  --generations 10 \
  --sbx_prob 0.9 --sbx_eta 20.0 \
  --pm_prob_var 0.15 --pm_eta 25.0 \
  --test_episodes 25 \
  --output my_qtable.pickle
```

**Uso con paralelismo** (8 workers, servidores gestionados automáticamente):
```bash
python ga_smac.py \
  --agent_script q_agent_feature_based.py \
  --actions registration_info.json \
  --states estadosSMALL.json \
  --port 9000 \
  --workers 8 \
  --task_config ./AIDojoCoordinator/netsecenv_conf.yaml \
  --population 20 --generations 15 \
  --output my_qtable_parallel.pickle
```

**Cuándo usar**:
- Ya conoces buenos hiperparámetros
- Quieres validar una configuración específica
- Tiempo limitado (1-2 horas con 8 workers)
- Debugging o pruebas rápidas

**Salidas**:
- `my_qtable.pickle`: Q-table optimizada
- Gráficas de progreso del GA

---

### Modo 2: SMAC (Meta-optimización)

**Uso básico**:
```bash
python ga_smac.py \
  --agent_script q_agent_feature_based.py \
  --smac \
  --smac_trials 100 \
  --workers 8 \
  --task_config ./AIDojoCoordinator/netsecenv_conf.yaml \
  --smac_output_dir smac_output
```

**Uso completo** (reanudar si se interrumpe — mismo comando):
```bash
python ga_smac.py \
  --agent_script /path/to/q_agent_feature_based.py \
  --actions registration_info.json \
  --states estadosSMALL.json \
  --host 127.0.0.1 --port 9000 \
  --smac \
  --smac_trials 100 \
  --workers 8 \
  --task_config ./AIDojoCoordinator/netsecenv_conf.yaml \
  --smac_output_dir results/smac_100trials \
  --output final_qtable.pickle
```

**Cuándo usar**:
- Primera vez optimizando Q-tables para un juego
- No sabes qué hiperparámetros funcionan mejor
- Tienes muchos recursos computacionales
- Quieres el mejor resultado posible

**Salidas**:
- `smac_output/`: Directorio con resultados de SMAC
- `smac_trial_N_wr_XX.pickle`: Q-tables de los mejores trials
- `smac_results_TIMESTAMP.json`: Historial completo
- `smac_plots/*.png`: 15+ visualizaciones
- `smac_logs/smac_TIMESTAMP.log`: Log detallado
- `final_qtable_smac_best.pickle`: Modelo final reentrenado

---

### Modo 3: SMAC sin Entrenamiento Final

**Uso**:
```bash
python ga_smac.py \
  --agent_script q_agent_feature_based.py \
  --smac \
  --smac_trials 10 \
  --skip_final_training
```

**Cuándo usar**:
- Solo quieres encontrar buenos hiperparámetros
- Planeas usar esos hiperparámetros en otro script
- Quieres ahorrar tiempo (evita el último GA completo)

**Salidas**:
- Hiperparámetros óptimos en `smac_results_*.json`
- Q-tables de trials individuales
- Visualizaciones de SMAC
- NO genera modelo final reentrenado



---

## Resultados y Visualizaciones

### Archivos de Salida

#### Del GA (Modo estándar)

1. **Q-table optimizada** (`*.pickle`):
   ```python
   # Formato interno
   {
       'q_table': {
           ((4,8,4,7,2), Action(SCAN)): 0.65,
           ((4,8,4,7,2), Action(EXPLOIT)): -0.23,
           ...
       },
       'state_mapping': {}
   }
   ```

2. **Gráfica de progreso** (`*.png`):
   - Subplot 1: Fitness por generación
   - Subplot 2: Win rate estimado por generación

#### De SMAC (Modo meta-optimización)

1. **Resultados JSON** (`smac_results_TIMESTAMP.json`):
   ```json
   {
       "best_win_rate_validated": 74.8,
       "best_win_rate_internal": 75.3,
       "best_trial_internal": 9,
       "incumbent_params": {
           "population_size": 16,
           "n_generations": 10,
           "sbx_prob": 0.95,
           "sbx_eta": 22.0,
           "pm_prob_var": 0.11,
           "pm_eta": 20.5
       },
       "n_trials": 10,
       "history": [
           {
               "trial": 1,
               "win_rate": 55.2,
               "cost": 0.448,
               "config": {...}
           },
           ...
       ]
   }
   ```

2. **Q-tables de trials** (`smac_trial_N_wr_XX.pickle`):
   - Guardadas solo para trials que mejoran el mejor resultado
   - Formato igual que Q-table de GA estándar

3. **Directorio SMAC** (`smac_output/`):
   - Archivos internos de SMAC3
   - RunHistory, configuraciones, etc.

4. **Logs** (`smac_logs/smac_TIMESTAMP.log`):
   - Log detallado de toda la ejecución
   - Útil para debugging

5. **Visualizaciones** (`smac_plots/`):
   - 15+ gráficas PNG de alta resolución (300 DPI)
   - `README.md` con tabla de resultados y todas las imágenes embebidas
   - Ver sección siguiente

---

### Visualizaciones Generadas por SMAC

Todos los archivos se guardan en `smac_plots/` con 300 DPI.

**1. `smac_win_rate_per_trial.png`**
- Línea + marcadores de win rate por trial
- **Interpretación**: Ver qué trials fueron exitosos y si hay tendencia de mejora

**2. `smac_cost_per_trial.png`**
- Línea del costo por trial (SMAC minimiza)
- **Interpretación**: ¿Está bajando el costo con el tiempo?

**3. `smac_win_rate_distribution.png`**
- Histograma de win rates obtenidos con línea del mejor resultado
- **Interpretación**: ¿Hay consistencia o mucha variabilidad?

**4. `smac_hyperparameters_vs_winrate.png`**
- Scatter 2D: `pm_prob_var` vs `sbx_prob`, coloreado por win rate
- **Interpretación**: Ver interacción entre estos dos hiperparámetros

**5–10. `smac_PARAM_vs_winrate.png`** (6 archivos, uno por hiperparámetro)
- Scatter individual con línea de tendencia lineal
- **Interpretación**: Correlación de cada hiperparámetro con win rate

**11. `smac_correlation_matrix.png`**
- Heatmap de correlación 7×7 (6 hiperparámetros + win rate)
- Valores de −1 a +1
- **Interpretación**: Fila `win_rate` → qué hiperparámetros tienen mayor impacto

**12. `smac_hyperparams_evolution.png`**
- 6 subplots: todos los trials en gris, top-5 resaltados por win rate
- **Interpretación**: ¿Converge SMAC hacia ciertos rangos?

**13–18. `smac_boxplot_PARAM.png`** (6 archivos, uno por hiperparámetro)
- Box plots por bins percentilílicos (4–8 bins adaptativos según `n_trials`)
- Diamantes azules = media; línea negra = mediana; jitter de puntos individuales
- Cajas coloreadas en gradiente verde (bins con mejor mediana = más verde)
- **Interpretación**: ¿Qué rangos de valores producen mejores resultados?

**`README.md`** (en `smac_plots/`):
- Tabla de espacio de hiperparámetros con rangos y tipos
- Tabla de todos los trials ordenados por win rate
- Todas las imágenes embebidas como inline Markdown


---

## Recomendaciones y Mejores Prácticas

### Parámetros Recomendados por Etapa

#### Exploración Inicial (rápida, 1 worker)
```bash
--population 5 --generations 3 --test_episodes 10
```
- Tiempo: ~10 min
- Objetivo: Validar que el pipeline completo funciona

#### Desarrollo (8 workers, modo multi-servidor)
```bash
--population 10 --generations 8 --test_episodes 25
--workers 8 --task_config ./AIDojoCoordinator/netsecenv_conf.yaml
```
- Tiempo: ~13 min
- Objetivo: Iterar y ajustar

#### Meta-optimización para Tesis (8 workers)
```bash
--smac --smac_trials 100
--workers 8 --task_config ./AIDojoCoordinator/netsecenv_conf.yaml
```
- Tiempo estimado: ~3-6 días
- Objetivo: Búsqueda científica reproducible de hiperparámetros
- Rango explotado por SMAC: `population_size (5,20)`, `n_generations (5,20)`
- **Reanudable**: relanzar el mismo comando si se interrumpe

### Configuración Recomendada para Tesis Académica

Para una tesis donde se dispone de ~12 días y 8 workers, la configuración que maximiza la calidad dentro del presupuesto temporal es:

```bash
python ga_smac.py \
  --agent_script q_agent_feature_based.py \
  --actions registration_info.json \
  --states estadosSMALL.json \
  --port 9000 \
  --smac \
  --smac_trials 100 \
  --workers 8 \
  --task_config ./AIDojoCoordinator/netsecenv_conf.yaml \
  --smac_output_dir results/smac_thesis \
  --output thesis_best.pickle
```

**Justificación**:
- `n_trials=100`: Suficiente para que el Random Forest de SMAC capture tendencias; justificable en literatura de HPO
- `n_workers=8`: Utiliza todos los cores disponibles; reduce tiempo total ~8×
- Espacio de búsqueda `(5,20)×(5,20)`: 256 combinaciones enteras; 100 trials cubre ~40% con exploitación bayesiana
- `test_episodes=25` (fijo en SMAC): balance entre evaluación confiable y velocidad

---

### Tips de Optimización

1. **Empezar pequeño**: Validar pipeline con `--population 5 --generations 3`
2. **Usar `--task_config` siempre**: El modo multi-servidor es el único que garantiza paralelismo real
3. **No cambiar `--smac_output_dir` entre ejecuciones**: El checkpoint y el RunHistory de SMAC dependen de la ruta
4. **Aumentar `test_episodes` gradualmente**: SMAC usa 25 (fijo); para el GA estándar puedes usar 50-100
5. **Analizar `smac_plots/README.md`**: Contiene la tabla ordenada por win rate y las imágenes embebidas

---

### Troubleshooting

**Problema**: GA no mejora entre generaciones.
- **Causa**: `test_episodes` muy bajo, evaluación ruidosa
- **Solución**: Aumentar a 25-30

**Problema**: SMAC sugiere configuraciones muy similares.
- **Causa**: Espacio de búsqueda muy estrecho o pocos trials iniciales
- **Solución**: Verificar `configspace`; si `n_trials < 10` aumentar al menos a 10

**Problema**: Timeout en evaluaciones (subprocess).
- **Causa**: Agente tarda más de 400 s por evaluación
- **Solución**: Reducir `test_episodes` o verificar si el servidor está respondiendo

**Problema**: Archivos temporales no se eliminan.
- **Causa**: Error crítico en evaluación antes del `finally`
- **Solución**: Verificar logs; buscar y borrar archivos `ga_qtable_*.pkl` en `%TEMP%`

**Problema**: `[Checkpoint] Advertencia: no se pudo cargar`.
- **Causa**: JSON corrupto por interrupción durante escritura
- **Solución**: Borrar `objective_checkpoint.json.tmp` y volver a lanzar; si el `.json` también está corrupto, eliminarlo (SMAC recomenzará desde cero)

**Problema**: `overwrite=False` pero SMAC dice que el directorio no existe.
- **Causa**: `smac_output_dir` cambió entre ejecuciones
- **Solución**: Usar exactamente el mismo `--smac_output_dir` que la ejecución original

**Problema**: Servidores no se detienen tras error.
- **Causa**: `stop_servers()` en `finally` podría no alcanzarse si Python se mata a nivel de OS
- **Solución**: Buscar y matar procesos con `pkill -f NSEGameCoordinator` (Linux) o Task Manager (Windows)

---

## Conclusión

`ga_smac.py` proporciona una solución completa y autónoma para optimizar Q-tables en Q-Learning mediante un esquema de dos niveles:

- **Nivel 1 (GA)**: Optimiza los valores iniciales de la Q-table usando pymoo con operadores SBX + PM
- **Nivel 2 (SMAC3)**: Meta-optimiza los 6 hiperparámetros del GA mediante Optimización Bayesiana con Random Forest surrogate

**Características de producción incluidas**:
- Paralelismo real vía `ThreadPoolExecutor` + N instancias de NetSecGame gestionadas automáticamente
- Checkpoint atómico con `os.replace()` para tolerar interrupciones
- Reanudación automática: relanzar el mismo comando continúa desde donde se detuvo
- 15+ visualizaciones generadas automáticamente + README Markdown con tabla de resultados

**Recursos adicionales**:
- Documentación PyMOO: https://pymoo.org/
- Documentación SMAC3: https://automl.github.io/SMAC3/
- Paper SBX/PM: Deb et al. (2002) - NSGA-II
- Paper SMAC: Hutter et al. (2011) - Sequential Model-based Algorithm Configuration
