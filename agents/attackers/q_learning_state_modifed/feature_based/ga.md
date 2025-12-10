# Especificación técnica: ga.py

## 1. Resumen

### Descripción general
`ga.py` es un sistema de optimización de Q-tables (tablas de valores Q para aprendizaje por refuerzo) mediante Algoritmos Genéticos (GA) y optimización bayesiana de hiperparámetros con Optuna. El archivo implementa un framework completo para inicializar y optimizar políticas de Q-Learning para agentes en el entorno NetSecGame.

### Objetivo principal
El problema que resuelve es la **inicialización inteligente de Q-tables** para agentes de Q-Learning en escenarios de ciberseguridad. En lugar de comenzar con valores aleatorios o ceros, el sistema:
- Utiliza algoritmos genéticos para evolucionar Q-tables candidatas
- Evalúa cada Q-table ejecutando el agente en el entorno real
- Encuentra combinaciones óptimas de valores Q que maximizan el win rate
- Opcionalmente, optimiza los hiperparámetros del GA usando Optuna

---

## 2. Estructura y componentes principales

### Clases principales

#### **QTableOptimizationProblem(Problem)**
Hereda de `pymoo.core.problem.Problem` y define el espacio de optimización para el algoritmo genético.

**Responsabilidades:**
- Cargar estados y acciones desde archivos JSON
- Definir el espacio de búsqueda (n_states × n_actions valores Q)
- Evaluar fitness de individuos ejecutando el agente con cada Q-table
- Gestionar la comunicación con el servidor del juego

**Métodos clave:**
- `__init__()`: Inicializa el problema, carga estados/acciones, define límites
- `_load_actions()`: Parsea `registration_info.json` y convierte a objetos Action
- `_load_states()`: Carga estados desde `estadosSMALL.json`
- `_create_q_table_from_params()`: Convierte vector de parámetros a diccionario Q-table
- `_evaluate()`: Evalúa población de Q-tables ejecutándolas en el agente
- `_run_agent_evaluation()`: Ejecuta subprocess con el agente y extrae métricas
- `_extract_metric()`: Parsea salida del agente para obtener win_rate

#### **QTableGeneticOptimizer**
Orquestador principal que encapsula el problema de optimización y gestiona el ciclo evolutivo.

**Responsabilidades:**
- Configurar y ejecutar el algoritmo genético (GA)
- Almacenar historial de optimización
- Guardar/cargar Q-tables optimizadas
- Generar visualizaciones del progreso

**Métodos clave:**
- `__init__()`: Crea instancia de QTableOptimizationProblem
- `optimize()`: Ejecuta el GA con pymoo (SBX crossover, PM mutation)
- `save_q_table()`: Serializa Q-table en formato pickle compatible con el agente
- `load_q_table()`: Carga Q-table desde archivo
- `plot_optimization_progress()`: Genera gráficas matplotlib del historial
- `analyze_q_table()`: Calcula estadísticas descriptivas de los valores Q

### Funciones de Nivel Superior

#### **Grupo Optuna (Optimización de Hiperparámetros)**

**`setup_optuna_logging(log_dir)`**
- Configura sistema de logging para Optuna
- Crea directorio de logs con timestamp

**`objective_function(trial, base_config)`**
- Función objetivo para Optuna que optimiza hiperparámetros del GA
- Sugiere valores para: `population_size`, `n_generations`, `crossover_prob`, `mutation_prob`, `test_episodes`
- Ejecuta GA completo con cada combinación de hiperparámetros
- Retorna `best_win_rate` como métrica a maximizar
- Implementa pruning progresivo reportando fitness intermedios

**`optimize_with_optuna(base_config, n_trials, study_name, storage)`**
- Orquesta el estudio de optimización con Optuna
- Configura TPESampler (Tree-structured Parzen Estimator) para exploración inteligente
- Configura MedianPruner para detener trials poco prometedores
- Ejecuta n_trials experimentos completos
- Guarda resultados en JSON

**`plot_optuna_results(study, save_dir)`**
- Genera 5 visualizaciones HTML interactivas:
  1. Historial de optimización
  2. Importancia de hiperparámetros
  3. Coordenadas paralelas
  4. Slice plots (relaciones individuales)
  5. Contour plots (relaciones entre pares)

#### **`main()`**
Punto de entrada principal que:
- Parsea argumentos de línea de comandos
- Valida existencia de archivos requeridos
- Ejecuta dos modos mutuamente excluyentes:
  - **Modo Optuna**: Optimización de hiperparámetros seguida de entrenamiento final
  - **Modo Estándar**: Ejecución única del GA con hiperparámetros fijos

### Dependencias Externas

**Librerías científicas:**
- `numpy`: Operaciones matriciales y manejo de arrays
- `matplotlib.pyplot`: Generación de gráficos

**Framework de optimización (pymoo):**
- `pymoo.core.problem.Problem`: Clase base para problemas de optimización
- `pymoo.algorithms.soo.nonconvex.ga.GA`: Algoritmo genético
- `pymoo.operators.crossover.sbx.SBX`: Simulated Binary Crossover
- `pymoo.operators.mutation.pm.PM`: Polynomial Mutation
- `pymoo.operators.sampling.rnd.FloatRandomSampling`: Muestreo aleatorio inicial
- `pymoo.optimize.minimize`: Motor de optimización
- `pymoo.termination.get_termination`: Criterios de detención

**Optimización bayesiana:**
- `optuna`: Framework de optimización de hiperparámetros
- `optuna.pruners.MedianPruner`: Pruning de trials
- `optuna.samplers.TPESampler`: Muestreo Bayesiano

**Sistema:**
- `subprocess`: Ejecución de agentes externos
- `pickle`: Serialización de Q-tables
- `json`: Lectura de configuraciones
- `tempfile`: Gestión de archivos temporales
- `os`, `sys`, `pathlib`: Operaciones del sistema

**Dominio específico:**
- `AIDojoCoordinator.game_components`: Clases `Action`, `ActionType` del entorno NetSecGame

**Archivos de datos requeridos:**
- `registration_info.json`: Define el espacio de acciones del agente
- `estadosSMALL.json`: Define el espacio de estados discreto
- Script del agente (ej: `q_agent_feature_based.py`): Agente que ejecuta las Q-tables

---

## 3. Flujo de Ejecución Detallado

### Punto de Inicio
La ejecución comienza en el bloque `if __name__ == "__main__":` que invoca `main()`.

### Secuencia de Operaciones

#### **Fase 1: Inicialización y Validación**

```
main()
  ├─ Parsear argumentos CLI con argparse
  ├─ Validar existencia de agent_script
  ├─ Validar existencia de actions/states files
  └─ Parsear reward_range (string "-1,1" → tuple (-1, 1))
```

#### **Fase 2: Bifurcación por Modo de Ejecución**

##### **MODO OPTUNA (--optuna flag activo)**

```
1. Crear base_config dict
   ├─ agent_script_path
   ├─ host, port
   └─ actions_file, states_file

2. optimize_with_optuna()
   ├─ setup_optuna_logging() → crear log file
   ├─ Crear estudio Optuna:
   │   ├─ TPESampler(seed=42) para reproducibilidad
   │   ├─ MedianPruner(n_startup_trials=3, n_warmup_steps=5)
   │   └─ direction='maximize' (win_rate)
   │
   ├─ LOOP: n_trials iteraciones
   │   │
   │   ├─ objective_function(trial, base_config)
   │   │   │
   │   │   ├─ Sugerir hiperparámetros:
   │   │   │   ├─ population_size: [10, 15]
   │   │   │   ├─ n_generations: [2, 5]
   │   │   │   ├─ crossover_prob: [0.5, 0.95]
   │   │   │   ├─ mutation_prob: [0.0001, 0.01] (log scale)
   │   │   │   └─ test_episodes: [250, 300] step 25
   │   │   │
   │   │   ├─ Crear QTableGeneticOptimizer
   │   │   │   └─ Crea QTableOptimizationProblem
   │   │   │       ├─ _load_actions() → parsear registration_info.json
   │   │   │       ├─ _load_states() → parsear estadosSMALL.json
   │   │   │       └─ Definir espacio: n_states × n_actions variables
   │   │   │
   │   │   ├─ optimizer.optimize()
   │   │   │   │
   │   │   │   ├─ Configurar GA(pymoo):
   │   │   │   │   ├─ sampling: FloatRandomSampling()
   │   │   │   │   ├─ crossover: SBX(prob, eta=15)
   │   │   │   │   ├─ mutation: PM(prob, eta=20)
   │   │   │   │   └─ eliminate_duplicates=True
   │   │   │   │
   │   │   │   ├─ minimize(problem, algorithm, termination)
   │   │   │   │   │
   │   │   │   │   └─ CICLO EVOLUTIVO (n_generations):
   │   │   │   │       │
   │   │   │   │       ├─ [Gen 0] Generar población inicial
   │   │   │   │       │   └─ population_size individuos con valores Q aleatorios
   │   │   │   │       │
   │   │   │   │       ├─ problem._evaluate(población)
   │   │   │   │       │   │
   │   │   │   │       │   └─ PARA CADA INDIVIDUO:
   │   │   │   │       │       │
   │   │   │   │       │       ├─ _create_q_table_from_params()
   │   │   │   │       │       │   ├─ Reshape vector → matriz (n_states × n_actions)
   │   │   │   │       │       │   └─ Crear dict {(state, Action): q_value}
   │   │   │   │       │       │
   │   │   │   │       │       ├─ Guardar Q-table en tempfile.pickle
   │   │   │   │       │       │
   │   │   │   │       │       ├─ _run_agent_evaluation(q_table_path)
   │   │   │   │       │       │   │
   │   │   │   │       │       │   ├─ subprocess.run([python, agent_script, 
   │   │   │   │       │       │   │                    --testing, True,
   │   │   │   │       │       │   │                    --previous_model, temp.pickle,
   │   │   │   │       │       │   │                    --episodes, test_episodes])
   │   │   │   │       │       │   │   │
   │   │   │   │       │       │   │   └─ Agente ejecuta test_episodes episodios
   │   │   │   │       │       │   │       en NetSecGame con la Q-table
   │   │   │   │       │       │   │
   │   │   │   │       │       │   ├─ Capturar stdout + stderr
   │   │   │   │       │       │   ├─ _extract_metric(output, "winrate")
   │   │   │   │       │       │   │   └─ Parsear líneas buscando "winrate=XX%"
   │   │   │   │       │       │   │
   │   │   │   │       │       │   └─ fitness = -win_rate (minimización)
   │   │   │   │       │       │
   │   │   │   │       │       └─ os.remove(temp.pickle) → limpieza
   │   │   │   │       │
   │   │   │   │       ├─ Asignar fitness a cada individuo
   │   │   │   │       │
   │   │   │   │       ├─ [Gen 1..n-1] OPERADORES GENÉTICOS:
   │   │   │   │       │   │
   │   │   │   │       │   ├─ Selección (basada en fitness)
   │   │   │   │       │   │
   │   │   │   │       │   ├─ Crossover SBX:
   │   │   │   │       │   │   └─ Combinar parejas de padres → hijos
   │   │   │   │       │   │       (mezcla valores Q)
   │   │   │   │       │   │
   │   │   │   │       │   ├─ Mutación PM:
   │   │   │   │       │   │   └─ Perturbar valores Q con distribución polinomial
   │   │   │   │       │   │
   │   │   │   │       │   └─ Eliminar duplicados
   │   │   │   │       │
   │   │   │   │       └─ Repetir evaluación para nueva generación
   │   │   │   │
   │   │   │   ├─ Extraer mejor solución (res.X)
   │   │   │   ├─ Convertir a Q-table dict
   │   │   │   └─ Guardar historial: [fitness_gen0, fitness_gen1, ...]
   │   │   │
   │   │   ├─ Convertir best_fitness (numpy array) → float
   │   │   │   └─ best_win_rate = -best_fitness
   │   │   │
   │   │   ├─ PRUNING PROGRESIVO:
   │   │   │   └─ Para cada generación:
   │   │   │       ├─ trial.report(-fitness_val, gen)
   │   │   │       └─ if trial.should_prune(): raise TrialPruned
   │   │   │
   │   │   ├─ Guardar Q-table si es mejor trial:
   │   │   │   └─ optuna_trial_{N}_wr_{XX.X}.pickle
   │   │   │
   │   │   └─ return best_win_rate
   │   │
   │   └─ Optuna actualiza modelo bayesiano interno
   │
   ├─ Guardar resultados JSON:
   │   └─ optuna_results_{study_name}.json
   │       ├─ best_value (win_rate)
   │       ├─ best_params (hiperparámetros)
   │       ├─ best_trial (número)
   │       └─ q_table_file (ruta)
   │
   └─ plot_optuna_results(study)
       └─ Generar 5 HTMLs en optuna_plots/

3. ENTRENAMIENTO FINAL con mejores hiperparámetros
   ├─ Crear QTableGeneticOptimizer con best_params
   ├─ optimizer.optimize() [GA completo]
   ├─ save_q_table(optimized_q_table_ga_optuna_best.pickle)
   └─ plot_optimization_progress()
```

##### **MODO ESTÁNDAR (sin --optuna flag)**

```
1. Crear QTableGeneticOptimizer
   ├─ agent_script_path, host, port
   ├─ test_episodes (default: 100)
   ├─ reward_range (default: (-1, 1))
   └─ Internamente crea QTableOptimizationProblem

2. optimizer.optimize()
   ├─ population_size (default: 20)
   ├─ n_generations (default: 10)
   ├─ crossover_prob (default: 0.8)
   └─ mutation_prob (default: 0.0005)
   
   [Sigue el mismo ciclo evolutivo descrito arriba]

3. save_q_table(optimized_q_table_ga.pickle)

4. plot_optimization_progress()
   ├─ Subplot 1: Fitness vs Generación
   └─ Subplot 2: Win Rate vs Generación
```

### Lógica Condicional Importante

#### **Evaluación con Manejo de Errores**
```python
try:
    crear Q-table → guardar pickle → ejecutar agente → extraer fitness
except Exception:
    fitness = 1000.0  # Penalización máxima
finally:
    eliminar archivo temporal
```

#### **Timeout en Subprocess**
```python
subprocess.run(..., timeout=400)  # 5 minutos máximo
except TimeoutExpired:
    return 1000.0  # Penalización
```

#### **Pruning de Optuna**
```python
for gen, fitness in enumerate(optimization_history):
    trial.report(-fitness, gen)
    if trial.should_prune():  # Compara con mediana de trials
        raise TrialPruned
```

#### **Guardar Q-table Solo si es Mejor**
```python
if trial.number == 0 or best_win_rate > trial.study.best_value:
    save Q-table
```

#### **Conversión Segura de NumPy Arrays**
```python
if isinstance(best_fitness, np.ndarray):
    best_fitness = float(best_fitness.item() if best_fitness.size == 1 else best_fitness[0])
best_win_rate = float(-best_fitness)
```

### Manejo de Errores/Excepciones

1. **Archivos faltantes:**
   - `actions_file` → `FileNotFoundError` → `sys.exit(0)`
   - `states_file` → Genera estados dummy y continúa

2. **Timeout de agente:**
   - `subprocess.TimeoutExpired` → fitness = 1000.0 (penalización)

3. **Errores de evaluación:**
   - Cualquier excepción en `_evaluate()` → fitness = 1000.0

4. **Interrupciones del usuario:**
   - `KeyboardInterrupt` → Mensaje de interrupción + `sys.exit(1)`

5. **Errores generales:**
   - `Exception` en `main()` → Traceback completo + `sys.exit(1)`

6. **Trials de Optuna fallidos:**
   - `TrialPruned` → Log y continuar con siguiente trial
   - `Exception` → return 0.0 (penalización) y continuar

7. **Limpieza de temporales:**
   - Doble garantía: `finally` block + loop de verificación final

---

## 4. Entradas (Inputs) Requeridas

### Fuente de Datos

**Argumentos de línea de comandos (CLI):**
```bash
python ga.py --agent_script PATH --actions FILE --states FILE [OPTIONS]
```

**Archivos del sistema:**
- `registration_info.json`: Espacio de acciones del agente
- `estadosSMALL.json`: Espacio de estados discreto
- Script del agente Python (ejecutable)

**Servidor del juego:**
- NetSecGame server corriendo en `host:port`

### Formato de Entradas

#### **Argumentos CLI (todos via argparse)**

| Argumento | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--agent_script` | str | **REQUERIDO** | Ruta absoluta a `q_agent_feature_based.py` |
| `--actions` | str | `"registration_info.json"` | Archivo JSON con definiciones de acciones |
| `--states` | str | `"estadosSMALL.json"` | Archivo JSON con estados discretos |
| `--host` | str | `"127.0.0.1"` | Host del servidor NetSecGame |
| `--port` | int | `9000` | Puerto del servidor |
| `--test_episodes` | int | `100` | Episodios para evaluar cada Q-table |
| `--population` | int | `20` | Tamaño de población del GA |
| `--generations` | int | `10` | Número de generaciones |
| `--crossover_prob` | float | `0.8` | Probabilidad de cruzamiento SBX |
| `--mutation_prob` | float | `0.0005` | Probabilidad de mutación PM |
| `--reward_range` | str | `"-1,1"` | Rango min,max para valores Q |
| `--output` | str | `"optimized_q_table_ga.pickle"` | Archivo de salida |
| `--plot` | str | `None` | Ruta para guardar gráfica (opcional) |
| `--optuna` | flag | `False` | Activar modo Optuna |
| `--optuna_trials` | int | `10` | Número de trials de Optuna |
| `--optuna_study` | str | `None` | Nombre del estudio (para persistencia) |
| `--optuna_storage` | str | `None` | URL de storage (ej: `sqlite:///optuna.db`) |

#### **Estructura de `registration_info.json`**
```json
{
  "all_actions": "[{\"action_type\": \"ActionType.SCAN_NETWORK\", \"parameters\": {...}}, ...]"
}
```
- **Formato:** JSON anidado (string serializado dentro de JSON externo)
- **Campo clave:** `all_actions` (string que contiene array JSON)
- **Elementos del array:**
  - `action_type`: String enum (ej: `"ActionType.SCAN_NETWORK"`)
  - `parameters`: Dict con parámetros de la acción

#### **Estructura de `estadosSMALL.json`**
```json
{
  "states": [
    "(4, 8, 4, 7, 2)",
    "(4, 8, 5, 7, 2)",
    ...
  ]
}
```
- **Formato:** Array de strings representando tuplas Python
- **Dimensión de estado:** `(n_redes, n_hosts_conocidos, n_hosts_controlados, n_servicios, n_datos)`
- **Parsing:** Usa `eval()` para convertir string → tupla

### Parámetros Críticos

1. **`--agent_script`** (CRÍTICO)
   - Debe ser ejecutable con Python
   - Debe aceptar argumentos: `--host`, `--port`, `--episodes`, `--testing`, `--previous_model`, `--experiment_id`
   - Debe imprimir línea con formato: `"winrate=XX.XX%"`

2. **`--test_episodes`** (CRÍTICO PARA RENDIMIENTO)
   - Mayor valor → evaluaciones más precisas pero más lentas
   - Valor típico: 100-300
   - Rango Optuna: 250-300

3. **`--population` y `--generations`** (CRÍTICO PARA CALIDAD)
   - Trade-off entre tiempo y calidad de solución
   - Más población → mayor diversidad, más evaluaciones por generación
   - Más generaciones → más refinamiento, más tiempo total
   - Ejemplo: 15 pop × 3 gen × 250 ep = 11,250 evaluaciones

4. **`--reward_range`** (CRÍTICO PARA ESPACIO DE BÚSQUEDA)
   - Define límites de valores Q
   - Debe contener rango de recompensas del entorno
   - Default `(-1, 1)` apropiado para recompensas normalizadas

5. **`--optuna_trials`** (CRÍTICO PARA OPTUNA)
   - Cada trial ejecuta GA completo
   - Recomendado: 10-20 trials
   - Tiempo total ≈ trial_time × n_trials

---

## 5. Resultados Esperados (Outputs)

### Salida Principal

#### **MODO ESTÁNDAR:**
1. **Archivo pickle serializado:**
   - Nombre: `optimized_q_table_ga.pickle` (o valor de `--output`)
   - Contenido:
     ```python
     {
         'q_table': {
             ((4, 8, 4, 7, 2), Action(...)): 0.234,
             ((4, 8, 5, 7, 2), Action(...)): -0.456,
             ...
         },
         'state_mapping': {}  # Vacío, el agente lo construye
     }
     ```

2. **Gráfica matplotlib (opcional):**
   - Formato: PNG de alta resolución (300 DPI)
   - 2 subplots:
     - Fitness vs Generación
     - Win Rate vs Generación

3. **Logs en consola:**
   ```
   ======================================================================
   OPTIMIZACIÓN COMPLETADA
   ======================================================================
   Mejor fitness: -15.600000
   Win rate estimado: 15.60%
   ======================================================================
   ```

#### **MODO OPTUNA:**
1. **Archivo JSON de resultados:**
   - Nombre: `optuna_results_{study_name}.json`
   - Estructura:
     ```json
     {
       "study_name": "ga_optimization_20251210_153000",
       "best_value": 18.5,
       "best_params": {
         "population_size": 15,
         "n_generations": 3,
         "crossover_prob": 0.829,
         "mutation_prob": 0.00157,
         "test_episodes": 275
       },
       "best_trial": 7,
       "n_trials": 10,
       "q_table_file": "optuna_trial_7_wr_18.5.pickle"
     }
     ```

2. **Q-tables intermedias:**
   - Formato: `optuna_trial_{N}_wr_{WR}.pickle`
   - Solo se guarda si es mejor que trials previos

3. **Q-table final optimizada:**
   - Nombre: `optimized_q_table_ga_optuna_best.pickle`
   - Entrenada con mejores hiperparámetros encontrados

4. **Visualizaciones HTML (directorio `optuna_plots/`):**
   - `optimization_history.html`: Evolución de win_rate por trial
   - `param_importances.html`: Impacto de cada hiperparámetro
   - `parallel_coordinate.html`: Relaciones multidimensionales
   - `slice_plot.html`: Efecto individual de cada parámetro
   - `contour_plot.html`: Interacciones entre pares de parámetros

5. **Log file de Optuna:**
   - Directorio: `optuna_logs/`
   - Nombre: `optuna_{timestamp}.log`
   - Contenido: Todos los eventos de Optuna (INFO level)

### Formato de Salida

#### **Q-table Pickle (estructura interna)**
```python
{
    'q_table': Dict[Tuple[Tuple[int, ...], Action], float],
    'state_mapping': Dict  # Vacío en este contexto
}
```

**Propiedades:**
- Compatible con `q_agent_feature_based.py`
- Serializado con `pickle.dump()` en modo binario
- Tamaño típico: ~10-50 MB (depende de n_states × n_actions)

#### **Logs de consola (pymoo)**
```
n_gen |  n_eval  | f_avg     | f_min
----- | -------- | --------- | ---------
    1 |       15 | -8.533333 | -12.800000
    2 |       30 | -10.26667 | -15.200000
    3 |       45 | -12.80000 | -17.600000
```
- `n_gen`: Generación actual
- `n_eval`: Evaluaciones acumuladas
- `f_avg`: Fitness promedio de la población
- `f_min`: Mejor fitness (MENOR es MEJOR, valores negativos)

### Comportamiento en Caso de Éxito

**Criterios de éxito:**
1. Ejecución completa sin excepciones
2. Generación de archivo(s) pickle válido(s)
3. Win rate > 0% (indica que el agente gana al menos algunos episodios)
4. Mejora progresiva observable en `f_min` a través de generaciones
5. (Optuna) best_value > 0 y al menos un trial completo exitoso

**Indicadores de éxito en logs:**
```
✓ Optimización completada exitosamente!

Q-table guardada en: optimized_q_table_ga.pickle
Formato: Explícito con 1680 estados × 156 acciones
Total de entradas: 262,080

Win rate estimado: 15.60%
```

**Uso posterior:**
```bash
python q_agent_feature_based.py \
    --previous_model optimized_q_table_ga.pickle \
    --host 127.0.0.1 \
    --port 9000 \
    --episodes 1000
```

**Ventajas de la Q-table optimizada:**
- Inicialización inteligente (vs. zeros)
- Exploración reducida en fases tempranas de entrenamiento
- Convergencia más rápida a políticas óptimas
- Baseline de rendimiento conocido (win_rate del GA)

---

## 6. Consideraciones Adicionales

### Complejidad Computacional
- **Evaluaciones por generación:** `population_size`
- **Evaluaciones totales:** `population_size × n_generations`
- **Tiempo por evaluación:** `~(test_episodes × episode_duration)`
- **Tiempo total estimado:**
  - Modo estándar: `population_size × n_generations × test_episodes × ~3s`
  - Modo Optuna: `n_trials × (tiempo GA) ≈ 10-50 horas`

### Recomendaciones de Uso

#### Para pruebas rápidas:
```bash
python ga.py --agent_script q_agent.py \
    --population 10 \
    --generations 2 \
    --test_episodes 50
```

#### Para producción (con Optuna):
```bash
python ga.py --agent_script q_agent.py \
    --optuna \
    --optuna_trials 20
```

#### Monitoreo:
- Observar `f_min` en tiempo real para detectar convergencia prematura
- Si `f_min` no mejora después de 3-5 generaciones, considerar detener
- Verificar que win_rate > 0 en las primeras generaciones

### Limitaciones Conocidas

1. **Dependencia del servidor:**
   - Requiere servidor NetSecGame activo durante toda la ejecución
   - Fallas del servidor causan penalizaciones masivas (fitness = 1000.0)

2. **Sin paralelización:**
   - No paraleliza evaluaciones (subprocess secuencial)
   - Posible optimización: usar multiprocessing.Pool

3. **Seguridad:**
   - Usa `eval()` para parsear estados (riesgo si JSON no es confiable)
   - Recomendación: usar `ast.literal_eval()` en producción

4. **Timeout fijo:**
   - 400s por evaluación puede ser insuficiente para episodios largos
   - Ajustar manualmente en `_run_agent_evaluation()` si es necesario

5. **Memoria:**
   - Cada Q-table ocupa `n_states × n_actions × 8 bytes` en memoria
   - Para espacios grandes (>1M entradas), considerar representaciones sparse

### Ejemplos de Uso Completos

#### Ejemplo 1: Ejecución estándar básica
```bash
python ga.py \
    --agent_script ../agents/q_agent_feature_based.py \
    --actions registration_info.json \
    --states estadosSMALL.json \
    --population 20 \
    --generations 10 \
    --test_episodes 100 \
    --output my_q_table.pickle
```

#### Ejemplo 2: Optimización con Optuna
```bash
python ga.py \
    --agent_script ../agents/q_agent_feature_based.py \
    --optuna \
    --optuna_trials 15 \
    --optuna_study my_study \
    --optuna_storage sqlite:///optuna_db.sqlite
```

#### Ejemplo 3: Con visualización y configuración personalizada
```bash
python ga.py \
    --agent_script ../agents/q_agent_feature_based.py \
    --population 15 \
    --generations 20 \
    --crossover_prob 0.9 \
    --mutation_prob 0.001 \
    --reward_range "-2,2" \
    --test_episodes 200 \
    --plot progress_chart.png \
    --output best_q_table.pickle
```

---

## 7. Troubleshooting

### Problemas Comunes

#### Error: "Timeout en evaluación del agente"
**Causa:** Episodios toman >400s  
**Solución:** Aumentar timeout o reducir `--test_episodes`

#### Error: "win_rate = 0.0%" consistentemente
**Causa:** Q-table inicial inadecuada o agente defectuoso  
**Solución:** 
- Verificar que el agente funciona sin Q-table previa
- Ajustar `--reward_range` para incluir valores esperados
- Aumentar `--test_episodes` para mejor estimación

#### Optuna no mejora después de varios trials
**Causa:** Espacio de hiperparámetros muy restrictivo o problema mal definido  
**Solución:**
- Ampliar rangos en `objective_function()` (líneas 615-621)
- Aumentar `--optuna_trials` para mejor exploración
- Verificar que el agente base funciona correctamente

#### Memoria insuficiente
**Causa:** Q-table muy grande (millones de entradas)  
**Solución:**
- Reducir `n_states` en `estadosSMALL.json`
- Reducir `--population` 
- Considerar implementación sparse de Q-table

---

## 8. Referencias Técnicas

### Algoritmos Implementados

**Algoritmo Genético (GA):**
- Representación: Vector real de dimensión `n_states × n_actions`
- Selección: Basada en fitness (implementado por pymoo)
- Crossover: SBX (Simulated Binary Crossover, Deb & Agrawal, 1995)
- Mutación: PM (Polynomial Mutation, Deb & Goyal, 1996)
- Parámetro eta=15 para crossover, eta=20 para mutación

**Optimización Bayesiana (Optuna):**
- Sampler: TPE (Tree-structured Parzen Estimator, Bergstra et al., 2011)
- Pruner: MedianPruner (detiene trials con rendimiento por debajo de la mediana)
- Direction: Maximize (maximizar win_rate)

### Métricas de Evaluación

**Fitness Function:**
```
fitness = -win_rate
```
- Problema de minimización (pymoo estándar)
- win_rate en rango [0, 100]
- Valores negativos indican mejor rendimiento

**Win Rate:**
```
win_rate = (episodios_ganados / total_episodios) × 100
```
- Métrica principal de rendimiento
- Extraída de salida del agente
- Formato esperado: "winrate=XX.XX%"

---
