import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from AIDojoCoordinator.game_components import Action, ActionType
import pickle
import matplotlib.pyplot as plt
import subprocess
import sys
import json
import os
import tempfile
from pathlib import Path
from ConfigSpace import ConfigurationSpace, Configuration, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory import RunHistory
import logging
from datetime import datetime
import argparse
import traceback
from multiprocessing import Pool, cpu_count
from functools import partial


# ============================================================================
# Función auxiliar para evaluación paralela (debe estar en el nivel superior)
# ============================================================================

def _evaluate_single_individual(params_tuple):
    """
    Evalúa un solo individuo. Esta función debe estar a nivel de módulo
    para ser serializable por multiprocessing.
    
    Args:
        params_tuple: Tupla con (idx, x, agent_script_path, host, port, 
                      test_episodes, actions, states, state_to_idx, 
                      n_actions, n_states, reward_range)
    
    Returns:
        tuple: (idx, fitness)
    """
    (idx, x, agent_script_path, host, port, test_episodes, 
     actions, states, state_to_idx, n_actions, n_states, reward_range) = params_tuple
    
    print(f"\n[Proceso] Evaluando individuo {idx + 1}...")
    q_table_path = None
    
    try:
        # Crear Q-table desde parámetros
        q_table_matrix = x.reshape(n_states, n_actions)
        
        # Crear diccionario Q-table
        q_table = {}
        for state_idx, state in enumerate(states):
            for action_idx in range(n_actions):
                q_value = q_table_matrix[state_idx, action_idx]
                q_table[(state, actions[action_idx])] = q_value
        
        q_table_data = {
            'q_table': q_table,
            'state_mapping': {}
        }
        
        # Crear archivo temporal
        tmp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pickle', delete=False)
        q_table_path = tmp_file.name
        
        # Guardar Q-table y cerrar el archivo
        pickle.dump(q_table_data, tmp_file)
        tmp_file.close()
        
        # Ejecutar agente en modo testing
        fitness = _run_agent_evaluation_static(
            q_table_path, agent_script_path, host, port, test_episodes
        )
        
        print(f"[Proceso] Individuo {idx + 1} -> Fitness: {fitness:.4f}")
        
        return (idx, fitness)
        
    except Exception as e:
        print(f"[Proceso] Error evaluando individuo {idx + 1}: {e}")
        return (idx, 1000.0)
    
    finally:
        # Eliminar el archivo temporal
        if q_table_path and os.path.exists(q_table_path):
            try:
                os.remove(q_table_path)
            except:
                pass


def _run_agent_evaluation_static(q_table_path, agent_script_path, host, port, test_episodes):
    """
    Versión estática de _run_agent_evaluation para uso con multiprocessing.
    
    Args:
        q_table_path (str): Ruta al archivo pickle con la Q-table
        agent_script_path (str): Ruta al script del agente
        host (str): Host del servidor
        port (int): Puerto del servidor
        test_episodes (int): Número de episodios de prueba
        
    Returns:
        float: Fitness (menor es mejor)
    """
    cmd = [
        sys.executable,
        agent_script_path,
        "--host", host,
        "--port", str(port),
        "--episodes", str(test_episodes),
        "--testing", "True",
        "--previous_model", q_table_path,
        "--experiment_id", "ga_eval",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=400
        )
        
        output = result.stdout + result.stderr
        win_rate = _extract_metric_static(output, "winrate")
        
        print(f"[Proceso] win_rate: {win_rate}%")
        fitness = -win_rate
        
        return fitness
        
    except subprocess.TimeoutExpired:
        print("[Proceso] Timeout en evaluación del agente")
        return 1000.0
    except Exception as e:
        print(f"[Proceso] Error ejecutando agente: {e}")
        return 1000.0


def _extract_metric_static(output, metric_name):
    """
    Versión estática de _extract_metric para uso con multiprocessing.
    
    Args:
        output (str): Salida del agente
        metric_name (str): Nombre de la métrica a extraer
        
    Returns:
        float: Valor de la métrica (0 si no se encuentra)
    """
    try:
        for line in output.split('\n'):
            if metric_name in line:
                parts = line.split('=')
                if len(parts) > 1:
                    value_str = parts[1].split()[0].strip('%,')
                    return float(value_str)
    except:
        pass
    return 0.0


# ============================================================================
# Clase principal del problema con soporte multiproceso
# ============================================================================

class QTableOptimizationProblemMulti(Problem):
    """
    Problema de optimización para inicializar una Q-table usando algoritmos genéticos.
    
    Versión con soporte para evaluación paralela usando multiprocessing.
    
    Cada individuo representa una Q-table completa donde:
    - state: tupla (n_redes, n_hosts_conocidos, n_hosts_controlados, n_servicios, n_datos)
    - action: acciones definidas en registration_info.json
    - q_value: valor Q para el par (state, action)
    
    El fitness se evalúa ejecutando el agente en modo testing con la Q-table del individuo.
    Los estados específicos son cargados desde estadosSMALL.json.
    """
    
    def __init__(self, 
                 agent_script_path,
                 host="127.0.0.1",
                 port=9000,
                 test_episodes=10,
                 reward_range=(-1, 1),
                 actions_file="registration_info.json",
                 states_file="estadosSMALL.json",
                 n_jobs=1):
        """
        Inicializa el problema de optimización de Q-table.
        
        Args:
            agent_script_path (str): Ruta al script q_agent_feature_based.py
            host (str): Host del servidor del juego
            port (int): Puerto del servidor del juego
            test_episodes (int): Número de episodios para evaluar cada Q-table
            reward_range (tuple): Rango de valores para los Q-values
            actions_file (str): Ruta al archivo registration_info.json con las acciones
            states_file (str): Ruta al archivo con los estados específicos
            n_jobs (int): Número de procesos paralelos (1=secuencial, -1=todos los CPUs)
        """
        self.agent_script_path = agent_script_path
        self.host = host
        self.port = port
        self.test_episodes = test_episodes
        self.reward_range = reward_range
        self.actions_file = actions_file
        self.states_file = states_file
        
        # Configurar número de trabajadores
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = max(1, n_jobs)
        
        # Cargar información de acciones y estados
        self.actions = self._load_actions()
        self.n_actions = len(self.actions)
        
        self.states = self._load_states()
        self.n_states = len(self.states)
        
        # Crear mapeo de estados a índices para acceso rápido
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        
        # El tamaño de la Q-table es: n_states × n_actions
        self.q_table_size = self.n_states * self.n_actions
        
        # Definir los límites del problema
        xl = np.full(self.q_table_size, reward_range[0])
        xu = np.full(self.q_table_size, reward_range[1])
        
        super().__init__(
            n_var=self.q_table_size,
            n_obj=1,
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )
        
        print(f"Problema inicializado con {self.n_states} estados y {self.n_actions} acciones")
        print(f"Tamaño de Q-table: {self.n_states} × {self.n_actions} = {self.q_table_size} valores")
        print(f"Modo de evaluación: {'PARALELO' if self.n_jobs > 1 else 'SECUENCIAL'}")
        if self.n_jobs > 1:
            print(f"Número de procesos: {self.n_jobs} (CPUs disponibles: {cpu_count()})")
    
    def _load_actions(self):
        """Carga las acciones desde registration_info.json y las convierte a objetos Action"""
        try:
            json_raw = open(self.actions_file).read()
            outer = json.loads(json_raw)
            actions_dicts = json.loads(outer["all_actions"])

            print("Cantidad de acciones:", len(actions_dicts))
            print("Ejemplo:", actions_dicts[0])

            if not actions_dicts:
                print("No se encontraron acciones en el archivo.")
                sys.exit(0)
                
            print(f"Cargadas {len(actions_dicts)} acciones desde {self.actions_file}")

            # Convertir diccionarios a objetos Action
            actions = []
            for action_dict in actions_dicts:
                # Key is "action_type" with value like "ActionType.ExfiltrateData"
                raw_type = action_dict["action_type"]
                # Strip the "ActionType." prefix if present
                type_name = raw_type.split(".")[-1]
                action_type = ActionType[type_name]
                params = self._make_params_hashable(action_dict["parameters"])
                action = Action(
                    action_type=action_type,
                    parameters=params
                )
                actions.append(action)
            
            print(f"Convertidas {len(actions)} acciones a objetos Action")
            print(f"Ejemplo de acción: {actions[0]}")
            
            return actions
        except FileNotFoundError:
            print(f"Archivo {self.actions_file} no encontrado.")
            sys.exit(0)
        except json.JSONDecodeError as e:
            print(f"Error al parsear {self.actions_file}: {e}")
            sys.exit(0)
    
    def _make_params_hashable(self, params):
        """Convierte parámetros con diccionarios anidados a estructura hashable"""
        hashable_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                # Convertir dict a tupla de tuplas
                hashable_params[key] = tuple(sorted(value.items()))
            elif isinstance(value, list):
                # Convertir lista a tupla
                hashable_params[key] = tuple(value)
            else:
                hashable_params[key] = value
        return hashable_params
    
    def _load_states(self):
        """Carga los estados específicos desde el archivo JSON"""
        try:
            with open(self.states_file, 'r') as f:
                states_data = json.load(f)
            states = [tuple(s) for s in states_data]
            print(f"Cargados {len(states)} estados desde {self.states_file}")
            return states
        except FileNotFoundError:
            print(f"Archivo {self.states_file} no encontrado. Usando estados dummy.")
            # Generar estados dummy
            return [(4, i, j, k, l) for i in range(3, 10) for j in range(2, 8) 
                    for k in range(0, 5) for l in range(0, 3)]
    
    def _create_q_table_from_params(self, params):
        """
        Crea una Q-table desde los parámetros del individuo.
        
        Los parámetros representan directamente la Q-table completa:
        matriz de (n_states × n_actions)
        """
        # Reshape parámetros a matriz Q-table
        q_table_matrix = params.reshape(self.n_states, self.n_actions)
        
        # Crear diccionario Q-table con mapeo estado -> Action object -> q-value
        q_table = {}
        for state_idx, state in enumerate(self.states):
            for action_idx in range(self.n_actions):
                q_value = q_table_matrix[state_idx, action_idx]
                q_table[(state, self.actions[action_idx])] = q_value
        
        # Crear estructura compatible con q_agent_feature_based.py
        q_table_data = {
            'q_table': q_table,
            'state_mapping': {}
        }
        
        return q_table_data
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evalúa la calidad de las Q-tables candidatas ejecutándolas en el agente.
        
        Versión paralela usando multiprocessing.Pool cuando n_jobs > 1.
        
        Para cada individuo:
        1. Crea una Q-table desde sus parámetros
        2. Guarda la Q-table en un archivo temporal
        3. Ejecuta el agente en modo testing con esa Q-table
        4. Extrae el fitness (win_rate, avg_return, etc.)
        """
        if self.n_jobs > 1:
            # Evaluación paralela
            objectives = self._evaluate_parallel(X)
        else:
            # Evaluación secuencial (original)
            objectives = self._evaluate_sequential(X)
        
        out["F"] = np.array(objectives)
    
    def _evaluate_sequential(self, X):
        """Evaluación secuencial (modo original)"""
        objectives = []
        temp_files_created = []
        
        for idx, x in enumerate(X):
            print(f"\nEvaluando individuo {idx + 1}/{len(X)}...")
            q_table_path = None
            
            try:
                # Crear Q-table desde parámetros
                q_table_data = self._create_q_table_from_params(x)
                
                # Crear archivo temporal
                tmp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pickle', delete=False)
                q_table_path = tmp_file.name
                temp_files_created.append(q_table_path)
                
                # Guardar Q-table y cerrar el archivo
                pickle.dump(q_table_data, tmp_file)
                tmp_file.close()
                
                # Ejecutar agente en modo testing
                fitness = self._run_agent_evaluation(q_table_path)
                objectives.append(fitness)
                print(f"  Fitness: {fitness:.4f}")
                
            except Exception as e:
                print(f"  Error evaluando individuo: {e}")
                objectives.append(1000.0)
            
            finally:
                # Eliminar el archivo temporal
                if q_table_path and os.path.exists(q_table_path):
                    try:
                        os.remove(q_table_path)
                        print(f"Temporal eliminado: {os.path.basename(q_table_path)}")
                    except Exception as e:
                        print(f"No se pudo eliminar {q_table_path}: {e}")
        
        # Limpieza adicional
        for temp_file in temp_files_created:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        return objectives
    
    def _evaluate_parallel(self, X):
        """
        Evaluación paralela usando multiprocessing.Pool.
        
        Distribuye la evaluación de individuos entre múltiples procesos.
        """
        print(f"\n{'='*70}")
        print(f"EVALUACIÓN PARALELA: {len(X)} individuos en {self.n_jobs} procesos")
        print(f"{'='*70}\n")
        
        # Preparar parámetros para cada individuo
        params_list = []
        for idx, x in enumerate(X):
            params_tuple = (
                idx, x, self.agent_script_path, self.host, self.port,
                self.test_episodes, self.actions, self.states,
                self.state_to_idx, self.n_actions, self.n_states,
                self.reward_range
            )
            params_list.append(params_tuple)
        
        # Ejecutar evaluaciones en paralelo
        objectives = [0.0] * len(X)  # Inicializar array de objetivos
        
        try:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(_evaluate_single_individual, params_list)
            
            # Ordenar resultados por índice
            for idx, fitness in results:
                objectives[idx] = fitness
            
            print(f"\n{'='*70}")
            print(f"EVALUACIÓN PARALELA COMPLETADA")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\nError en evaluación paralela: {e}")
            traceback.print_exc()
            # Fallback a penalización máxima
            objectives = [1000.0] * len(X)
        
        return objectives
    
    def _run_agent_evaluation(self, q_table_path):
        """
        Ejecuta el agente con la Q-table proporcionada y extrae el fitness.
        
        Args:
            q_table_path (str): Ruta al archivo pickle con la Q-table
            
        Returns:
            float: Fitness (menor es mejor). Usamos: -win_rate
        """
        return _run_agent_evaluation_static(
            q_table_path, self.agent_script_path, self.host, 
            self.port, self.test_episodes
        )
    
    def _extract_metric(self, output, metric_name):
        """
        Extrae una métrica de la salida del agente.
        
        Args:
            output (str): Salida del agente
            metric_name (str): Nombre de la métrica a extraer
            
        Returns:
            float: Valor de la métrica (0 si no se encuentra)
        """
        return _extract_metric_static(output, metric_name)


class QTableGeneticOptimizerMulti:
    """
    Optimizador de Q-table usando algoritmos genéticos con pymoo.
    
    Versión con soporte para evaluación paralela de individuos.
    
    Cada individuo es una Q-table completa que se evalúa ejecutándola en el agente
    de NetSecGame en modo testing. Los estados son cargados desde estadosSMALL.json.
    """
    
    def __init__(self, 
                 agent_script_path,
                 host="127.0.0.1",
                 port=9000,
                 test_episodes=10,
                 reward_range=(-1, 1),
                 actions_file="registration_info.json",
                 states_file="estadosSMALL.json",
                 n_jobs=1):
        """
        Inicializa el optimizador.
        
        Args:
            agent_script_path (str): Ruta al script q_agent_feature_based.py
            host (str): Host del servidor del juego
            port (int): Puerto del servidor del juego
            test_episodes (int): Número de episodios para evaluar cada Q-table
            reward_range (tuple): Rango de valores para la Q-table
            actions_file (str): Ruta al archivo con las acciones registradas
            states_file (str): Ruta al archivo con los estados específicos
            n_jobs (int): Número de procesos paralelos (1=secuencial, -1=todos los CPUs)
        """
        self.agent_script_path = agent_script_path
        self.host = host
        self.port = port
        self.test_episodes = test_episodes
        self.reward_range = reward_range
        self.actions_file = actions_file
        self.states_file = states_file
        self.n_jobs = n_jobs
        
        # Crear el problema de optimización
        self.problem = QTableOptimizationProblemMulti(
            agent_script_path=agent_script_path,
            host=host,
            port=port,
            test_episodes=test_episodes,
            reward_range=reward_range,
            actions_file=actions_file,
            states_file=states_file,
            n_jobs=n_jobs
        )
        
        self.best_q_table = None
        self.optimization_history = []
        self.best_params = None
    
    def optimize(self, 
                 population_size=10,
                 n_generations=1,
                 sbx_prob=0.9,
                 sbx_eta=15,
                 pm_prob_var=0.1,
                 pm_eta=20,
                 verbose=True):
        """
        Ejecuta la optimización usando algoritmo genético.
        
        Cada individuo (Q-table) es evaluado ejecutándolo en el agente
        durante varios episodios de testing.
        
        Args:
            population_size (int): Tamaño de la población
            n_generations (int): Número de generaciones
            sbx_prob (float): Probabilidad de cruzamiento SBX
            sbx_eta (float): Índice de distribución SBX
            pm_prob_var (float): Probabilidad por variable de mutación PM
            pm_eta (float): Índice de distribución PM
            verbose (bool): Mostrar progreso
        
        Returns:
            dict: Q-table optimizada con metadata
        """
        # Valores fijos
        sbx_prob_var = 1.0  # prob_var para SBX fijo en 1
        pm_prob = 1.0       # prob para PM fijo en 1
        
        print("\n" + "="*70)
        print("OPTIMIZACIÓN DE Q-TABLE CON ALGORITMO GENÉTICO (MULTIPROCESO)")
        print("="*70)
        print(f"Agente: {self.agent_script_path}")
        print(f"Servidor: {self.host}:{self.port}")
        print(f"Episodios de evaluación: {self.test_episodes}")
        print(f"Población: {population_size}")
        print(f"Generaciones: {n_generations}")
        print(f"Rango de Q-values: {self.reward_range}")
        print(f"Procesos paralelos: {self.problem.n_jobs}")
        print(f"SBX prob: {sbx_prob}, eta: {sbx_eta}, prob_var: {sbx_prob_var} (fijo)")
        print(f"PM prob: {pm_prob} (fijo), prob_var: {pm_prob_var}, eta: {pm_eta}")
        print("="*70 + "\n")
        
        # Configurar algoritmo genético
        algorithm = GA(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=sbx_prob, prob_var=sbx_prob_var, eta=sbx_eta),
            mutation=PM(prob=pm_prob, prob_var=pm_prob_var, eta=pm_eta),
            eliminate_duplicates=True
        )
        
        # Configurar terminación
        termination = get_termination("n_gen", n_generations)
        
        # Ejecutar optimización
        print("Iniciando evolución...\n")
        res = minimize(
            self.problem,
            algorithm,
            termination,
            verbose=verbose,
            save_history=True
        )
        
        # Guardar el mejor resultado
        self.best_params = res.X
        self.best_q_table = self.problem._create_q_table_from_params(res.X)
        
        # Guardar historial de optimización
        self.optimization_history = [entry.opt.get("F")[0] for entry in res.history]
        
        print("\n" + "="*70)
        print("OPTIMIZACIÓN COMPLETADA")
        print("="*70)
        print(f"Mejor fitness: {res.F[0]:.6f}")
        print(f"Win rate estimado: {-res.F[0]:.2f}%")
        print("="*70 + "\n")
        
        return self.best_q_table
    
    def save_q_table(self, filename="optimized_q_table_ga_multi.pickle"):
        """
        Guarda la Q-table optimizada en un archivo compatible con el agente.
        
        Args:
            filename (str): Nombre del archivo donde guardar la Q-table
        """
        if self.best_q_table is None:
            raise ValueError("No hay Q-table optimizada para guardar. Ejecuta optimize() primero.")
        
        # Preparar datos para guardar en formato compatible con q_agent_feature_based.py
        save_data = {
            'q_table': self.best_q_table['q_table'],
            'state_mapping': self.best_q_table['state_mapping'],
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nQ-table guardada en: {filename}")
        print(f"Formato: Explícito con {self.problem.n_states} estados × {self.problem.n_actions} acciones")
        print(f"Total de entradas: {len(self.best_q_table['q_table'])}")
    
    def load_q_table(self, filename):
        """
        Carga una Q-table desde un archivo.
        
        Args:
            filename (str): Nombre del archivo a cargar
        
        Returns:
            dict: Q-table cargada
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Cargar formato simplificado
        self.best_q_table = {
            'q_table': data.get('q_table', {}),
            'state_mapping': data.get('state_mapping', {})
        }
        
        print(f"Q-table cargada desde: {filename}")
        print(f"Total de entradas: {len(self.best_q_table['q_table'])}")
        return self.best_q_table
    
    def plot_optimization_progress(self, save_path=None):
        """
        Grafica el progreso de la optimización.
        
        Args:
            save_path (str): Ruta donde guardar la figura (opcional)
        """
        if not self.optimization_history:
            print("No hay historial de optimización para graficar.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Fitness por generación
        plt.subplot(1, 2, 1)
        plt.plot(self.optimization_history, marker='o', linewidth=2)
        plt.title('Progreso de Optimización de Q-table (Multi)', fontsize=14, fontweight='bold')
        plt.xlabel('Generación', fontsize=12)
        plt.ylabel('Fitness (menor es mejor)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Win rate estimado por generación
        plt.subplot(1, 2, 2)
        win_rates = [-f for f in self.optimization_history]
        plt.plot(win_rates, marker='o', linewidth=2, color='green')
        plt.title('Win Rate Estimado por Generación', fontsize=14, fontweight='bold')
        plt.xlabel('Generación', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")
        
        plt.show()
    
    def analyze_q_table(self):
        """
        Analiza las propiedades estadísticas de la Q-table optimizada.
        """
        if self.best_q_table is None:
            print("No hay Q-table optimizada para analizar.")
            return
        
        print("\n" + "="*70)
        print("ANÁLISIS DE Q-TABLE OPTIMIZADA")
        print("="*70)
        
        q_table_dict = self.best_q_table['q_table']
        print(f"Total de entradas: {len(q_table_dict):,}")
        
        # Extraer todos los Q-values
        all_q_values = np.array(list(q_table_dict.values()))
        
        print(f"\nEstadísticas de Q-values:")
        print(f"  Valor mínimo: {np.min(all_q_values):.6f}")
        print(f"  Valor máximo: {np.max(all_q_values):.6f}")
        print(f"  Media: {np.mean(all_q_values):.6f}")
        print(f"  Desviación estándar: {np.std(all_q_values):.6f}")
        print(f"  Mediana: {np.median(all_q_values):.6f}")
        
        # Análisis de distribución de valores
        print(f"\nDistribución de Q-values:")
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(all_q_values, p)
            print(f"  Percentil {p}: {val:.3f}")
        
        if self.optimization_history:
            print(f"\nHistorial de optimización:")
            initial_fitness = float(self.optimization_history[0])
            final_fitness = float(self.optimization_history[-1])
            print(f"  Fitness inicial: {initial_fitness:.6f}")
            print(f"  Fitness final: {final_fitness:.6f}")
            print(f"  Mejora: {initial_fitness - final_fitness:.6f}")
            print(f"  Win rate estimado final: {-final_fitness:.2f}%")
        
        print("="*70 + "\n")


# ============================================================================
# SMAC3: Optimización Bayesiana de hiperparámetros del GA (con multiproceso)
# ============================================================================

def setup_smac_logging(log_dir="smac_logs"):
    """Configura logging para SMAC"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"smac_multi_{timestamp}.log")
    
    # Configurar logger
    logger = logging.getLogger("smac_ga_multi")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return log_file, logger


class SMACGAObjectiveMulti:
    """
    Clase que encapsula la función objetivo para SMAC3.
    
    Versión con soporte para multiproceso en las evaluaciones del GA.
    
    SMAC3 requiere una función target_function(config, seed) -> cost.
    Esta clase almacena la configuración base y los mejores resultados.
    """
    
    def __init__(self, base_config):
        """
        Args:
            base_config (dict): Configuración base con parámetros fijos
                - agent_script_path, host, port, actions_file, states_file, n_jobs
        """
        self.base_config = base_config
        self.trial_count = 0
        self.best_win_rate = 0.0
        self.best_trial = -1
        self.results_history = []
    
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Define el espacio de configuración (hiperparámetros) para SMAC.
        
        Hiperparámetros optimizados:
          - population_size: tamaño de la población del GA [5, 20]
          - n_generations: número de generaciones del GA [5, 20]
          - sbx_prob: probabilidad de cruzamiento SBX [0.8, 1.0]
          - sbx_eta: índice de distribución SBX [0, 60]
          - pm_prob_var: probabilidad por variable de mutación PM [0.01, 0.6]
          - pm_eta: índice de distribución PM [0, 60]
        """
        cs = ConfigurationSpace(seed=42)
        
        population_size = Integer("population_size", (5, 20))
        n_generations = Integer("n_generations", (5, 20))
        sbx_prob = Float("sbx_prob", (0.8, 1.0))
        sbx_eta = Float("sbx_eta", (0.0, 60.0))
        pm_prob_var = Float("pm_prob_var", (0.01, 0.6))
        pm_eta = Float("pm_eta", (0.0, 60.0))
        
        cs.add([population_size, n_generations, sbx_prob, sbx_eta, pm_prob_var, pm_eta])
        
        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        """
        Función objetivo que SMAC minimizará.
        
        Crea un QTableGeneticOptimizerMulti con los hiperparámetros de 'config',
        ejecuta la optimización GA y devuelve el costo (negativo del win_rate,
        porque SMAC minimiza).
        
        Args:
            config: Configuración de SMAC con los hiperparámetros
            seed: Semilla para reproducibilidad (no usada directamente aquí)
            
        Returns:
            float: Costo a minimizar (100 - win_rate) / 100. Menor es mejor.
        """
        self.trial_count += 1
        trial_num = self.trial_count
        
        population_size = config["population_size"]
        n_generations = config["n_generations"]
        sbx_prob = config["sbx_prob"]
        sbx_eta = config["sbx_eta"]
        pm_prob_var = config["pm_prob_var"]
        pm_eta = config["pm_eta"]
        
        # Valores fijos
        test_episodes = 25
        reward_range = (-1, 1)
        n_jobs = self.base_config.get('n_jobs', 1)
        
        print(f"\n{'='*70}")
        print(f"SMAC TRIAL {trial_num} (Multiproceso: {n_jobs} workers)")
        print(f"{'='*70}")
        print(f"Hiperparámetros sugeridos:")
        print(f"  population_size: {population_size}")
        print(f"  n_generations: {n_generations}")
        print(f"  sbx_prob: {sbx_prob:.4f}")
        print(f"  sbx_eta: {sbx_eta:.2f}")
        print(f"  pm_prob_var: {pm_prob_var:.4f}")
        print(f"  pm_eta: {pm_eta:.2f}")
        print(f"  test_episodes: {test_episodes} (fijo)")
        print(f"  n_jobs: {n_jobs} (fijo)")
        print(f"  seed: {seed}")
        print(f"{'='*70}\n")
        
        try:
            # Crear optimizador con hiperparámetros sugeridos
            optimizer = QTableGeneticOptimizerMulti(
                agent_script_path=self.base_config['agent_script_path'],
                host=self.base_config['host'],
                port=self.base_config['port'],
                test_episodes=test_episodes,
                reward_range=reward_range,
                actions_file=self.base_config['actions_file'],
                states_file=self.base_config['states_file'],
                n_jobs=n_jobs
            )
            
            # Ejecutar optimización GA
            optimized_q_table = optimizer.optimize(
                population_size=population_size,
                n_generations=n_generations,
                sbx_prob=sbx_prob,
                sbx_eta=sbx_eta,
                pm_prob_var=pm_prob_var,
                pm_eta=pm_eta,
                verbose=True
            )
            
            # Obtener el mejor fitness (último valor del historial)
            best_fitness = optimizer.optimization_history[-1]
            if isinstance(best_fitness, np.ndarray):
                best_fitness = float(best_fitness.item() if best_fitness.size == 1 else best_fitness[0])
            best_win_rate = float(-best_fitness)
            
            # SMAC minimiza, así que devolvemos (100 - win_rate) / 100
            cost = (100.0 - best_win_rate) / 100.0
            
            # Registrar resultado
            self.results_history.append({
                'trial': trial_num,
                'win_rate': best_win_rate,
                'cost': cost,
                'config': dict(config),
            })
            
            # Guardar Q-table si es el mejor resultado
            if best_win_rate > self.best_win_rate:
                self.best_win_rate = best_win_rate
                self.best_trial = trial_num
                output_file = f"smac_multi_trial_{trial_num}_wr_{best_win_rate:.1f}.pickle"
                optimizer.save_q_table(output_file)
                print(f"  -> Nuevo mejor resultado! Q-table guardada en: {output_file}")
            
            print(f"\nTrial {trial_num} completado: win_rate = {best_win_rate:.2f}%, cost = {cost:.4f}\n")
            
            return cost
            
        except Exception as e:
            print(f"\nError en trial {trial_num}: {e}\n")
            traceback.print_exc()
            return 1.0  # Penalización máxima (win_rate = 0%)


def optimize_with_smac_multi(base_config, n_trials=20, output_dir="smac_output_multi"):
    """
    Ejecuta optimización de hiperparámetros con SMAC3.
    
    Versión con soporte para multiproceso en las evaluaciones del GA.
    
    Args:
        base_config: Diccionario con configuración base (agent_script, host, port, etc.)
        n_trials: Número de trials (evaluaciones) objetivo
        output_dir: Directorio para resultados de SMAC
        
    Returns:
        tuple: (incumbent_config, objective_instance, smac_instance)
    """
    # Configurar logging
    log_file, logger = setup_smac_logging()
    print(f"SMAC logs guardados en: {log_file}\n")
    
    # Crear instancia del objetivo
    objective = SMACGAObjectiveMulti(base_config)
    
    # Crear Scenario de SMAC
    scenario = Scenario(
        configspace=objective.configspace,
        deterministic=False,
        n_trials=n_trials,
        seed=42,
        output_directory=Path(output_dir),
        name="ga_qtable_optimization_multi",
    )
    
    # Configurar el número de configuraciones iniciales
    n_initial_configs = max(2, int(n_trials * 0.2))
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario, n_configs=n_initial_configs
    )
    
    print(f"\n{'='*70}")
    print(f"INICIANDO OPTIMIZACIÓN CON SMAC3 (MULTIPROCESO)")
    print(f"{'='*70}")
    print(f"Trials a ejecutar: {n_trials}")
    print(f"Modelo surrogate: Random Forest (default de SMAC)")
    print(f"Configuraciones iniciales: {n_initial_configs} (~{(n_initial_configs/n_trials)*100:.0f}% de trials)")
    print(f"Procesos paralelos por trial: {base_config.get('n_jobs', 1)}")
    print(f"Directorio de salida: {output_dir}")
    print(f"{'='*70}\n")
    
    # Crear facade SMAC
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective.train,
        initial_design=initial_design,
        overwrite=True,
    )
    
    # Ejecutar optimización
    incumbent = smac.optimize()
    
    # Validar la configuración incumbente
    incumbent_cost = smac.validate(incumbent)
    incumbent_win_rate = 100.0 * (1.0 - incumbent_cost)
    
    # Mostrar resultados
    print(f"\n{'='*70}")
    print(f"OPTIMIZACIÓN CON SMAC3 COMPLETADA")
    print(f"{'='*70}")
    print(f"Mejor win_rate (incumbente validado): {incumbent_win_rate:.2f}%")
    print(f"Mejor trial interno: {objective.best_trial} (win_rate: {objective.best_win_rate:.2f}%)")
    print(f"\nMejores hiperparámetros (incumbente):")
    for key, value in dict(incumbent).items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    # Guardar resultados en JSON
    results_file = f"smac_multi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        'best_win_rate_validated': incumbent_win_rate,
        'best_win_rate_internal': objective.best_win_rate,
        'best_trial_internal': objective.best_trial,
        'incumbent_params': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in dict(incumbent).items()},
        'n_trials': n_trials,
        'n_jobs': base_config.get('n_jobs', 1),
        'output_dir': output_dir,
        'history': objective.results_history,
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"Resultados guardados en: {results_file}\n")
    
    return incumbent, objective, smac


def main():
    """
    Función principal para ejecutar la optimización de Q-table con algoritmo genético.
    
    Versión con soporte para multiproceso.
    
    Soporta dos modos:
    1. Modo estándar: Ejecuta GA con hiperparámetros fijos
    2. Modo SMAC: Optimiza hiperparámetros del GA usando SMAC3
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimizador de Q-table con Algoritmo Genético (Multiproceso)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  # Modo estándar con 4 procesos paralelos
  python ga_smac_multi.py --agent_script path/to/q_agent_feature_based.py --n_jobs 4

  # Modo SMAC con multiproceso
  python ga_smac_multi.py --agent_script path/to/q_agent_feature_based.py --smac --n_jobs 4 --smac_trials 15
  
  # Usar todos los CPUs disponibles
  python ga_smac_multi.py --agent_script path/to/q_agent_feature_based.py --n_jobs -1
        """
    )
    
    parser.add_argument("--agent_script", 
                       help="Ruta al script q_agent_feature_based.py",
                       required=True,
                       type=str)
    parser.add_argument("--actions",
                       help="Ruta al archivo registration_info.json con las acciones",
                       default="registration_info.json",
                       type=str)
    parser.add_argument("--states",
                       help="Ruta al archivo con los estados específicos (JSON)",
                       default="estadosSMALL.json",
                       type=str)
    parser.add_argument("--host",
                       help="Host del servidor del juego",
                       default="127.0.0.1",
                       type=str)
    parser.add_argument("--port",
                       help="Puerto del servidor del juego",
                       default=9000,
                       type=int)
    parser.add_argument("--test_episodes",
                       help="Número de episodios para evaluar cada individuo",
                       default=100,
                       type=int)
    parser.add_argument("--population",
                       help="Tamaño de la población del GA",
                       default=20,
                       type=int)
    parser.add_argument("--generations",
                       help="Número de generaciones del GA",
                       default=10,
                       type=int)
    parser.add_argument("--sbx_prob",
                       help="Probabilidad de cruzamiento SBX",
                       default=0.9,
                       type=float)
    parser.add_argument("--sbx_eta",
                       help="Índice de distribución SBX",
                       default=15.0,
                       type=float)
    parser.add_argument("--pm_prob_var",
                       help="Probabilidad por variable de mutación PM",
                       default=0.1,
                       type=float)
    parser.add_argument("--pm_eta",
                       help="Índice de distribución PM",
                       default=20.0,
                       type=float)
    parser.add_argument("--reward_range",
                       help="Rango de valores para Q-values (min,max)",
                       default="-1,1",
                       type=str)
    parser.add_argument("--output",
                       help="Nombre del archivo para guardar la Q-table optimizada",
                       default="optimized_q_table_ga_multi.pickle",
                       type=str)
    parser.add_argument("--plot",
                       help="Ruta donde guardar la gráfica de progreso (opcional)",
                       default=None,
                       type=str)
    parser.add_argument("--n_jobs",
                       help="Número de procesos paralelos (1=secuencial, -1=todos los CPUs disponibles)",
                       default=1,
                       type=int)
    # Argumentos SMAC
    parser.add_argument("--smac",
                       help="Activar optimización de hiperparámetros con SMAC3",
                       action='store_true')
    parser.add_argument("--smac_trials",
                       help="Número de trials para SMAC3",
                       default=10,
                       type=int)
    parser.add_argument("--smac_output_dir",
                       help="Directorio de salida para SMAC3",
                       default="smac_output_multi",
                       type=str)
    parser.add_argument("--skip_final_training",
                       help="Omitir el entrenamiento final con mejores hiperparámetros (solo en modo SMAC)",
                       action='store_true')
    
    args = parser.parse_args()
    
    # Parsear reward_range
    reward_min, reward_max = map(float, args.reward_range.split(','))
    reward_range = (reward_min, reward_max)
    
    # Verificar archivos
    if not os.path.exists(args.agent_script):
        print(f"ERROR: No se encuentra el script del agente: {args.agent_script}")
        sys.exit(1)
    
    if not os.path.exists(args.actions):
        print(f"ADVERTENCIA: No se encuentra el archivo de acciones: {args.actions}")
    
    if not os.path.exists(args.states):
        print(f"ADVERTENCIA: No se encuentra el archivo de estados: {args.states}")
    
    print("\n" + "="*70)
    print("OPTIMIZADOR DE Q-TABLE CON ALGORITMO GENÉTICO (MULTIPROCESO)")
    print("Para NetSecGame - Q-Learning Agent")
    print("="*70)
    print(f"Procesos paralelos: {args.n_jobs} (-1 = todos los CPUs)")
    if args.n_jobs == -1:
        print(f"CPUs disponibles: {cpu_count()}")
    print("="*70)
    
    # ================================================================
    # Modo SMAC: Optimización de hiperparámetros con SMAC3
    # ================================================================
    if args.smac:
        print("\nMODO: Optimización de hiperparámetros con SMAC3 (Multiproceso)\n")
        
        base_config = {
            'agent_script_path': args.agent_script,
            'host': args.host,
            'port': args.port,
            'actions_file': args.actions,
            'states_file': args.states,
            'n_jobs': args.n_jobs
        }
        
        # Ejecutar optimización con SMAC3
        incumbent, objective, smac_instance = optimize_with_smac_multi(
            base_config=base_config,
            n_trials=args.smac_trials,
            output_dir=args.smac_output_dir
        )
        
        # Entrenar modelo final con mejores hiperparámetros (opcional)
        if not args.skip_final_training:
            print(f"\n{'='*70}")
            print("ENTRENANDO MODELO FINAL CON MEJORES HIPERPARÁMETROS (SMAC)")
            print(f"{'='*70}\n")
            
            best_params = dict(incumbent)
            optimizer = QTableGeneticOptimizerMulti(
                agent_script_path=args.agent_script,
                host=args.host,
                port=args.port,
                test_episodes=25,
                reward_range=(-1, 1),
                actions_file=args.actions,
                states_file=args.states,
                n_jobs=args.n_jobs
            )
            
            # Ejecutar optimización final
            optimizer.optimize(
                population_size=best_params['population_size'],
                n_generations=best_params['n_generations'],
                sbx_prob=best_params['sbx_prob'],
                sbx_eta=best_params['sbx_eta'],
                pm_prob_var=best_params['pm_prob_var'],
                pm_eta=best_params['pm_eta'],
                verbose=True
            )
            
            # Guardar modelo final
            final_output = args.output.replace('.pickle', '_smac_best.pickle')
            optimizer.save_q_table(final_output)
            optimizer.plot_optimization_progress(save_path=args.plot)
        
        print(f"\n{'='*70}")
        print("OPTIMIZACIÓN CON SMAC3 COMPLETADA")
        print(f"{'='*70}\n")
    
    # ================================================================
    # Modo normal: Ejecutar GA con hiperparámetros fijos
    # ================================================================
    else:
        print("\nMODO: Ejecución estándar con hiperparámetros fijos (Multiproceso)\n")
        
        # Crear el optimizador
        optimizer = QTableGeneticOptimizerMulti(
            agent_script_path=args.agent_script,
            host=args.host,
            port=args.port,
            test_episodes=args.test_episodes,
            reward_range=reward_range,
            actions_file=args.actions,
            states_file=args.states,
            n_jobs=args.n_jobs
        )
        
        # Ejecutar optimización
        optimizer.optimize(
            population_size=args.population,
            n_generations=args.generations,
            sbx_prob=args.sbx_prob,
            sbx_eta=args.sbx_eta,
            pm_prob_var=args.pm_prob_var,
            pm_eta=args.pm_eta,
            verbose=True
        )
        
        # Guardar la Q-table optimizada
        optimizer.save_q_table(args.output)
        
        # Mostrar progreso de optimización
        optimizer.plot_optimization_progress(save_path=args.plot)
        
        print("\n" + "="*70)
        print("INSTRUCCIONES PARA USAR LA Q-TABLE OPTIMIZADA")
        print("="*70)
        print(f"1. La Q-table ha sido guardada en: {args.output}")
        print(f"2. Para usarla en el agente, cárgala con:")
        print(f"   python q_agent_feature_based.py --previous_model {args.output} ...")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Ejecutar optimización
    try:
        main()
        print("\n✓ Optimización completada exitosamente!")
    except KeyboardInterrupt:
        print("\n\n✗ Optimización interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error durante la optimización: {e}")
        traceback.print_exc()
        sys.exit(1)
