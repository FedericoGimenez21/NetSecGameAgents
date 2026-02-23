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
import threading
import time
import socket as _socket
import queue as stdlib_queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
from ConfigSpace import ConfigurationSpace, Configuration, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory import RunHistory
import logging
from datetime import datetime


class QTableOptimizationProblem(Problem):
    """
    Problema de optimización para inicializar una Q-table usando algoritmos genéticos.
    
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
                 n_workers=1,
                 required_players=6,
                 task_config_path=None):
        """
        Inicializa el problema de optimización de Q-table.
        
        Args:
            agent_script_path (str): Ruta al script q_agent_feature_based.py
            host (str): Host del servidor del juego
            port (int): Puerto base. En modo multi-servidor se usan port, port+1, ...
            test_episodes (int): Número de episodios para evaluar cada Q-table
            reward_range (tuple): Rango de valores para los Q-values
            actions_file (str): Ruta al archivo registration_info.json con las acciones
            states_file (str): Ruta al archivo con los estados específicos
            n_workers (int): Número de workers paralelos para evaluar individuos
            required_players (int): (Legado) Solo se usa cuando task_config_path=None.
            task_config_path (str): Ruta al netsecenv_conf.yaml. Si se provee, el GA
                                    gestiona automáticamente N instancias de NetSecGame
                                    en puertos consecutivos (modo multi-servidor).
        """
        self.agent_script_path = agent_script_path
        self.host = host
        self.port = port
        self.test_episodes = test_episodes
        self.reward_range = reward_range
        self.actions_file = actions_file
        self.states_file = states_file
        self.n_workers = max(1, n_workers)
        self.required_players = max(1, required_players)
        self.task_config_path = task_config_path
        self._print_lock = threading.Lock()
        self._server_procs = None   # Servidores persistentes (ciclo de vida externo)
        self._port_pool = None      # Pool de puertos para workers
        
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
        print(f"Workers paralelos para evaluación: {self.n_workers}")
        if self.task_config_path:
            print(f"Modo multi-servidor: {self.n_workers} instancia(s) de NetSecGame")
            print(f"  Config: {self.task_config_path}")
            print(f"  Puertos: {self.port} – {self.port + self.n_workers - 1}")
        else:
            print(f"Modo servidor externo (required_players={self.required_players})")

    # ----------------------------------------------------------------
    # Soporte de pickling: threading.Lock no es serializable, se omite
    # en __getstate__ y se recrea en __setstate__.
    # ----------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_print_lock', None)
        state.pop('_server_procs', None)
        state.pop('_port_pool', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._print_lock = threading.Lock()
        self._server_procs = None
        self._port_pool = None
        if 'required_players' not in state:
            self.required_players = 6
        if 'task_config_path' not in state:
            self.task_config_path = None

    def _load_actions(self):
        """Carga las acciones desde registration_info.json y las convierte a objetos Action"""
        try:
            json_raw = open("registration_info.json").read()

            outer = json.loads(json_raw)          # convierte primer nivel
            actions_dicts = json.loads(outer["all_actions"])  # convierte el string interno


            print("Cantidad de acciones:", len(actions_dicts))
            print("Ejemplo:", actions_dicts[0])

            if not actions_dicts:
                print(f"ADVERTENCIA: No se encontraron acciones en {self.actions_file}")
                sys.exit(0)
                
            print(f"Cargadas {len(actions_dicts)} acciones desde {self.actions_file}")

            # Convertir diccionarios a objetos Action
            actions = []
            for action_dict in actions_dicts:
                if 'action_type' in action_dict and 'parameters' in action_dict:
                    # Convertir action_type string a ActionType enum
                    # Limpiar el prefijo 'ActionType.' si existe
                    action_type_str = action_dict['action_type'].replace('ActionType.', '')
                    action_type = ActionType[action_type_str]
                    
                    # Convertir parámetros a estructura hashable
                    params = self._make_params_hashable(action_dict['parameters'])
                    action = Action(action_type, parameters=params)
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
                # Convertir diccionarios a tuplas de items ordenados
                hashable_params[key] = tuple(sorted(value.items()))
            elif isinstance(value, list):
                # Convertir listas a tuplas
                hashable_params[key] = tuple(value)
            else:
                hashable_params[key] = value
        return hashable_params
    
    def _load_states(self):
        """Carga los estados específicos desde el archivo JSON"""
        try:
            with open(self.states_file, 'r') as f:
                data = json.load(f)
                states_list = data.get('states', [])
                
                # Convertir strings de tuplas a tuplas reales
                states = []
                for state_str in states_list:
                    # Parsear "(4, 8, 4, 7, 2)" -> (4, 8, 4, 7, 2)
                    state_tuple = eval(state_str)
                    states.append(state_tuple)
                
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
                # Usar el objeto Action en lugar del índice
                action = self.actions[action_idx]
                q_table[(state, action)] = q_table_matrix[state_idx, action_idx]
        
        # Crear estructura compatible con q_agent_feature_based.py
        # El agente solo usa 'q_table' y 'state_mapping'
        q_table_data = {
            'q_table': q_table,           # Diccionario {(state_tuple, Action): q_value}
            'state_mapping': {}            # Vacío, el agente lo construye dinámicamente
        }
        
        return q_table_data

    # ----------------------------------------------------------------
    # Gestión de servidores NetSecGame (modo multi-servidor)
    # ----------------------------------------------------------------

    def _start_game_servers(self, n_servers):
        """
        Lanza n_servers instancias del coordinator NetSecGame en puertos
        consecutivos comenzando desde self.port.

        Returns:
            dict: {port: subprocess.Popen}
        """
        server_procs = {}
        for i in range(n_servers):
            port = self.port + i
            cmd = [
                sys.executable, '-m', 'AIDojoCoordinator.worlds.NSEGameCoordinator',
                f'--task_config={self.task_config_path}',
                f'--game_port={port}',
                f'--game_host={self.host}',
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            server_procs[port] = proc
            with self._print_lock:
                print(f"  [Servidor] Puerto {port} iniciado (PID {proc.pid})")
        return server_procs

    def _stop_game_servers(self, server_procs):
        """Termina todas las instancias del coordinator lanzadas por este proceso."""
        for port, proc in server_procs.items():
            try:
                proc.terminate()
                proc.wait(timeout=15)
                with self._print_lock:
                    print(f"  [Servidor] Puerto {port} detenido.")
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _wait_for_server(self, port, timeout=60):
        """
        Espera hasta que el servidor TCP en el puerto dado acepte conexiones.

        Returns:
            bool: True si respondió antes del timeout, False en caso contrario.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with _socket.create_connection((self.host, port), timeout=1):
                    return True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        return False

    def start_servers(self):
        """
        Inicia los servidores NetSecGame una sola vez y prepara el pool de puertos.
        Debe llamarse antes de iniciar la optimización. Los servidores permanecen
        activos hasta que se llame a stop_servers().
        """
        if self.task_config_path is None:
            return
        n_servers = self.n_workers
        self._port_pool = stdlib_queue.Queue()
        for i in range(n_servers):
            self._port_pool.put(self.port + i)
        print(
            f"\nIniciando {n_servers} instancia(s) de NetSecGame "
            f"(puertos {self.port}\u2013{self.port + n_servers - 1})..."
        )
        self._server_procs = self._start_game_servers(n_servers)
        print("Esperando a que los servidores estén listos...")
        for i in range(n_servers):
            p = self.port + i
            ok = self._wait_for_server(p)
            status = "listo" if ok else "TIMEOUT (se continúa de todas formas)"
            print(f"  Puerto {p}: {status}")
        print("Servidores listos. Se reutilizarán en todas las generaciones.\n")

    def stop_servers(self):
        """Detiene todos los servidores iniciados con start_servers()."""
        if self._server_procs:
            print("\nDeteniendo instancias de NetSecGame...")
            self._stop_game_servers(self._server_procs)
            self._server_procs = None
            self._port_pool = None

    def _evaluate_single(self, idx, x, total, port=None):
        """
        Evalúa un único individuo de la población.

        Args:
            idx (int): Índice del individuo
            x (np.ndarray): Parámetros del individuo
            total (int): Tamaño total de la población (para logging)

        Returns:
            tuple: (idx, fitness)
        """
        with self._print_lock:
            print(f"\nEvaluando individuo {idx + 1}/{total}...")
        q_table_path = None
        try:
            q_table_data = self._create_q_table_from_params(x)
            tmp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pickle', delete=False)
            q_table_path = tmp_file.name
            pickle.dump(q_table_data, tmp_file)
            tmp_file.close()
            fitness = self._run_agent_evaluation(q_table_path, port=port)
            with self._print_lock:
                print(f"  Individuo {idx + 1}/{total} -> Fitness: {fitness:.4f}")
            return idx, fitness
        except Exception as e:
            with self._print_lock:
                print(f"  Error evaluando individuo {idx + 1}: {e}")
            return idx, 1000.0
        finally:
            if q_table_path and os.path.exists(q_table_path):
                try:
                    os.remove(q_table_path)
                except Exception:
                    pass

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evalúa la calidad de las Q-tables candidatas ejecutándolas en el agente.

        Modos de evaluación:

        **Modo multi-servidor** (task_config_path provisto):
          El GA lanza N instancias independientes de NetSecGame en puertos
          consecutivos (port, port+1, ...). Los individuos se distribuyen sobre
          un pool de puertos y se evalúan con paralelismo real sin batching.
          Los servidores se inician al principio de cada generación y se detienen
          al finalizar, incluso ante errores.

        **Modo servidor externo** (task_config_path=None):
          - n_workers=1: secuencial en self.port.
          - n_workers>1: lotes de required_players (legado, para required_players>1).
        """
        total = len(X)
        objectives = [None] * total

        if self.task_config_path is not None:
            # --------------------------------------------------------
            # Modo multi-servidor: usa los servidores ya iniciados por
            # QTableGeneticOptimizer.optimize() via start_servers().
            # El ciclo de vida (inicio/parada) es externo a _evaluate.
            # --------------------------------------------------------
            port_pool = self._port_pool
            n_servers = len(self._server_procs) if self._server_procs else self.n_workers

            def _worker_with_port(idx):
                port = port_pool.get()
                try:
                    return self._evaluate_single(idx, X[idx], total, port=port)
                finally:
                    port_pool.put(port)

            print(f"\nEvaluando {total} individuos con {n_servers} servidor(es) en paralelo...")
            with ThreadPoolExecutor(max_workers=n_servers) as executor:
                futures = {
                    executor.submit(_worker_with_port, idx): idx
                    for idx in range(total)
                }
                for future in as_completed(futures):
                    idx, fitness = future.result()
                    objectives[idx] = fitness

        elif self.n_workers == 1:
            # --------------------------------------------------------
            # Servidor externo, evaluación secuencial
            # --------------------------------------------------------
            for idx, x in enumerate(X):
                _, fitness = self._evaluate_single(idx, x, total)
                objectives[idx] = fitness
        else:
            # --------------------------------------------------------
            # Servidor externo, lotes de required_players (legado)
            # --------------------------------------------------------
            batch_size = min(self.n_workers, self.required_players)
            n_batches = (total + batch_size - 1) // batch_size
            print(
                f"\nEvaluando {total} individuos en {n_batches} lote(s) "
                f"de hasta {batch_size} workers "
                f"(required_players={self.required_players})..."
            )
            for batch_num in range(n_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, total)
                batch_indices = list(range(batch_start, batch_end))
                print(
                    f"  Lote {batch_num + 1}/{n_batches}: "
                    f"individuos {batch_start + 1}–{batch_end} de {total}"
                )
                with ThreadPoolExecutor(max_workers=len(batch_indices)) as executor:
                    futures = {
                        executor.submit(self._evaluate_single, idx, X[idx], total): idx
                        for idx in batch_indices
                    }
                    for future in as_completed(futures):
                        idx, fitness = future.result()
                        objectives[idx] = fitness
                print(f"  Lote {batch_num + 1}/{n_batches} completado.")

        out["F"] = np.array(objectives)
    
    def _run_agent_evaluation(self, q_table_path, port=None):
        """
        Ejecuta el agente con la Q-table proporcionada y extrae el fitness.
        
        Args:
            q_table_path (str): Ruta al archivo pickle con la Q-table
            port (int): Puerto del servidor a usar. None usa self.port.
            
        Returns:
            float: Fitness (menor es mejor). Usamos: -win_rate - avg_return/100
        """
        effective_port = port if port is not None else self.port
        # Comando para ejecutar el agente en modo testing
        cmd = [
            sys.executable,  # Python interpreter
            self.agent_script_path,
            "--host", self.host,
            "--port", str(effective_port),
            "--episodes", str(self.test_episodes),
            "--testing", "True",
            "--previous_model", q_table_path,
            "--experiment_id", "ga_eval",
        ]
        
        try:
            # Ejecutar el agente
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=400  # Timeout de 5 minutos
            )
            
            # Parsear la salida para extraer métricas
            output = result.stdout + result.stderr
            
            # Buscar win rate en la salida (porcentaje 0-100)
            win_rate = self._extract_metric(output, "winrate")
            
            # Fitness: minimizar (maximizar win_rate)
            # Convertimos a problema de minimización con signo negativo
            # win_rate está en porcentaje (0-100), así que -win_rate será negativo
            print(f"win_rate: {win_rate}%")
            fitness = -win_rate
            
            return fitness
            
        except subprocess.TimeoutExpired:
            print("  Timeout en evaluación del agente")
            return 1000.0
        except Exception as e:
            print(f"  Error ejecutando agente: {e}")
            return 1000.0
    
    def _extract_metric(self, output, metric_name):
        """
        Extrae una métrica de la salida del agente.
        
        Args:
            output (str): Salida del agente
            metric_name (str): Nombre de la métrica a extraer
            
        Returns:
            float: Valor de la métrica (0 si no se encuentra)
        """
        try:
            # Buscar líneas que contienen la métrica
            for line in output.split('\n'):
                if metric_name in line:
                    # Extraer el valor numérico
                    parts = line.split('=')
                    if len(parts) > 1:
                        value_str = parts[1].split()[0].strip('%,')
                        return float(value_str)
        except:
            pass
        return 0.0


class QTableGeneticOptimizer:
    """
    Optimizador de Q-table usando algoritmos genéticos con pymoo.
    
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
                 n_workers=1,
                 required_players=6,
                 task_config_path=None):
        """
        Inicializa el optimizador.
        
        Args:
            agent_script_path (str): Ruta al script q_agent_feature_based.py
            host (str): Host del servidor del juego
            port (int): Puerto base. En modo multi-servidor se usan port, port+1, ...
            test_episodes (int): Número de episodios para evaluar cada Q-table
            reward_range (tuple): Rango de valores para la Q-table
            actions_file (str): Ruta al archivo con las acciones registradas
            states_file (str): Ruta al archivo con los estados específicos
            n_workers (int): Número de workers paralelos para evaluar individuos
            required_players (int): (Legado) Solo se usa cuando task_config_path=None.
            task_config_path (str): Ruta al netsecenv_conf.yaml. Si se provee, el GA
                                    gestiona sus propias instancias de NetSecGame.
        """
        self.agent_script_path = agent_script_path
        self.host = host
        self.port = port
        self.test_episodes = test_episodes
        self.reward_range = reward_range
        self.actions_file = actions_file
        self.states_file = states_file
        self.n_workers = n_workers
        self.required_players = required_players
        self.task_config_path = task_config_path
        
        # Crear el problema de optimización
        self.problem = QTableOptimizationProblem(
            agent_script_path=agent_script_path,
            host=host,
            port=port,
            test_episodes=test_episodes,
            reward_range=reward_range,
            actions_file=actions_file,
            states_file=states_file,
            n_workers=n_workers,
            required_players=required_players,
            task_config_path=task_config_path
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
        print("OPTIMIZACIÓN DE Q-TABLE CON ALGORITMO GENÉTICO")
        print("="*70)
        print(f"Agente: {self.agent_script_path}")
        print(f"Servidor: {self.host}:{self.port}")
        print(f"Episodios de evaluación: {self.test_episodes}")
        print(f"Población: {population_size}")
        print(f"Generaciones: {n_generations}")
        print(f"Rango de Q-values: {self.reward_range}")
        print(f"SBX prob: {sbx_prob}, eta: {sbx_eta}, prob_var: {sbx_prob_var} (fijo)")
        print(f"PM prob: {pm_prob} (fijo), prob_var: {pm_prob_var}, eta: {pm_eta}")
        print(f"Workers paralelos: {self.n_workers}")

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
        self.problem.start_servers()
        try:
            res = minimize(
                self.problem,
                algorithm,
                termination,
                verbose=verbose,
                save_history=True
            )
        finally:
            self.problem.stop_servers()
        
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
    
    def save_q_table(self, filename="optimized_q_table_ga.pickle"):
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
        plt.title('Progreso de Optimización de Q-table', fontsize=14, fontweight='bold')
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
            # Convertir a float para evitar problemas con arrays de numpy
            initial_fitness = float(self.optimization_history[0])
            final_fitness = float(self.optimization_history[-1])
            print(f"  Fitness inicial: {initial_fitness:.6f}")
            print(f"  Fitness final: {final_fitness:.6f}")
            print(f"  Mejora: {initial_fitness - final_fitness:.6f}")
            print(f"  Win rate estimado final: {-final_fitness:.2f}%")
        
        print("="*70 + "\n")


# ============================================================================
# SMAC3: Optimización Bayesiana de hiperparámetros del GA
# ============================================================================

def setup_smac_logging(log_dir="smac_logs"):
    """Configura logging para SMAC"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"smac_{timestamp}.log")
    
    # Configurar logger
    logger = logging.getLogger("smac_ga")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return log_file, logger


class SMACGAObjective:
    """
    Clase que encapsula la función objetivo para SMAC3.
    
    SMAC3 requiere una función target_function(config, seed) -> cost.
    Esta clase almacena la configuración base y los mejores resultados.
    """
    
    def __init__(self, base_config):
        """
        Args:
            base_config (dict): Configuración base con parámetros fijos
                - agent_script_path, host, port, actions_file, states_file
                - n_workers (opcional): número de workers paralelos
        """
        self.base_config = base_config
        self.trial_count = 0
        self.best_win_rate = 0.0
        self.best_trial = -1
        self.results_history = []
        self.n_workers = base_config.get('n_workers', 1)
        self.required_players = base_config.get('required_players', 6)
        self.task_config_path = base_config.get('task_config_path', None)
    
    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Define el espacio de configuración (hiperparámetros) para SMAC.
        
        Hiperparámetros optimizados:
          - population_size: tamaño de la población del GA [2, 4]
          - n_generations: número de generaciones del GA [2, 4]
          - sbx_prob: probabilidad de cruzamiento SBX [0.8, 1.0]
          - sbx_eta: índice de distribución SBX [0, 60]
          - pm_prob_var: probabilidad por variable de mutación PM [0.01, 0.6]
          - pm_eta: índice de distribución PM [0, 60]
        
        Valores fijos (no varían en el análisis):
          - sbx_prob_var = 1.0
          - pm_prob = 1.0
          - test_episodes = 1
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
        
        Crea un QTableGeneticOptimizer con los hiperparámetros de 'config',
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
        
        print(f"\n{'='*70}")
        print(f"SMAC TRIAL {trial_num}")
        print(f"{'='*70}")
        print(f"Hiperparámetros sugeridos:")
        print(f"  population_size: {population_size}")
        print(f"  n_generations: {n_generations}")
        print(f"  sbx_prob: {sbx_prob:.4f}")
        print(f"  sbx_eta: {sbx_eta:.2f}")
        print(f"  pm_prob_var: {pm_prob_var:.4f}")
        print(f"  pm_eta: {pm_eta:.2f}")
        print(f"  test_episodes: {test_episodes} (fijo)")
        print(f"  seed: {seed}")
        print(f"{'='*70}\n")
        
        try:
            # Crear optimizador con hiperparámetros sugeridos
            optimizer = QTableGeneticOptimizer(
                agent_script_path=self.base_config['agent_script_path'],
                host=self.base_config['host'],
                port=self.base_config['port'],
                test_episodes=test_episodes,
                reward_range=reward_range,
                actions_file=self.base_config['actions_file'],
                states_file=self.base_config['states_file'],
                n_workers=self.n_workers,
                required_players=self.required_players,
                task_config_path=self.task_config_path
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
            # De esta forma: win_rate=100% -> cost=0.0, win_rate=0% -> cost=1.0
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
                output_file = f"smac_trial_{trial_num}_wr_{best_win_rate:.1f}.pickle"
                optimizer.save_q_table(output_file)
                print(f"  -> Nuevo mejor resultado! Q-table guardada en: {output_file}")
            
            print(f"\nTrial {trial_num} completado: win_rate = {best_win_rate:.2f}%, cost = {cost:.4f}\n")
            
            return cost
            
        except Exception as e:
            print(f"\nError en trial {trial_num}: {e}\n")
            import traceback
            traceback.print_exc()
            return 1.0  # Penalización máxima (win_rate = 0%)


def optimize_with_smac(base_config, n_trials=20, output_dir="smac_output", n_workers=1):
    """
    Ejecuta optimización de hiperparámetros con SMAC3.
    
    SMAC3 usa optimización Bayesiana con Random Forest como modelo surrogate,
    combinado con racing agresivo para decidir eficientemente qué configuraciones
    son mejores.
    
    El diseño inicial usa 20% de n_trials (mínimo 2) para exploración aleatoria
    inicial, lo cual es una práctica recomendada en optimización bayesiana.
    
    IMPORTANTE: SMAC3 puede ejecutar más evaluaciones que n_trials debido a:
    - Configuraciones del diseño inicial (20% de n_trials)
    - Configuraciones por defecto automáticas
    - Validación de la configuración incumbente al final
    
    Para n_trials muy bajos (< 10), el número real de evaluaciones puede ser
    significativamente mayor. Se recomienda usar n_trials >= 10.
    
    Args:
        base_config: Diccionario con configuración base (agent_script, host, port, etc.)
        n_trials: Número de trials (evaluaciones) objetivo (el número real puede ser mayor)
        output_dir: Directorio para resultados de SMAC
        
    Returns:
        tuple: (incumbent_config, objective_instance, smac_instance)
    """
    # Configurar logging
    log_file, logger = setup_smac_logging()
    print(f"SMAC logs guardados en: {log_file}\n")
    
    # Inyectar n_workers en la configuración base si no está presente
    base_config = {**base_config, 'n_workers': n_workers}

    # Crear instancia del objetivo
    objective = SMACGAObjective(base_config)
    
    # Crear Scenario de SMAC
    scenario = Scenario(
        configspace=objective.configspace,
        deterministic=False,       # La evaluación no es determinista (juego estocástico)
        n_trials=n_trials,         # Número total de evaluaciones
        seed=42,
        output_directory=Path(output_dir),
        name="ga_qtable_optimization",
    )
    
    # Configurar el número de configuraciones iniciales (exploración)
    # Buena práctica: 20% de n_trials para diseño inicial (mínimo 2)
    # Esto balancea exploración inicial vs. optimización bayesiana
    n_initial_configs = max(2, int(n_trials * 0.2))
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario, n_configs=n_initial_configs
    )
    
    print(f"\n{'='*70}")
    print(f"INICIANDO OPTIMIZACIÓN CON SMAC3")
    print(f"{'='*70}")
    print(f"Trials a ejecutar: {n_trials}")
    print(f"Modelo surrogate: Random Forest (default de SMAC)")
    print(f"Configuraciones iniciales: {n_initial_configs} (~{(n_initial_configs/n_trials)*100:.0f}% de trials)")
    print(f"Directorio de salida: {output_dir}")
    print(f"\nNOTA: SMAC3 puede ejecutar trials adicionales debido a:")
    print(f"      - Diseño inicial (warm-up del Random Forest)")
    print(f"      - Validación de la configuración incumbente")
    print(f"      - Configuraciones por defecto del framework")
    print(f"{'='*70}\n")
    
    # Crear facade SMAC
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective.train,
        initial_design=initial_design,
        overwrite=True,  # Sobreescribir ejecuciones previas (o usar False para continuar)
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
    results_file = f"smac_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        'best_win_rate_validated': incumbent_win_rate,
        'best_win_rate_internal': objective.best_win_rate,
        'best_trial_internal': objective.best_trial,
        'incumbent_params': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in dict(incumbent).items()},
        'n_trials': n_trials,
        'output_dir': output_dir,
        'history': objective.results_history,
    }
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"Resultados guardados en: {results_file}\n")
    
    return incumbent, objective, smac


def plot_smac_results(objective, save_dir="smac_plots"):
    """
    Genera visualizaciones de los resultados de SMAC.
    
    Args:
        objective: Instancia de SMACGAObjective con el historial
        save_dir: Directorio donde guardar las gráficas
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not objective.results_history:
        print("No hay historial de resultados para graficar.")
        return
    
    trials = [r['trial'] for r in objective.results_history]
    win_rates = [r['win_rate'] for r in objective.results_history]
    costs = [r['cost'] for r in objective.results_history]
    
    # Calcular mejor acumulado
    best_so_far = []
    current_best = 0.0
    for wr in win_rates:
        current_best = max(current_best, wr)
        best_so_far.append(current_best)
    
    # ================================================================
    # 1. Win rate por trial
    # ================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(trials, win_rates, 'o-', color='steelblue', linewidth=2, markersize=5, label='Win Rate')
    plt.title('Win Rate por Trial (SMAC)', fontsize=13, fontweight='bold')
    plt.xlabel('Trial')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path_1 = os.path.join(save_dir, "smac_win_rate_per_trial.png")
    plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {plot_path_1}")
    plt.close()
    
    # ================================================================
    # 2. Costo por trial
    # ================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(trials, costs, 'o-', color='darkorange', linewidth=2, markersize=5)
    plt.title('Costo (SMAC minimiza) por Trial', fontsize=13, fontweight='bold')
    plt.xlabel('Trial')
    plt.ylabel('Costo')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path_2 = os.path.join(save_dir, "smac_cost_per_trial.png")
    plt.savefig(plot_path_2, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {plot_path_2}")
    plt.close()
    
    # ================================================================
    # 3. Distribución de win rates
    # ================================================================
    plt.figure(figsize=(10, 6))
    plt.hist(win_rates, bins=max(5, len(win_rates)//3), color='mediumseagreen', 
             alpha=0.7, edgecolor='black')
    plt.axvline(max(win_rates), color='red', linestyle='--', linewidth=2, label=f'Mejor: {max(win_rates):.1f}%')
    plt.title('Distribución de Win Rates', fontsize=13, fontweight='bold')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path_3 = os.path.join(save_dir, "smac_win_rate_distribution.png")
    plt.savefig(plot_path_3, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {plot_path_3}")
    plt.close()
    
    # ================================================================
    # 4. Hiperparámetros vs win_rate (scatter pm_prob_var vs sbx_prob)
    # ================================================================
    if objective.results_history:
        plt.figure(figsize=(10, 6))
        pm_prob_vars = [r['config'].get('pm_prob_var', 0) for r in objective.results_history]
        sbx_probs = [r['config'].get('sbx_prob', 0) for r in objective.results_history]
        scatter = plt.scatter(pm_prob_vars, sbx_probs, c=win_rates, 
                            cmap='RdYlGn', s=80, edgecolors='black', linewidths=0.5)
        plt.title('PM prob_var vs SBX prob (color=Win Rate)', fontsize=13, fontweight='bold')
        plt.xlabel('PM prob_var')
        plt.ylabel('SBX prob')
        plt.colorbar(scatter, label='Win Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path_4 = os.path.join(save_dir, "smac_hyperparameters_vs_winrate.png")
        plt.savefig(plot_path_4, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada: {plot_path_4}")
        plt.close()
    
    # ================================================================
    # 5. Análisis individual de cada hiperparámetro vs win_rate
    # ================================================================
    hyperparams = ['population_size', 'n_generations', 'sbx_prob', 'sbx_eta', 'pm_prob_var', 'pm_eta']
    hyperparam_labels = {
        'population_size': 'Population Size',
        'n_generations': 'Number of Generations',
        'sbx_prob': 'SBX Probability',
        'sbx_eta': 'SBX Eta',
        'pm_prob_var': 'PM Prob Var',
        'pm_eta': 'PM Eta'
    }
    
    for param in hyperparams:
        plt.figure(figsize=(10, 6))
        param_values = [r['config'].get(param, 0) for r in objective.results_history]
        
        # Crear scatter plot con línea de tendencia
        plt.scatter(param_values, win_rates, c=win_rates, cmap='RdYlGn', 
                   s=100, alpha=0.6, edgecolors='black', linewidths=0.5)
        
        # Añadir línea de tendencia si hay suficientes datos
        if len(param_values) > 2:
            z = np.polyfit(param_values, win_rates, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(param_values), max(param_values), 100)
            plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Tendencia')
        
        plt.title(f'{hyperparam_labels[param]} vs Win Rate', fontsize=13, fontweight='bold')
        plt.xlabel(hyperparam_labels[param])
        plt.ylabel('Win Rate (%)')
        plt.colorbar(label='Win Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path_param = os.path.join(save_dir, f"smac_{param}_vs_winrate.png")
        plt.savefig(plot_path_param, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada: {plot_path_param}")
        plt.close()
    
    # ================================================================
    # 6. Correlación entre hiperparámetros y win_rate
    # ================================================================
    plt.figure(figsize=(10, 8))
    
    # Preparar datos para correlación
    data_for_corr = []
    for r in objective.results_history:
        row = [r['win_rate']]
        for param in hyperparams:
            row.append(r['config'].get(param, 0))
        data_for_corr.append(row)
    
    data_array = np.array(data_for_corr)
    
    # Calcular matriz de correlación
    corr_matrix = np.corrcoef(data_array.T)
    
    # Crear heatmap
    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Configurar etiquetas
    labels = ['Win Rate'] + [hyperparam_labels[p] for p in hyperparams]
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)
    
    # Añadir valores de correlación en las celdas
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, label='Correlación')
    plt.title('Matriz de Correlación: Hiperparámetros vs Win Rate', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    plot_path_corr = os.path.join(save_dir, "smac_correlation_matrix.png")
    plt.savefig(plot_path_corr, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {plot_path_corr}")
    plt.close()
    
    # ================================================================
    # 7. Evolución de hiperparámetros de los mejores trials
    # ================================================================
    # Ordenar por win_rate y tomar top 5
    top_trials = sorted(objective.results_history, key=lambda x: x['win_rate'], reverse=True)[:min(5, len(objective.results_history))]
    
    if len(top_trials) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, param in enumerate(hyperparams):
            ax = axes[idx]
            
            # Valores para todos los trials
            all_values = [r['config'].get(param, 0) for r in objective.results_history]
            all_trials_num = [r['trial'] for r in objective.results_history]
            
            # Valores para top trials
            top_values = [r['config'].get(param, 0) for r in top_trials]
            top_trials_num = [r['trial'] for r in top_trials]
            top_winrates = [r['win_rate'] for r in top_trials]
            
            # Scatter de todos los trials
            ax.scatter(all_trials_num, all_values, c='lightgray', s=50, alpha=0.5, label='Todos')
            
            # Destacar top trials
            scatter = ax.scatter(top_trials_num, top_values, c=top_winrates, 
                               cmap='RdYlGn', s=150, edgecolors='black', 
                               linewidths=1.5, label='Top 5', zorder=5)
            
            ax.set_title(hyperparam_labels[param], fontsize=11, fontweight='bold')
            ax.set_xlabel('Trial')
            ax.set_ylabel(hyperparam_labels[param])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path_evolution = os.path.join(save_dir, "smac_hyperparams_evolution.png")
        plt.savefig(plot_path_evolution, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada: {plot_path_evolution}")
        plt.close()
    
    # ================================================================
    # 8. Box plots de win_rate por rangos de hiperparámetros (un archivo por param)
    # ================================================================
    n_trials_total = len(objective.results_history)
    from matplotlib.lines import Line2D

    # Número de bins adaptativo según cantidad de trials
    if n_trials_total < 20:
        n_bins = 4
    elif n_trials_total < 50:
        n_bins = 5
    elif n_trials_total < 100:
        n_bins = 6
    else:
        n_bins = 8

    use_notch  = (n_trials_total >= 40)
    show_points = (n_trials_total <= 200)

    plot_paths_boxplots = []

    for param in hyperparams:
        param_values  = np.array([r['config'].get(param, 0) for r in objective.results_history])
        win_rates_arr = np.array(win_rates)

        fig, ax = plt.subplots(figsize=(10, 6))

        if len(set(param_values)) > 1:
            # Bins por percentiles equiespaciados → cada bin ≈ mismo nº de trials
            percentile_edges = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.unique(np.percentile(param_values, percentile_edges))
            actual_bins = len(bin_edges) - 1

            groups_data    = []
            groups_labels  = []
            groups_medians = []

            for b in range(actual_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                mask = (param_values >= lo) & (param_values <= hi) \
                       if b == actual_bins - 1 \
                       else (param_values >= lo) & (param_values < hi)
                group_wr = win_rates_arr[mask]
                if len(group_wr) == 0:
                    continue
                groups_data.append(group_wr)
                lbl = f'{lo:.3g}' if lo == hi else f'[{lo:.3g}, {hi:.3g}]'
                groups_labels.append(f'{lbl}\n(n={len(group_wr)})')
                groups_medians.append(np.median(group_wr))

            if groups_data:
                cmap = plt.cm.RdYlGn
                med_min, med_max = min(groups_medians), max(groups_medians)
                med_range = med_max - med_min if med_max != med_min else 1.0
                box_colors = [cmap((m - med_min) / med_range) for m in groups_medians]

                bp = ax.boxplot(
                    groups_data,
                    labels=groups_labels,
                    patch_artist=True,
                    notch=use_notch and all(len(g) >= 5 for g in groups_data),
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='navy',
                                   markeredgecolor='navy', markersize=6),
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4,
                                    markerfacecolor='gray'),
                )
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)

                # Jitter overlay
                if show_points:
                    rng = np.random.default_rng(seed=42)
                    for i, group_wr in enumerate(groups_data, start=1):
                        jitter = rng.uniform(-0.18, 0.18, size=len(group_wr))
                        ax.scatter(
                            np.full(len(group_wr), i) + jitter,
                            group_wr,
                            alpha=0.35, s=14, color='steelblue', zorder=3
                        )

        ax.set_title(
            f'{hyperparam_labels[param]} vs Win Rate\n'
            f'({n_trials_total} trials · {n_bins} bins por percentil)',
            fontsize=13, fontweight='bold'
        )
        ax.set_xlabel(hyperparam_labels[param], fontsize=11)
        ax.set_ylabel('Win Rate (%)', fontsize=11)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='navy',
                   markersize=7, label='Media'),
            Line2D([0], [0], color='black', linewidth=2, label='Mediana'),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

        plt.tight_layout()
        plot_path_bp = os.path.join(save_dir, f"smac_boxplot_{param}.png")
        plt.savefig(plot_path_bp, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada: {plot_path_bp}")
        plt.close()
        plot_paths_boxplots.append(plot_path_bp)

    print(f"\nTodas las gráficas guardadas en: {save_dir}/")
    print(f"  - {os.path.basename(plot_path_1)}")
    print(f"  - {os.path.basename(plot_path_2)}")
    print(f"  - {os.path.basename(plot_path_3)}")
    if objective.results_history:
        print(f"  - {os.path.basename(plot_path_4)}")
    print(f"  - 6 gráficos de hiperparámetros individuales vs win_rate (scatter)")
    print(f"  - {os.path.basename(plot_path_corr)}")
    if len(top_trials) > 1:
        print(f"  - {os.path.basename(plot_path_evolution)}")
    for pp in plot_paths_boxplots:
        print(f"  - {os.path.basename(pp)}")
    
    # Tabla resumen
    print(f"\n{'='*70}")
    print("RESUMEN DE TODOS LOS TRIALS (SMAC)")
    print(f"{'='*70}")
    print(f"{'Trial':>6} | {'Win Rate':>10} | {'Costo':>8} | {'Pop':>4} | {'Gens':>4} | {'SBX_P':>7} | {'SBX_eta':>8} | {'PM_pv':>7} | {'PM_eta':>8}")
    print("-" * 95)
    for r in sorted(objective.results_history, key=lambda x: x['win_rate'], reverse=True):
        c = r['config']
        print(f"{r['trial']:>6} | {r['win_rate']:>9.2f}% | {r['cost']:>8.4f} | "
              f"{c.get('population_size', '?'):>4} | {c.get('n_generations', '?'):>4} | "
              f"{c.get('sbx_prob', 0):>7.4f} | {c.get('sbx_eta', 0):>8.2f} | "
              f"{c.get('pm_prob_var', 0):>7.4f} | {c.get('pm_eta', 0):>8.2f}")
    print(f"{'='*70}\n")


def main():
    """
    Función principal para ejecutar la optimización de Q-table con algoritmo genético.
    
    Soporta dos modos:
    1. Modo estándar: Ejecuta GA con hiperparámetros fijos
    2. Modo SMAC: Optimiza hiperparámetros del GA usando SMAC3
       (Bayesian Optimization con Random Forest)
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimizador de Q-table con Algoritmo Genético para NetSecGame (SMAC3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  # Modo estándar (GA con hiperparámetros fijos)
  python ga_smac.py --agent_script path/to/q_agent_feature_based.py --actions registration_info.json

  # Modo SMAC (optimización de hiperparámetros con SMAC3)
  python ga_smac.py --agent_script path/to/q_agent_feature_based.py --smac --smac_trials 15
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
                       default="optimized_q_table_ga.pickle",
                       type=str)
    parser.add_argument("--plot",
                       help="Ruta donde guardar la gráfica de progreso (opcional)",
                       default=None,
                       type=str)
    # Argumentos SMAC
    parser.add_argument("--smac",
                       help="Activar optimización de hiperparámetros con SMAC3",
                       action='store_true')
    parser.add_argument("--smac_trials",
                       help="Número de trials para SMAC3 (recomendado: 10-20). NOTA: 20%% se usa para diseño inicial + SMAC puede ejecutar trials adicionales",
                       default=10,
                       type=int)
    parser.add_argument("--smac_output_dir",
                       help="Directorio de salida para SMAC3",
                       default="smac_output",
                       type=str)
    parser.add_argument("--skip_final_training",
                       help="Omitir el entrenamiento final con mejores hiperparámetros (solo en modo SMAC)",
                       action='store_true')
    parser.add_argument("--workers",
                       help="Número de workers paralelos para evaluar individuos de la población. "
                            "Cada worker lanza un subprocess del agente. "
                            f"Por defecto 1 (secuencial). Máximo recomendado: {multiprocessing.cpu_count()}.",
                       default=1,
                       type=int)
    parser.add_argument("--required_players",
                       help="(Legado) Jugadores requeridos por el servidor. Solo se usa cuando "
                            "--task_config no se proporciona. Por defecto 1.",
                       default=1,
                       type=int)
    parser.add_argument("--task_config",
                       help="Ruta al netsecenv_conf.yaml. Si se provee, el GA gestiona "
                            "automáticamente N instancias de NetSecGame en puertos consecutivos "
                            "(modo multi-servidor). Cada worker tiene su propio servidor, "
                            "permitiendo paralelismo real sin batching ni esperas. "
                            "Ejemplo: --task_config ./AIDojoCoordinator/netsecenv_conf.yaml",
                       default=None,
                       type=str)
    
    args = parser.parse_args()
    
    # Parsear reward_range
    reward_min, reward_max = map(float, args.reward_range.split(','))
    reward_range = (reward_min, reward_max)
    
    # Verificar que el script del agente existe
    if not os.path.exists(args.agent_script):
        print(f"ERROR: No se encuentra el script del agente: {args.agent_script}")
        sys.exit(1)
    
    # Verificar que los archivos necesarios existen
    if not os.path.exists(args.actions):
        print(f"ADVERTENCIA: No se encuentra el archivo de acciones: {args.actions}")
        print("Se usarán acciones dummy para la prueba.")
    
    if not os.path.exists(args.states):
        print(f"ADVERTENCIA: No se encuentra el archivo de estados: {args.states}")
        print("Se usarán estados dummy para la prueba.")
    
    print("\n" + "="*70)
    print("OPTIMIZADOR DE Q-TABLE CON ALGORITMO GENÉTICO")
    print("Para NetSecGame - Q-Learning Agent")
    print("="*70)
    
    # ================================================================
    # Modo SMAC: Optimización de hiperparámetros con SMAC3
    # ================================================================
    if args.smac:
        print("\nMODO: Optimización de hiperparámetros con SMAC3\n")
        
        # Configuración base para SMAC
        base_config = {
            'agent_script_path': args.agent_script,
            'host': args.host,
            'port': args.port,
            'actions_file': args.actions,
            'states_file': args.states,
            'n_workers': args.workers,
            'required_players': args.required_players,
            'task_config_path': args.task_config,
        }
        
        # Ejecutar optimización con SMAC3
        incumbent, objective, smac_instance = optimize_with_smac(
            base_config=base_config,
            n_trials=args.smac_trials,
            output_dir=args.smac_output_dir,
            n_workers=args.workers
        )
        
        # Generar visualizaciones
        plot_smac_results(objective)
        
        # Entrenar modelo final con mejores hiperparámetros (opcional)
        if not args.skip_final_training:
            print(f"\n{'='*70}")
            print("ENTRENANDO MODELO FINAL CON MEJORES HIPERPARÁMETROS (SMAC)")
            print(f"{'='*70}\n")
            
            best_params = dict(incumbent)
            optimizer = QTableGeneticOptimizer(
                agent_script_path=args.agent_script,
                host=args.host,
                port=args.port,
                test_episodes=25,
                reward_range=(-1, 1),
                actions_file=args.actions,
                states_file=args.states,
                n_workers=args.workers,
                required_players=args.required_players,
                task_config_path=args.task_config
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
        else:
            print(f"\n{'='*70}")
            print("ENTRENAMIENTO FINAL OMITIDO (--skip_final_training activado)")
            print(f"{'='*70}\n")
            final_output = None
        
        print(f"\n{'='*70}")
        print("OPTIMIZACIÓN CON SMAC3 COMPLETADA")
        print(f"{'='*70}")
        print(f"Resultados guardados en directorio: {args.smac_output_dir}/")
        print(f"Visualizaciones en: smac_plots/")
        if final_output:
            print(f"Mejor Q-table guardada en: {final_output}")
        else:
            print(f"Nota: No se ejecutó entrenamiento final (--skip_final_training)")
        print(f"{'='*70}\n")
    
    # ================================================================
    # Modo normal: Ejecutar GA con hiperparámetros fijos
    # ================================================================
    else:
        print("\nMODO: Ejecución estándar con hiperparámetros fijos\n")
        
        # Crear el optimizador
        optimizer = QTableGeneticOptimizer(
            agent_script_path=args.agent_script,
            host=args.host,
            port=args.port,
            test_episodes=args.test_episodes,
            reward_range=reward_range,
            actions_file=args.actions,
            states_file=args.states,
            n_workers=args.workers,
            required_players=args.required_players,
            task_config_path=args.task_config
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
        print(f"3. La Q-table usa una representación paramétrica basada en características")
        print(f"4. Durante el entrenamiento, se irán agregando entradas específicas")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
