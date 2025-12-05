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
import gc


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
                 states_file="estadosSMALL.json"):
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
        """
        self.agent_script_path = agent_script_path
        self.host = host
        self.port = port
        self.test_episodes = test_episodes
        self.reward_range = reward_range
        self.actions_file = actions_file
        self.states_file = states_file
        
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
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evalúa la calidad de las Q-tables candidatas ejecutándolas en el agente.
        
        Para cada individuo:
        1. Crea una Q-table desde sus parámetros
        2. Guarda la Q-table en un archivo temporal
        3. Ejecuta el agente en modo testing con esa Q-table
        4. Extrae el fitness (win_rate, avg_return, etc.)
        """
        objectives = []
        temp_files_created = []  # Mantener registro de archivos temporales
        
        for idx, x in enumerate(X):
            print(f"\nEvaluando individuo {idx + 1}/{len(X)}...")
            q_table_path = None
            
            try:
                # Crear Q-table desde parámetros
                q_table_data = self._create_q_table_from_params(x)
                
                # Crear archivo temporal sin delete=False para control manual
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
                # Penalización máxima en caso de error
                objectives.append(1000.0)
            
            finally:
                # Eliminar el archivo temporal INMEDIATAMENTE después de evaluar
                if q_table_path and os.path.exists(q_table_path):
                    try:
                        os.remove(q_table_path)
                        print(f"Temporal eliminado: {os.path.basename(q_table_path)}")
                    except Exception as e:
                        print(f"No se pudo eliminar {q_table_path}: {e}")
                gc.collect()
        # Limpieza adicional: asegurar que todos los archivos temporales fueron eliminados
        for temp_file in temp_files_created:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Limpieza final: {os.path.basename(temp_file)} eliminado")
                except:
                    pass
        
        out["F"] = np.array(objectives)
    
    def _run_agent_evaluation(self, q_table_path):
        """
        Ejecuta el agente con la Q-table proporcionada y extrae el fitness.
        
        Args:
            q_table_path (str): Ruta al archivo pickle con la Q-table
            
        Returns:
            float: Fitness (menor es mejor). Usamos: -win_rate - avg_return/100
        """
        # Comando para ejecutar el agente en modo testing
        cmd = [
            sys.executable,  # Python interpreter
            self.agent_script_path,
            "--host", self.host,
            "--port", str(self.port),
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
                 states_file="estadosSMALL.json"):
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
        """
        self.agent_script_path = agent_script_path
        self.host = host
        self.port = port
        self.test_episodes = test_episodes
        self.reward_range = reward_range
        self.actions_file = actions_file
        self.states_file = states_file
        
        # Crear el problema de optimización
        self.problem = QTableOptimizationProblem(
            agent_script_path=agent_script_path,
            host=host,
            port=port,
            test_episodes=test_episodes,
            reward_range=reward_range,
            actions_file=actions_file,
            states_file=states_file
        )
        
        self.best_q_table = None
        self.optimization_history = []
        self.best_params = None
    
    def optimize(self, 
                 population_size=10,
                 n_generations=1,
                 crossover_prob=0.8,
                 mutation_prob=0.2,
                 verbose=True):
        """
        Ejecuta la optimización usando algoritmo genético.
        
        Cada individuo (Q-table) es evaluado ejecutándolo en el agente
        durante varios episodios de testing.
        
        Args:
            population_size (int): Tamaño de la población
            n_generations (int): Número de generaciones
            crossover_prob (float): Probabilidad de cruzamiento
            mutation_prob (float): Probabilidad de mutación
            verbose (bool): Mostrar progreso
        
        Returns:
            dict: Q-table optimizada con metadata
        """
        print("\n" + "="*70)
        print("OPTIMIZACIÓN DE Q-TABLE CON ALGORITMO GENÉTICO")
        print("="*70)
        print(f"Agente: {self.agent_script_path}")
        print(f"Servidor: {self.host}:{self.port}")
        print(f"Episodios de evaluación: {self.test_episodes}")
        print(f"Población: {population_size}")
        print(f"Generaciones: {n_generations}")
        print(f"Rango de Q-values: {self.reward_range}")
        print(f"Mutation_prob: {mutation_prob}")
        print(f"Crossover_prob: {crossover_prob}")

        print("="*70 + "\n")
        
        # Configurar algoritmo genético
        algorithm = GA(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=15),
            mutation=PM(prob=mutation_prob, eta=20),
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


def main():
    """
    Función principal para ejecutar la optimización de Q-table con algoritmo genético.
    
    La optimización funciona de la siguiente manera:
    1. Cada individuo del GA representa una Q-table paramétrica
    2. La Q-table usa una matriz de pesos (características_estado x acciones)
    3. Cada individuo se evalúa ejecutándolo en el agente durante varios episodios
    4. El fitness es el win rate y el return promedio obtenido
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimizador de Q-table con Algoritmo Genético para NetSecGame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python ga.py --agent_script ../Codigo/NetSecGame/NetSecGameAgents/agents/attackers/q_learning_state_modifed/feature_based/q_agent_feature_based.py --actions registration_info.json --population 20 --generations 30
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
    parser.add_argument("--crossover_prob",
                       help="Probabilidad de cruzamiento",
                       default=0.8,
                       type=float)
    parser.add_argument("--mutation_prob",
                       help="Probabilidad de mutación",
                       default=0.0005,
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
    
    # Crear el optimizador
    optimizer = QTableGeneticOptimizer(
        agent_script_path=args.agent_script,
        host=args.host,
        port=args.port,
        test_episodes=args.test_episodes,
        reward_range=reward_range,
        actions_file=args.actions,
        states_file=args.states
    )
    
    # Ejecutar optimización
    optimized_q_table = optimizer.optimize(
        population_size=args.population,
        n_generations=args.generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        verbose=True
    )
    
    # Analizar resultados
    #optimizer.analyze_q_table()
    
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
    
    return optimized_q_table


if __name__ == "__main__":
    # Ejecutar optimización
    try:
        q_table = main()
        print("\n✓ Optimización completada exitosamente!")
    except KeyboardInterrupt:
        print("\n\n✗ Optimización interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error durante la optimización: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
