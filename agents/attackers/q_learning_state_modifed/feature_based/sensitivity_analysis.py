"""
Análisis de Sensibilidad de Hiperparámetros del Algoritmo Genético
usando SALib (Sobol Sensitivity Analysis)

Este script cuantifica la influencia de cada hiperparámetro del GA
en el win_rate final del agente Q-learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol
import pickle
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Importar el optimizador
from ga import QTableGeneticOptimizer


class SensitivityAnalyzer:
    """
    Realiza análisis de sensibilidad de Sobol sobre los hiperparámetros
    del algoritmo genético para optimización de Q-tables.
    """
    
    def __init__(self, 
                 agent_script_path,
                 host="127.0.0.1",
                 port=9000,
                 actions_file="registration_info.json",
                 states_file="estadosSMALL.json",
                 output_dir="sensitivity_results"):
        """
        Inicializa el analizador de sensibilidad.
        
        Args:
            agent_script_path (str): Ruta al script del agente
            host (str): Host del servidor
            port (int): Puerto del servidor
            actions_file (str): Archivo de acciones
            states_file (str): Archivo de estados
            output_dir (str): Directorio para guardar resultados
        """
        self.agent_script_path = agent_script_path
        self.host = host
        self.port = port
        self.actions_file = actions_file
        self.states_file = states_file
        self.output_dir = output_dir
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Definir el espacio de hiperparámetros
        self.problem = {
            'num_vars': 6,
            'names': [
                'population_size',
                'n_generations', 
                'sbx_prob',
                'sbx_eta',
                'pm_prob_var',
                'pm_eta'
            ],
            'bounds': [
                [10, 20],          # population_size: 10-20
                [10, 20],          # n_generations: 20-30
                [0.8, 1.0],      # sbx_prob: probabilidad de SBX
                [0, 60],         # sbx_eta: índice de distribución SBX
                [0.01, 0.6],     # pm_prob_var: probabilidad por variable de PM
                [0, 60]          # pm_eta: índice de distribución PM
            ]
        }
        
        # Valores fijos (no varían en el análisis)
        self.sbx_prob_var = 1.0  # prob_var para SBX fijo en 1
        self.pm_prob = 1.0       # prob para PM fijo en 1
        self.test_episodes = 20  # test_episodes fijo
        
        # Rango de reward adaptado al análisis previo
        self.reward_range = (-1000, 1000)
        
        print("="*70)
        print("ANALIZADOR DE SENSIBILIDAD DE HIPERPARÁMETROS")
        print("="*70)
        print(f"Hiperparámetros a analizar: {self.problem['names']}")
        print(f"Rangos definidos:")
        for name, bounds in zip(self.problem['names'], self.problem['bounds']):
            print(f"  {name}: [{bounds[0]}, {bounds[1]}]")
        print("="*70 + "\n")
    
    def generate_samples(self, n_samples=1024, calc_second_order=True):
        """
        Genera muestras usando el método de Saltelli para análisis de Sobol.
        
        Args:
            n_samples (int): Número de muestras base (el total será n*(2*d+2))
            calc_second_order (bool): Calcular índices de segundo orden
            
        Returns:
            np.ndarray: Matriz de muestras
        """
        print(f"Generando muestras de Saltelli...")
        print(f"Muestras base: {n_samples}")
        
        # Saltelli genera n*(2*d+2) muestras donde d = num_vars
        # Para second order: n*(2*d+2), sin second order: n*(d+2)
        param_values = saltelli.sample(
            self.problem, 
            n_samples,
            calc_second_order=calc_second_order
        )
        
        total_samples = param_values.shape[0]
        print(f"Total de muestras generadas: {total_samples}")
        print(f"Cada muestra ejecutará un GA completo\n")
        
        # Guardar muestras
        samples_file = os.path.join(self.output_dir, 'parameter_samples.npy')
        np.save(samples_file, param_values)
        print(f"Muestras guardadas en: {samples_file}\n")
        
        return param_values
    
    def evaluate_sample(self, sample_idx, params):
        """
        Evalúa una muestra de hiperparámetros ejecutando el GA.
        
        Args:
            sample_idx (int): Índice de la muestra
            params (np.ndarray): Vector de hiperparámetros
            
        Returns:
            float: Win rate obtenido
        """
        # Extraer hiperparámetros (6 variables del problema)
        population_size = int(params[0])
        n_generations = int(params[1])
        sbx_prob = float(params[2])      # prob: probabilidad de cruzar un par de padres
        sbx_eta = float(params[3])       # eta: índice de distribución SBX
        pm_prob_var = float(params[4])   # prob_var: probabilidad de mutar cada gen
        pm_eta = float(params[5])        # eta: índice de distribución PM
        
        print(f"\n{'='*70}")
        print(f"EVALUANDO MUESTRA {sample_idx + 1}")
        print(f"{'='*70}")
        print(f"Hiperparámetros:")
        print(f"  population_size: {population_size}")
        print(f"  n_generations: {n_generations}")
        print(f"  SBX - prob: {sbx_prob:.4f}, prob_var: {self.sbx_prob_var}, eta: {sbx_eta:.2f}")
        print(f"  PM  - prob: {self.pm_prob}, prob_var: {pm_prob_var:.4f}, eta: {pm_eta:.2f}")
        print(f"{'='*70}\n")
        
        # Crear directorio para esta muestra
        sample_dir = os.path.join(self.output_dir, f"sample_{sample_idx + 1:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Preparar diccionario de hiperparámetros
        hyperparams = {
            'sample_idx': sample_idx + 1,
            'population_size': population_size,
            'n_generations': n_generations,
            'sbx_prob': sbx_prob,
            'sbx_prob_var': self.sbx_prob_var,
            'sbx_eta': sbx_eta,
            'pm_prob': self.pm_prob,
            'pm_prob_var': pm_prob_var,
            'pm_eta': pm_eta,
            'test_episodes': self.test_episodes,
            'reward_range': self.reward_range,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Crear optimizador
            optimizer = QTableGeneticOptimizer(
                agent_script_path=self.agent_script_path,
                host=self.host,
                port=self.port,
                test_episodes=self.test_episodes,
                reward_range=self.reward_range,
                actions_file=self.actions_file,
                states_file=self.states_file
            )
            
            # Ejecutar optimización
            optimizer.optimize(
                population_size=population_size,
                n_generations=n_generations,
                sbx_prob=sbx_prob,
                sbx_prob_var=self.sbx_prob_var,
                sbx_eta=sbx_eta,
                pm_prob=self.pm_prob,
                pm_prob_var=pm_prob_var,
                pm_eta=pm_eta,
                verbose=False
            )
            
            # Obtener win_rate del mejor individuo
            best_fitness = optimizer.optimization_history[-1]
            if isinstance(best_fitness, np.ndarray):
                best_fitness = float(best_fitness.item() if best_fitness.size == 1 else best_fitness[0])
            win_rate = float(-best_fitness)
            
            # Agregar resultados a los hiperparámetros
            hyperparams['win_rate'] = win_rate
            hyperparams['best_fitness'] = best_fitness
            hyperparams['optimization_history'] = [float(f) for f in optimizer.optimization_history]
            hyperparams['status'] = 'success'
            
            # Guardar la mejor Q-table
            q_table_path = os.path.join(sample_dir, 'best_q_table.pickle')
            optimizer.save_q_table(q_table_path)
            print(f"  Q-table guardada: {q_table_path}")
            
            # Guardar hiperparámetros en JSON
            hyperparams_path = os.path.join(sample_dir, 'hyperparameters.json')
            with open(hyperparams_path, 'w', encoding='utf-8') as f:
                json.dump(hyperparams, f, indent=2, ensure_ascii=False)
            print(f"  Hiperparámetros guardados: {hyperparams_path}")
            
            # Guardar historial de optimización
            history_path = os.path.join(sample_dir, 'optimization_history.npy')
            np.save(history_path, np.array(optimizer.optimization_history))
            print(f"  Historial guardado: {history_path}")
            
            print(f"\nMuestra {sample_idx + 1} completada: win_rate = {win_rate:.2f}%\n")
            
            return win_rate
            
        except Exception as e:
            print(f"\nError en muestra {sample_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            
            # Guardar información del error
            hyperparams['status'] = 'error'
            hyperparams['error_message'] = str(e)
            hyperparams['win_rate'] = 0.0
            
            hyperparams_path = os.path.join(sample_dir, 'hyperparameters.json')
            with open(hyperparams_path, 'w', encoding='utf-8') as f:
                json.dump(hyperparams, f, indent=2, ensure_ascii=False)
            
            return 0.0  # Retornar 0 en caso de error
    
    def run_sensitivity_analysis(self, param_values, resume_from=None):
        """
        Ejecuta el análisis de sensibilidad evaluando todas las muestras.
        Detecta automáticamente el progreso previo y reanuda desde el último punto.
        
        Args:
            param_values (np.ndarray): Matriz de parámetros a evaluar
            resume_from (int): Índice desde donde resumir (si se interrumpió).
                              Si es None, detecta automáticamente.
            
        Returns:
            np.ndarray: Array de win_rates obtenidos
        """
        n_samples = param_values.shape[0]
        results_file = os.path.join(self.output_dir, 'evaluation_results.npy')
        progress_file = os.path.join(self.output_dir, 'progress.json')
        
        # Intentar cargar resultados previos y detectar progreso
        Y = np.zeros(n_samples)
        
        if os.path.exists(results_file):
            Y = np.load(results_file)
            print(f"Resultados previos cargados desde: {results_file}")
            
            # Asegurar que el array tiene el tamaño correcto
            if len(Y) != n_samples:
                print(f"ADVERTENCIA: Tamaño de resultados ({len(Y)}) != muestras ({n_samples})")
                old_Y = Y.copy()
                Y = np.zeros(n_samples)
                Y[:min(len(old_Y), n_samples)] = old_Y[:min(len(old_Y), n_samples)]
        
        # Detectar automáticamente desde dónde resumir
        if resume_from is None:
            resume_from = self._detect_last_completed_sample()
            if resume_from > 0:
                print(f"\n{'='*70}")
                print(f"REANUDACIÓN AUTOMÁTICA DETECTADA")
                print(f"{'='*70}")
                print(f"Última muestra completada: {resume_from}")
                print(f"Reanudando desde muestra: {resume_from + 1}")
                print(f"{'='*70}\n")
        
        # Cargar resultados de muestras ya completadas desde sus directorios
        Y = self._load_completed_results(Y, resume_from)
        
        # Guardar información de progreso
        self._save_progress(resume_from, n_samples)
        
        print(f"Evaluando {n_samples} muestras...")
        print(f"Muestras completadas: {resume_from}/{n_samples}")
        print(f"Muestras pendientes: {n_samples - resume_from}")
        remaining_time = (n_samples - resume_from) * 0.5
        print(f"Tiempo estimado restante: ~{remaining_time:.1f} horas\n")
        
        try:
            for i in range(resume_from, n_samples):
                # Evaluar muestra
                Y[i] = self.evaluate_sample(i, param_values[i])
                
                # Guardar resultados parciales cada muestra (para seguridad)
                np.save(results_file, Y)
                
                # Actualizar progreso
                self._save_progress(i + 1, n_samples)
                
                # Mostrar estadísticas cada 10 muestras
                if (i + 1) % 10 == 0:
                    print(f"\n{'='*50}")
                    print(f"PROGRESO: {i + 1}/{n_samples} muestras completadas ({(i+1)/n_samples*100:.1f}%)")
                    print(f"Win rate promedio hasta ahora: {np.mean(Y[:i+1]):.2f}%")
                    print(f"Win rate máximo hasta ahora: {np.max(Y[:i+1]):.2f}%")
                    remaining = n_samples - (i + 1)
                    print(f"Muestras restantes: {remaining}")
                    print(f"{'='*50}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print(f"INTERRUPCIÓN DEL USUARIO")
            print(f"{'='*70}")
            np.save(results_file, Y)
            self._save_progress(i, n_samples, interrupted=True)
            print(f"Progreso guardado. Última muestra completada: {i}")
            print(f"Para resumir, simplemente ejecute el script de nuevo.")
            print(f"El progreso se detectará automáticamente.")
            print(f"{'='*70}\n")
            raise
        
        except Exception as e:
            print(f"\n\n{'='*70}")
            print(f"ERROR DURANTE LA EVALUACIÓN")
            print(f"{'='*70}")
            print(f"Error: {e}")
            np.save(results_file, Y)
            self._save_progress(i, n_samples, interrupted=True, error=str(e))
            print(f"Progreso guardado. Última muestra intentada: {i + 1}")
            print(f"Para resumir, simplemente ejecute el script de nuevo.")
            print(f"{'='*70}\n")
            raise
        
        # Guardar resultados finales
        np.save(results_file, Y)
        self._save_progress(n_samples, n_samples, completed=True)
        
        print(f"\n{'='*70}")
        print(f"EVALUACIÓN COMPLETADA")
        print(f"{'='*70}")
        print(f"Resultados guardados en: {results_file}")
        print(f"Win rate promedio: {np.mean(Y):.2f}%")
        print(f"Win rate std: {np.std(Y):.2f}%")
        print(f"Win rate min: {np.min(Y):.2f}%")
        print(f"Win rate max: {np.max(Y):.2f}%")
        print(f"{'='*70}\n")
        
        return Y
    
    def _detect_last_completed_sample(self):
        """
        Detecta automáticamente la última muestra completada exitosamente.
        
        Returns:
            int: Índice de la última muestra completada (0 si ninguna)
        """
        last_completed = 0
        
        # Método 1: Verificar archivo de progreso
        progress_file = os.path.join(self.output_dir, 'progress.json')
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    last_completed = progress.get('last_completed', 0)
                    print(f"Progreso cargado desde archivo: {last_completed} muestras completadas")
            except:
                pass
        
        # Método 2: Verificar directorios de muestras existentes
        sample_dirs = []
        for item in os.listdir(self.output_dir):
            if item.startswith('sample_') and os.path.isdir(os.path.join(self.output_dir, item)):
                try:
                    # Extraer número de muestra
                    sample_num = int(item.replace('sample_', ''))
                    # Verificar que tenga los archivos necesarios (éxito)
                    sample_dir = os.path.join(self.output_dir, item)
                    hyperparams_file = os.path.join(sample_dir, 'hyperparameters.json')
                    
                    if os.path.exists(hyperparams_file):
                        with open(hyperparams_file, 'r') as f:
                            hyperparams = json.load(f)
                            if hyperparams.get('status') == 'success':
                                sample_dirs.append(sample_num)
                except:
                    continue
        
        if sample_dirs:
            dir_last_completed = max(sample_dirs)
            # Usar el mayor entre archivo de progreso y directorios
            if dir_last_completed > last_completed:
                last_completed = dir_last_completed
                print(f"Detectadas {len(sample_dirs)} muestras completadas en directorios")
        
        return last_completed
    
    def _load_completed_results(self, Y, up_to_sample):
        """
        Carga los resultados de muestras completadas desde sus directorios.
        
        Args:
            Y (np.ndarray): Array de resultados a llenar
            up_to_sample (int): Cargar hasta esta muestra
            
        Returns:
            np.ndarray: Array actualizado con resultados cargados
        """
        loaded_count = 0
        
        for sample_idx in range(up_to_sample):
            sample_dir = os.path.join(self.output_dir, f"sample_{sample_idx + 1:04d}")
            hyperparams_file = os.path.join(sample_dir, 'hyperparameters.json')
            
            if os.path.exists(hyperparams_file):
                try:
                    with open(hyperparams_file, 'r') as f:
                        hyperparams = json.load(f)
                        if 'win_rate' in hyperparams:
                            Y[sample_idx] = hyperparams['win_rate']
                            loaded_count += 1
                except:
                    continue
        
        if loaded_count > 0:
            print(f"Cargados {loaded_count} resultados desde directorios de muestras")
        
        return Y
    
    def _save_progress(self, last_completed, total_samples, interrupted=False, completed=False, error=None):
        """
        Guarda el estado de progreso del análisis.
        
        Args:
            last_completed (int): Última muestra completada
            total_samples (int): Total de muestras
            interrupted (bool): Si fue interrumpido
            completed (bool): Si se completó todo
            error (str): Mensaje de error si hubo uno
        """
        progress_file = os.path.join(self.output_dir, 'progress.json')
        
        progress = {
            'last_completed': last_completed,
            'total_samples': total_samples,
            'percentage': (last_completed / total_samples * 100) if total_samples > 0 else 0,
            'status': 'completed' if completed else ('interrupted' if interrupted else 'in_progress'),
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': error
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def analyze_results(self, param_values, Y, calc_second_order=True):
        """
        Analiza los resultados usando el método de Sobol.
        
        Args:
            param_values (np.ndarray): Matriz de parámetros evaluados
            Y (np.ndarray): Array de resultados (win_rates)
            calc_second_order (bool): Analizar interacciones de segundo orden
            
        Returns:
            dict: Índices de Sobol
        """
        print("Calculando índices de Sobol...")
        
        # Análisis de Sobol
        Si = sobol.analyze(
            self.problem, 
            Y,
            calc_second_order=calc_second_order,
            print_to_console=True
        )
        
        # Guardar índices en el directorio principal
        indices_file = os.path.join(self.output_dir, 'sobol_indices.json')
        indices_dict = {
            'S1': Si['S1'].tolist(),
            'S1_conf': Si['S1_conf'].tolist(),
            'ST': Si['ST'].tolist(),
            'ST_conf': Si['ST_conf'].tolist(),
            'parameter_names': self.problem['names'],
            'parameter_bounds': self.problem['bounds'],
            'num_samples': len(Y),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if calc_second_order:
            indices_dict['S2'] = Si['S2'].tolist()
            indices_dict['S2_conf'] = Si['S2_conf'].tolist()
        
        with open(indices_file, 'w') as f:
            json.dump(indices_dict, f, indent=2)
        
        print(f"\nÍndices de Sobol guardados en: {indices_file}")
        
        # Crear resumen con ranking de parámetros
        self._create_summary(Si, Y, param_values)
        
        # Actualizar cada directorio de muestra con los índices globales
        self._update_sample_directories_with_sobol(Si, Y, param_values)
        
        print(f"Índices de Sobol guardados en: {indices_file}\n")
        
        return Si
    
    def _create_summary(self, Si, Y, param_values):
        """
        Crea un archivo resumen con las mejores configuraciones.
        
        Args:
            Si (dict): Índices de Sobol
            Y (np.ndarray): Array de resultados
            param_values (np.ndarray): Matriz de parámetros
        """
        summary_path = os.path.join(self.output_dir, 'best_configurations.json')
        
        # Encontrar las mejores configuraciones
        sorted_indices = np.argsort(Y)[::-1]  # Ordenar por win_rate descendente
        
        best_configs = []
        for rank, idx in enumerate(sorted_indices[:10], 1):  # Top 10
            config = {
                'rank': rank,
                'sample_idx': int(idx + 1),
                'win_rate': float(Y[idx]),
                'hyperparameters': {
                    'population_size': int(param_values[idx, 0]),
                    'n_generations': int(param_values[idx, 1]),
                    'sbx_prob': float(param_values[idx, 2]),
                    'sbx_eta': float(param_values[idx, 3]),
                    'pm_prob_var': float(param_values[idx, 4]),
                    'pm_eta': float(param_values[idx, 5])
                },
                'sample_dir': f"sample_{idx + 1:04d}"
            }
            best_configs.append(config)
        
        # Ranking de importancia de parámetros
        param_ranking = []
        sorted_st_indices = np.argsort(Si['ST'])[::-1]
        for rank, idx in enumerate(sorted_st_indices, 1):
            param_ranking.append({
                'rank': rank,
                'parameter': self.problem['names'][idx],
                'S1': float(Si['S1'][idx]),
                'ST': float(Si['ST'][idx]),
                'interaction': float(Si['ST'][idx] - Si['S1'][idx])
            })
        
        summary = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(Y),
            'statistics': {
                'mean_win_rate': float(np.mean(Y)),
                'std_win_rate': float(np.std(Y)),
                'min_win_rate': float(np.min(Y)),
                'max_win_rate': float(np.max(Y)),
                'median_win_rate': float(np.median(Y))
            },
            'parameter_importance_ranking': param_ranking,
            'top_10_configurations': best_configs
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Resumen de mejores configuraciones guardado: {summary_path}")
    
    def _update_sample_directories_with_sobol(self, Si, Y, param_values):
        """
        Actualiza cada directorio de muestra con los índices de Sobol globales
        y su posición en el ranking.
        
        Args:
            Si (dict): Índices de Sobol
            Y (np.ndarray): Array de resultados
            param_values (np.ndarray): Matriz de parámetros
        """
        print("Actualizando directorios de muestras con índices de Sobol...")
        
        # Calcular ranking de cada muestra
        sorted_indices = np.argsort(Y)[::-1]
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        
        for sample_idx in range(len(Y)):
            sample_dir = os.path.join(self.output_dir, f"sample_{sample_idx + 1:04d}")
            
            if not os.path.exists(sample_dir):
                continue
            
            # Crear archivo con índices de Sobol para esta muestra
            sobol_info = {
                'sample_idx': sample_idx + 1,
                'win_rate': float(Y[sample_idx]),
                'rank': rank_map[sample_idx],
                'total_samples': len(Y),
                'percentile': float((1 - rank_map[sample_idx] / len(Y)) * 100),
                'global_sobol_indices': {
                    'S1': {name: float(Si['S1'][i]) for i, name in enumerate(self.problem['names'])},
                    'ST': {name: float(Si['ST'][i]) for i, name in enumerate(self.problem['names'])},
                    'S1_conf': {name: float(Si['S1_conf'][i]) for i, name in enumerate(self.problem['names'])},
                    'ST_conf': {name: float(Si['ST_conf'][i]) for i, name in enumerate(self.problem['names'])}
                },
                'parameter_values': {
                    'population_size': int(param_values[sample_idx, 0]),
                    'n_generations': int(param_values[sample_idx, 1]),
                    'sbx_prob': float(param_values[sample_idx, 2]),
                    'sbx_eta': float(param_values[sample_idx, 3]),
                    'pm_prob_var': float(param_values[sample_idx, 4]),
                    'pm_eta': float(param_values[sample_idx, 5])
                }
            }
            
            if 'S2' in Si:
                sobol_info['global_sobol_indices']['S2'] = Si['S2'].tolist()
            
            sobol_path = os.path.join(sample_dir, 'sobol_indices.json')
            with open(sobol_path, 'w', encoding='utf-8') as f:
                json.dump(sobol_info, f, indent=2, ensure_ascii=False)
        
        print(f"Directorios de muestras actualizados con índices de Sobol.")
    
    def plot_results(self, Si, save_dir=None):
        """
        Genera visualizaciones de los resultados del análisis de sensibilidad.
        
        Args:
            Si (dict): Índices de Sobol
            save_dir (str): Directorio donde guardar las gráficas
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Gráfica de barras: Índices de primer orden (S1)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: S1 (First-order indices)
        ax1 = axes[0, 0]
        params = self.problem['names']
        s1_values = Si['S1']
        s1_conf = Si['S1_conf']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
        bars1 = ax1.bar(params, s1_values, yerr=s1_conf, capsize=5, 
                        color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('First-order Sobol Index (S1)', fontsize=12, fontweight='bold')
        ax1.set_title('Efecto Directo de Hiperparámetros', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Agregar valores en las barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Subplot 2: ST (Total-order indices)
        ax2 = axes[0, 1]
        st_values = Si['ST']
        st_conf = Si['ST_conf']
        
        bars2 = ax2.bar(params, st_values, yerr=st_conf, capsize=5,
                       color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Total-order Sobol Index (ST)', fontsize=12, fontweight='bold')
        ax2.set_title('Efecto Total (Directo + Interacciones)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Agregar valores en las barras
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Subplot 3: Comparación S1 vs ST
        ax3 = axes[1, 0]
        x = np.arange(len(params))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, s1_values, width, label='S1 (Directo)',
                        color='skyblue', alpha=0.8, edgecolor='black')
        bars3b = ax3.bar(x + width/2, st_values, width, label='ST (Total)',
                        color='lightcoral', alpha=0.8, edgecolor='black')
        
        ax3.set_ylabel('Índice de Sobol', fontsize=12, fontweight='bold')
        ax3.set_title('S1 vs ST: Efectos Directos vs Totales', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(params, rotation=45)
        ax3.legend(fontsize=11)
        ax3.grid(axis='y', alpha=0.3)
        
        # Subplot 4: Índices de interacción (ST - S1)
        ax4 = axes[1, 1]
        interaction = st_values - s1_values
        
        bars4 = ax4.bar(params, interaction, color='orange', alpha=0.8, 
                       edgecolor='black')
        ax4.set_ylabel('Índice de Interacción (ST - S1)', fontsize=12, fontweight='bold')
        ax4.set_title('Efectos de Interacción entre Hiperparámetros', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Agregar valores en las barras
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        
        # Guardar figura
        fig_path = os.path.join(save_dir, 'sobol_indices_summary.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada: {fig_path}")
        plt.close()
        
        # 2. Heatmap de interacciones de segundo orden (si están disponibles)
        if 'S2' in Si:
            fig2, ax = plt.subplots(figsize=(10, 8))
            
            s2_matrix = Si['S2']
            im = ax.imshow(s2_matrix, cmap='RdYlGn', aspect='auto')
            
            # Configurar ejes
            ax.set_xticks(np.arange(len(params)))
            ax.set_yticks(np.arange(len(params)))
            ax.set_xticklabels(params, rotation=45, ha='right')
            ax.set_yticklabels(params)
            
            # Agregar valores en cada celda
            for i in range(len(params)):
                for j in range(len(params)):
                    if i != j:  # No mostrar diagonal
                        text = ax.text(j, i, f'{s2_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=9)
            
            ax.set_title('Índices de Sobol de Segundo Orden (S2)\nInteracciones entre Pares de Hiperparámetros',
                        fontsize=14, fontweight='bold', pad=20)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Índice S2', rotation=270, labelpad=20, fontsize=12)
            
            plt.tight_layout()
            
            # Guardar figura
            fig2_path = os.path.join(save_dir, 'sobol_second_order_heatmap.png')
            plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap guardado: {fig2_path}")
            plt.close()
        
        # 3. Ranking de importancia
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        # Ordenar por ST (efecto total)
        sorted_indices = np.argsort(st_values)[::-1]
        sorted_params = [params[i] for i in sorted_indices]
        sorted_st = st_values[sorted_indices]
        sorted_s1 = s1_values[sorted_indices]
        
        x_pos = np.arange(len(sorted_params))
        bars_st = ax.barh(x_pos, sorted_st, alpha=0.7, label='ST (Total)', 
                         color='coral', edgecolor='black')
        bars_s1 = ax.barh(x_pos, sorted_s1, alpha=0.7, label='S1 (Directo)', 
                         color='skyblue', edgecolor='black')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(sorted_params, fontsize=11)
        ax.set_xlabel('Índice de Sobol', fontsize=12, fontweight='bold')
        ax.set_title('Ranking de Importancia de Hiperparámetros', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        
        # Agregar valores
        for i, (st, s1) in enumerate(zip(sorted_st, sorted_s1)):
            ax.text(st + 0.01, i, f'{st:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Guardar figura
        fig3_path = os.path.join(save_dir, 'parameter_ranking.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"Ranking guardado: {fig3_path}")
        plt.close()
        
        print(f"\nTodas las visualizaciones guardadas en: {save_dir}/\n")
    
    def generate_report(self, Si, Y, save_path=None):
        """
        Genera un reporte en texto con los resultados del análisis.
        
        Args:
            Si (dict): Índices de Sobol
            Y (np.ndarray): Resultados de win_rate
            save_path (str): Ruta donde guardar el reporte
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sensitivity_report.txt')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE ANÁLISIS DE SENSIBILIDAD\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. RESUMEN EJECUTIVO\n")
            f.write("-"*70 + "\n")
            f.write(f"Número de muestras evaluadas: {len(Y)}\n")
            f.write(f"Win rate promedio: {np.mean(Y):.2f}%\n")
            f.write(f"Desviación estándar: {np.std(Y):.2f}%\n")
            f.write(f"Win rate mínimo: {np.min(Y):.2f}%\n")
            f.write(f"Win rate máximo: {np.max(Y):.2f}%\n\n")
            
            f.write("2. HIPERPARÁMETROS ANALIZADOS\n")
            f.write("-"*70 + "\n")
            for name, bounds in zip(self.problem['names'], self.problem['bounds']):
                f.write(f"  • {name}: [{bounds[0]}, {bounds[1]}]\n")
            f.write("\n")
            
            f.write("3. ÍNDICES DE SOBOL DE PRIMER ORDEN (S1)\n")
            f.write("-"*70 + "\n")
            f.write("Miden el efecto directo de cada hiperparámetro en el win_rate.\n\n")
            for i, name in enumerate(self.problem['names']):
                f.write(f"  {name:20s}: {Si['S1'][i]:.4f} ± {Si['S1_conf'][i]:.4f}\n")
            f.write("\n")
            
            f.write("4. ÍNDICES DE SOBOL TOTALES (ST)\n")
            f.write("-"*70 + "\n")
            f.write("Miden el efecto total (directo + interacciones) de cada hiperparámetro.\n\n")
            for i, name in enumerate(self.problem['names']):
                f.write(f"  {name:20s}: {Si['ST'][i]:.4f} ± {Si['ST_conf'][i]:.4f}\n")
            f.write("\n")
            
            f.write("5. ÍNDICES DE INTERACCIÓN (ST - S1)\n")
            f.write("-"*70 + "\n")
            f.write("Miden el efecto de interacciones con otros hiperparámetros.\n\n")
            for i, name in enumerate(self.problem['names']):
                interaction = Si['ST'][i] - Si['S1'][i]
                f.write(f"  {name:20s}: {interaction:.4f}\n")
            f.write("\n")
            
            f.write("6. RANKING DE IMPORTANCIA (por ST)\n")
            f.write("-"*70 + "\n")
            sorted_indices = np.argsort(Si['ST'])[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                name = self.problem['names'][idx]
                st_value = Si['ST'][idx]
                f.write(f"  {rank}. {name:20s}: {st_value:.4f}\n")
            f.write("\n")
            
            if 'S2' in Si:
                f.write("7. INTERACCIONES DE SEGUNDO ORDEN MÁS SIGNIFICATIVAS\n")
                f.write("-"*70 + "\n")
                
                # Extraer las interacciones más significativas
                s2_matrix = Si['S2']
                interactions = []
                for i in range(len(self.problem['names'])):
                    for j in range(i+1, len(self.problem['names'])):
                        if not np.isnan(s2_matrix[i, j]):
                            interactions.append((
                                self.problem['names'][i],
                                self.problem['names'][j],
                                s2_matrix[i, j]
                            ))
                
                interactions.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for param1, param2, value in interactions[:10]:
                    f.write(f"  {param1:15s} × {param2:15s}: {value:.4f}\n")
                f.write("\n")
            
            f.write("8. INTERPRETACIÓN Y RECOMENDACIONES\n")
            f.write("-"*70 + "\n")
            
            # Identificar hiperparámetros más importantes
            most_important_idx = np.argmax(Si['ST'])
            most_important_name = self.problem['names'][most_important_idx]
            most_important_value = Si['ST'][most_important_idx]
            
            f.write(f"El hiperparámetro más influyente es '{most_important_name}' ")
            f.write(f"(ST = {most_important_value:.4f}),\n")
            f.write(f"lo que indica que tiene el mayor impacto en el win_rate del agente.\n\n")
            
            # Identificar hiperparámetros con alta interacción
            high_interaction = []
            for i, name in enumerate(self.problem['names']):
                interaction = Si['ST'][i] - Si['S1'][i]
                if interaction > 0.1:  # Umbral de interacción significativa
                    high_interaction.append((name, interaction))
            
            if high_interaction:
                f.write("Hiperparámetros con interacciones significativas:\n")
                for name, value in high_interaction:
                    f.write(f"  • {name}: interacción = {value:.4f}\n")
                f.write("\nEsto sugiere que estos hiperparámetros deben ser optimizados\n")
                f.write("conjuntamente para obtener los mejores resultados.\n\n")
            
            f.write("Recomendaciones:\n")
            f.write("  1. Priorizar la optimización del hiperparámetro más influyente\n")
            f.write("  2. Considerar interacciones al ajustar hiperparámetros relacionados\n")
            f.write("  3. Hiperparámetros con bajo ST pueden usar valores por defecto\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("="*70 + "\n")
        
        print(f"Reporte generado: {save_path}\n")


def main():
    """Función principal para ejecutar el análisis de sensibilidad."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Análisis de Sensibilidad de Hiperparámetros del GA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ejecutar análisis completo
  python sensitivity_analysis.py --agent_script path/to/agent.py
  
  # Reanudar automáticamente (detecta progreso previo)
  python sensitivity_analysis.py --agent_script path/to/agent.py
  
  # Forzar reinicio desde cero
  python sensitivity_analysis.py --agent_script path/to/agent.py --force_restart
  
  # Solo analizar resultados existentes
  python sensitivity_analysis.py --agent_script path/to/agent.py --only_analyze
        """
    )
    
    parser.add_argument("--agent_script", 
                       help="Ruta al script q_agent_feature_based.py",
                       required=True,
                       type=str)
    parser.add_argument("--n_samples",
                       help="Número de muestras base para Saltelli (default: 512)",
                       default=512,
                       type=int)
    parser.add_argument("--second_order",
                       help="Calcular índices de segundo orden",
                       default=True,
                       type=bool)
    parser.add_argument("--output_dir",
                       help="Directorio para resultados",
                       default="sensitivity_results",
                       type=str)
    parser.add_argument("--resume_from",
                       help="Índice desde donde resumir evaluación (auto-detectado si no se especifica)",
                       default=None,
                       type=int)
    parser.add_argument("--only_analyze",
                       help="Solo analizar resultados existentes (no evaluar)",
                       action='store_true')
    parser.add_argument("--force_restart",
                       help="Forzar reinicio desde cero (ignora progreso previo)",
                       action='store_true')
    
    args = parser.parse_args()
    
    # Crear analizador
    analyzer = SensitivityAnalyzer(
        agent_script_path=args.agent_script,
        output_dir=args.output_dir
    )
    
    if not args.only_analyze:
        # Verificar si existen muestras previas
        samples_file = os.path.join(args.output_dir, 'parameter_samples.npy')
        
        if os.path.exists(samples_file) and not args.force_restart:
            # Cargar muestras existentes
            param_values = np.load(samples_file)
            print(f"Muestras existentes cargadas desde: {samples_file}")
            print(f"Total de muestras: {len(param_values)}")
            
            # Verificar compatibilidad con el número de parámetros actual
            expected_params = analyzer.problem['num_vars']
            actual_params = param_values.shape[1] if len(param_values.shape) > 1 else 0
            
            if actual_params != expected_params:
                print(f"\n{'='*70}")
                print(f"¡ADVERTENCIA! INCOMPATIBILIDAD DETECTADA")
                print(f"{'='*70}")
                print(f"Las muestras existentes tienen {actual_params} parámetros,")
                print(f"pero el análisis actual requiere {expected_params} parámetros.")
                print(f"Regenerando muestras automáticamente...")
                print(f"{'='*70}\n")
                
                param_values = analyzer.generate_samples(
                    n_samples=args.n_samples,
                    calc_second_order=args.second_order
                )
        else:
            # Generar nuevas muestras
            if args.force_restart:
                print("\n¡REINICIO FORZADO! Generando nuevas muestras...\n")
            param_values = analyzer.generate_samples(
                n_samples=args.n_samples,
                calc_second_order=args.second_order
            )
        
        # Configurar resume_from
        resume_from = 0 if args.force_restart else args.resume_from
        
        # Evaluar muestras (detecta automáticamente progreso si resume_from es None)
        Y = analyzer.run_sensitivity_analysis(
            param_values,
            resume_from=resume_from
        )
    else:
        # Cargar resultados existentes
        param_values = np.load(os.path.join(args.output_dir, 'parameter_samples.npy'))
        Y = np.load(os.path.join(args.output_dir, 'evaluation_results.npy'))
        print("Resultados cargados desde archivos existentes.\n")
    
    # Analizar resultados
    Si = analyzer.analyze_results(
        param_values, 
        Y,
        calc_second_order=args.second_order
    )
    
    # Generar visualizaciones
    analyzer.plot_results(Si)
    
    # Generar reporte
    analyzer.generate_report(Si, Y)
    
    print("\n" + "="*70)
    print("ANÁLISIS DE SENSIBILIDAD COMPLETADO")
    print("="*70)
    print(f"Resultados guardados en: {args.output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
