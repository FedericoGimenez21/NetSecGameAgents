#!/usr/bin/env python3
"""
Script para extraer todos los estados de una Q-table guardada en formato pickle.
Maneja tanto el formato de diccionario tradicional como el formato numpy del GA.
"""
import pickle
import numpy as np
import argparse
import json
import sys
import os
from collections import defaultdict

# Agregar el path del proyecto para importar módulos necesarios
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def extract_states_from_dict_format(q_table):
    """
    Extrae estados únicos de una Q-table en formato diccionario.
    
    Args:
        q_table (dict): Q-table con claves (state_id, action)
    
    Returns:
        set: Conjunto de state_ids únicos
    """
    states = set()
    for key in q_table.keys():
        if isinstance(key, tuple) and len(key) >= 2:
            state_id = key[0]
            states.add(state_id)
    return states

def extract_states_from_numpy_format(q_table):
    """
    Extrae información de una Q-table en formato numpy.
    
    Args:
        q_table (numpy.ndarray): Q-table como matriz
    
    Returns:
        dict: Información sobre la matriz
    """
    return {
        'format': 'numpy_matrix',
        'shape': q_table.shape,
        'n_states': q_table.shape[0],
        'n_actions': q_table.shape[1],
        'total_entries': q_table.size,
        'min_value': float(np.min(q_table)),
        'max_value': float(np.max(q_table)),
        'mean_value': float(np.mean(q_table)),
        'std_value': float(np.std(q_table)),
        'non_zero_entries': int(np.count_nonzero(q_table))
    }

def analyze_state_ids(states):
    """
    Analiza los state_ids extraídos.
    
    Args:
        states (set): Conjunto de state_ids
    
    Returns:
        dict: Estadísticas sobre los estados
    """
    stats = {
        'total_unique_states': len(states),
        'state_id_types': defaultdict(int),
        'state_id_lengths': defaultdict(int)
    }
    
    for state_id in states:
        stats['state_id_types'][type(state_id).__name__] += 1
        if isinstance(state_id, tuple):
            stats['state_id_lengths'][len(state_id)] += 1
    
    # Convertir defaultdict a dict normal para JSON serialization
    stats['state_id_types'] = dict(stats['state_id_types'])
    stats['state_id_lengths'] = dict(stats['state_id_lengths'])
    
    return stats

def extract_action_statistics(q_table):
    """
    Extrae estadísticas sobre las acciones en la Q-table.
    
    Args:
        q_table (dict): Q-table en formato diccionario
    
    Returns:
        dict: Estadísticas sobre acciones
    """
    actions = defaultdict(int)
    state_action_counts = defaultdict(int)
    q_values_stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'values': []
    }
    
    for key, value in q_table.items():
        if isinstance(key, tuple) and len(key) >= 2:
            state_id, action = key[0], key[1]
            actions[str(action)] += 1
            state_action_counts[state_id] += 1
            
            q_values_stats['values'].append(value)
            q_values_stats['min'] = min(q_values_stats['min'], value)
            q_values_stats['max'] = max(q_values_stats['max'], value)
    
    # Calcular estadísticas de Q-values
    if q_values_stats['values']:
        q_values_stats['mean'] = float(np.mean(q_values_stats['values']))
        q_values_stats['std'] = float(np.std(q_values_stats['values']))
        q_values_stats['median'] = float(np.median(q_values_stats['values']))
    
    del q_values_stats['values']  # Remover la lista para ahorrar espacio
    
    return {
        'total_actions': len(actions),
        'action_distribution': dict(actions),
        'actions_per_state_stats': {
            'min': int(min(state_action_counts.values())) if state_action_counts else 0,
            'max': int(max(state_action_counts.values())) if state_action_counts else 0,
            'mean': float(np.mean(list(state_action_counts.values()))) if state_action_counts else 0
        },
        'q_values': q_values_stats
    }

def save_states_to_file(states, output_file, format_type='txt'):
    """
    Guarda los estados en un archivo.
    
    Args:
        states (set or dict): Estados o información a guardar
        output_file (str): Nombre del archivo de salida
        format_type (str): Formato del archivo ('txt', 'json', 'pickle')
    """
    if format_type == 'txt':
        with open(output_file, 'w') as f:
            if isinstance(states, set):
                f.write(f"Total de estados únicos: {len(states)}\n")
                f.write("=" * 80 + "\n\n")
                for i, state in enumerate(sorted(states, key=str), 1):
                    f.write(f"{i}. {state}\n")
            else:
                f.write(str(states))
    
    elif format_type == 'json':
        with open(output_file, 'w') as f:
            if isinstance(states, set):
                json.dump({
                    'total_states': len(states),
                    'states': [str(s) for s in states]
                }, f, indent=2)
            else:
                json.dump(states, f, indent=2)
    
    elif format_type == 'pickle':
        with open(output_file, 'wb') as f:
            pickle.dump(states, f)

def main():
    parser = argparse.ArgumentParser(
        description='Extrae estados de una Q-table guardada en formato pickle'
    )
    parser.add_argument(
        'pickle_file',
        help='Ruta al archivo pickle de la Q-table'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Archivo de salida para guardar los estados',
        default='extracted_states.txt'
    )
    parser.add_argument(
        '--format',
        '-f',
        choices=['txt', 'json', 'pickle'],
        default='txt',
        help='Formato del archivo de salida'
    )
    parser.add_argument(
        '--stats',
        '-s',
        action='store_true',
        help='Mostrar estadísticas detalladas'
    )
    parser.add_argument(
        '--stats-output',
        help='Archivo para guardar estadísticas en formato JSON',
        default='qtable_statistics.json'
    )
    
    args = parser.parse_args()
    
    print(f"Cargando Q-table desde: {args.pickle_file}")
    print("=" * 80)
    
    try:
        # Intentar cargar el archivo pickle
        try:
            with open(args.pickle_file, 'rb') as f:
                data = pickle.load(f)
        except ModuleNotFoundError as e:
            print(f"⚠️  Advertencia: Módulo no encontrado ({e})")
            print("Intentando cargar con importaciones opcionales...")
            
            # Intentar importar los módulos necesarios
            try:
                from AIDojoCoordinator.game_components import Action, Observation, GameState, AgentStatus
                print("✓ Módulos importados exitosamente")
            except ImportError:
                print("⚠️  No se pudieron importar todos los módulos, continuando de todos modos...")
            
            # Reintentar la carga
            with open(args.pickle_file, 'rb') as f:
                data = pickle.load(f)
        
        print(f"✓ Archivo cargado exitosamente")
        print(f"Tipo de datos: {type(data)}")
        
        if isinstance(data, dict) and "q_table" in data:
            q_table = data["q_table"]
            print(f"Tipo de Q-table: {type(q_table)}")
            
            all_stats = {}
            
            if isinstance(q_table, np.ndarray):
                # Formato numpy (del GA)
                print("\n📊 Q-table en formato NUMPY (Algoritmo Genético)")
                print("-" * 80)
                
                info = extract_states_from_numpy_format(q_table)
                for key, value in info.items():
                    print(f"{key}: {value}")
                
                all_stats = info
                
                # Guardar información
                save_states_to_file(info, args.output, args.format)
                print(f"\n✓ Información guardada en: {args.output}")
                
            elif isinstance(q_table, dict):
                # Formato diccionario tradicional
                print("\n📊 Q-table en formato DICCIONARIO")
                print("-" * 80)
                
                # Extraer estados
                states = extract_states_from_dict_format(q_table)
                print(f"Total de estados únicos: {len(states)}")
                print(f"Total de entradas (state, action): {len(q_table)}")
                
                # Analizar estados
                if args.stats:
                    print("\n📈 Analizando estados...")
                    state_stats = analyze_state_ids(states)
                    
                    print("\nEstadísticas de Estados:")
                    print(f"  - Estados únicos: {state_stats['total_unique_states']}")
                    print(f"  - Tipos de state_id: {state_stats['state_id_types']}")
                    print(f"  - Longitudes de state_id: {state_stats['state_id_lengths']}")
                    
                    print("\n📊 Analizando acciones y Q-values...")
                    action_stats = extract_action_statistics(q_table)
                    
                    print("\nEstadísticas de Acciones:")
                    print(f"  - Acciones únicas: {action_stats['total_actions']}")
                    print(f"  - Acciones por estado (min/max/mean): "
                          f"{action_stats['actions_per_state_stats']['min']}/"
                          f"{action_stats['actions_per_state_stats']['max']}/"
                          f"{action_stats['actions_per_state_stats']['mean']:.2f}")
                    
                    print("\nEstadísticas de Q-values:")
                    for key, value in action_stats['q_values'].items():
                        print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
                    
                    # Combinar estadísticas
                    all_stats = {
                        'format': 'dictionary',
                        'total_entries': len(q_table),
                        'state_statistics': state_stats,
                        'action_statistics': action_stats
                    }
                    
                    # Guardar estadísticas
                    with open(args.stats_output, 'w') as f:
                        json.dump(all_stats, f, indent=2)
                    print(f"\n✓ Estadísticas guardadas en: {args.stats_output}")
                
                # Guardar estados
                save_states_to_file(states, args.output, args.format)
                print(f"✓ Estados guardados en: {args.output}")
                
                # Mostrar muestra de estados
                print("\n📋 Muestra de estados (primeros 10):")
                for i, state in enumerate(list(states)[:10], 1):
                    print(f"  {i}. {state}")
                if len(states) > 10:
                    print(f"  ... y {len(states) - 10} estados más")
            
            else:
                print(f"❌ Formato de Q-table no reconocido: {type(q_table)}")
        
        else:
            print("❌ Estructura del pickle no reconocida")
            print(f"Claves disponibles: {data.keys() if isinstance(data, dict) else 'No es un diccionario'}")
    
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado: {args.pickle_file}")
    except Exception as e:
        print(f"❌ Error al procesar el archivo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
