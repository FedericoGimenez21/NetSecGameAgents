from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Any
import numpy as np
import hashlib
from collections import defaultdict
from AIDojoCoordinator.game_components import GameState

@dataclass
class StateTransformerConfig:
    """Configuración para el transformador de estados"""
    n_network_buckets: int = 4  # Número de buckets para redes
    n_host_buckets: int = 8     # Número de buckets para hosts
    n_service_buckets: int = 4  # Número de buckets para servicios
    n_data_buckets: int = 4     # Número de buckets para datos
    relevant_services: Set[str] = None  # Servicios específicos a monitorear
    hash_seed: int = 42         # Semilla para hashing consistente
    
    def __post_init__(self):
        if self.relevant_services is None:
            self.relevant_services = {'smb', 'ssh', 'ms-wbt-server', 
        'microsoft-ds', 'postgresql', 'bash', 
        'listener', 'ftp'}

class StateTransformer:
    """
    Transforma estados crudos de NetSecGame en representaciones compactas 
    para Q-learning mediante técnicas de agregación y bucketing.
    """
    
    def __init__(self, config: StateTransformerConfig = None):
        self.config = config or StateTransformerConfig()
        np.random.seed(self.config.hash_seed)
        
    def _hash_to_bucket(self, item: str, n_buckets: int) -> int:
        """Asigna un item a un bucket usando hashing consistente"""
        hash_val = int(hashlib.md5(item.encode()).hexdigest(), 16)
        return hash_val % n_buckets
    
    def _count_by_bucket(self, items: Set[str], n_buckets: int) -> List[int]:
        """Cuenta items por bucket"""
        counts = [0] * n_buckets
        for item in items:
            bucket = self._hash_to_bucket(item, n_buckets)
            counts[bucket] += 1
        return counts
    
    def _aggregate_services(self, services: Dict[str, Set[str]]) -> List[int]:
        """Agrega servicios por tipo en buckets"""
        service_counts = defaultdict(int)
        for host_services in services.values():
            for service in host_services:
                if service in self.config.relevant_services:
                    service_counts[service] += 1
                    
        # Convertir conteos a vector de buckets
        counts = [0] * self.config.n_service_buckets
        for service, count in service_counts.items():
            bucket = self._hash_to_bucket(service, self.config.n_service_buckets)
            counts[bucket] = max(counts[bucket], count)
        return counts
        
    def _parse_networks(self, nets_str: str) -> Set[str]:
        """Parse networks section from state string"""
        nets = nets_str.replace('nets:[', '').replace(']', '')
        return set(nets.split(',')) if nets else set()

    def _parse_hosts(self, hosts_str: str, prefix: str) -> Set[str]:
        """Parse hosts or controlled hosts section from state string"""
        hosts = hosts_str.replace(f'{prefix}:[', '').replace(']', '')
        return set(hosts.split(',')) if hosts else set()

    def _parse_services(self, services_str: str) -> Dict[str, List[dict]]:
        """Parse services section from state string"""
        services_dict = {}
        if not services_str:
            return services_dict
        
        # Remove outer brackets and split by host
        services = services_str.replace('services:{', '').replace('}', '')
        host_services = services.split(']')
        
        for host_service in host_services:
            if not host_service:
                continue
            # Split host and its services
            host_ip, services_list = host_service.split(':[')
            services_list = services_list.split(',')
            
            parsed_services = []
            for service in services_list:
                if 'Service' in service:
                    # Extract service name from the Service object string
                    name = service.split("name='")[1].split("'")[0]
                    parsed_services.append({'name': name})
            
            if parsed_services:
                services_dict[host_ip] = parsed_services
                
        return services_dict

    def _parse_data(self, data_str: str) -> Dict[str, List[dict]]:
        """Parse data section from state string"""
        data_dict = {}
        if not data_str or 'data:{' not in data_str:
            return data_dict
        
        try:
            # Remove outer brackets and split by host
            data = data_str.replace('data:{', '').split('},blocks')[0]  # Remove blocks section
            host_data = data.split(']')
            
            for host_datum in host_data:
                if not host_datum or '[' not in host_datum:
                    continue
                    
                # Split host and its data
                parts = host_datum.split(':[')
                if len(parts) != 2:
                    continue
                    
                host_ip, data_list = parts
                data_entries = data_list.split('Data(')
                
                parsed_data = []
                for datum in data_entries:
                    if not datum:
                        continue
                        
                    try:
                        # Extract data properties more safely
                        owner = datum.split("owner='")[1].split("'")[0] if "owner='" in datum else "unknown"
                        data_type = datum.split("type='")[1].split("'")[0] if "type='" in datum else ""
                        size = int(datum.split("size=")[1].split(",")[0]) if "size=" in datum else 0
                        
                        parsed_data.append({
                            'owner': owner,
                            'type': data_type,
                            'size': size
                        })
                    except (IndexError, ValueError) as e:
                        # Log error but continue processing other entries
                        print(f"Error parsing data entry: {datum}. Error: {str(e)}")
                        continue
                
                if parsed_data:
                    # Clean host_ip of any remaining special characters
                    host_ip = host_ip.strip(',{}[] ')
                    data_dict[host_ip] = parsed_data
                    
        except Exception as e:
            print(f"Error in _parse_data: {str(e)}")
            return {}
            
        return data_dict

    def transform_state(self, state_str: str) -> Tuple:
        """
        Transforma un string de estado en una representación compacta.
        
        Args:
            state_str: String representation of the state containing:
                - networks list
                - known hosts list
                - controlled hosts list
                - services dictionary
                - data dictionary
                    
        Returns:
            Tuple: Representación agregada del estado, usable como clave en Q-table
        """
        # Split the state string into sections
        sections = state_str.split(',blocks:')
        main_section = sections[0]
        
        # Parse networks
        nets_end = main_section.find('],hosts:')
        networks = self._parse_networks(main_section[:nets_end+1])
        
        # Parse known hosts
        hosts_start = main_section.find('hosts:[')
        hosts_end = main_section.find('],controlled:')
        known_hosts = self._parse_hosts(main_section[hosts_start:hosts_end+1], 'hosts')
        
        # Parse controlled hosts
        controlled_start = main_section.find('controlled:[')
        controlled_end = main_section.find('],services:')
        controlled_hosts = self._parse_hosts(main_section[controlled_start:controlled_end+1], 'controlled')
        
        # Parse services
        services_start = main_section.find('services:{')
        services_end = main_section.find('},data:')
        services = self._parse_services(main_section[services_start:services_end+1])
        
        # Parse data
        data_start = main_section.find('data:{')
        data = self._parse_data(main_section[data_start:])
        
        # Agregación de redes
        network_counts = self._count_by_bucket(networks, self.config.n_network_buckets)
        
        # Agregación de hosts conocidos y controlados
        known_host_counts = self._count_by_bucket(known_hosts, self.config.n_host_buckets)
        controlled_host_counts = self._count_by_bucket(controlled_hosts, self.config.n_host_buckets)
        
        # Agregación de servicios por tipo
        service_counts = [0] * self.config.n_service_buckets
        for host_services in services.values():
            for service in host_services:
                if service['name'] in self.config.relevant_services:
                    bucket = self._hash_to_bucket(service['name'], self.config.n_service_buckets)
                    service_counts[bucket] += 1
        
        # Agregación de datos por tipo/tamaño
        data_counts = [0] * self.config.n_data_buckets
        for host_data in data.values():
            for datum in host_data:
                data_key = f"{datum['type']}_{datum['owner']}_{datum['size']}"
                bucket = self._hash_to_bucket(data_key, self.config.n_data_buckets)
                data_counts[bucket] += 1
        
        # Combinar todas las características en un único tuple
        return tuple(
            network_counts +
            known_host_counts +
            controlled_host_counts +
            service_counts +
            data_counts
        )