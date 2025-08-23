from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Any
import numpy as np

@dataclass
class FeatureExtractorConfig:
    """Configuración para la extracción de características"""
    use_service_types: bool = True
    use_network_counts: bool = True
    use_host_ratios: bool = True
    use_data_features: bool = True
    relevant_services: Set[str] = None
    feature_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.relevant_services is None:
            self.relevant_services = {'smb', 'ssh', 'http', 'ms-wbt-server', 
                                    'microsoft-ds', 'postgresql', 'bash', 'listener'}
        if self.feature_weights is None:
            self.feature_weights = {
                'network_coverage': 1.0,
                'host_control_ratio': 2.0,
                'service_diversity': 1.5,
                'data_value': 1.0
            }

class FeatureExtractor:
    """
    Extrae características relevantes del estado para generalización
    """
    
    def __init__(self, config: FeatureExtractorConfig = None):
        self.config = config or FeatureExtractorConfig()
    
    def _parse_state_string(self, state_str: str) -> Dict:
        """Parse estado desde string a diccionario estructurado"""
        networks = []
        known_hosts = []
        controlled_hosts = []
        services = {}
        data = {}
        
        # Extract networks
        net_start = state_str.find('nets:[') + 6
        net_end = state_str.find('],hosts:')
        networks = state_str[net_start:net_end].split(',')
        
        # Extract known hosts
        host_start = state_str.find('hosts:[') + 7
        host_end = state_str.find('],controlled:')
        known_hosts = state_str[host_start:host_end].split(',')
        
        # Extract controlled hosts
        ctrl_start = state_str.find('controlled:[') + 11
        ctrl_end = state_str.find('],services:')
        controlled_hosts = state_str[ctrl_start:ctrl_end].split(',')
        
        # Extract services
        services_start = state_str.find('services:{') + 10
        services_end = state_str.find('},data:')
        services_str = state_str[services_start:services_end]
        
        if services_str:
            # Split by closing bracket to separate host entries
            host_entries = services_str.split(']')
            for entry in host_entries:
                if '[' in entry:  # Valid host entry
                    host_ip, services_list = entry.split(':[')
                    host_ip = host_ip.strip(',{}[] ')  # Clean host IP
                    
                    # Extract individual services
                    services[host_ip] = []
                    if 'Service' in services_list:
                        service_entries = services_list.split('Service(')
                        for service in service_entries:
                            if 'name=' in service:
                                service_name = service.split("name='")[1].split("'")[0]
                                services[host_ip].append({'name': service_name})
        
        # Extract data
        data_start = state_str.find('data:{') + 6
        data_end = state_str.find('},blocks:')
        data_str = state_str[data_start:data_end]
        
        if data_str:
            # Split by closing bracket to separate host entries
            host_entries = data_str.split(']')
            for entry in host_entries:
                if '[' in entry:  # Valid host entry
                    host_ip, data_list = entry.split(':[')
                    host_ip = host_ip.strip(',{}[] ')  # Clean host IP
                    
                    # Extract individual data entries
                    data[host_ip] = []
                    if 'Data' in data_list:
                        data_entries = data_list.split('Data(')
                        for datum in data_entries:
                            if 'owner=' in datum:
                                # Extract basic data properties
                                data[host_ip].append({
                                    'owner': datum.split("owner='")[1].split("'")[0],
                                    'id': datum.split("id='")[1].split("'")[0],
                                    'size': int(datum.split("size=")[1].split(",")[0])
                                })
        
        return {
            'known_networks': set(networks),
            'known_hosts': set(known_hosts),
            'controlled_hosts': set(controlled_hosts),
            'services': services,
            'data': data
        }
        
    def extract_features(self, state_str: str) -> np.ndarray:
        """
        Extrae vector de características básicas del estado contando elementos
        
        Args:
            state_str: Representación del estado como string
                
        Returns:
            np.ndarray: Vector con cantidades de elementos por tipo [networks, hosts, controlled, services, data]
        """
        state = self._parse_state_string(state_str)
        features = []
        
        # Count known networks
        num_networks = len(state['known_networks'])
        features.append(num_networks)
        
        # Count known hosts
        num_known_hosts = len(state['known_hosts'])
        features.append(num_known_hosts)
        
        # Count controlled hosts
        num_controlled_hosts = len(state['controlled_hosts'])
        features.append(num_controlled_hosts)
        
        # Count total services across all hosts
        total_services = sum(len(services) for services in state['services'].values())
        features.append(total_services)
        
        # Count total data entries across all hosts
        total_data = sum(len(data_entries) for data_entries in state['data'].values())
        features.append(total_data)
        
        return np.array(features)