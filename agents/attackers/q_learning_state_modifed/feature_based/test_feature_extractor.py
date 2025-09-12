import unittest
import numpy as np
from feature_extractor import FeatureExtractor, FeatureExtractorConfig

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        """Initialize feature extractor and test states before each test"""
        self.feature_extractor = FeatureExtractor()
        
        # Test state with multiple networks, hosts, services and data
        self.complex_state = """nets:[192.168.1.0/24,192.168.2.0/24,192.168.3.0/24,213.47.23.192/26],hosts:[192.168.1.2,192.168.1.3,192.168.1.4,192.168.2.1,192.168.2.2,192.168.2.3,192.168.2.4,192.168.2.5,192.168.2.6,213.47.23.195],controlled:[192.168.2.2,192.168.2.4,192.168.2.6,213.47.23.195],services:{192.168.1.3:[Service(name='postgresql', type='passive', version='14.3.0', is_local=False),Service(name='ssh', type='passive', version='8.1.0', is_local=False)]192.168.2.2:[Service(name='ms-wbt-server', type='passive', version='10.0.19041', is_local=False)]192.168.2.3:[Service(name='ms-wbt-server', type='passive', version='10.0.19041', is_local=False)]192.168.2.4:[Service(name='bash', type='passive', version='5.0.0', is_local=True),Service(name='ssh', type='passive', version='8.1.0', is_local=False)]192.168.2.5:[Service(name='ssh', type='passive', version='8.1.0', is_local=False)]192.168.2.6:[Service(name='bash', type='passive', version='5.0.0', is_local=True)]213.47.23.195:[Service(name='bash', type='passive', version='5.0.0', is_local=True),Service(name='listener', type='passive', version='1.0.0', is_local=False)]},data:{192.168.2.2:[Data(owner='system', id='logfile', size=439, type='log')]192.168.2.4:[Data(owner='system', id='logfile', size=439, type='log')]213.47.23.195:[Data(owner='system', id='logfile', size=676, type='log')]},blocks:{}"""

        # Test state with minimal information
        self.minimal_state = """nets:[192.168.1.0/24],hosts:[192.168.1.2],controlled:[],services:{},data:{},blocks:{}"""
            
        # Test state with empty sections
        self.empty_state = """nets:[],hosts:[],controlled:[],services:{},data:{},blocks:{}"""
        
        # Test state with only services, no data
        self.services_only_state = """nets:[192.168.1.0/24],hosts:[192.168.1.2,192.168.1.3],controlled:[192.168.1.2],services:{192.168.1.2:[Service(name='ssh', type='passive', version='8.1.0', is_local=False)]192.168.1.3:[Service(name='http', type='passive', version='1.4.54', is_local=False),Service(name='mysql', type='passive', version='8.0.0', is_local=False)]},data:{},blocks:{}"""
        
        # Test state with only data, no services
        self.data_only_state = """nets:[192.168.1.0/24,192.168.2.0/24],hosts:[192.168.1.2,192.168.2.3],controlled:[192.168.1.2,192.168.2.3],services:{},data:{192.168.1.2:[Data(owner='user', id='config', size=256, type='config')]192.168.2.3:[Data(owner='admin', id='secrets', size=1024, type='credentials'),Data(owner='system', id='backup', size=2048, type='backup')]},blocks:{}"""
        
        # Test state with many services on single host
        self.many_services_state = """nets:[10.0.0.0/24],hosts:[10.0.0.100],controlled:[10.0.0.100],services:{10.0.0.100:[Service(name='ssh', type='passive', version='8.1.0', is_local=False),Service(name='http', type='passive', version='1.4.54', is_local=False),Service(name='https', type='passive', version='1.4.54', is_local=False),Service(name='mysql', type='passive', version='8.0.0', is_local=False),Service(name='ftp', type='passive', version='2.0.0', is_local=False),Service(name='smtp', type='passive', version='1.0.0', is_local=False)]},data:{},blocks:{}"""
        
        # Test state with many data entries on single host
        self.many_data_state = """nets:[172.16.0.0/24],hosts:[172.16.0.50],controlled:[172.16.0.50],services:{},data:{172.16.0.50:[Data(owner='user1', id='file1', size=100, type='document'),Data(owner='user2', id='file2', size=200, type='image'),Data(owner='admin', id='config', size=300, type='config'),Data(owner='system', id='logs', size=400, type='log'),Data(owner='backup', id='archive', size=500, type='backup')]},blocks:{}"""
        
        # Test state with mixed scenarios
        self.mixed_state = """nets:[192.168.0.0/24,10.0.0.0/24,172.16.0.0/24],hosts:[192.168.0.1,192.168.0.2,10.0.0.1,172.16.0.1],controlled:[192.168.0.1,10.0.0.1],services:{192.168.0.1:[Service(name='ssh', type='passive', version='8.1.0', is_local=False)]10.0.0.1:[Service(name='http', type='passive', version='1.4.54', is_local=False),Service(name='mysql', type='passive', version='8.0.0', is_local=False)]},data:{192.168.0.1:[Data(owner='user', id='test', size=123, type='test')]172.16.0.1:[Data(owner='admin', id='config', size=456, type='config')]},blocks:{}"""

    def test_extract_features_complex_state(self):
        """Test feature extraction from a complex state"""
        features = self.feature_extractor.extract_features(self.complex_state)
        
        expected = np.array([
            4,  # 4 networks
            10, # 10 known hosts
            4,  # 4 controlled hosts
            10, # 10 total services
            3   # 3 data entries
        ])
        
        np.testing.assert_array_equal(features, expected)
        self.assertEqual(len(features), 5)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_extract_features_empty_state(self):
        """Test feature extraction from an empty state"""
        features = self.feature_extractor.extract_features(self.empty_state)
        expected = np.array([0, 0, 0, 0, 0])
        np.testing.assert_array_equal(features, expected)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_extract_features_minimal_state(self):
        """Test feature extraction from a minimal state"""
        features = self.feature_extractor.extract_features(self.minimal_state)
        expected = np.array([1, 1, 0, 0, 0])
        np.testing.assert_array_equal(features, expected)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_extract_features_services_only(self):
        """Test feature extraction with only services, no data"""
        features = self.feature_extractor.extract_features(self.services_only_state)
        
        expected = np.array([
            1,  # 1 network
            2,  # 2 known hosts
            1,  # 1 controlled host
            3,  # 3 services (1 ssh + 2 on second host)
            0   # 0 data entries
        ])
        
        np.testing.assert_array_equal(features, expected)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_extract_features_data_only(self):
        """Test feature extraction with only data, no services"""
        features = self.feature_extractor.extract_features(self.data_only_state)
        
        expected = np.array([
            2,  # 2 networks
            2,  # 2 known hosts
            2,  # 2 controlled hosts
            0,  # 0 services
            3   # 3 data entries (1 + 2)
        ])
        
        np.testing.assert_array_equal(features, expected)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_extract_features_many_services(self):
        """Test feature extraction with many services on single host"""
        features = self.feature_extractor.extract_features(self.many_services_state)
        
        expected = np.array([
            1,  # 1 network
            1,  # 1 known host
            1,  # 1 controlled host
            6,  # 6 services on single host
            0   # 0 data entries
        ])
        
        np.testing.assert_array_equal(features, expected)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_extract_features_many_data(self):
        """Test feature extraction with many data entries on single host"""
        features = self.feature_extractor.extract_features(self.many_data_state)
        
        expected = np.array([
            1,  # 1 network
            1,  # 1 known host
            1,  # 1 controlled host
            0,  # 0 services
            5   # 5 data entries on single host
        ])
        
        np.testing.assert_array_equal(features, expected)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_extract_features_mixed_scenario(self):
        """Test feature extraction with mixed services and data distribution"""
        features = self.feature_extractor.extract_features(self.mixed_state)
        
        expected = np.array([
            3,  # 3 networks
            4,  # 4 known hosts
            2,  # 2 controlled hosts
            3,  # 3 services (1 + 2)
            2   # 2 data entries (1 + 1)
        ])
        
        np.testing.assert_array_equal(features, expected)
        # print(f"Expected: {expected}")
        # result = tuple(int(x) for x in features)
        # print(f"Resultado: {result}")

    def test_feature_vector_type(self):
        """Test that the feature vector is a numpy array with correct dtype"""
        features = self.feature_extractor.extract_features(self.complex_state)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.int64)

    def test_feature_vector_consistency(self):
        """Test that same input produces same output"""
        features1 = self.feature_extractor.extract_features(self.complex_state)
        features2 = self.feature_extractor.extract_features(self.complex_state)
        
        np.testing.assert_array_equal(features1, features2)

    def test_different_states_different_features(self):
        """Test that different states produce different feature vectors"""
        features_complex = self.feature_extractor.extract_features(self.complex_state)
        features_minimal = self.feature_extractor.extract_features(self.minimal_state)
        features_empty = self.feature_extractor.extract_features(self.empty_state)
        
        # All should be different
        self.assertFalse(np.array_equal(features_complex, features_minimal))
        self.assertFalse(np.array_equal(features_minimal, features_empty))
        self.assertFalse(np.array_equal(features_complex, features_empty))


def get_int(self, vec:np.ndarray) -> int:
    """
    Extrae características del estado y las usa como identificador
    """
    return tuple(np.round(vec, decimals=1))

if __name__ == '__main__':
    unittest.main()