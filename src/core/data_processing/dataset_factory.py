"""
Dataset Factory Module

Consolidates the duplicate TurbulenceDataset instantiation patterns found across
6+ training files. Provides standardized dataset creation with predefined configurations.
"""

from typing import Dict, List, Optional, Tuple, Union
from src.core.data_processing.turbulence_dataset import TurbulenceDataset


class DatasetFactory:
    """Factory class for creating standardized TurbulenceDataset instances."""

    # Predefined dataset configurations to replace hardcoded parameters
    DATASET_CONFIGS = {
        'inc_training': {
            'name': "Training",
            'data_dirs': ["data"],
            'filter_top': ["128_inc"],
            'filter_sim': [(10, 81)],
            'filter_frame': [(800, 1300)],
            'exclude_filter_sim': True,
            'print_level': "sim"
        },
        'inc_test_low_rey': {
            'name': "Test Low Reynolds 100-200",
            'data_dirs': ["data"],
            'filter_top': ["128_inc"],
            'filter_sim': [[82, 84, 86, 88, 90]],
            'filter_frame': [(1000, 1150)],
            'sequence_length': [[60, 2]],
            'print_level': "sim"
        },
        'inc_test_high_rey': {
            'name': "Test High Reynolds 900-1000",
            'data_dirs': ["data"],
            'filter_top': ["128_inc"],
            'filter_sim': [[0, 2, 4, 6, 8]],
            'filter_frame': [(1000, 1150)],
            'sequence_length': [[60, 2]],
            'print_level': "sim"
        },
        'inc_test_var_rey': {
            'name': "Test Varying Reynolds Number (200-900)",
            'data_dirs': ["data"],
            'filter_top': ["128_reyVar"],
            'filter_sim': [[0]],
            'filter_frame': [(300, 800)],
            'sequence_length': [[250, 2]],
            'print_level': "sim"
        },
        'iso_training': {
            'name': "Training",
            'data_dirs': ["data"],
            'filter_top': ["128_iso"],
            'filter_sim': [(200, 351)],
            'exclude_filter_sim': True,
            'filter_frame': [(0, 1000)],
            'print_level': "sim"
        },
        'iso_test': {
            'name': "Test Isotropic Turbulence",
            'data_dirs': ["data"],
            'filter_top': ["128_iso"],
            'filter_sim': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
            'filter_frame': [(400, 500)],
            'sequence_length': [[60, 1]],
            'print_level': "sim"
        },
        'tra_training': {
            'name': "Training",
            'data_dirs': ["data"],
            'filter_top': ["128_tra"],
            'filter_sim': [[0, 1, 2, 14, 15, 16, 17, 18]],
            'exclude_filter_sim': True,
            'filter_frame': [(0, 1000)],
            'print_level': "sim"
        },
        'tra_test': {
            'name': "Test Transonic Cylinder Flow",
            'data_dirs': ["data"],
            'filter_top': ["128_tra"],
            'filter_sim': [[19, 20, 21, 22, 23, 24, 25, 26, 27]],
            'filter_frame': [(400, 500)],
            'sequence_length': [[60, 2]],
            'print_level': "sim"
        }
    }

    @classmethod
    def create_dataset(
        cls,
        config_name: str,
        sequence_length: Optional[List[List[int]]] = None,
        rand_seq_offset: bool = True,
        sim_fields: Optional[List[str]] = None,
        sim_params: Optional[List[str]] = None,
        override_config: Optional[Dict] = None
    ) -> TurbulenceDataset:
        """
        Create a TurbulenceDataset with predefined or custom configuration.

        Args:
            config_name: Name of predefined configuration from DATASET_CONFIGS
            sequence_length: Sequence length override
            rand_seq_offset: Whether to use random sequence offset
            sim_fields: Simulation fields to load
            sim_params: Simulation parameters
            override_config: Additional configuration overrides

        Returns:
            Configured TurbulenceDataset instance

        Raises:
            ValueError: If config_name is not found in DATASET_CONFIGS
        """
        if config_name not in cls.DATASET_CONFIGS:
            available_configs = list(cls.DATASET_CONFIGS.keys())
            raise ValueError(f"Unknown config '{config_name}'. Available: {available_configs}")

        # Get base configuration
        config = cls.DATASET_CONFIGS[config_name].copy()

        # Apply overrides
        if override_config:
            config.update(override_config)

        # Override specific parameters if provided
        if sequence_length is not None:
            config['sequence_length'] = sequence_length
        if sim_fields is not None:
            config['sim_fields'] = sim_fields
        if sim_params is not None:
            config['sim_params'] = sim_params

        # Set default values for common parameters
        config.setdefault('sequence_length', [[2, 2]])
        config.setdefault('rand_seq_offset', rand_seq_offset)
        config.setdefault('sim_fields', ["pres"])
        config.setdefault('sim_params', [])
        config.setdefault('exclude_filter_sim', False)

        # Create dataset
        return TurbulenceDataset(
            name=config['name'],
            dataDirs=config['data_dirs'],
            filterTop=config.get('filter_top'),
            filterSim=config.get('filter_sim'),
            excludefilterSim=config.get('exclude_filter_sim', False),
            filterFrame=config.get('filter_frame'),
            sequenceLength=config['sequence_length'],
            randSeqOffset=config['rand_seq_offset'],
            simFields=config['sim_fields'],
            simParams=config['sim_params'],
            printLevel=config.get('print_level', 'none')
        )

    @classmethod
    def create_training_dataset(
        cls,
        dataset_type: str,
        sequence_length: Optional[List[List[int]]] = None,
        sim_fields: Optional[List[str]] = None,
        sim_params: Optional[List[str]] = None,
        **kwargs
    ) -> TurbulenceDataset:
        """
        Create a training dataset for the specified type.

        Args:
            dataset_type: Type of dataset ('inc', 'iso', 'tra')
            sequence_length: Sequence length configuration
            sim_fields: Fields to load
            sim_params: Simulation parameters
            **kwargs: Additional overrides

        Returns:
            Training dataset instance
        """
        config_name = f"{dataset_type}_training"
        return cls.create_dataset(
            config_name=config_name,
            sequence_length=sequence_length,
            sim_fields=sim_fields,
            sim_params=sim_params,
            override_config=kwargs
        )

    @classmethod
    def create_test_datasets(
        cls,
        dataset_type: str,
        sim_fields: Optional[List[str]] = None,
        sim_params: Optional[List[str]] = None
    ) -> Dict[str, TurbulenceDataset]:
        """
        Create all test datasets for the specified type.

        Args:
            dataset_type: Type of dataset ('inc', 'iso', 'tra')
            sim_fields: Fields to load
            sim_params: Simulation parameters

        Returns:
            Dictionary mapping test set names to dataset instances
        """
        test_configs = {
            'inc': ['inc_test_low_rey', 'inc_test_high_rey', 'inc_test_var_rey'],
            'iso': ['iso_test'],
            'tra': ['tra_test']
        }

        if dataset_type not in test_configs:
            raise ValueError(f"Unknown dataset type '{dataset_type}'. Available: {list(test_configs.keys())}")

        test_datasets = {}
        for config_name in test_configs[dataset_type]:
            # Extract short name for the key
            short_name = config_name.replace(f'{dataset_type}_test_', '').replace(f'{dataset_type}_test', 'test')
            if short_name.startswith('inc_'):
                short_name = short_name[4:]  # Remove 'inc_' prefix

            test_datasets[short_name] = cls.create_dataset(
                config_name=config_name,
                sim_fields=sim_fields,
                sim_params=sim_params
            )

        return test_datasets

    @classmethod
    def list_available_configs(cls) -> List[str]:
        """List all available dataset configurations."""
        return list(cls.DATASET_CONFIGS.keys())

    @classmethod
    def get_config_info(cls, config_name: str) -> Dict:
        """Get information about a specific configuration."""
        if config_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown config '{config_name}'")
        return cls.DATASET_CONFIGS[config_name].copy()


# Convenience functions for backward compatibility
def create_inc_datasets(sequence_length=None, sim_fields=None, sim_params=None):
    """Create Inc dataset and test sets (backward compatibility)."""
    factory = DatasetFactory()
    train_set = factory.create_training_dataset('inc', sequence_length, sim_fields, sim_params)
    test_sets = factory.create_test_datasets('inc', sim_fields, sim_params)
    return train_set, test_sets


def create_iso_datasets(sequence_length=None, sim_fields=None, sim_params=None):
    """Create Iso dataset and test sets (backward compatibility)."""
    factory = DatasetFactory()
    train_set = factory.create_training_dataset('iso', sequence_length, sim_fields, sim_params)
    test_sets = factory.create_test_datasets('iso', sim_fields, sim_params)
    return train_set, test_sets


def create_tra_datasets(sequence_length=None, sim_fields=None, sim_params=None):
    """Create Tra dataset and test sets (backward compatibility)."""
    factory = DatasetFactory()
    train_set = factory.create_training_dataset('tra', sequence_length, sim_fields, sim_params)
    test_sets = factory.create_test_datasets('tra', sim_fields, sim_params)
    return train_set, test_sets