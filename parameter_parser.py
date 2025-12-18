#!/usr/bin/env python3
"""
R-DeepONet Data Factory - Parameter Parser
파라미터 조합 생성기: config.yaml을 해석하여 모든 시뮬레이션 케이스 조합을 생성

Author: R-DeepONet Data Factory Architect
License: MIT
"""

import yaml
import numpy as np
import itertools
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParameterParser:
    """config.yaml 파싱 및 파라미터 조합 생성 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path (str): config.yaml 파일 경로
        """
        self.config_path = Path(config_path)
        self.config = None
        
    def load_config(self) -> Dict[str, Any]:
        """config.yaml 파일을 로드하고 유효성 검사
        
        Returns:
            Dict: 파싱된 설정 딕셔너리
            
        Raises:
            FileNotFoundError: 설정 파일이 존재하지 않을 때
            yaml.YAMLError: YAML 파싱 오류
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8-sig') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Config loaded: {self.config_path}")
            self._validate_config()
            return self.config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            raise
    
    def _validate_config(self) -> None:
        """설정 파일의 필수 섹션과 키 유효성 검사"""
        required_sections = ['scenarios', 'paths', 'grid']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # 시나리오 유효성 검사
        enabled_scenarios = [s for s in self.config['scenarios'] if s.get('enabled', False)]
        if not enabled_scenarios:
            raise ValueError("No enabled scenarios found")
        
        logger.info(f"Config validation passed. {len(enabled_scenarios)} scenarios enabled.")
    
    def generate_frequency_values(self, freq_config: Dict[str, Any]) -> List[float]:
        """주파수 범위 설정에서 실제 주파수 값들을 생성
        
        Args:
            freq_config (Dict): 주파수 설정 (values list 또는 start/stop range)
            
        Returns:
            List[float]: 생성된 주파수 값들 (Hz)
        """
        # New format: explicit values list (JASA_Benchmark)
        if 'values' in freq_config:
            return list(freq_config['values'])
        
        # Legacy format: start/stop ranges  
        start = freq_config['start']
        stop = freq_config['stop']
        scale = freq_config.get('scale', 'linear')
        
        if scale == 'linear':
            if 'step' in freq_config:
                values = np.arange(start, stop + freq_config['step'], freq_config['step'])
            elif 'num' in freq_config:
                values = np.linspace(start, stop, freq_config['num'])
            else:
                raise ValueError("Either 'step' or 'num' must be specified for linear scale")
                
        elif scale == 'log':
            if 'num' not in freq_config:
                raise ValueError("'num' must be specified for log scale")
            values = np.logspace(np.log10(start), np.log10(stop), freq_config['num'])
            
        else:
            raise ValueError(f"Unknown scale: {scale}. Use 'linear' or 'log'")
        
        return values.tolist()
    
    def generate_depth_values(self, depth_config: Dict[str, Any]) -> List[float]:
        """깊이 범위 설정에서 실제 깊이 값들을 생성
        
        Args:
            depth_config (Dict): 깊이 설정 (values list 또는 start/stop range)
            
        Returns:
            List[float]: 생성된 깊이 값들 (m)
        """
        # New format: explicit values list
        if 'values' in depth_config:
            return list(depth_config['values'])
        
        # Legacy format: list of values (backward compatibility)
        if isinstance(depth_config, list):
            return depth_config
        
        # Legacy format: start/stop range
        start = depth_config['start']
        stop = depth_config['stop']
        step = depth_config.get('step', 0)
        
        if step == 0:
            return [start]  # Single depth
        
        # Generate values from start to stop with step interval
        return np.arange(start, stop + step, step).tolist()
    
    def generate_range_values(self, range_config: Dict[str, Any]) -> List[float]:
        """거리 범위 설정에서 실제 거리 값들을 생성
        
        Args:
            range_config (Dict): 거리 설정
            
        Returns:
            List[float]: 생성된 거리 값들 (km)
        """
        if isinstance(range_config, list):
            return range_config
            
        start = range_config['start']
        stop = range_config['stop']
        num = range_config['num']
        scale = range_config.get('scale', 'linear')
        
        if scale == 'linear':
            values = np.linspace(start, stop, num)
        elif scale == 'log':
            values = np.logspace(np.log10(start), np.log10(stop), num)
        else:
            raise ValueError(f"Unknown scale: {scale}")
        
        return values.tolist()
    
    def generate_beam_angles(self, angle_config: Dict[str, Any]) -> List[float]:
        """빔 각도 범위 설정에서 실제 각도 값들을 생성
        
        Args:
            angle_config (Dict): 각도 설정
            
        Returns:
            List[float]: 생성된 각도 값들 (degrees)
        """
        start = angle_config['start']
        stop = angle_config['stop']
        num = angle_config['num']
        
        return np.linspace(start, stop, num).tolist()
    
    def generate_scenario_combinations(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """단일 시나리오에 대한 모든 파라미터 조합 생성
        
        Args:
            scenario (Dict): 시나리오 설정
            
        Returns:
            List[Dict]: 파라미터 조합 리스트
        """
        # 각 변수의 값 생성
        frequencies = self.generate_frequency_values(scenario['frequencies'])
        source_depths = self.generate_depth_values(scenario['source_depths'])
        receiver_depths = self.generate_depth_values(scenario['receiver_depths'])
        ranges = self.generate_range_values(scenario['ranges'])
        beam_angles = self.generate_beam_angles(scenario['beam_angles'])
        
        logger.info(f"Scenario '{scenario['name']}': "
                   f"{len(frequencies)} freqs × {len(source_depths)} src depths × "
                   f"{len(receiver_depths)} rcv depths = "
                   f"{len(frequencies) * len(source_depths) * len(receiver_depths)} combinations")
        
        # 모든 조합 생성 (Cartesian product)
        combinations = []
        
        for freq, src_depth, rcv_depths, beam_angles_set in itertools.product(
            frequencies, source_depths, [receiver_depths], [beam_angles]):
            
            param_dict = {
                'scenario_name': scenario['name'],
                'frequency': freq,
                'source_depth': src_depth,
                'receiver_depths': rcv_depths,
                'ranges': ranges,
                'beam_angles': beam_angles_set,
                'environment': self.config.get('environment', {}),
                'grid': self.config.get('grid', {}),
                'paths': self.config.get('paths', {}),
                'simulation_space': self.config.get('simulation_space', {}),
                'tl_model': scenario.get('tl_model', 'KRAKEN'),
                'ground_truth_model': scenario.get('ground_truth_model', scenario.get('tl_model', 'KRAKEN'))
            }
            combinations.append(param_dict)
        
        return combinations
    
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """모든 활성화된 시나리오에 대한 파라미터 조합 생성
        
        Returns:
            List[Dict]: 전체 파라미터 조합 리스트
        """
        if self.config is None:
            self.load_config()
        
        all_combinations = []
        
        for scenario in self.config['scenarios']:
            if scenario.get('enabled', False):
                scenario_combinations = self.generate_scenario_combinations(scenario)
                all_combinations.extend(scenario_combinations)
                logger.info(f"Added {len(scenario_combinations)} combinations from scenario '{scenario['name']}'")
        
        logger.info(f"Total parameter combinations generated: {len(all_combinations)}")
        return all_combinations
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """파라미터 조합의 유효성 검사
        
        Args:
            params (Dict): 파라미터 딕셔너리
            
        Returns:
            bool: 유효성 검사 결과
        """
        try:
            # 필수 키 확인
            required_keys = ['frequency', 'source_depth', 'receiver_depths', 'ranges', 'beam_angles']
            for key in required_keys:
                if key not in params:
                    logger.error(f"Missing required parameter: {key}")
                    return False
            
            # 값 범위 확인
            if params['frequency'] <= 0:
                logger.error(f"Invalid frequency: {params['frequency']}")
                return False
            
            if params['source_depth'] < 0:
                logger.error(f"Invalid source depth: {params['source_depth']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False


def main():
    """테스트 및 예제 실행 함수"""
    try:
        # 파라미터 파서 초기화
        parser = ParameterParser('config.yaml')
        
        # 설정 로드
        config = parser.load_config()
        print(f"✓ Config loaded from {parser.config_path}")
        
        # 파라미터 조합 생성
        combinations = parser.generate_parameter_combinations()
        print(f"✓ Generated {len(combinations)} parameter combinations")
        
        # 첫 번째 조합 출력 (예제)
        if combinations:
            print("\n=== First Parameter Combination ===")
            first_combo = combinations[0]
            for key, value in first_combo.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"{key}: [{value[0]:.2f}, ..., {value[-1]:.2f}] ({len(value)} values)")
                else:
                    print(f"{key}: {value}")
        
        print("\n✓ parameter_parser.py test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()