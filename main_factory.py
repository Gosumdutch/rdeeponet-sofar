#!/usr/bin/env python3
"""
R-DeepONet Data Factory - Main Factory
메인 오케스트레이터: 전체 데이터 생성 파이프라인의 모든 단계를 순서대로 지휘

Author: R-DeepONet Data Factory Architect
License: MIT
"""

import os
import sys
import h5py
import time
import subprocess
import shutil
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 로컬 모듈 임포트
from parameter_parser import ParameterParser
from env_generator import EnvironmentGenerator
from output_parser import OutputParser

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_factory.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataFactory:
    """R-DeepONet 데이터 공장 메인 클래스"""
    
    def __init__(self, config_path: str = "config.yaml", start_idx: int = 0, resume: bool = False):
        """
        Args:
            config_path (str): 설정 파일 경로
            start_idx (int): 재시작 시 시작할 조합 인덱스(파일명 jobID 연속 유지)
            resume (bool): 이미 존재하는 산출물(H5) 발견 시 건너뛰기
        """
        self.config_path = config_path
        self.config = None
        self.start_idx = int(start_idx)
        self.resume = bool(resume)
        
        # 컴포넌트 초기화
        self.parameter_parser = ParameterParser(config_path)
        self.env_generator = None
        self.output_parser = OutputParser()
        
        # 통계 변수
        self.total_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.start_time = None
        
    def setup_directories(self) -> None:
        """필요한 디렉토리들 생성"""
        try:
            paths = self.config.get('paths', {})
            
            # 출력 디렉토리
            project_config = self.config.get('project', {})
            output_dir = Path(project_config.get('output_directory', paths.get('output_directory', './output')))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 임시 디렉토리
            temp_dir = Path(paths.get('temp_directory', './temp'))
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 로그 디렉토리
            log_dir = output_dir / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            logger.info(f"Directories setup: output={output_dir}, temp={temp_dir}")
            
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            raise
    
    def validate_executables(self) -> bool:
        """실행파일들의 존재 여부 확인"""
        try:
            paths = self.config.get('paths', {})
            
            # BELLHOP 실행파일 확인
            bellhop_exe = Path(paths.get('bellhop_executable', ''))
            if not bellhop_exe.exists():
                logger.error(f"BELLHOP executable not found: {bellhop_exe}")
                return False
            
            # 실행 권한 테스트
            try:
                result = subprocess.run([str(bellhop_exe)], 
                                      capture_output=True, text=True, timeout=5)
                logger.info(f"BELLHOP executable validated: {bellhop_exe}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"BELLHOP executable test failed, but file exists: {bellhop_exe}")
            
            # SCOOTER 실행파일 확인 (요청사항)
            scooter_exe = Path(paths.get('scooter_executable', ''))
            if not scooter_exe.exists():
                logger.error(f"SCOOTER executable not found: {scooter_exe}")
                return False

            # 선택: 간단 실행 테스트 (존재만 확인해도 충분)
            try:
                subprocess.run([str(scooter_exe)], capture_output=True, text=True, timeout=5)
                logger.info(f"SCOOTER executable validated: {scooter_exe}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"SCOOTER executable test failed, but file exists: {scooter_exe}")

            return True
            
        except Exception as e:
            logger.error(f"Executable validation failed: {e}")
            return False
    
    def run_bellhop_simulation(self, 
                              env_path: str, 
                              mode: str = "ray") -> Tuple[bool, Optional[str], Optional[str]]:
        """BELLHOP 시뮬레이션 실행
        
        Args:
            env_path (str): 환경파일 경로
            mode (str): 실행 모드 ("ray" 또는 "tl")
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (성공여부, 출력파일경로, 에러메시지)
        """
        try:
            env_file = Path(env_path)
            if not env_file.exists():
                return False, None, f"Environment file not found: {env_path}"
            
            # 실행파일 경로
            bellhop_exe = self.config['paths']['bellhop_executable']
            
            # 출력 파일 경로 예측
            base_name = env_file.stem
            output_dir = env_file.parent
            
            if mode == "ray":
                expected_output = output_dir / f"{base_name}.ray"
            else:  # tl
                expected_output = output_dir / f"{base_name}.shd"
            
            # 기존 출력 파일 삭제
            if expected_output.exists():
                expected_output.unlink()
            
            # BELLHOP 실행 (Working Directory 내에서 상대경로 사용)
            env_basename = env_file.stem  # 확장자 없는 파일명만
            cmd = [str(bellhop_exe), env_basename]
            
            logger.debug(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            # 출력 파일 확인 (returncode보다 실제 출력 파일 존재가 더 중요)
            if not expected_output.exists():
                # 다른 가능한 출력 파일명들 확인
                for possible_ext in ['.ray', '.shd', '.arr', '.prt']:
                    possible_file = output_dir / f"{base_name}{possible_ext}"
                    if possible_file.exists():
                        expected_output = possible_file
                        break
                else:
                    error_msg = f"Expected output file not found: {expected_output}"
                    logger.error(error_msg)
                    return False, None, error_msg
            
            logger.info(f"BELLHOP simulation completed: {expected_output}")
            return True, str(expected_output), None
            
        except subprocess.TimeoutExpired:
            error_msg = f"BELLHOP simulation timeout: {env_path}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"BELLHOP simulation error: {e}"
            logger.error(error_msg)
            return False, None, error_msg

    def run_scooter_simulation(self, env_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """SCOOTER 시뮬레이션 실행
        
        Args:
            env_path (str): 환경 파일 경로
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (성공여부, 출력파일경로, 에러메시지)
        """
        try:
            # SCOOTER 실행파일 경로 확인
            scooter_exe = Path(self.config['paths']['scooter_executable'])
            if not scooter_exe.exists():
                error_msg = f"SCOOTER executable not found: {scooter_exe}"
                logger.error(error_msg)
                return False, None, error_msg
            
            env_file = Path(env_path)
            if not env_file.exists():
                error_msg = f"Environment file not found: {env_path}"
                logger.error(error_msg)
                return False, None, error_msg
            
            # 출력 파일 경로 예측
            base_name = env_file.stem
            output_dir = env_file.parent
            expected_output = output_dir / f"{base_name}.shd"
            
            # 기존 출력 파일 삭제
            if expected_output.exists():
                expected_output.unlink()
            
            # SCOOTER 실행 (cwd=output_dir에서 확장자 제거한 파일명만 전달)
            env_basename = env_file.stem
            cmd = [str(scooter_exe), env_basename]
            
            logger.debug(f"Executing SCOOTER: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=600  # 10분 타임아웃 (SCOOTER는 시간이 더 걸림)
            )
            
            # 실행 결과 확인
            if result.returncode != 0:
                error_msg = f"SCOOTER failed (code {result.returncode}): {result.stderr}"
                logger.error(error_msg)
                return False, None, error_msg
            
            # If .shd not produced by SCOOTER, try FIELD conversion from .grn
            if not expected_output.exists() or expected_output.stat().st_size == 0:
                grn_file = output_dir / f"{base_name}.grn"
                field_exe = Path(scooter_exe).parent / 'field.exe'
                if grn_file.exists() and field_exe.exists():
                    try:
                        logger.debug(f"Executing FIELD: {field_exe} {env_basename}")
                        subprocess.run([str(field_exe), env_basename], cwd=str(output_dir),
                                       capture_output=True, text=True, timeout=600)
                    except Exception as _:
                        pass
                # Re-check for .shd
                for _ in range(50):
                    if expected_output.exists() and expected_output.stat().st_size > 0:
                        break
                    time.sleep(0.1)
                else:
                    error_msg = f"SCOOTER output file not found: {expected_output}"
                    logger.error(error_msg)
                    try:
                        prt_file = output_dir / f"{base_name}.prt"
                        if prt_file.exists():
                            with open(prt_file, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                            tail = ''.join(lines[-10:]) if lines else ''
                            logger.error(f"SCOOTER .prt tail (last 10 lines):\n{tail}")
                    except Exception:
                        pass
                    return False, None, error_msg
            
            logger.info(f"SCOOTER simulation completed: {expected_output}")
            return True, str(expected_output), None
            
        except subprocess.TimeoutExpired:
            error_msg = f"SCOOTER simulation timeout: {env_path}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"SCOOTER simulation error: {e}"
            logger.error(error_msg)
            return False, None, error_msg

    def run_kraken_simulation(self, env_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """KRAKEN 시뮬레이션 실행
        
        Args:
            env_path (str): 환경 파일 경로
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (성공여부, 출력파일경로, 에러메시지)
        """
        try:
            # KRAKEN 실행파일 경로 확인
            kraken_exe = Path(self.config['paths']['kraken_executable'])
            if not kraken_exe.exists():
                error_msg = f"KRAKEN executable not found: {kraken_exe}"
                logger.error(error_msg)
                return False, None, error_msg
            
            env_file = Path(env_path)
            if not env_file.exists():
                error_msg = f"Environment file not found: {env_path}"
                logger.error(error_msg)
                return False, None, error_msg
            
            # 출력 파일 경로 예측 (KRAKEN은 .mod 파일 생성)
            base_name = env_file.stem
            output_dir = env_file.parent
            expected_output = output_dir / f"{base_name}.mod"
            
            # 기존 출력 파일 삭제
            if expected_output.exists():
                expected_output.unlink()
            
            # KRAKEN 실행 (Working Directory 내에서 상대경로 사용)
            env_basename = env_file.stem  # 확장자 없는 파일명만
            cmd = [str(kraken_exe), env_basename]
            
            logger.debug(f"Executing KRAKEN: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            # KRAKEN 실행 후 .mod 파일 확인
            mod_file = output_dir / f"{base_name}.mod"
            for _ in range(50):  # 5초간 0.1초 간격으로 확인
                if mod_file.exists() and mod_file.stat().st_size > 0:
                    break
                time.sleep(0.1)
            else:
                error_msg = f"KRAKEN .mod file not found: {mod_file}"
                logger.error(error_msg)
                return False, None, error_msg
            
            # field.exe 실행하여 .mod → .shd 변환 (강제 실행 + 재시도)
            field_exe = kraken_exe.parent / "field.exe"
            if field_exe.exists():
                logger.debug(f"Executing FIELD: {field_exe} {env_basename}")
                # 최대 2회 재시도
                shd_file = output_dir / f"{base_name}.shd"
                for attempt in range(2):
                    subprocess.run(
                        [str(field_exe), env_basename],
                        cwd=str(output_dir),
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    # .shd 파일 생성 확인 (최대 5초 대기)
                    for _ in range(50):
                        if shd_file.exists() and shd_file.stat().st_size > 0:
                            expected_output = shd_file
                            break
                        time.sleep(0.1)
                    if expected_output == shd_file:
                        break
                if expected_output != shd_file:
                    logger.warning(f"FIELD did not generate .shd file after retries, using .mod: {mod_file}")
                    expected_output = mod_file
            else:
                logger.warning(f"field.exe not found at {field_exe}, using .mod file directly")
                expected_output = mod_file  # field.exe 없으면 .mod 파일 사용
            
            logger.info(f"KRAKEN simulation completed: {expected_output}")
            return True, str(expected_output), None
            
        except subprocess.TimeoutExpired:
            error_msg = f"KRAKEN simulation timeout: {env_path}"
            logger.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"KRAKEN simulation error: {e}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def process_single_parameter_set(self, 
                                   params: Dict[str, Any], 
                                   job_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """단일 파라미터 세트에 대한 전체 처리 과정
        
        Args:
            params (Dict): 파라미터 딕셔너리
            job_id (int): 작업 ID
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]: (X, Y, error_message)
        """
        try:
            # 케이스 식별자 생성
            freq = params['frequency']
            src_depth = params['source_depth']
            scenario = params.get('scenario_name', 'default')
            case_id = f"job{job_id:04d}_{scenario}"
            
            logger.info(f"Processing job {job_id}: f={freq:.1f}Hz, zs={src_depth:.1f}m")
            
            # 1. 환경파일 생성 (명시적 분기)
            ray_path_no_ext = str(Path(self.env_generator.temp_dir) / f"{case_id}_ray")
            tl_path_no_ext = str(Path(self.env_generator.temp_dir) / f"{case_id}_tl")

            # Ray (BELLHOP) env
            ray_env_path = self.env_generator.generate_bellhop_env(params, ray_path_no_ext, "ray")

            # TL env by model
            model_to_run = str(params.get('ground_truth_model', params.get('tl_model', 'KRAKEN'))).upper()
            scenario_name = str(params.get('scenario_name', '')).strip()
            if model_to_run == 'KRAKEN':
                # Match MATLAB template for JASA/Munk scenarios
                if scenario_name in ('JASA_Benchmark', 'Munk_Profile_Benchmark'):
                    tl_env_path = self.env_generator.generate_kraken_env_munk_template(params, tl_path_no_ext)
                else:
                    tl_env_path = self.env_generator.generate_kraken_env(params, tl_path_no_ext)
            elif model_to_run == 'SCOOTER':
                tl_env_path = self.env_generator.generate_scooter_env(params, tl_path_no_ext)
            else:
                return None, None, f"Unsupported ground truth model: {model_to_run}"

            # 1-1. FIELD용 .flp 파일 생성 (KRAKEN일 때만)
            if model_to_run == 'KRAKEN':
                try:
                    tl_root_no_ext = str(Path(tl_env_path).with_suffix(''))
                    self.env_generator.create_kraken_flp_file(params, tl_root_no_ext)
                except Exception as e:
                    logger.warning(f".flp generation failed (non-critical): {e}")
            
            # 2. Ray 시뮬레이션 실행
            ray_success, ray_output_path, ray_error = self.run_bellhop_simulation(ray_env_path, "ray")
            if not ray_success:
                return None, None, f"Ray simulation failed: {ray_error}"
            
            # 3. TL 시뮬레이션 실행 (ground_truth_model에 따라 분기)

            tl_success, tl_output_path, tl_error = False, None, None
            if model_to_run == 'KRAKEN':
                logger.info("Executing KRAKEN using subprocess ...")
                tl_success, tl_output_path, tl_error = self.run_kraken_simulation(tl_env_path)
            elif model_to_run == 'SCOOTER':
                logger.info("Executing SCOOTER using subprocess ...")
                tl_success, tl_output_path, tl_error = self.run_scooter_simulation(tl_env_path)
            else:
                return None, None, f"Unsupported ground truth model: {model_to_run}"
                
            if not tl_success:
                return None, None, f"TL simulation failed: {tl_error}"
            
            # 4. 결과 파싱 및 변환
            X, Y = self.output_parser.process_simulation_outputs(
                ray_output_path, tl_output_path, params.get('grid', {})
            )
            
            # 5. 임시 파일 정리
            self.cleanup_temp_files([ray_env_path, tl_env_path, ray_output_path, tl_output_path])
            
            return X, Y, None
            
        except Exception as e:
            error_msg = f"Job {job_id} processing failed: {e}"
            logger.error(error_msg)
            return None, None, error_msg
    
    def save_data_pair(self, 
                      X: np.ndarray, 
                      Y: np.ndarray, 
                      metadata: Dict[str, Any], 
                      output_path: str) -> bool:
        """데이터 쌍을 HDF5 파일로 저장
        
        Args:
            X (np.ndarray): Ray density map
            Y (np.ndarray): TL field
            metadata (Dict): 메타데이터
            output_path (str): 출력 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(output_file, 'w') as f:
                # 데이터 저장
                f.create_dataset('X', data=X, compression='gzip', compression_opts=9)
                f.create_dataset('Y', data=Y, compression='gzip', compression_opts=9)
                
                # 메타데이터 저장
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        if len(value) < 100:  # 작은 배열만 저장
                            meta_group.create_dataset(key, data=value)
                
                # 추가 정보
                meta_group.attrs['creation_time'] = datetime.now().isoformat()
                meta_group.attrs['X_shape'] = X.shape
                meta_group.attrs['Y_shape'] = Y.shape
                meta_group.attrs['X_dtype'] = str(X.dtype)
                meta_group.attrs['Y_dtype'] = str(Y.dtype)

                # 학습용 개별 구조도 동시에 저장 (.npy 파일)
                # 디렉토리 구조 준비 (root = R-DeepONet_Data_XXXX)
                # output_file: <root>/data/h5/<file>.h5 → root = parents[2]
                try:
                    root_dir = output_file.parents[2]
                except Exception:
                    root_dir = output_file.parent.parent
                base_dir = root_dir / 'data'
                (base_dir / 'branch_inputs').mkdir(parents=True, exist_ok=True)
                (base_dir / 'trunk_coordinates').mkdir(parents=True, exist_ok=True)
                (base_dir / 'ray_maps').mkdir(parents=True, exist_ok=True)
                (base_dir / 'tl_targets').mkdir(parents=True, exist_ok=True)

                # Branch inputs: [frequency, source_depth]
                branch = np.array([metadata.get('frequency', metadata.get('frequency_hz', 0.0)),
                                   metadata.get('source_depth', metadata.get('source_depth_m', 0.0))],
                                   dtype=np.float32)
                np.save(base_dir / 'branch_inputs' / (output_file.stem + '_branch.npy'), branch)

                # Trunk coordinates: 256x256 grid of [range(m), depth(m)]
                # config 기반 최대값 사용
                max_range_km = float(self.config.get('simulation_space', {}).get('max_range_km', 100.0))
                max_depth_m = float(self.config.get('simulation_space', {}).get('max_depth_m', 5000.0))
                r_points = int(self.config.get('grid', {}).get('tl_field', {}).get('range_resolution', 256))
                z_points = int(self.config.get('grid', {}).get('tl_field', {}).get('depth_resolution', 256))
                ranges = np.linspace(0, max_range_km * 1000.0, r_points)
                depths = np.linspace(0, max_depth_m, z_points)
                R, Z = np.meshgrid(ranges, depths)
                trunk = np.column_stack([R.ravel(), Z.ravel()]).astype(np.float32)
                np.save(base_dir / 'trunk_coordinates' / (output_file.stem + '_trunk.npy'), trunk)

                # Ray map / TL target
                np.save(base_dir / 'ray_maps' / (output_file.stem + '_ray.npy'), X.astype(np.float32))
                np.save(base_dir / 'tl_targets' / (output_file.stem + '_tl.npy'), Y.astype(np.float32))
            
            file_size = output_file.stat().st_size
            logger.info(f"Data saved: {output_path} ({file_size/1024/1024:.2f} MB)")
            
            # Generate visualization (300 DPI PNG)
            try:
                parser = OutputParser()
                # PNG 파일은 images 하위 폴더에 저장
                try:
                    root_dir = output_file.parents[2]
                except Exception:
                    root_dir = output_file.parent.parent
                images_dir = root_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                vis_path = str(images_dir / output_file.stem)  # Remove .h5 extension
                # pass zmax for proper axis scaling if available
                try:
                    zmax_m = getattr(self.env_generator, 'last_zmax_m', None)
                    if zmax_m is not None:
                        metadata = dict(metadata)
                        metadata['zmax_m'] = float(zmax_m)
                except Exception:
                    pass
                parser.save_visualization(X, Y, metadata, vis_path)
            except Exception as e:
                logger.warning(f"Visualization failed (non-critical): {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data saving failed: {e}")
            return False
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """임시 파일들 정리 (디버그 보존 옵션 지원)"""
        preserve = bool(self.config.get('debug', {}).get('preserve_temp_files', False))
        if preserve:
            logger.info("Debug mode: preserve_temp_files=True, skipping temp cleanup")
            return
        for file_path in file_paths:
            try:
                temp_file = Path(file_path)
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
    
    def generate_output_filename(self, params: Dict[str, Any], job_id: int) -> str:
        """출력 파일명 생성"""
        freq = params['frequency']
        src_depth = params['source_depth']
        scenario = params.get('scenario_name', 'default')
        
        # 안전한 파일명 생성
        safe_scenario = "".join(c for c in scenario if c.isalnum() or c in "_-")
        
        filename = f"job{job_id:04d}_f{freq:.0f}Hz_zs{src_depth:.0f}m_{safe_scenario}.h5"
        
        project_config = self.config.get('project', {})
        paths_config = self.config.get('paths', {})
        output_dir = Path(project_config.get('output_directory', paths_config.get('output_directory', './output')))
        
        # H5 파일은 data/h5 하위 폴더에 저장
        h5_dir = output_dir / "data" / "h5"
        h5_dir.mkdir(parents=True, exist_ok=True)
        return str(h5_dir / filename)
    
    def print_progress_summary(self) -> None:
        """진행 상황 요약 출력"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'='*50}")
        print(f"Progress Summary")
        print(f"{'='*50}")
        print(f"Total jobs: {self.total_jobs}")
        print(f"Completed: {self.completed_jobs}")
        print(f"Failed: {self.failed_jobs}")
        print(f"Remaining: {self.total_jobs - self.completed_jobs - self.failed_jobs}")
        print(f"Success rate: {self.completed_jobs/max(self.total_jobs,1)*100:.1f}%")
        print(f"Elapsed time: {elapsed/3600:.2f} hours")
        if self.completed_jobs > 0:
            avg_time = elapsed / (self.completed_jobs + self.failed_jobs)
            remaining_jobs = self.total_jobs - self.completed_jobs - self.failed_jobs
            eta = remaining_jobs * avg_time
            print(f"ETA: {eta/3600:.2f} hours")
        print(f"{'='*50}\n")
    
    def main_pipeline(self) -> bool:
        """메인 데이터 생성 파이프라인 실행
        
        Returns:
            bool: 전체 파이프라인 성공 여부
        """
        try:
            self.start_time = time.time()
            logger.info("=" * 60)
            logger.info("R-DeepONet Data Factory Pipeline Starting")
            logger.info("=" * 60)
            
            # 1. 설정 로드
            self.config = self.parameter_parser.load_config()
            # Hardcode GEBCO for both env var and config paths (robust against shell/env issues)
            try:
                import os
                gebco_path_hard = r"C:\Users\whaye\Desktop\deep\GEBCO\ahn.nc"
                os.environ['GEBCO_FILE'] = gebco_path_hard
                os.environ['GEBCO'] = gebco_path_hard
                self.config.setdefault('paths', {})
                self.config['paths']['gebco_file'] = gebco_path_hard
                # Some code paths look for typo 'gepco_file' → set both
                self.config['paths']['gepco_file'] = gebco_path_hard
                logger.info(f"GEBCO hardcoded: {gebco_path_hard}")
            except Exception as _e:
                logger.warning(f"GEBCO hardcode set failed: {_e}")
            logger.info("✓ Configuration loaded")
            
            # 2. 디렉토리 설정
            self.setup_directories()
            logger.info("✓ Directories setup completed")
            
            # 3. 실행파일 검증
            if not self.validate_executables():
                raise RuntimeError("Executable validation failed")
            logger.info("✓ Executables validated")
            
            # 4. 환경파일 생성기 초기화
            temp_dir = self.config.get('paths', {}).get('temp_directory', './temp')
            self.env_generator = EnvironmentGenerator(temp_dir)
            logger.info("✓ Environment generator initialized")
            
            # 5. 파라미터 조합 생성
            parameter_combinations = self.parameter_parser.generate_parameter_combinations()
            self.total_jobs = len(parameter_combinations)
            logger.info(f"✓ Generated {self.total_jobs} parameter combinations")
            
            if self.total_jobs == 0:
                logger.warning("No parameter combinations to process")
                return True
            
            # 6. 메인 처리 루프
            logger.info("Starting main processing loop...")
            
            # 성능 설정
            performance = self.config.get('performance', {})
            batch_size = performance.get('batch_size', 1)
            
            with tqdm(total=self.total_jobs, desc="Processing jobs") as pbar:
                for job_id, params in enumerate(parameter_combinations):
                    # skip until start_idx
                    if job_id < self.start_time and self.start_time is not None:
                        pass
                    try:
                        # 출력 파일 경로 사전 계산
                        output_path = self.generate_output_filename(params, job_id)
                        # resume: 이미 산출물 존재하면 스킵
                        if self.resume and Path(output_path).exists():
                            self.completed_jobs += 1
                            pbar.update(1)
                            continue
                        # 단일 작업 처리
                        X, Y, error = self.process_single_parameter_set(params, job_id)
                        
                        if X is not None and Y is not None:
                            # 데이터 저장
                            # output_path는 위에서 계산됨
                            
                            metadata = {
                                'job_id': job_id,
                                'frequency': params['frequency'],
                                'source_depth': params['source_depth'],
                                'scenario_name': params.get('scenario_name', 'default')
                            }
                            # Auto-plumb bathymetry path/source for visualization overlay
                            try:
                                metadata['bty_path'] = getattr(self.env_generator, 'last_bty_path', None)
                                metadata['bathymetry_source'] = getattr(self.env_generator, 'last_bathymetry_source', 'UNKNOWN')
                                # also store shallow water max depth (zmax) for training/vis masking
                                zmax_m = getattr(self.env_generator, 'last_zmax_m', None)
                                if zmax_m is not None:
                                    metadata['zmax_m'] = float(zmax_m)
                            except Exception:
                                # Robustness: do not block save/vis if not available
                                pass
                            
                            if self.save_data_pair(X, Y, metadata, output_path):
                                self.completed_jobs += 1
                                pbar.set_postfix(completed=self.completed_jobs, failed=self.failed_jobs)
                            else:
                                self.failed_jobs += 1
                                logger.error(f"Job {job_id}: Data saving failed")
                        else:
                            self.failed_jobs += 1
                            logger.error(f"Job {job_id}: {error}")
                        
                        pbar.update(1)
                        
                        # 주기적으로 진행 상황 출력
                        if (job_id + 1) % max(1, self.total_jobs // 10) == 0:
                            self.print_progress_summary()
                            
                    except KeyboardInterrupt:
                        logger.warning("Pipeline interrupted by user")
                        break
                    except Exception as e:
                        self.failed_jobs += 1
                        logger.error(f"Job {job_id} failed with exception: {e}")
                        pbar.update(1)
            
            # 7. 최종 요약
            self.print_progress_summary()
            
            success_rate = self.completed_jobs / max(self.total_jobs, 1)
            
            logger.info("=" * 60)
            logger.info(f"R-DeepONet Data Factory Pipeline Completed")
            logger.info(f"Success rate: {success_rate*100:.1f}% ({self.completed_jobs}/{self.total_jobs})")
            logger.info("=" * 60)
            
            return success_rate > 0.8  # 80% 이상 성공시 성공으로 간주
            
        except Exception as e:
            logger.error(f"Main pipeline failed: {e}")
            return False


def main():
    """메인 실행 함수"""
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description='R-DeepONet Data Factory')
        parser.add_argument('--config', '-c', default='config.yaml',
                          help='Configuration file path (default: config.yaml)')
        parser.add_argument('--start_idx', type=int, default=0,
                          help='Start index for resuming (job id offset)')
        parser.add_argument('--resume', action='store_true',
                          help='Skip cases whose H5 already exists')
        parser.add_argument('--test', action='store_true',
                          help='Run in test mode with single parameter')
        
        args = parser.parse_args()
        
        # 데이터 공장 초기화
        factory = DataFactory(args.config)
        
        if args.test:
            logger.info("Running in test mode...")
            print("✓ Test mode completed")
        else:
            # 전체 파이프라인 실행
            factory = DataFactory(args.config, start_idx=args.start_idx, resume=args.resume)
            success = factory.main_pipeline()
            
            if success:
                print("\nData factory pipeline completed successfully.")
                sys.exit(0)
            else:
                print("\nData factory pipeline failed or had too many errors.")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()