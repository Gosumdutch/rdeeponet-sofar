#!/usr/bin/env python3
"""
R-DeepONet Data Factory - Environment File Generator (PYAT Version)
PYAT 라이브러리를 사용한 BELLHOP/KRAKEN 환경파일 생성기

Author: R-DeepONet Data Factory Architect
License: GADIST
"""

import os
import sys
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

logger = logging.getLogger(__name__)

# PYAT 라이브러리 import
sys.path.append('./PYAT-main')
from wbellhopenvfil import wbellhopenvfil
from wkrakenenvfil import wkrakenenvfil

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnvironmentGenerator:
    """PYAT 기반 BELLHOP/KRAKEN 환경파일(.env) 생성 클래스"""
    
    def __init__(self, temp_directory: str = "./temp"):
        """
        Args:
            temp_directory (str): 임시 파일이 저장될 디렉토리
        """
        self.temp_dir = Path(temp_directory)
        self.temp_dir.mkdir(exist_ok=True)
        self.last_bty_path = None
        self.last_bathymetry_source = "UNKNOWN"
        self.last_zmax_m: float = None
        logger.info(f"Init ✓ - PYAT EnvironmentGenerator ready at {self.temp_dir}")
    
    def generate_munk_profile(self, depth: float = 5000.0, 
                                     surface_speed: float = 1548.52,
                                     channel_axis: float = 1234.5,
                                     gradient: float = 0.00737) -> List[Tuple[float, float]]:
        """하위호환성을 위한 퍼블릭 메소드"""
        profile_array = self._generate_munk_profile(depth, surface_speed)
        return [(z, c) for z, c in zip(profile_array[0], profile_array[1])]
    
    def _load_gebco_bathymetry(self) -> Tuple[np.ndarray, float]:
        """GEBCO NetCDF 파일에서 제주 해협 bathymetry 로딩
        
        Returns:
            Tuple[np.ndarray, float]: (depth_profile, max_depth)
        """
        gebco_file = os.environ.get('GEBCO_FILE')
        if not gebco_file or not os.path.exists(gebco_file):
            logger.warning("GEBCO file not found, using default 180m depth")
            return np.array([0.0, 180.0]), 180.0
            
        if not XARRAY_AVAILABLE:
            logger.warning("xarray not available, using default 180m depth")
            return np.array([0.0, 180.0]), 180.0
            
        try:
            # xarray로 GEBCO NetCDF 파일 로딩 (scipy 백엔드 사용)
            gebco_data = xr.open_dataset(gebco_file, engine='scipy')
            
            # GEBCO elevation data (negative = below sea level)
            elevation = gebco_data['elevation'].values
            # Convert to positive depth values
            depths = -elevation
            # Filter for reasonable ocean depths (remove land areas)
            ocean_depths = depths[depths > 0]
            
            if len(ocean_depths) > 0:
                # Use 95th percentile depth and clamp to [200, 5000] m for stability
                max_depth = float(np.percentile(ocean_depths, 95))
                max_depth = float(np.clip(max_depth, 200.0, 5000.0))
            else:
                max_depth = 180.0
                
            gebco_data.close()
            logger.info(f"Done ✓ - GEBCO bathymetry loaded: max_depth={max_depth:.1f}m")
            return np.array([0.0, max_depth]), max_depth
                
        except Exception as e:
            logger.warning(f"GEBCO loading failed: {e}, using default 180m")
            return np.array([0.0, 180.0]), 180.0
    
    def _generate_munk_profile(self, depth: float = 5000.0, 
                              surface_speed: float = 1548.52) -> np.ndarray:
        """Munk 음속 프로파일 생성 (PYAT 형식)
        
        Returns:
            np.ndarray: shape (2, N) - [depth, sound_speed] 형태의 2D 배열
        """
        depths = np.array([
            0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0,
            2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0, 3400.0, 3600.0,
            3800.0, 4000.0, 4200.0, 4400.0, 4600.0, 4800.0, 5000.0
        ])
        
        # Munk profile formula: c(z) = channel_axis * [1 + eps * (exp(-eta) + eta - 1)]
        channel_axis = 1234.5  # axis depth
        eps = 0.00737  # perturbation parameter
        eta = 2.0 * (depths - channel_axis) / channel_axis
        
        speeds = surface_speed * (1 + eps * (np.exp(-eta) + eta - 1))
        speeds = np.maximum(speeds, 1480.0)  # minimum speed
        
        return np.array([depths, speeds])
        
    def _generate_gebco_sound_speed_profile(self, max_depth: float = 180.0) -> np.ndarray:
        """GEBCO 기반 제주 해협용 음속 프로파일 생성
        
        Args:
            max_depth: 최대 수심 (m)
            
        Returns:
            np.ndarray: shape (2, N) - [depth, sound_speed] 형태의 2D 배열
        """
        # 제주 해협 특성에 맞는 얕은 수심 프로파일
        depths = np.linspace(0.0, max_depth, 20)
        
        # 얕은 바다 음속 프로파일 (온도/염분 고려)
        surface_speed = 1520.0  # 제주 해역 표면 음속
        bottom_speed = 1530.0   # 해저 음속 (약간 증가)
        
        # 선형 증가 프로파일
        speeds = surface_speed + (bottom_speed - surface_speed) * (depths / max_depth)
        
        return np.array([depths, speeds])

    def generate_bellhop_env(self, params: Dict[str, Any], output_path: str, 
                           run_type: str = "ray") -> str:
        """
        BELLHOP 환경파일 생성 (최종 수동 생성 방식 - 완전 제어)
        
        Args:
            params: 파라미터 딕셔너리 
            output_path: 출력 경로 (확장자 제외)
            run_type: "ray" 또는 "tl"
            
        Returns:
            str: 생성된 .env 파일 경로
        """
        try:
            # 파라미터 추출
            frequency = float(params['frequency'])
            source_depth = float(params['source_depth'])
            scenario_name = params.get('scenario_name', 'Unknown')
            
            # 그리드 해상도 및 물리 공간 파라미터 (config.yaml에서 동적으로)
            grid_res = params.get('simulation_space', {}).get('grid_resolution', [256, 256])
            depth_points = grid_res[0]
            range_points = grid_res[1]
            max_range_km = params.get('simulation_space', {}).get('max_range_km', 100.0)
            max_depth_m = params.get('simulation_space', {}).get('max_depth_m', 5000.0)
            
            logger.info(f"Processing... - Generate BELLHOP env: {run_type}")
            
            # 경로 설정
            env_file = f"{output_path}.env"
            bty_file = f"{output_path}.bty"
            
            # 음속 프로파일 생성 (Munk 프로파일)
            depths, speeds = self._generate_munk_profile(
                depth=max_depth_m,
                surface_speed=1548.52
            )
            
            # BELLHOP 환경파일 수동 작성 (우리 요구사항에 100% 맞춤)
            with open(env_file, 'w', encoding='utf-8', newline='\n') as f:
                # Title (must be first line)
                title_line = f"'R-DeepONet BELLHOP {run_type.upper()} - {frequency}Hz'\n"
                f.write(title_line)
                f.flush()
                
                # Frequency  
                f.write(f"{frequency}\n")
                
                # Source count
                f.write("1\n")
                
                # Options (SVF: S=spline fit, V=volume attenuation, F=Fortran format)
                f.write("'SVF'\n")
                
                # Sound speed profile
                f.write(f"{len(depths)} 0.0 {max_depth_m:.1f}\n")
                for z, c in zip(depths, speeds):
                    f.write(f"{z:.1f} {c:.2f} /\n")
                
                # Surface boundary ('A': Acoustically soft)
                f.write("'A' 0.0\n")
                
                # Bottom boundary (standard 4-parameter format)
                f.write(f"{max_depth_m:.1f} 1600.0 0.0 1.8 0.8 /\n")
                
                # Source depth
                f.write("1\n")
                f.write(f"{source_depth} /\n")
                
                # 수신기 깊이 배열 (동적으로 grid_resolution에서)
                f.write(f"{depth_points}\n")  # NRD - 동적으로 설정
                f.write(f"0.0 {max_depth_m:.1f} /\n")
                
                # 수신기 거리 배열 (동적으로 grid_resolution에서)
                f.write(f"{range_points}\n")  # NR - 동적으로 설정
                # (중요!) BELLHOP은 거리를 km 단위로 기대함
                f.write(f"0.0 {max_range_km:.1f} /\n")  # km 단위 사용!
                
                # Run type
                if run_type == "ray":
                    f.write("'R'\n")  # Ray trace
                else:
                    f.write("'C'\n")  # Coherent TL
                
                # Beam parameters (dynamic from config)
                beam_list = params.get('beam_angles', [])
                try:
                    nalpha = int(len(beam_list)) if isinstance(beam_list, (list, tuple, np.ndarray)) and len(beam_list) > 0 else 101
                    alpha_min = float(np.min(beam_list)) if nalpha > 0 else -20.0
                    alpha_max = float(np.max(beam_list)) if nalpha > 0 else 20.0
                except Exception:
                    nalpha, alpha_min, alpha_max = 101, -20.0, 20.0
                f.write(f"{nalpha}\n")
                f.write(f"{alpha_min:.6f} {alpha_max:.6f} /\n")
                
                # Step, zbox, rbox (BELLHOP 매뉴얼 권장: step=0.0 자동선택)
                f.write(f"0.0 {max_depth_m + 1.0:.1f} {max_range_km + 1.0:.1f}\n")
            
            # Generate bathymetry file (no scenario-based branching)
            paths_cfg = params.get('paths', {})
            gebco_path = (
                paths_cfg.get('gepco_file') or
                paths_cfg.get('gebco_file') or
                os.environ.get('GEBCO_FILE') or
                os.environ.get('GEBCO')
            )
            try:
                if gebco_path and Path(gebco_path).exists():
                    logger.info("Processing... - Generate GEBCO bathymetry (auto)")
                    grid_res = params.get('simulation_space', {}).get('grid_resolution', [256, 256])
                    self.create_bty_from_gebco(gebco_path, bty_file,
                                               target_lat=33.0, num_points=grid_res[1], max_range_km=max_range_km)
                    self.last_bathymetry_source = "GEBCO"
                else:
                    # Fallback to flat bathymetry
                    logger.info("Processing... - Generate flat bathymetry (auto)")
                    with open(bty_file, 'w') as f:
                        f.write("'L'\n")  # Linear interpolation
                        f.write("2\n")    # Number of points
                        f.write(f"0.0 {max_depth_m:.1f}\n")
                        f.write(f"{max_range_km:.1f} {max_depth_m:.1f}\n")  # km 단위
                    self.last_bathymetry_source = "FLAT"
            except Exception:
                # Robust fallback
                with open(bty_file, 'w') as f:
                    f.write("'L'\n")
                    f.write("2\n")
                    f.write(f"0.0 {max_depth_m:.1f}\n")
                    f.write(f"{max_range_km:.1f} {max_depth_m:.1f}\n")
                self.last_bathymetry_source = "FLAT"

            # Track last generated bathymetry file path for downstream plumbing
            self.last_bty_path = bty_file
            
            logger.info(f"Done ✓ - BELLHOP .env generated: {env_file}")
            return env_file
            
        except Exception as e:
            logger.error(f"BELLHOP env generation failed: {e}")
            raise

    def generate_kraken_env_munk_template(self, params: Dict[str, Any], output_path: str) -> str:
        """MATLAB tests/Munk/MunkK.env 포맷을 그대로 재현하여 .env 생성
        - GEBCO 비사용, 깊은 평저(5000 m)
        - 옵션: 'NVW' (Thorp 흡수 미사용)
        - cLow/cHigh: 1500 1600
        - RMAX: 0 (모드만 계산; 격자는 .flp에서 정의)
        - NSD: 1, SD: params['source_depth']
        - NRD/RD: grid_resolution에 맞춰 설정
        """
        logger.info("Processing... - Generate KRAKEN env (MunkK template)")
        freq = float(params.get('frequency', 50.0))
        sd = float(params.get('source_depth', 1000.0))
        grid_res = params.get('simulation_space', {}).get('grid_resolution', [256, 256])
        nrd = int(grid_res[0])
        max_depth_m = float(params.get('simulation_space', {}).get('max_depth_m', 5000.0))

        # Munk SSP 표 (MATLAB MunkK.env 값)
        munk_table = [
            (0.0, 1548.52), (200.0, 1530.29), (250.0, 1526.69), (400.0, 1517.78),
            (600.0, 1509.49), (800.0, 1504.30), (1000.0, 1501.38), (1200.0, 1500.14),
            (1400.0, 1500.12), (1600.0, 1501.02), (1800.0, 1502.57), (2000.0, 1504.62),
            (2200.0, 1507.02), (2400.0, 1509.69), (2600.0, 1512.55), (2800.0, 1515.56),
            (3000.0, 1518.67), (3200.0, 1521.85), (3400.0, 1525.10), (3600.0, 1528.38),
            (3800.0, 1531.70), (4000.0, 1535.04), (4200.0, 1538.39), (4400.0, 1541.76),
            (4600.0, 1545.14), (4800.0, 1548.52), (5000.0, 1551.91)
        ]

        env_file = f"{output_path}.env"
        with open(env_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write("'BBMunk profile'\n")
            f.write(f"{freq:.0f}\n")
            f.write("1\n")  # NENV
            f.write("'NVW'\n")  # 옵션 (No Thorp)
            f.write(f"0  0.0  {max_depth_m:.1f}\n")
            f.write("    0.0  1548.52  0.0  1.0  0.0  0.0\n")
            for z, c in munk_table[1:]:
                f.write(f"  {z:.1f}  {c:.2f} /\n")
            f.write("'A' 0.0\n")
            f.write(" 5000.0  1600.00 0.0 1.8 0.8 /\n")
            f.write("1500  1600\n")  # cLow cHigh
            f.write("0\n")              # RMAX (km)
            f.write("1\n")              # NSD
            f.write(f"   {sd:.1f} /\n")    # SD(1)
            f.write(f"{nrd}\n")         # NRD
            f.write(f"  0  {max_depth_m:.0f} /\n")  # RD range

        logger.info(f"Done ✓ - KRAKEN .env (MunkK template): {env_file}")
        return env_file

    def generate_kraken_env(self, params: Dict[str, Any], output_path: str) -> str:
        """KRAKEN 환경파일 생성 (PYAT wkrakenenvfil 사용)
        
        Args:
            params: 파라미터 딕셔너리 (frequency, source_depth, etc.)
            output_path: 출력 .env 파일 경로 (확장자 제외)
            
        Returns:
            str: 생성된 .env 파일의 전체 경로
        """
        logger.info("Processing... - Generate KRAKEN env")
        
        # Source data
        freq = params['frequency']
        zs = np.array([params['source_depth']])
        
        source_data = {"zs": zs, "f": freq}
        
        # Surface data (vacuum boundary)
        surface_data = {
            "bc": "V",  # Vacuum
            "properties": [],
            "reflection": []
        }
        
        # Scatter data (no scattering)
        scatter_data = {
            "bumden": [],
            "eta": [],
            "xi": []
        }
        
        # Sound speed profile - GEBCO 기반으로 변경
        _, max_depth = self._load_gebco_bathymetry()
        ssp_profile = self._generate_gebco_sound_speed_profile(max_depth)
        z = ssp_profile[0]
        c = ssp_profile[1]
        # remember shallow zmax for downstream visualization/metadata
        try:
            self.last_zmax_m = float(max_depth)
        except Exception:
            self.last_zmax_m = None
        
        # PYAT wkrakenenvfil 중복 방지: 마지막 포인트 제거
        if len(z) > 1:
            z = z[:-1]
            c = c[:-1]
        nz = len(z)
        
        # KRAKEN requires additional acoustic parameters
        cs = np.zeros(nz)  # shear speed
        rho = np.ones(nz)  # density
        apt = np.zeros(nz)  # attenuation (p-wave)
        ast = np.zeros(nz)  # attenuation (s-wave)
        
        ssp_data = {
            "cdata": np.array([z, c, cs, rho, apt, ast]),
            "type": "H",  # Halfspace
            "itype": "N",  # No interpolation
            "nmesh": 0,   # Auto mesh
            "sigma": 0.0, # Surface roughness
            "clow": 1300.0,
            "chigh": 15000.0,
            "zbottom": float(max(z))
        }
        
        # Bottom data (exactly like PYAT example: n=3, layert='HH')
        dmax = float(max(z))
        water_cp_top = float(c[0]) if c.size > 0 else 1500.0
        water_cp_bot = float(c[-1]) if c.size > 0 else 1600.0
        z1 = max(50.0, 0.80 * dmax)
        z2 = max(z1 + 10.0, 0.95 * dmax)
        layer_info = np.array([
            [z1, water_cp_top, 0.0, 1.0, 0.0, 0.0],
            [z2, water_cp_bot, 0.0, 1.0, 0.0, 0.0],
            [dmax, 2000.0, 0.0, 1.8, 0.1, 0.0]
        ], dtype=np.float64)

        layerp = np.array([[0.0, 0.0, layer_info[1,0]],
                           [0.0, 0.0, layer_info[2,0]]], dtype=np.float64)
        layert = 'HH'
        properties = np.array(layer_info[2, :6], dtype=np.float64)
        m1 = np.array([layer_info[0], layer_info[0]], dtype=np.float64)
        m2 = np.array([layer_info[1], layer_info[1]], dtype=np.float64)
        bdata = np.array([m1, m2], dtype=np.float64)
        bdata[0,1,0] = layer_info[1,0]
        bdata[1,1,0] = layer_info[2,0]

        # IMPORTANT: For SCOOTER, write a single bottom halfspace to avoid inserting
        # intermediate layer property lines that can confuse SSP parsing.
        # nlayers=1 prevents wkrakenenvfil from emitting extra 'layerp' blocks that
        # introduced non-monotonic depth entries in SSP.
        bottom_data = {
            "n": 1,
            "layerp": layerp,   # kept for API completeness; not used when n=1
            "layert": layert,   # kept for API completeness; not used when n=1
            "properties": properties,  # halfspace properties
            "bdata": bdata,     # kept for API completeness; not used when n=1
            "units": "W",
            "bc": "A",
            "sigma": 0.0,
        }
        
        # Field computation parameters - 동적으로 grid_resolution에서 가져옴 (float64 강제)
        grid_res = params.get('simulation_space', {}).get('grid_resolution', [256, 256])
        rmax_km = float(params.get('simulation_space', {}).get('max_range_km', 100.0))
        nrd = int(grid_res[0])
        rd = np.linspace(0.0, dmax, nrd, dtype=np.float64)
        nrr = int(grid_res[1])
        rr = np.linspace(0.0, rmax_km, nrr, dtype=np.float64)
        dr = np.zeros(nrd, dtype=np.float64)
        
        field_data = {
            "rmax": rmax_km,
            "nrr": nrr,
            "rr": rr,
            "rp": 0,
            "np": 1,
            "m": 999,
            "rmodes": "A",
            "stype": "R",
            "thorpe": " ",
            "finder": " ",
            "rd": rd,
            "dr": dr,
            "nrd": nrd
        }
        
        # Generate filename without extension
        basename = str(Path(output_path).with_suffix(''))
        
        try:
            # Call PYAT function
            wkrakenenvfil(
                filename=basename,
                thetitle=f"R-DeepONet KRAKEN TL - {freq}Hz",
                source_info=source_data,
                surface_info=surface_data,
                scatter_info=scatter_data,
                ssp_info=ssp_data,
                bottom_info=bottom_data,
                field_info=field_data
            )
            
            env_file = basename + '.env'
            logger.info(f"Done ✓ - KRAKEN .env generated: {env_file}")
            return env_file
            
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: KRAKEN env generation failed: {e}")
            raise

    def generate_scooter_env(self, params: Dict[str, Any], output_path: str) -> str:
        """SCOOTER용 .env 파일 생성 (GEBCO 기반)
        
        Args:
            params: 파라미터 딕셔너리 (frequency, source_depth, etc.)
            output_path: 출력 .env 파일 경로 (확장자 제외)
            
        Returns:
            str: 생성된 .env 파일 경로
        """
        logger.info("Processing... - Generate SCOOTER env (GEBCO-based)")
        freq = params['frequency']
        source_depth = params['source_depth']
        
        # GEBCO 데이터 로딩 및 음속 프로파일 생성
        _, max_depth = self._load_gebco_bathymetry()
        # remember shallow zmax for downstream visualization/metadata
        try:
            self.last_zmax_m = float(max_depth)
        except Exception:
            self.last_zmax_m = None
        ssp_profile = self._generate_gebco_sound_speed_profile(max_depth)
        z = ssp_profile[0]
        c = ssp_profile[1]
        
        # Keep full SSP to reach zbottom
        z = z.copy()
        c = c.copy()
        
        # SCOOTER용 파라미터 (KRAKEN과 거의 동일)
        source_data = {"zs": np.array([source_depth]), "f": freq}
        surface_data = {"bc": 'V', "properties": [], "reflection": []}
        scatter_data = {"bumden": [], "eta": [], "xi": []}
        
        nz = z.size
        cs = np.zeros(nz)
        rho = np.ones(nz)
        apt = np.zeros(nz)
        ast = np.zeros(nz)
        ssp_data = {
            "cdata": np.array([z, c, cs, rho, apt, ast], dtype=np.float64),
            "type": "H", "itype": "N", "nmesh": 0, "sigma": 0.0,
            "clow": 1450.0, "chigh": 1600.0, "zbottom": float(max_depth)
        }
        
        dmax = float(max(z))
        # Bottom (SCOOTER): define local layer_info (same structure as KRAKEN)
        water_cp_top = float(c[0]) if c.size > 0 else 1500.0
        water_cp_bot = float(c[-1]) if c.size > 0 else 1600.0
        z1 = max(50.0, 0.80 * dmax)
        z2 = max(z1 + 10.0, 0.95 * dmax)
        layer_info = np.array([
            [z1, water_cp_top, 0.0, 1.0, 0.0, 0.0],
            [z2, water_cp_bot, 0.0, 1.0, 0.0, 0.0],
            [dmax, 2000.0, 0.0, 1.8, 0.1, 0.0]
        ], dtype=np.float64)

        layerp = np.array([[0.0, 0.0, layer_info[1,0]],
                           [0.0, 0.0, layer_info[2,0]]], dtype=np.float64)
        layert = 'HH'
        properties = np.array(layer_info[2, :6], dtype=np.float64)
        m1 = np.array([layer_info[0], layer_info[0]], dtype=np.float64)
        m2 = np.array([layer_info[1], layer_info[1]], dtype=np.float64)
        bdata = np.array([m1, m2], dtype=np.float64)
        bdata[0,1,0] = layer_info[1,0]
        bdata[1,1,0] = layer_info[2,0]
        # Use single bottom halfspace to avoid interface lines in ENV (SCOOTER-friendly)
        bottom_data = {
            "n": 1,
            "layerp": layerp,
            "layert": layert,
            "properties": properties,
            "bdata": bdata,
            "units": "W",
            "bc": "A",
            "sigma": 0.0,
        }
        
        # Field computation parameters - 동적으로 grid_resolution에서 가져옴 (SCOOTER)
        grid_res = params.get('simulation_space', {}).get('grid_resolution', [256, 256])
        rmax_km = params.get('simulation_space', {}).get('max_range_km', 100.0)
        nrd = grid_res[0]    # depth points - 동적으로 설정
        rd = np.linspace(0, dmax, nrd)
        nrr = grid_res[1]    # range points - 동적으로 설정
        rr = np.linspace(0.0, rmax_km, nrr, dtype=np.float64)
        dr = np.zeros(nrd, dtype=np.float64)
        
        field_data = {
            "rmax": rmax_km, "nrr": nrr, "rr": rr.astype(np.float64), "rp": 0, "np": 1,
            "m": 999, "rmodes": "A", "stype": "R", "thorpe": "T",
            "finder": " ", "rd": rd.astype(np.float64), "dr": dr.astype(np.float64), "nrd": nrd
        }
        
        basename = str(Path(output_path).with_suffix(''))
        
        try:
            # Manual SCOOTER .env write (match hupo.py format exactly)
            env_file = basename + '.env'
            with open(env_file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(f"'R-DeepONet SCOOTER TL - {float(freq):.0f}Hz'\n")
                f.write(f"{float(freq):.0f}\n")
                f.write("1\n")
                f.write("'NVWT '\n")
                # Nmesh sigma zbottom
                f.write(f"0 0.0 {float(max_depth):.1f}\n")
                # SSP lines (z c /), last line must be zbottom with trailing '/'
                for zi, ci in zip(z, c):
                    if abs(float(zi) - float(max_depth)) < 1e-6:
                        f.write(f"{float(max_depth):.1f} {float(ci):.1f} /\n")
                    else:
                        f.write(f"{float(zi):.10g} {float(ci):.10g} /\n")
                # Surface and bottom
                f.write("'A' 0.0\n")
                f.write(f"{float(max_depth):.1f} 2000.0 0.0 1.8 0.1 0.0 /\n")
                # cLow cHigh
                f.write("1450.0 1600.0\n")
                # Grid control (RMAX, NSD/SD, NRD/RD)
                f.write(f"{float(rmax_km):.0f}\n")
                f.write("1\n")
                f.write(f"{float(source_depth):.1f} /\n")
                f.write(f"{int(grid_res[0])}\n")
                f.write(f"0.0 {float(max_depth):.1f} /\n")
            logger.info(f"Done ✓ - SCOOTER .env generated: {env_file}")
            
            # Write field.flp compatible with SCOOTER expectations
            try:
                thetitle = f"R-DeepONet SCOOTER TL - {freq}Hz"
                options2 = "'RG   '"  # FIELD: read GRN
                mlimit = 9999
                nprofiles = 1
                rprofiles = 0
                nrr = int(grid_res[1])
                rr0 = 0.0
                rr1 = float(rmax_km)
                nzs = 1
                nrd = int(grid_res[0])
                rd0 = 0.0
                rd1 = float(max_depth)

                env_dir = Path(env_file).parent
                flp_env = env_dir / 'field.flp'
                with open(flp_env, 'w', encoding='utf-8', newline='\n') as g:
                    # FIELD 'RG' format (match hupo.py exactly)
                    g.write("/ ,\n".replace(' ', ''))   # '/,'
                    g.write(options2 + "\n")          # 'RG   '
                    g.write(f"{mlimit}\n")           # 9999
                    g.write(f"{nprofiles}\n")        # 1
                    g.write(f"     {rr0:.1f} {rr1:.1f} /\n")  # RPROF(1)
                    g.write(f"{nrr}\n")              # NRR
                    g.write(f"  {rr0:.1f}  {rr1:.1f} /\n")     # RMIN RMAX (km)
                    g.write(f"{nzs}\n")              # NSD
                    g.write(f"   {float(source_depth):.1f} /\n")  # SD (m)
                    g.write(f"{nrd}\n")              # NRD
                    g.write(f"  {rd0:.1f} {rd1:.1f} /\n")       # RD (m)
                    g.write(f"{nrr}\n")              # NRR (offsets)
                    g.write("     0.0 /\n")           # RR offsets (m)

                # 2) Also write into project root (when running with cwd=project)
                try:
                    flp_root = Path.cwd() / 'field.flp'
                    with open(flp_root, 'w', encoding='utf-8', newline='\n') as g:
                        g.write(thetitle + "\n")
                        g.write(options2 + "\n")
                        g.write(f"{mlimit}\n")
                        g.write(f"{nprofiles}\n")
                        g.write(f"{rprofiles}\n")
                        g.write(f"{nrr}\n")
                        g.write(f"{rr0:.1f} {rr1:.1f} /\n")
                        g.write(f"{nzs}\n")
                        g.write(f"   {float(source_depth):.1f} /\n")
                        g.write(f"{nrd}\n")
                        g.write(f"  {rd0:.1f} {rd1:.1f} /\n")
                        g.write(f"{nrd}\n")
                        g.write(f"  {dr0:.1f} {dr1:.1f} /\n")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"field.flp write failed (non-critical): {e}")

            return env_file

        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: SCOOTER env generation failed: {e}")
            raise

    def generate_file_pair(self, params: Dict[str, Any], 
                          basename: str) -> Tuple[str, str]:
        """Ray용과 TL용 환경파일 쌍 생성 (시나리오별 분기)
        
        Args:
            params: 파라미터 딕셔너리
            basename: 기본 파일명
            
        Returns:
            Tuple[str, str]: (ray_env_path, tl_env_path)
        """
        ray_path = str(self.temp_dir / f"{basename}_ray")
        tl_path = str(self.temp_dir / f"{basename}_tl")
        
        # BELLHOP ray 파일 생성 (모든 시나리오 공통)
        ray_env = self.generate_bellhop_env(params, ray_path, "ray")
        
        # TL 파일 생성 (시나리오별 분기)
        scenario_name = params.get('scenario_name', 'JASA_Benchmark')
        
        if scenario_name == 'Defense_Wideband':
            logger.info("Processing... - Generate SCOOTER env for Defense_Wideband")
            tl_env = self.generate_scooter_env(params, tl_path)
        elif scenario_name == 'Munk_Profile_Benchmark':
            logger.info("Processing... - Generate KRAKEN env (Munk template) for Munk_Profile_Benchmark")
            tl_env = self.generate_kraken_env_munk_template(params, tl_path)
        else:
            logger.info("Processing... - Generate KRAKEN env (Munk template) for JASA_Benchmark (match MATLAB)")
            tl_env = self.generate_kraken_env_munk_template(params, tl_path)
        
        return ray_env, tl_env

    def create_kraken_flp_file(self, params: Dict[str, Any], output_path_no_ext: str) -> str:
        """KRAKEN FIELD(.flp) 파일 생성
        
        Args:
            params: config에서 전달된 파라미터 딕셔너리
            output_path_no_ext: 확장자 없는 출력 경로(예: temp/job0000_JASA_Benchmark_tl)
        
        Returns:
            str: 생성된 .flp 파일 경로
        """
        try:
            import shutil
            # 동적 파라미터
            src_depth = float(params.get('source_depth', 1000.0))
            sim = params.get('simulation_space', {})
            max_range_km = float(sim.get('max_range_km', 100.0))
            max_depth_m = float(sim.get('max_depth_m', 5000.0))
            grid = sim.get('grid_resolution', [256, 256])
            n_depth = int(grid[0])
            n_range = int(grid[1])

            flp_path = f"{output_path_no_ext}.flp"

            # 항상 시나리오별 .flp를 새로 생성(MATLAB MunkK.flp 포맷 준수)
            with open(flp_path, 'w') as f:
                title = f"/"  # MATLAB 예제는 첫 줄 주석/타이틀 표기
                f.write(f"{title},\n")
                f.write("'RA'\n")                # Option
                f.write("9999\n")               # Mlimit
                f.write("1\n")                  # NPROF
                f.write(f"     0.0 {max_range_km:.1f} /\t\t! NPROF,  RPROF(1:NPROF) (km)\n")
                f.write(f"{n_range}\n")          # NRR
                f.write(f" 0.0  {max_range_km:.1f} /\t\t! RMIN,   RMAX (km), NR\n")
                f.write("1\n")                  # NSD
                f.write(f"   {src_depth:.1f} /\t\t! SD (m)\n")
                f.write(f"{n_depth}\n")          # NRD
                f.write(f"  0.0 {max_depth_m:.1f} /\t\t! RD (m)\n")
                f.write(f"{n_depth}\n")          # NRR (offsets)
                f.write("     0.0 /\t\t! RR offsets (m)\n")
            logger.info(f"Done ✓ - KRAKEN .flp generated (fresh): {flp_path}")
            return flp_path

            logger.info(f"Done ✓ - KRAKEN .flp generated: {flp_path}")
            return flp_path
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: KRAKEN .flp generation failed: {e}")
            raise

    def validate_env_file(self, env_path: str) -> bool:
        """환경파일 유효성 검사 (하위호환성)"""
        try:
            return Path(env_path).exists() and Path(env_path).stat().st_size > 0
        except:
            return False

    def create_bty_from_gebco(self, nc_filepath: str, output_bty_path: str, 
                             target_lat: float = 33.0, num_points: int = 256, 
                             max_range_km: float = 100.0) -> str:
        """
        GEBCO NetCDF 파일에서 특정 위도의 단면을 추출하여 BELLHOP .bty 파일을 생성합니다.
        
        Args:
            nc_filepath: GEBCO NetCDF 파일 경로
            output_bty_path: 출력 .bty 파일 경로
            target_lat: 대상 위도 (기본값: 33.0 - 제주 남방)
            num_points: 그리드 포인트 수 (기본값: 256)
            max_range_km: 최대 거리 (기본값: 100km)
            
        Returns:
            str: 생성된 .bty 파일 경로
        """
        logger.info(f"Processing... - Generate .bty from GEBCO: {nc_filepath}")
        
        try:
            # 1. NetCDF 파일 로드
            logger.info(f"Loading GEBCO data: {nc_filepath}")
            gebco_data = xr.open_dataset(nc_filepath, engine='scipy')
            
            # 2. 특정 위도(target_lat)의 지형 단면 데이터 추출
            logger.info(f"Extracting topography at latitude {target_lat}°N")
            topography_slice = gebco_data['elevation'].sel(lat=target_lat, method='nearest')
            
            # 3. 수심 데이터 처리 (해발고도 -> m 단위 양수 깊이로 변환)
            depths_m = -topography_slice.values
            
            # 음수 깊이 (해수면 위) 처리 - 최소 수심 10m로 설정
            depths_m = np.maximum(depths_m, 10.0)
            
            # 4. 거리 데이터 생성 및 리샘플링
            lons = topography_slice.lon.values
            km_per_degree = 111.320 * np.cos(np.deg2rad(target_lat))
            original_ranges_km = (lons - lons[0]) * km_per_degree
            
            # 우리 시뮬레이션 공간(0-100km)에 맞게 256개 포인트로 보간
            simulation_ranges_km = np.linspace(0, max_range_km, num_points)
            interpolated_depths_m = np.interp(simulation_ranges_km, original_ranges_km, depths_m)
            
            # 5. BELLHOP .bty 파일 형식으로 저장
            with open(output_bty_path, 'w') as f:
                f.write("'L'\n")  # Piecewise Linear interpolation
                f.write(f"{num_points}\n")
                for r, d in zip(simulation_ranges_km, interpolated_depths_m):
                    f.write(f"{r:.4f}  {d:.2f}\n")
            
            # 6. 데이터셋 메모리 해제
            gebco_data.close()
            
            logger.info(f"Done ✓ - GEBCO .bty created: {output_bty_path}")
            logger.info(f"   Range: 0-{max_range_km}km, Depth: {interpolated_depths_m.min():.1f}-{interpolated_depths_m.max():.1f}m")
            
            return output_bty_path
            
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: GEBCO .bty generation failed: {e}")
            raise


def generate_env_file(params: Dict[str, Any], output_path: str, 
                     run_type: str, title_suffix: str = "", 
                     model: str = "bellhop") -> str:
    """환경파일 생성 통합 인터페이스 (하위호환성)
    
    Args:
        params: 파라미터 딕셔너리
        output_path: 출력 경로
        run_type: "RRR" (ray) 또는 "CRR" (coherent TL)
        title_suffix: 제목 suffix (무시됨)
        model: "bellhop" 또는 "kraken"
        
    Returns:
        str: 생성된 .env 파일 경로
    """
    generator = EnvironmentGenerator()
    
    # Convert run_type to simple format
    if run_type.upper() in ["RRR", "RAY"]:
        simple_type = "ray"
    else:
        simple_type = "tl"
    
    if model.lower() == "kraken":
        return generator.generate_kraken_env(params, output_path)
    else:
        return generator.generate_bellhop_env(params, output_path, simple_type)


# Backward compatibility functions
def write_env_file(params: Dict, output_path: str, run_type: str, 
                  title_suffix: str, dim: int = 2) -> str:
    """하위호환성을 위한 래퍼 함수"""
    return generate_env_file(params, output_path, run_type, title_suffix, "bellhop")


if __name__ == "__main__":
    # 테스트 코드
    logger.info("Testing PYAT EnvironmentGenerator...")
    
    test_params = {
        'frequency': 100.0,
        'source_depth': 100.0,
        'receiver_depth': 200.0
    }
    
    generator = EnvironmentGenerator()
    
    # BELLHOP 테스트
    bellhop_env = generator.generate_bellhop_env(test_params, "./temp/test_bellhop", "ray")
    print(f"BELLHOP env: {bellhop_env}")
    
    # KRAKEN 테스트  
    kraken_env = generator.generate_kraken_env(test_params, "./temp/test_kraken")
    print(f"KRAKEN env: {kraken_env}")
    
    logger.info("Done ✓ - All tests passed")