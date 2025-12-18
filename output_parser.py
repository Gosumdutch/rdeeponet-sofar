#!/usr/bin/env python3
"""
R-DeepONet Data Factory - Output Parser (PYAT Version)
PYAT 라이브러리를 사용한 시뮬레이션 결과 파서

Author: R-DeepONet Data Factory Architect
License: MIT
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import logging
from typing import Dict, List, Tuple, Any, Optional
from scipy.io import FortranFile  # kept for reference; custom block parser used instead
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# PYAT 라이브러리 import
sys.path.append('./PYAT-main')
from readshd import readshd
from plotray import plotray

def parse_ray_file_final(filepath):
    """
    BELLHOP .ray 파일의 반복 구조를 정확히 파싱
    Header 형식 (plotray.py 참조):
      1: title
      2: freq
      3: Nsxyz (Nsx Nsy Nsz)
      4: NBeamAngles (Nalpha Nbeta)
      5: DEPTHT
      6: DEPTHB
      7: Type
    반복: for isz in range(Nsz): for ibeam in range(Nalpha):
      angle
      nsteps NumTopBnc NumBotBnc
      [r z] x nsteps
    """
    try:
        with open(filepath, 'r') as f:
            # Some builds have no title line. Detect numeric-first.
            first = f.readline()
            if first is None:
                return {"rays": []}
            try:
                float(first.strip())
                # numeric → freq
                freq_line = first
                nsxyz_line = f.readline()
                nbeam_line = f.readline()
                depth_top = f.readline()
                depth_bot = f.readline()
                typeline = f.readline()
            except ValueError:
                # non-numeric → title exists
                freq_line = f.readline()
                nsxyz_line = f.readline()
                nbeam_line = f.readline()
                depth_top = f.readline()
                depth_bot = f.readline()
                typeline = f.readline()
            if not (nsxyz_line and nbeam_line and typeline):
                return {"rays": []}

            ns_tokens = nsxyz_line.strip().split()
            nb_tokens = nbeam_line.strip().split()
            if len(ns_tokens) < 3 or len(nb_tokens) < 2:
                return {"rays": []}
            Nsx, Nsy, Nsz = map(int, ns_tokens[:3])
            Nalpha, Nbeta = map(int, nb_tokens[:2])

            all_rays = []
            for _isz in range(Nsz):
                for _ in range(Nalpha):
                    angle_line = f.readline()
                    if not angle_line:
                        break
                    angle_line = angle_line.strip()
                    if angle_line == "":
                        continue
                    launch_angle = float(angle_line)

                    counts_line = f.readline()
                    if not counts_line:
                        break
                    counts = counts_line.strip().split()
                    if len(counts) < 1:
                        continue
                    try:
                        nsteps = int(counts[0])
                    except ValueError:
                        continue

                    coords = []
                    for _step in range(nsteps):
                        coord_line = f.readline()
                        if not coord_line:
                            break
                        parts = coord_line.strip().split()
                        if len(parts) >= 2:
                            r = float(parts[0])
                            z = float(parts[1])
                            coords.append([r, z])
                    if coords:
                        all_rays.append({"angle": launch_angle, "coords": coords})

        return {"rays": all_rays}
    except Exception as e:
        print(f"--- PARSING ERROR in {filepath}: {e} ---")
        return {"rays": []}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _axis_is_bad(axis: np.ndarray, expected_len: Optional[int] = None, rmax_hint_m: Optional[float] = None) -> bool:
    """Return True if axis is invalid (NaN/Inf, too short, non-increasing, or absurd scale).
    - expected_len: if given, mismatch triggers True
    - rmax_hint_m: if given, values far beyond 1.5×hint treated as invalid
    """
    try:
        arr = np.asarray(axis, dtype=float)
        if arr.ndim != 1:
            return True
        if expected_len is not None and arr.size != int(expected_len):
            return True
        if arr.size < 2:
            return True
        if not np.all(np.isfinite(arr)):
            return True
        # strictly increasing check
        diffs = np.diff(arr)
        if not np.all(diffs > 0):
            return True
        mx = float(np.nanmax(arr))
        if mx > 1e12:  # clearly corrupt scale
            return True
        if rmax_hint_m is not None and mx > 1.5 * float(rmax_hint_m):
            return True
        return False
    except Exception:
        return True


class OutputParser:
    """PYAT 기반 BELLHOP/KRAKEN 시뮬레이션 결과 파서"""
    
    def __init__(self):
        """초기화"""
        logger.info("Init ✓ - PYAT OutputParser ready")
    
    def load_ray_file(self, ray_path: str) -> Dict[str, Any]:
        """Ray 파일(.ray) 파싱 - 최종 파서 사용"""
        logger.info(f"Processing... - Parse ray file: {ray_path}")
        try:
            if not Path(ray_path).exists():
                raise FileNotFoundError(f"Ray file not found: {ray_path}")

            parsed = parse_ray_file_final(ray_path)

            rays_list: List[np.ndarray] = []
            for item in parsed.get('rays', []):
                coords = item.get('coords') if isinstance(item, dict) else item
                if not coords:
                    continue
                arr = np.asarray(coords, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    rays_list.append(arr[:, :2])

            result = {'rays': rays_list}
            logger.info(f"Done ✓ - Ray parsed: {len(rays_list)} rays")
            return result
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: Ray parsing failed: {e}")
            raise
    def convert_rays_to_density_map(self, ray_data: Dict[str, Any], 
                                  grid_config: Dict[str, Any]) -> np.ndarray:
        """Ray 데이터를 Ray Density Map으로 변환
        
        Args:
            ray_data: load_ray_file()의 출력 또는 테스트 데이터
            grid_config: 그리드 설정
            
        Returns:
            np.ndarray: 2D Ray Density Map
        """
        logger.info("Processing... - Convert rays to density map")
        
        try:
            # Grid parameters - nested config 또는 flat config 모두 지원
            if 'ray_density_map' in grid_config:
                ray_config = grid_config['ray_density_map']
                r_max = ray_config.get('max_range_km', 10.0) * 1000.0  # km to m
                z_max = ray_config.get('max_depth_m', 5000.0)
                # key 호환성: range_resolution/depth_resolution 우선
                r_points = ray_config.get('range_resolution', ray_config.get('x_resolution', 256))
                z_points = ray_config.get('depth_resolution', ray_config.get('z_resolution', 256))
            else:
                r_max = grid_config.get('r_max', 10000.0)  # 10 km
                z_max = grid_config.get('z_max', 5000.0)   # 5 km depth
                r_points = grid_config.get('range_resolution', grid_config.get('r_points', 256))
                z_points = grid_config.get('depth_resolution', grid_config.get('z_points', 256))
            
            # Initialize density map
            density_map = np.zeros((z_points, r_points))
            
            # Process each ray
            rays = ray_data.get('rays', [])
            
            for ray in rays:
                # Handle different ray data formats
                if isinstance(ray, dict) and 'data' in ray:
                    # Test format: {'data': [(r, z), ...]}
                    ray_coords = ray['data']
                elif isinstance(ray, np.ndarray) and ray.ndim == 2:
                    # PYAT format: numpy array [[r, z], ...]
                    ray_coords = ray
                elif isinstance(ray, list):
                    # Simple list format
                    ray_coords = ray
                else:
                    continue
                
                if len(ray_coords) == 0:
                    continue
                
                # Convert to numpy array if needed
                if not isinstance(ray_coords, np.ndarray):
                    ray_coords = np.array(ray_coords)
                
                if ray_coords.ndim != 2 or ray_coords.shape[1] < 2:
                    continue
                
                # Ray coordinates
                r_coords = ray_coords[:, 0]  # range (km 또는 m)
                z_coords = ray_coords[:, 1]  # depth (m)
                
                # Convert km to m if needed (휴리스틱: 값이 100 미만이면 km로 가정)
                if np.max(r_coords) < 100:
                    r_coords = r_coords * 1000.0
                
                # Filter valid coordinates
                valid_mask = (r_coords >= 0) & (r_coords <= r_max) & \
                           (z_coords >= 0) & (z_coords <= z_max)
                
                if not np.any(valid_mask):
                    continue
                
                r_valid = r_coords[valid_mask]
                z_valid = z_coords[valid_mask]
                
                # Convert to grid indices
                r_indices = np.clip(
                    ((r_valid / r_max) * (r_points - 1)).astype(int),
                    0, r_points - 1
                )
                z_indices = np.clip(
                    ((z_valid / z_max) * (z_points - 1)).astype(int),
                    0, z_points - 1
                )
                
                # Increment density (vectorized)
                for zi, ri in zip(z_indices, r_indices):
                    if 0 <= zi < z_points and 0 <= ri < r_points:
                        density_map[zi, ri] += 1
            
            # Seabed masking using optional .bty (terrain) if provided in grid_config
            try:
                bty_path = None
                if 'ray_density_map' in grid_config:
                    bty_path = grid_config['ray_density_map'].get('bty_path')
                if bty_path is None:
                    bty_path = grid_config.get('bty_path')
                if bty_path:
                    # parse simple .bty: pairs of (range_km, depth_m)
                    r_km, z_m = None, None
                    try:
                        with open(bty_path, 'r') as f:
                            lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith(('!', '#', '%', 'L', "'"))]
                        # detect count header
                        idx = 0
                        toks = lines[0].replace(',', ' ').split() if lines else []
                        if len(toks) == 1 and toks[0].replace('-', '').isdigit():
                            try:
                                npts = int(toks[0]); idx = 1
                            except Exception:
                                npts = None
                        else:
                            npts = None
                        pairs = []
                        if npts is not None:
                            for k in range(npts):
                                if idx + k >= len(lines):
                                    break
                                tt = lines[idx + k].replace(',', ' ').split()
                                if len(tt) >= 2:
                                    try:
                                        pairs.append((float(tt[0]), float(tt[1])))
                                    except Exception:
                                        pass
                        else:
                            for s in lines[idx:]:
                                tt = s.replace(',', ' ').split()
                                if len(tt) >= 2:
                                    try:
                                        pairs.append((float(tt[0]), float(tt[1])))
                                    except Exception:
                                        pass
                        if len(pairs) >= 2:
                            r_km = np.array([p[0] for p in pairs], dtype=float)
                            z_m = np.array([p[1] for p in pairs], dtype=float)
                            order = np.argsort(r_km)
                            r_km = r_km[order]; z_m = z_m[order]
                            keep = np.concatenate(([True], np.diff(r_km) > 0))
                            r_km = r_km[keep]; z_m = z_m[keep]
                    except Exception:
                        r_km, z_m = None, None
                    if r_km is not None and z_m is not None and r_km.size > 1:
                        # build mask grid
                        r_grid_m = np.linspace(0.0, r_max, r_points)
                        z_grid_m = np.linspace(0.0, z_max, z_points)
                        # bty range in m
                        r_bty_m = r_km * 1000.0
                        # interpolate bottom depth at grid columns
                        bot_m = np.interp(r_grid_m, r_bty_m, z_m, left=z_m[0], right=z_m[-1])
                        for j in range(r_points):
                            zb = float(bot_m[j])
                            if zb <= 0:
                                continue
                            # zero out cells deeper than seabed
                            zi = int(np.floor((zb / z_max) * (z_points - 1)))
                            zi = max(0, min(z_points - 1, zi))
                            density_map[zi+1:, j] = 0.0
            except Exception:
                # masking is best-effort; ignore errors
                pass
            
            # Normalize to [0, 1]
            if density_map.max() > 0:
                density_map = density_map / density_map.max()
            
            logger.info(f"Done ✓ - Ray density map: {density_map.shape}")
            return density_map
            
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: Ray density conversion failed: {e}")
            raise

    def parse_shd_file(self, shd_path: str) -> Dict[str, Any]:
        """SHD 파일(.shd) 파싱 - 완전 수동 파서 (MATLAB read_shd_bin.m 모방)
        - scipy.io.FortranFile 미사용
        - 레코드 마커는 무시하고 recl(첫 int32) 기반 4*recl 바이트 도약으로 처리
        - rarray는 float32로 읽고, 압력은 float32 interleaved(Re/Im)
        """
        logger.info(f"Processing... - Parse SHD file: {shd_path}")
        if not Path(shd_path).exists():
            raise FileNotFoundError(f"SHD file not found: {shd_path}")

        # Prefer robust PYAT readshd (handles FIELD 'RG' format)
        try:
            pressure, geometry = readshd(filename=str(shd_path), xs=np.nan, ys=np.nan, freq=np.nan)
            result: Dict[str, Any] = {}
            result['pressure'] = pressure
            result['zs'] = np.asarray(geometry.get('zs'))
            result['freqVec'] = np.asarray(geometry.get('f'))
            result['theta'] = np.asarray(geometry.get('thetas'))
            result['rarray'] = np.asarray(geometry.get('rarray'))
            result['zarray'] = np.asarray(geometry.get('zarray'))
            # Header best-effort
            try:
                result['Nfreq'] = int(result['freqVec'].size) if result['freqVec'] is not None else 1
            except Exception:
                result['Nfreq'] = 1
            try:
                shp = getattr(pressure, 'shape', (1,1,1,1))
                result['Ntheta'], result['Nsz'], result['Nrz'], result['Nrr'] = [int(x) for x in shp]
            except Exception:
                result['Ntheta'] = 1; result['Nsz'] = 1; result['Nrz'] = 1; result['Nrr'] = 1
            result['title'] = 'PYAT readshd'
            result['PlotType'] = 'TL'
            result['atten'] = 0.0
            result['file'] = str(shd_path)
            # Validate axes; if clearly bad, fall back to manual parser
            r_hint = None
            try:
                # best-effort hint from filename/stem used in our pipeline (0..100 km)
                if 'job0000' in str(shd_path).lower():
                    r_hint = 100_000.0  # meters
            except Exception:
                r_hint = None
            if _axis_is_bad(result['rarray'], expected_len=result['Nrr'], rmax_hint_m=r_hint):
                logger.warning("PYAT readshd delivered invalid rarray → fallback to manual parser")
                raise RuntimeError('Invalid rarray from readshd')

            logger.info(f"Done ✓ - SHD parsed (PYAT readshd): Nrr={result['Nrr']}, Nrz={result['Nrz']}")
            return result
        except Exception as e:
            logger.warning(f"PYAT readshd failed: {e}; falling back to manual parser")

        # Prefer FortranFile parser for Bellhop TL files
        try:
            if '_bhtl' in os.path.basename(str(shd_path)).lower():
                return self._parse_shd_file_manual(shd_path)
        except Exception:
            pass

        import struct
        result: Dict[str, Any] = {}
        with open(shd_path, 'rb') as fid:
            # 첫 레코드 길이(recl): bytes 단위가 아니라 MATLAB과 동일하게 '4*recl = record_bytes' 가정
            recl_bytes = struct.unpack('<i', fid.read(4))[0]  # MATLAB: fread int32
            # Title (80 chars)
            title = fid.read(80).decode('utf-8', errors='ignore').strip()
            # move to end of first record
            fid.seek(4 * recl_bytes, 0)

            # PlotType (10 chars)
            plot_type = fid.read(10).decode('utf-8', errors='ignore')
            # end of second record
            fid.seek(2 * 4 * recl_bytes, 0)

            # Integers: Nfreq, Ntheta, Nsx, Nsy, Nsz, Nrz, Nrr
            ints = np.fromfile(fid, dtype='<i4', count=7)
            if ints.size != 7:
                raise RuntimeError('Failed to read SHD integer header')
            Nfreq, Ntheta, Nsx, Nsy, Nsz, Nrz, Nrr = [int(x) for x in ints]
            # freq0, atten (float32)
            freq0 = np.fromfile(fid, dtype='<f4', count=1)[0]
            atten = np.fromfile(fid, dtype='<f4', count=1)[0]

            # end of record 3
            fid.seek(3 * 4 * recl_bytes, 0)
            # freqVec (float32) Nfreq (FIELD RG format)
            freqVec = np.fromfile(fid, dtype='<f4', count=Nfreq)

            # end of record 4
            fid.seek(4 * 4 * recl_bytes, 0)
            # theta (float32) Ntheta
            theta = np.fromfile(fid, dtype='<f4', count=Ntheta)

            # Xs/Ys
            if plot_type.startswith('TL'):
                fid.seek(5 * 4 * recl_bytes, 0)
                Pos_S_x = np.fromfile(fid, dtype='<f4', count=2)
                Xs = np.linspace(Pos_S_x[0], Pos_S_x[1], Nsx)
                fid.seek(6 * 4 * recl_bytes, 0)
                Pos_S_y = np.fromfile(fid, dtype='<f4', count=2)
                Ys = np.linspace(Pos_S_y[0], Pos_S_y[1], Nsy)
            else:
                fid.seek(5 * 4 * recl_bytes, 0)
                Xs = np.fromfile(fid, dtype='<f4', count=Nsx)
                fid.seek(6 * 4 * recl_bytes, 0)
                Ys = np.fromfile(fid, dtype='<f4', count=Nsy)

            # zs, zarray, rarray(float32)
            fid.seek(7 * 4 * recl_bytes, 0)
            zs = np.fromfile(fid, dtype='<f4', count=Nsz)
            fid.seek(8 * 4 * recl_bytes, 0)
            zarray = np.fromfile(fid, dtype='<f4', count=Nrz)
            fid.seek(9 * 4 * recl_bytes, 0)
            rarray = np.fromfile(fid, dtype='<f4', count=Nrr).astype(np.float64)

            # Pressure/TL 배열
            pt = plot_type.strip().lower()
            # Important: BELLHOP commonly stores complex pressure even for RunType 'C'.
            # Only treat as TL(dB) if PlotType explicitly equals 'TL'.
            fname_l = os.path.basename(str(shd_path)).lower()
            is_tl_db = pt.startswith('tl')
            if pt.startswith('irregular'):
                Nrcvrs_per_range = 1
                pressure = np.zeros((Ntheta, Nsz, 1, Nrr), dtype=(np.float32 if is_tl_db else np.complex64))
            else:
                Nrcvrs_per_range = Nrz
                pressure = np.zeros((Ntheta, Nsz, Nrz, Nrr), dtype=(np.float32 if is_tl_db else np.complex64))

            # 주파수 인덱스 선택 (MATLAB과 동일한 블록 오프셋 적용)
            # 파일명에서 fXXHz 패턴을 추출해 목표 주파수 추정, 없으면 freq0 사용
            import re
            target_freq = None
            m = re.search(r"_f(\d+)Hz", os.path.basename(shd_path))
            if m:
                try:
                    target_freq = float(m.group(1))
                except Exception:
                    target_freq = None
            if target_freq is None:
                try:
                    target_freq = float(freq0)
                except Exception:
                    target_freq = None
            if Nfreq and Nfreq > 1 and target_freq is not None:
                diffs = np.abs(freqVec - target_freq)
                ifreq = int(np.argmin(diffs))  # 0-based
            else:
                ifreq = 0
            logger.info(f"SHD header: Nfreq={Nfreq}, ifreq={ifreq}, freq0={freq0:.3f}")

            # 첫 데이터 레코드 시작점 기준으로 MATLAB처럼 recnum*(4*recl) 만큼 이동
            # 현재 파일 포인터는 rarray 끝 위치에 있음
            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        recnum = 10 \
                                 + (ifreq) * Ntheta * Nsz * Nrcvrs_per_range \
                                 + itheta * Nsz * Nrcvrs_per_range \
                                 + isz * Nrcvrs_per_range + irz
                        # MATLAB: fseek(fid, recnum * 4 * recl, 'bof') → 절대 오프셋
                        fid.seek(recnum * (4 * recl_bytes), 0)
                        if is_tl_db:
                            temp = np.fromfile(fid, dtype='<f4', count=Nrr)
                            if temp.size < Nrr:
                                temp = np.pad(temp, (0, Nrr - temp.size))
                            pressure[itheta, isz, irz, :] = temp.astype(np.float32)
                        else:
                            temp = np.fromfile(fid, dtype='<f4', count=2 * Nrr)
                            if temp.size < 2 * Nrr:
                                temp = np.pad(temp, (0, 2 * Nrr - temp.size))
                            real = temp[0::2]
                            imag = temp[1::2]
                            pressure[itheta, isz, irz, :] = real + 1j * imag
                        
            # rarray 품질 로그 (비정상 구간 진단)
            try:
                dr = np.diff(rarray)
                if dr.size > 0:
                    logger.info(f"rarray diff stats: min={np.nanmin(dr):.3f}, median={np.nanmedian(dr):.3f}, max={np.nanmax(dr):.3f} (units as read)")
            except Exception:
                pass

        result['title'] = title
        result['PlotType'] = plot_type
        result['Nfreq'] = Nfreq; result['Ntheta'] = Ntheta
        result['Nsx'] = Nsx; result['Nsy'] = Nsy
        result['Nsz'] = Nsz; result['Nrz'] = Nrz; result['Nrr'] = Nrr
        result['atten'] = float(atten)
        result['freqVec'] = freqVec; result['theta'] = theta
        result['Xs'] = Xs; result['Ys'] = Ys; result['zs'] = zs
        result['zarray'] = zarray; result['rarray'] = rarray
        result['pressure'] = pressure
        result['file'] = str(shd_path)
        logger.info(f"Done ✓ - SHD parsed (manual recl/seek): Nrr={Nrr}, Nrz={Nrz}")
        # If still suspicious, try the FortranFile-based parser as last resort
        try:
            bad_r = _axis_is_bad(result.get('rarray', np.array([])), expected_len=int(Nrr))
            bad_z = _axis_is_bad(result.get('zarray', np.array([])), expected_len=int(Nrz))
            if bad_r or bad_z:
                logger.warning("Manual recl/seek parse produced invalid axes → trying FortranFile parser")
                return self._parse_shd_file_manual(shd_path)
        except Exception:
            pass
        return result
        # 단위: rarray km→m 또는 그대로 m로 통일
        rarray = np.asarray(rarray, dtype=float)
        if rarray.size > 0 and np.nanmax(rarray) <= 1000.0:
            rarray = rarray * 1000.0

        out = {
            'title': title,
            'PlotType': plot_type,
            'Nfreq': Nfreq, 'Ntheta': Ntheta, 'Nsx': Nsx, 'Nsy': Nsy,
            'Nsz': Nsz, 'Nrz': Nrz, 'Nrr': Nrr, 'atten': atten,
            'freqVec': np.asarray(freqVec), 'theta': np.asarray(theta),
            'Xs': np.asarray(Xs), 'Ys': np.asarray(Ys), 'zs': np.asarray(zs),
            'zarray': np.asarray(zarray, dtype=float), 'rarray': rarray,
            'pressure': pressure
        }
        logger.info(f"Done ✓ - SHD parsed (FortranFile): Nrr={Nrr}, Nrz={Nrz}")
        return out

    def parse_mod_file(self, mod_path: str) -> Dict[str, Any]:
        """MOD 파일(.mod) 파싱 - PYAT readmod 사용"""
        logger.info(f"Processing... - Parse MOD file: {mod_path}")
        try:
            if not Path(mod_path).exists():
                raise FileNotFoundError(f"MOD file not found: {mod_path}")
            # PYAT-main 경로는 상단에서 추가됨
            from readmod import readmod
            mod_data = readmod(filename=str(mod_path))
            if hasattr(mod_data, '__dict__'):
                mod_data = vars(mod_data)
            return mod_data
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: MOD parsing failed: {e}")
            return {}

    def convert_mod_to_tl_field(self, mod_data: Dict[str, Any], grid_config: Dict[str, Any]) -> np.ndarray:
        """MOD 데이터를 TL Field로 근사 변환 (임시)"""
        try:
            # tl_field 섹션 우선 적용
            if 'tl_field' in grid_config:
                tl_cfg = grid_config['tl_field']
                range_points = tl_cfg.get('range_resolution', 256)
                depth_points = tl_cfg.get('depth_resolution', 256)
                r_max = tl_cfg.get('max_range_km', 100.0) * 1000.0
                z_max = tl_cfg.get('max_depth_m', 5000.0)
            else:
                range_points = grid_config.get('r_points', 256)
                depth_points = grid_config.get('z_points', 256)
                r_max = grid_config.get('r_max', 100000.0)
                z_max = grid_config.get('z_max', 5000.0)

            if not mod_data or 'Nmodes' not in mod_data:
                logger.warning("Invalid MOD data, creating placeholder TL field")
                return np.zeros((depth_points, range_points))

            # 임시 근사(합성 TL): 추후 실제 모드합으로 교체
            R = np.linspace(0, r_max, range_points)
            Z = np.linspace(0, z_max, depth_points)
            Rg, Zg = np.meshgrid(R, Z)
            tl_field = 20 * np.log10(np.maximum(Rg, 1.0)) + 0.05 * (Rg / 1000.0)
            tl_field += 6.0 * np.sin(Zg / 800.0 * np.pi)
            tl_field = np.clip(tl_field, 40.0, 120.0)
            return tl_field
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: MOD to TL conversion failed: {e}")
            return np.zeros((grid_config.get('depth_resolution', 256), grid_config.get('range_resolution', 256)))
    
    def _parse_shd_file_manual(self, shd_path: str) -> Dict[str, Any]:
        """SHD 파일 수동 파싱 (Fortran unformatted, record markers 사용)
        - scipy.io.FortranFile로 안전하게 레코드 경계를 처리
        - 좌표축 단위(r: km→m) 통일
        """
        logger.info("Processing... - Manual SHD parsing (FortranFile)")
        from scipy.io import FortranFile  # kept for reference; custom block parser used instead

        with FortranFile(shd_path, 'r') as f:
            # 공통 헤더
            title = f.read_record('S80')[0].decode('utf-8', errors='ignore').strip()
            PlotType = f.read_record('S10')[0].decode('utf-8', errors='ignore').strip()
            Nfreq, Ntheta, Nsx, Nsy, Nsz, Nrz, Nrr = f.read_record('i4')
            atten = f.read_record('f4')[0]
            freqVec = f.read_record('f4')
            theta = f.read_record('f4')

            # 좌표축 블록
            if PlotType == 'TL':
                Pos_S_x = f.read_record('f4')  # 2 floats
                Xs = np.linspace(Pos_S_x[0], Pos_S_x[1], Nsx)
                Pos_S_y = f.read_record('f4')
                Ys = np.linspace(Pos_S_y[0], Pos_S_y[1], Nsy)
            else:
                Xs = f.read_record('f4')
                Ys = f.read_record('f4')
            zs = f.read_record('f4')
            zarray = f.read_record('f4')
            # Bellhop writes float32 axes as well; use f4 (km)
            rarray = f.read_record('f4')  # km

            # 배열 준비 (Bellhop TL은 실수 TL[dB])
            is_tl_db = (PlotType.strip().upper() == 'TL') or ('_bhtl' in os.path.basename(str(shd_path)).lower())
            if PlotType.lower() == 'irregular':
                Nrcvrs_per_range = 1
                pressure = np.zeros((Ntheta, Nsz, 1, Nrr), dtype=(np.float32 if is_tl_db else np.complex64))
            else:
                Nrcvrs_per_range = Nrz
                pressure = np.zeros((Ntheta, Nsz, Nrz, Nrr), dtype=(np.float32 if is_tl_db else np.complex64))

            # 데이터 레코드 순회
            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        if is_tl_db:
                            temp = f.read_record('f4')  # Nrr float32 TL(dB)
                            if temp.size < Nrr:
                                pad = np.zeros(Nrr - temp.size, dtype=np.float32)
                            temp = np.concatenate([temp, pad])
                            pressure[itheta, isz, irz, :] = temp.astype(np.float32)
                        else:
                            temp = f.read_record('f4')  # 2*Nrr float32 interleaved
                            if temp.size < 2 * Nrr:
                                temp = np.pad(temp, (0, 2 * Nrr - temp.size))
                            real = temp[0::2]
                            imag = temp[1::2]
                        pressure[itheta, isz, irz, :] = real + 1j * imag

            # 단위 정합 및 반환
            rarray_m = np.asarray(rarray, dtype=float) * 1000.0
            return {
                'title': title,
                'PlotType': PlotType,
                'Nfreq': int(Nfreq), 'Ntheta': int(Ntheta), 'Nsx': int(Nsx), 'Nsy': int(Nsy),
                'Nsz': int(Nsz), 'Nrz': int(Nrz), 'Nrr': int(Nrr), 'atten': float(atten),
                'freqVec': np.asarray(freqVec), 'theta': np.asarray(theta),
                'Xs': np.asarray(Xs), 'Ys': np.asarray(Ys), 'zs': np.asarray(zs),
                'zarray': np.asarray(zarray), 'rarray': rarray_m, 'pressure': pressure
            }

    def convert_shd_to_tl_field(self, shd_data: Dict[str, Any], 
                               grid_config: Dict[str, Any]) -> np.ndarray:
        """SHD 데이터를 TL Field로 변환
        
        Args:
            shd_data: parse_shd_file()의 출력
            grid_config: 그리드 설정
            
        Returns:
            np.ndarray: 2D TL Field (dB)
        """
        logger.info("Processing... - Convert SHD to TL field")
        
        try:
            pressure = np.asarray(shd_data['pressure'])
            zarray = np.asarray(shd_data['zarray'], dtype=float)
            rarray = np.asarray(shd_data['rarray'], dtype=float)
            
            # 거리축 단위 정합: rarray를 항상 m로 통일 (1000 미만이면 km로 판단하여 변환)
            if rarray.size > 0 and np.nanmax(rarray) <= 1000.0:
                rarray = rarray * 1000.0
            
            # 첫 번째 source와 첫 번째 angle 사용
            if pressure.ndim == 4:
                pressure_2d = pressure[0, 0, :, :]  # (Nrz, Nrr)
            elif pressure.ndim == 3:
                pressure_2d = pressure[0, :, :]
            else:
                pressure_2d = pressure
            # data grid size from file (rows: depth, cols: range)
            try:
                n_r_data = int(pressure_2d.shape[1])
            except Exception:
                n_r_data = int(pressure_2d.size)
            
            # TL 계산 (물리 보정 포함)
            # 1) 기본: TLdB = -20*log10(|p| * r) + C, r[m] 보정 + 앵커 오프셋 C
            # Bellhop TL(dB) 여부는 헤더 PlotType=='TL'로만 판정 (파일명 무시)
            plot_type_hdr = str(shd_data.get('PlotType', '')).strip().lower()
            pressure_is_tl = plot_type_hdr.startswith('tl')
            if pressure_is_tl:
                tl_field = np.array(pressure_2d, dtype=np.float64)
                # Fallback: if TL(dB) looks invalid (all same/NaN), recompute from magnitude
                try:
                    if not np.any(np.isfinite(tl_field)) or float(np.nanstd(tl_field)) < 1e-6:
                        logger.warning("Bellhop TL(dB) constant/invalid → recomputing from |p|")
                        pressure_is_tl = False
                except Exception:
                    pressure_is_tl = False
            else:
                pressure_mag = np.abs(pressure_2d)
                pressure_mag = np.nan_to_num(pressure_mag, nan=1e-12, posinf=1e6, neginf=1e6)
                pressure_mag = np.clip(pressure_mag, 1e-12, 1e6)
                # rarray(m) 준비
                r_use = np.asarray(rarray, dtype=float)
                # unit fix: km→m if small
                if r_use.size and np.nanmax(r_use) <= 1000.0:
                    r_use = r_use * 1000.0
                # if invalid axis, reconstruct from grid config and disable geometric spreading
                disable_spread = False
                rmax_hint_m = float(grid_config.get('tl_field', {}).get('max_range_km', 100.0)) * 1000.0
                if _axis_is_bad(r_use, expected_len=n_r_data, rmax_hint_m=rmax_hint_m):
                    logger.warning("rarray invalid → reconstructed linspace(0, r_max, Nrr) and disabled -20log10(r)")
                    r_use = np.linspace(0.0, rmax_hint_m, n_r_data)
                    disable_spread = True
                if not pressure_is_tl:
                    r_mat = np.maximum(r_use[None, :], 1.0)
                    # Use TL from amplitude only; optional range spreading via config flag
                    tl_from_amp = -20.0 * np.log10(pressure_mag)
                    apply_spread = bool(grid_config.get('tl_field', {}).get('apply_range_spread', False))
                    if apply_spread and not disable_spread:
                        tl_from_amp = tl_from_amp - 20.0 * np.log10(r_mat)
                    tl_field = np.nan_to_num(tl_from_amp, nan=100.0, posinf=120.0, neginf=40.0)

            # 축 유효성 검증 및 복구(fallback)
            n_z, n_r = tl_field.shape
            if not np.all(np.isfinite(rarray)) or rarray.size != n_r or _axis_is_bad(rarray, expected_len=n_r, rmax_hint_m=rmax_hint_m):
                rarray = np.linspace(0.0, rmax_hint_m, n_r)
                logger.warning("rarray invalid → linspace(0, r_max, Nrr)")
            else:
                # ensure strictly increasing (repair duplicates/zeros) and convert km→m if needed
                r_work = np.array(rarray, dtype=float)
                if np.nanmax(r_work) <= 1000.0:
                    r_work = r_work * 1000.0
                r_sorted = np.sort(r_work)
                diff = np.diff(r_sorted)
                if not np.all(diff > 0):
                    span = max(1.0, float(np.nanmax(r_sorted) - np.nanmin(r_sorted)))
                    eps = span * 1e-6
                    for i in range(1, r_sorted.size):
                        if not (r_sorted[i] > r_sorted[i-1]):
                            r_sorted[i] = r_sorted[i-1] + eps
                    logger.warning("rarray had non-increasing entries → repaired with epsilon jitter")
                rarray = r_sorted

            if not np.all(np.isfinite(zarray)) or zarray.size != n_z or (np.diff(np.sort(zarray)).min(initial=1.0) <= 0):
                if 'tl_field' in grid_config:
                    z_max_cfg = grid_config['tl_field'].get('max_depth_m', 5000.0)
                else:
                    z_max_cfg = grid_config.get('z_max', 5000.0)
                zarray = np.linspace(0.0, z_max_cfg, n_z)
                logger.warning("zarray invalid → reconstructed linspace(0, z_max, Nrz)")

            # 좌표/필드 정합 준비: 축 정렬 + 고유 인덱스 확보(np.unique)
            if rarray.ndim == 1:
                r_sorted_idx = np.argsort(rarray)
                r_sorted = rarray[r_sorted_idx]
                tl_sorted = tl_field[..., r_sorted_idx]
                r_unique, r_unique_idx = np.unique(r_sorted, return_index=True)
                tl_sorted = tl_sorted[..., r_unique_idx]
                rarray = r_unique
                tl_field = tl_sorted
            if zarray.ndim == 1:
                z_sorted_idx = np.argsort(zarray)
                z_sorted = zarray[z_sorted_idx]
                tl_sorted = tl_field[z_sorted_idx, ...]
                z_unique, z_unique_idx = np.unique(z_sorted, return_index=True)
                tl_sorted = tl_sorted[z_unique_idx, ...]
                zarray = z_unique
                tl_field = tl_sorted

            # 축 길이-픽셀 정합 점검: Nrr, Nrz가 256 미만이라면 최근접 보간으로 가로/세로를 정확히 256에 맞춤
            # (MATLAB pcolor 대비 imsave 타일 고정 크기 보장)
            try:
                desired_r = grid_config.get('tl_field', {}).get('range_resolution', 256)
                desired_z = grid_config.get('tl_field', {}).get('depth_resolution', 256)
            except Exception:
                desired_r, desired_z = 256, 256
            n_z, n_r = tl_field.shape
            # 리샘플 비활성화: 원 해상도 유지 (FIELD가 256x256 생성하므로 추가 재샘플 금지)
            # if (n_r != desired_r) or (n_z != desired_z):
            #     r_idx = (np.linspace(0, n_r - 1, desired_r)).astype(int)
            #     z_idx = (np.linspace(0, n_z - 1, desired_z)).astype(int)
            #     tl_field = tl_field[np.ix_(z_idx, r_idx)]
            #     if rarray.size == n_r:
            #         rarray = rarray[r_idx]
            #     else:
            #         rarray = np.linspace(rarray.min(), rarray.max(), desired_r)
            #     if zarray.size == n_z:
            #         zarray = zarray[z_idx]
            #     else:
            #         zarray = np.linspace(zarray.min(), zarray.max(), desired_z)

            # 원시 좌표/필드 저장(시각화 pcolormesh용)
            self._last_tl_raw = {
                'tl': tl_field.copy(),
                'rarray': rarray.copy(),
                'zarray': zarray.copy(),
            }

            # Grid 설정 읽기 및 보간 여부
            if 'tl_field' in grid_config:
                tl_cfg = grid_config['tl_field']
                r_max = tl_cfg.get('max_range_km', 100.0) * 1000.0
                z_max = tl_cfg.get('max_depth_m', 5000.0)
                r_points = tl_cfg.get('range_resolution', 256)
                z_points = tl_cfg.get('depth_resolution', 256)
                do_interp = bool(tl_cfg.get('interpolate', False))
            else:
                r_max = grid_config.get('r_max', 10000.0)
                z_max = grid_config.get('z_max', 5000.0)
                r_points = grid_config.get('range_resolution', grid_config.get('r_points', 256))
                z_points = grid_config.get('depth_resolution', grid_config.get('z_points', 256))
                do_interp = False

            logger.info(f"Axis stats - rarray[{rarray.size}]: {np.nanmin(rarray) if rarray.size else np.nan}..{np.nanmax(rarray) if rarray.size else np.nan} m, zarray[{zarray.size}]: {np.nanmin(zarray) if zarray.size else np.nan}..{np.nanmax(zarray) if zarray.size else np.nan} m")
            # Log TL stats to detect constant field issues
            try:
                tl_min = float(np.nanmin(tl_field))
                tl_max = float(np.nanmax(tl_field))
                tl_std = float(np.nanstd(tl_field))
                logger.info(f"TL stats: min={tl_min:.3f}, max={tl_max:.3f}, std={tl_std:.6f}")
            except Exception:
                pass
            
            # 보간 비활성화: 원 해상도 그대로 반환
            if not do_interp:
                # Guard against near-constant images by applying robust percentile window
                tmin = float(np.nanmin(tl_field)) if np.isfinite(tl_field).any() else 100.0
                tmax = float(np.nanmax(tl_field)) if np.isfinite(tl_field).any() else 100.0
                if not np.isfinite([tmin, tmax]).all() or abs(tmax - tmin) < 1e-5:
                    logger.warning("TL field collapsed (constant) → applying small jitter to enable visualization")
                    tl_field = tl_field + 1e-3 * np.random.RandomState(0).standard_normal(size=tl_field.shape)
                logger.info(f"Done ✓ - TL field (no interpolation): {tl_field.shape}, range: {tl_field.min():.1f}-{tl_field.max():.1f} dB")
                return tl_field

            # 아래는 필요한 경우에만 수행되는 보간 경로
            from scipy.interpolate import RegularGridInterpolator
            r_new = np.linspace(0, r_max, r_points)
            z_new = np.linspace(0, z_max, z_points)
            r_mask = (rarray >= 0) & (rarray <= r_max)
            z_mask = (zarray >= 0) & (zarray <= z_max)
            r_vals = rarray[r_mask]
            z_vals = zarray[z_mask]
            r_orig, r_unique_idx = np.unique(r_vals, return_index=True)
            z_orig, z_unique_idx = np.unique(z_vals, return_index=True)
            logger.info(f"Interp grid - r_max={r_max:.1f} m, z_max={z_max:.1f} m, points={r_points}x{z_points}")
            logger.info(f"Masked axes - r_orig[{r_orig.size}]: {np.nanmin(r_orig) if r_orig.size else np.nan}..{np.nanmax(r_orig) if r_orig.size else np.nan}, z_orig[{z_orig.size}]: {np.nanmin(z_orig) if z_orig.size else np.nan}..{np.nanmax(z_orig) if z_orig.size else np.nan}")
            if len(r_orig) > 1 and len(z_orig) > 1:
                tl_masked = tl_field[np.ix_(z_mask, r_mask)]
                tl_subset = tl_masked[z_unique_idx, :][:, r_unique_idx]
                tl_subset = np.nan_to_num(tl_subset, nan=100.0, posinf=120.0, neginf=40.0)
                interpolator = RegularGridInterpolator((z_orig, r_orig), tl_subset, bounds_error=False, fill_value=100.0)
                Z_new, R_new = np.meshgrid(z_new, r_new, indexing='ij')
                tl_field_interp = interpolator((Z_new, R_new))
                tl_field_interp = np.nan_to_num(tl_field_interp, nan=100.0, posinf=120.0, neginf=40.0)
            else:
                logger.warning("Insufficient axis points after masking; returning 100 dB field")
                tl_field_interp = np.full((z_points, r_points), 100.0)
            
            logger.info(f"Done ✓ - TL field: {tl_field_interp.shape}, range: {tl_field_interp.min():.1f}-{tl_field_interp.max():.1f} dB")
            return tl_field_interp
            
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: TL field conversion failed: {e}")
            raise

    def process_simulation_outputs(self, ray_file_path: str, tl_file_path: str, 
                                 grid_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        BELLHOP과 KRAKEN/SCOOTER의 모든 출력 파일을 파싱하여
        R-DeepONet 훈련에 필요한 최종 데이터 배열을 반환합니다.
        
        Args:
            ray_file_path: BELLHOP ray 파일 경로 (.ray)
            tl_file_path: KRAKEN/SCOOTER TL 파일 경로 (.shd 또는 .mod)
            grid_config: 그리드 설정 정보
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (ray_map, tl_field)
        """
        logger.info(f"Processing... - Parse simulation outputs")
        logger.info(f"Ray file: {ray_file_path}")
        logger.info(f"TL file: {tl_file_path}")
        
        try:
            # 1. Ray 파일 파싱
            ray_data = self.load_ray_file(ray_file_path)
            ray_map = self.convert_rays_to_density_map(ray_data, grid_config)
            
            # 2. TL 파일 파싱 (.shd 또는 .mod)
            if tl_file_path.endswith('.shd'):
                tl_data = self.parse_shd_file(tl_file_path)
                tl_field = self.convert_shd_to_tl_field(tl_data, grid_config)
            elif tl_file_path.endswith('.mod'):
                # .mod 파일 파싱 (PYAT readmod 사용)
                logger.info(f"Processing... - Parse MOD file: {tl_file_path}")
                mod_data = self.parse_mod_file(tl_file_path)
                tl_field = self.convert_mod_to_tl_field(mod_data, grid_config)
            else:
                raise ValueError(f"Unsupported TL file format: {tl_file_path}")
            
            logger.info(f"Done ✓ - Ray map: {ray_map.shape}, TL field: {tl_field.shape}")
            return ray_map, tl_field
            
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: Simulation output processing failed: {e}")
            raise

    def save_visualization(self, ray_map: np.ndarray, tl_field: np.ndarray, 
                          metadata: Dict[str, Any], output_path: str) -> None:
        """사람 확인용(합성) + 학습용(개별 타일) PNG 저장
        
        Args:
            ray_map: Ray density map (X)
            tl_field: TL field (Y) 
        """
        logger.info("Processing... - Generate visualization (combined + tiles)")
        
        # Paths and metadata
        out_path = output_path
        out_p = Path(out_path)
        base_dir = out_p.parent.parent  # Go up to R-DeepONet_Data level
        stem = out_p.stem
        images_root = base_dir / 'images'
        check_dir = images_root / 'check'
        images_dir = images_root
        try:
            check_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        
        freq = metadata.get('frequency', 'unknown')
        src_depth = metadata.get('source_depth', 'unknown')
        scenario = metadata.get('scenario_name', 'default')

        def _load_bathymetry(bty_hint: Optional[str], out_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            """Load .bty and return (range_km, depth_m). Heuristics included."""
            try:
                candidates: List[Path] = []
                if bty_hint:
                    p = Path(bty_hint)
                    if p.exists() and p.is_file():
                        candidates.append(p)
                # same stem .bty
                p_same = Path(out_path).with_suffix('.bty')
                if p_same.exists():
                    candidates.append(p_same)
                # search in directory (latest first, up to 10)
                parent = Path(out_path).parent
                try:
                    found = sorted([x for x in parent.glob('*.bty') if x.is_file()], key=lambda x: x.stat().st_mtime, reverse=True)
                    candidates.extend(found[:10])
                except Exception:
                    pass
                # de-dup
                uniq: List[Path] = []
                seen = set()
                for c in candidates:
                    if str(c) not in seen:
                        uniq.append(c)
                        seen.add(str(c))
                candidates = uniq
                # try parse
                for cand in candidates:
                    try:
                        with open(cand, 'r') as f:
                            lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith(('!', '#', '%'))]
                        if not lines:
                            continue
                        # header detect
                        idx = 0
                        if lines[0][0].isalpha():
                            idx += 1
                        # N or data
                        npts: Optional[int] = None
                        toks = lines[idx].replace(',', ' ').split()
                        if len(toks) == 1 and toks[0].replace('-', '').isdigit():
                            try:
                                npts = int(toks[0])
                                idx += 1
                            except Exception:
                                npts = None
                        pairs: List[Tuple[float, float]] = []
                        # read exact npts
                        if npts is not None:
                            for k in range(npts):
                                if idx + k >= len(lines):
                                    break
                                tt = lines[idx + k].replace(',', ' ').split()
                                if len(tt) >= 2:
                                    try:
                                        x = float(tt[0]); z = float(tt[1])
                                        pairs.append((x, z))
                                    except Exception:
                                        continue
                        else:
                            # scan all
                            for s in lines[idx:]:
                                tt = s.replace(',', ' ').split()
                                if len(tt) >= 2:
                                    try:
                                        x = float(tt[0]); z = float(tt[1])
                                        pairs.append((x, z))
                                    except Exception:
                                        continue
                        if len(pairs) < 2:
                            continue
                        r = np.array([p[0] for p in pairs], dtype=float)
                        z = np.array([p[1] for p in pairs], dtype=float)
                        # unit heuristic: if range > 1000, treat as meters → km
                        if np.nanmax(r) > 1000.0:
                            r_km = r / 1000.0
                        else:
                            r_km = r.copy()
                        z_m = z.copy()
                        # sort and unique
                        order = np.argsort(r_km)
                        r_km = r_km[order]
                        z_m = z_m[order]
                        keep = np.concatenate(([True], np.diff(r_km) > 0))
                        r_km = r_km[keep]
                        z_m = z_m[keep]
                        logger.info(f"Done ✓ - Bathymetry loaded: {cand} ({r_km.size} pts)")
                        return r_km, z_m
                    except Exception:
                        continue
            except Exception:
                pass
            logger.warning("Bathymetry .bty not found or unreadable; skip overlay")
            return None

        # Create figure with subplots (human-check)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # Guard to contain following visualization block (fix prior indentation issue)
        try:
            
            # Physical space definition (as per specification)
            max_range_km = 100.0  # 100 km
            # prefer shallow zmax from metadata if available (visual-only)
            try:
                max_depth_m = float(metadata.get('zmax_m', 5000.0))
            except Exception:
                max_depth_m = 5000.0
            
            extent = [0, max_range_km, max_depth_m, 0]  # [left, right, bottom, top]
            # Preload bathymetry for masking and overlay
            bty_data = None
            try:
                bty_hint = metadata.get('bty_path') or metadata.get('bathymetry_path')
                bty_data = _load_bathymetry(bty_hint, output_path)
            except Exception:
                bty_data = None
            
            # Compute TL offset to map into 40–120 dB for visualization
            def _compute_tl_offset(tl_arr: np.ndarray, z_arr: Optional[np.ndarray] = None) -> float:
                try:
                    arr = np.asarray(tl_arr, dtype=float)
                    mask = np.isfinite(arr)
                    if z_arr is not None:
                        z = np.asarray(z_arr, dtype=float)
                        if z.ndim == 1 and z.size == arr.shape[0]:
                            thr = min(200.0, 0.15 * float(max_depth_m))
                            zmask = z > thr
                            if np.any(zmask):
                                mask = mask & zmask[:, None]
                    vals = arr[mask]
                    if vals.size == 0:
                        return 0.0
                    med = float(np.median(vals))
                    return med - 80.0  # target median ≈ 80 dB
                except Exception:
                    return 0.0
            tl_offset = 0.0
            if hasattr(self, '_last_tl_raw'):
                try:
                    raw = self._last_tl_raw
                    zraw_tmp = np.asarray(raw.get('zarray', []), dtype=float)
                    tlraw_tmp = np.asarray(raw.get('tl', []), dtype=float)
                    if tlraw_tmp.ndim == 2 and zraw_tmp.ndim == 1 and zraw_tmp.size == tlraw_tmp.shape[0]:
                        tl_offset = _compute_tl_offset(tlraw_tmp, zraw_tmp)
                except Exception:
                    tl_offset = 0.0
            if tl_offset == 0.0:
                try:
                    tl_offset = _compute_tl_offset(tl_field, None)
                except Exception:
                    tl_offset = 0.0
            
            # Geometric spreading correction: subtract 20*log10(r)
            try:
                if hasattr(self, '_last_tl_raw'):
                    raw = self._last_tl_raw
                    rraw_for_spread = np.asarray(raw.get('rarray', []), dtype=float)
                    if rraw_for_spread.ndim != 1 or rraw_for_spread.size != tl_field.shape[1]:
                        rraw_for_spread = np.linspace(0.0, max_range_km*1000.0, tl_field.shape[1])
                else:
                    rraw_for_spread = np.linspace(0.0, max_range_km*1000.0, tl_field.shape[1])
                spread = 20.0 * np.log10(np.maximum(rraw_for_spread, 1.0))
                spread_row = spread[None, :]
            except Exception:
                spread_row = 0.0

            # Recompute TL offset after spreading correction (use TL' = TL - 20log10 r)
            try:
                tl_offset2 = 0.0
                if hasattr(self, '_last_tl_raw'):
                    raw = self._last_tl_raw
                    zraw_tmp = np.asarray(raw.get('zarray', []), dtype=float)
                    tlraw_tmp = np.asarray(raw.get('tl', []), dtype=float)
                    if tlraw_tmp.ndim == 2 and spread_row is not None:
                        tlp = tlraw_tmp - spread_row
                        tl_offset2 = _compute_tl_offset(tlp, zraw_tmp)
                else:
                    tlp = np.asarray(tl_field, dtype=float)
                    if tlp.ndim == 2 and spread_row is not None:
                        tlp = tlp - spread_row
                    tl_offset2 = _compute_tl_offset(tlp, None)
                tl_offset = float(tl_offset2)
            except Exception:
                pass
            
            # Plot 1: Ray Density Map (X) — use the same normalization as training tiles
            # Prepare normalized ray (same pipeline as tiles below)
            _ray = np.array(ray_map, dtype=np.float32)
            if np.max(_ray) > 0:
                hi = np.percentile(_ray, 99.5)
            if hi > 0:
                _ray = np.clip(_ray / hi, 0.0, 1.0)

            im1 = ax1.imshow(_ray, extent=extent, aspect='auto', 
                           cmap='hot', origin='upper', vmin=0.0, vmax=1.0,
                           interpolation='none')
            ax1.set_title(f'Ray Density Map (X)\n{freq}Hz, Src: {src_depth}m', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel('Range (km)', fontsize=11)
            ax1.set_ylabel('Depth (m)', fontsize=11)
            ax1.grid(True, alpha=0.3)
            # Seabed overlay on Ray panel
            try:
                if bty_data is not None:
                    r_bty_km, z_bty_m = bty_data
                    ax1.plot(r_bty_km, z_bty_m, color='w', lw=0.9, alpha=0.9)
            except Exception:
                pass
            
            # Colorbar for ray map
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Ray Density', fontsize=10)
            
            # Plot 2: TL Field (Y) with MATLAB-like auto scaling in dB domain
            def _matlab_auto_limits_db(tl_db: np.ndarray) -> tuple:
                t = np.array(tl_db, dtype=np.float64)
                t[~np.isfinite(t)] = np.nan
                finite = np.isfinite(t)
                if not finite.any():
                    return 40.0, 120.0
                vals = t[finite]
                tlmed = float(np.median(vals))
                tlstd = float(np.std(vals))
                tlmax = 10.0 * np.round((tlmed + 0.75 * tlstd) / 10.0)
                tlmin = tlmax - 50.0
                return float(tlmin), float(tlmax)

            im2 = None
            # Fixed scale for combined view to stabilize colors
            vmin_auto = 40.0
            vmax_auto = 120.0
            using_pcolor = False
            if hasattr(self, '_last_tl_raw'):
                raw = self._last_tl_raw
                rraw = np.asarray(raw.get('rarray', []), dtype=float)
                zraw = np.asarray(raw.get('zarray', []), dtype=float)
                tlraw = np.asarray(raw.get('tl', tl_field), dtype=float)
                if rraw.size > 1 and zraw.size > 1 and tlraw.ndim == 2:
                    # 좌표 정렬 및 중복 제거 후 pcolormesh
                    r_idx = np.argsort(rraw); rraw = rraw[r_idx]; tlraw = tlraw[..., r_idx]
                    z_idx = np.argsort(zraw); zraw = zraw[z_idx]; tlraw = tlraw[z_idx, ...]
                    r_keep = np.concatenate(([True], np.diff(rraw) > 0)) if rraw.size>1 else np.array([True])
                    z_keep = np.concatenate(([True], np.diff(zraw) > 0)) if zraw.size>1 else np.array([True])
                    rraw = rraw[r_keep]; tlraw = tlraw[..., r_keep]
                    zraw = zraw[z_keep]; tlraw = tlraw[z_keep, ...]
                    # For Bellhop TL(dB) do not apply any offset
                    tl_vis = tlraw
                    # Mask below seabed using .bty with epsilon to preserve reflections just above seabed
                    try:
                        if bty_data is not None:
                            r_bty_km, z_bty_m = bty_data
                            bot = np.interp(rraw, np.asarray(r_bty_km, dtype=float) * 1000.0, np.asarray(z_bty_m, dtype=float))
                            dz = float(np.nanmedian(np.diff(zraw))) if zraw.size > 1 else 0.0
                            eps = max(0.0, 0.5 * dz)
                            # keep one pixel above interface for reflections
                            msk = (zraw[:, None] > (bot[None, :] + eps))
                            tl_vis = tl_vis.copy()
                            tl_vis[msk] = np.nan
                    except Exception:
                        pass
                    # Keep fixed [40,120] scale for combined figure
                    im2 = ax2.pcolormesh(
                        rraw/1000.0, zraw, tl_vis,
                        shading='nearest', cmap='jet_r', vmin=vmin_auto, vmax=vmax_auto,
                        antialiased=False, rasterized=True)
                    ax2.set_ylim(float(max_depth_m), 0.0)
                    using_pcolor = True
            if im2 is None:
                vmin_auto, vmax_auto = 40.0, 120.0
                tl_vis2 = tl_field
                # Mask below seabed using .bty on pixel grid (use NaN to reveal seabed fill)
                try:
                    if bty_data is not None and isinstance(tl_vis2, np.ndarray) and tl_vis2.ndim == 2:
                        r_pix = np.linspace(0.0, max_range_km * 1000.0, tl_vis2.shape[1])
                        z_pix = np.linspace(0.0, max_depth_m, tl_vis2.shape[0])
                        r_bty_km, z_bty_m = bty_data
                        bot = np.interp(r_pix, np.asarray(r_bty_km, dtype=float) * 1000.0, np.asarray(z_bty_m, dtype=float))
                        dz = float(np.nanmedian(np.diff(z_pix))) if z_pix.size > 1 else 0.0
                        eps = max(0.0, 0.5 * dz)
                        msk = (z_pix[:, None] > (bot[None, :] + eps))
                        tl_vis2 = tl_vis2.copy()
                        tl_vis2[msk] = np.nan
                except Exception:
                    pass
                im2 = ax2.imshow(tl_vis2, extent=extent, aspect='auto', cmap='jet_r', vmin=vmin_auto, vmax=vmax_auto, origin='upper', interpolation='none')
                ax2.set_ylim(float(max_depth_m), 0.0)
            ax2.set_title(f'Transmission Loss Field (Y)\n{freq}Hz, Src: {src_depth}m', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('Range (km)', fontsize=11)
            ax2.set_ylabel('Depth (m)', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # Colorbar for TL field
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('TL (dB)', fontsize=10)

            # Bathymetry overlay (brown fill under TL + boundary line)
            try:
                if bty_data is not None:
                    r_bty_km, z_bty_m = bty_data
                    # Fill below seabed with brown; draw behind TL (lower zorder)
                    try:
                        ax2.fill_between(r_bty_km, z_bty_m, y2=max_depth_m, color='#42150d', alpha=0.60, linewidth=0, zorder=-1)
                    except Exception:
                        pass
                    ax2.plot(r_bty_km, z_bty_m, color='k', lw=1.0, alpha=0.9, label='Seabed', zorder=5)
            except Exception:
                logger.warning("Bathymetry overlay failed; continue without overlay")

            # Keep dynamic scaling for combined figure; tiles remain fixed at 40–120
            
            # Add scenario info
            fig.suptitle(f'R-DeepONet Data Factory - {scenario}', 
                        fontsize=14, fontweight='bold')

            # Zoom inset removed as per request
            
            # Adjust layout
            plt.tight_layout()
            
            # Save combined to check folder
            combined_file = str(check_dir / f"{stem}_combined.png")
            plt.savefig(combined_file, dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            plt.close()
            logger.info(f"Done ✓ - Combined visualization saved: {combined_file}")

            # 학습용 타일 저장 (정확히 256x256 픽셀)
            # Ray tile: enhance density contrast
            ray = np.array(ray_map, dtype=np.float32)
            if np.max(ray) > 0:
                hi = np.percentile(ray, 99.5)
            if hi > 0:
                ray = np.clip(ray / hi, 0.0, 1.0)

            # TL tile: dynamic clip using percentiles
            tl = np.array(tl_field, dtype=np.float32)
            # Apply seabed mask on tile image as well
            try:
                if bty_data is not None and tl.ndim == 2:
                    r_pix = np.linspace(0.0, max_range_km * 1000.0, tl.shape[1])
                    z_pix = np.linspace(0.0, max_depth_m, tl.shape[0])
                    r_bty_km, z_bty_m = bty_data
                    bot = np.interp(r_pix, np.asarray(r_bty_km, dtype=float) * 1000.0, np.asarray(z_bty_m, dtype=float))
                    msk = (z_pix[:, None] > bot[None, :])
                    tl[msk] = np.nan
            except Exception:
                pass
            finite = np.isfinite(tl)
            if finite.any():
                q05, q95 = np.percentile(tl[finite], [5, 95])
                vmin = float(max(40.0, q05 - 5.0))
                vmax = float(min(120.0, q95 + 5.0))
                tl = np.clip(tl, vmin, vmax)
            else:
                tl = np.clip(tl, 40.0, 120.0)

            # 저장 경로
            ray_tile = str(images_dir / f"{stem}_ray.png")
            tl_tile = str(images_dir / f"{stem}_tl.png")

            # imsave는 배열 크기를 그대로 픽셀로 저장
            plt.imsave(ray_tile, ray, cmap='hot', vmin=0.0, vmax=1.0)
            plt.imsave(tl_tile, np.nan_to_num(tl, nan=120.0), cmap='jet_r', vmin=40.0, vmax=120.0)
            logger.info(f"Done ✓ - Tiles saved: {ray_tile}, {tl_tile}")
            
        except Exception as e:
            logger.error(f"L{e.__traceback__.tb_lineno}: Visualization failed: {e}")
            # Don't raise - visualization failure shouldn't stop data generation


# Backward compatibility functions
def load_ray_file(ray_path: str) -> Dict[str, Any]:
    """하위호환성을 위한 래퍼 함수"""
    parser = OutputParser()
    return parser.load_ray_file(ray_path)


def convert_rays_to_density_map(ray_data: Dict[str, Any], grid_config: Dict[str, Any]) -> np.ndarray:
    """하위호환성을 위한 래퍼 함수"""
    parser = OutputParser()
    return parser.convert_rays_to_density_map(ray_data, grid_config)


def parse_shd_file(shd_path: str) -> Dict[str, Any]:
    """하위호환성을 위한 래퍼 함수"""
    parser = OutputParser()
    return parser.parse_shd_file(shd_path)


def convert_shd_to_tl_field(shd_data: Dict[str, Any], grid_config: Dict[str, Any]) -> np.ndarray:
    """하위호환성을 위한 래퍼 함수"""
    parser = OutputParser()
    return parser.convert_shd_to_tl_field(shd_data, grid_config)


if __name__ == "__main__":
    # 테스트 코드
    logger.info("Testing PYAT OutputParser...")
    
    parser = OutputParser()
    
    # 테스트용 그리드 설정
    grid_config = {
        'r_max': 10000.0,
        'z_max': 5000.0,
        'r_points': 256,
        'z_points': 256
    }
    
    logger.info("Done ✓ - OutputParser self-test stub")