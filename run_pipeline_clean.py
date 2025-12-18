#!/usr/bin/env python3
"""
SCOOTER 파이프라인 실행 스크립트 (터미널 없이)
캐시 정리 + 파이프라인 실행을 한 번에 처리
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def clear_all_caches():
    """모든 캐시 파일 정리"""
    print("Processing... - Clear all caches")
    
    # PYAT 캐시 삭제
    pyat_cache = Path("PYAT-main/__pycache__")
    if pyat_cache.exists():
        shutil.rmtree(pyat_cache)
        print("✓ PYAT __pycache__ deleted")
    
    # 메인 캐시 삭제
    main_cache = Path("__pycache__")
    if main_cache.exists():
        shutil.rmtree(main_cache)
        print("✓ Main __pycache__ deleted")
    
    # temp 폴더 정리
    temp_dir = Path("temp")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            if file.suffix in ['.prt', '.env', '.shd', '.ray', '.bty']:
                try:
                    file.unlink()
                    print(f"✓ Deleted {file.name}")
                except:
                    pass
    
    # Python 모듈 캐시 정리
    modules_to_clear = ['wkrakenenvfil', 'env_generator', 'dataset', 'models']
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    print("Done ✓ - All caches cleared")

def run_pipeline():
    """파이프라인 실행"""
    print("Processing... - Run SCOOTER pipeline")
    
    try:
        # test_pipeline.py 직접 import하여 실행
        import test_pipeline
        print("Done ✓ - Pipeline execution completed")
        return True
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("SCOOTER Pipeline Runner (No Terminal)")
    print("=" * 50)
    
    # 1. 캐시 정리
    clear_all_caches()
    
    # 2. 파이프라인 실행
    success = run_pipeline()
    
    # 3. 결과 확인
    shd_file = Path("temp/job0000_Defense_Wideband_tl.shd")
    if shd_file.exists():
        print("✓ SCOOTER .shd file generated successfully!")
        print(f"✓ File size: {shd_file.stat().st_size} bytes")
    else:
        print("✗ SCOOTER .shd file not found")
        
        # .prt 파일 확인
        prt_file = Path("temp/job0000_Defense_Wideband_tl.prt")
        if prt_file.exists():
            print("Checking .prt file for errors...")
            with open(prt_file, 'r') as f:
                content = f.read()
                if "FATAL ERROR" in content:
                    print("✗ FATAL ERROR found in .prt file")
                    # 오류 라인 출력
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if "Bad depth" in line or "FATAL ERROR" in line:
                            print(f"L{i+1}: {line}")
                else:
                    print("✓ No FATAL ERROR in .prt file")
    
    print("=" * 50)
    return success

if __name__ == "__main__":
    main()
