# Vast.ai 원격 SSH 연결 가이드

## 방법 1: GitHub 사용 (가장 간단) ⭐ 권장

### 1단계: GitHub에 코드 Push

```bash
# 1. GitHub에서 새 저장소 생성 (private 권장)
# https://github.com/new

# 2. 로컬에서 remote 추가
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 3. 변경사항 커밋
git add .
git commit -m "Add physics-informed loss implementation"

# 4. Push (처음에는 -u 필요)
git push -u origin master
```

**GitHub 로그인 필요 여부:**
- **공개 저장소**: 로그인 불필요 (누구나 clone 가능)
- **비공개 저장소**: 로그인 필요 (SSH key 또는 Personal Access Token)

### 2단계: Vast.ai 인스턴스에서 Clone

```bash
# Vast.ai 인스턴스 접속 후
cd ~
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 또는 SSH key 사용 (비공개 저장소)
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git
```

---

## 방법 2: SSH로 직접 파일 전송

### 1단계: Vast.ai 인스턴스 SSH 정보 확인
- Vast.ai 대시보드에서 인스턴스의 SSH 연결 정보 확인
- 예: `ssh root@ssh1.vast.ai -p 12345`

### 2단계: 로컬에서 rsync로 전송

```bash
# Windows PowerShell (WSL 또는 Git Bash 사용)
rsync -avz --exclude 'R-DeepONet_Data' \
  --exclude '__pycache__' \
  --exclude 'experiments' \
  --exclude '*.pt' \
  ./ root@ssh1.vast.ai:/root/sofar_deep/

# 또는 scp 사용
scp -r -P 12345 ./ root@ssh1.vast.ai:/root/sofar_deep/
```

**주의**: 대용량 데이터(`R-DeepONet_Data`)는 제외하고 코드만 전송

---

## 방법 3: GitHub + SSH Key (비공개 저장소용)

### 1단계: SSH Key 생성 (로컬)

```bash
# Windows PowerShell
ssh-keygen -t ed25519 -C "your_email@example.com"
# 파일 위치: C:\Users\NAVL\.ssh\id_ed25519.pub
```

### 2단계: GitHub에 SSH Key 등록
1. GitHub → Settings → SSH and GPG keys
2. New SSH key 클릭
3. `id_ed25519.pub` 내용 복사해서 등록

### 3단계: SSH로 Clone

```bash
# Vast.ai 인스턴스에서
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git
```

---

## Vast.ai 인스턴스 설정 체크리스트

### 필수 설치

```bash
# 1. Git 설치 확인
git --version

# 2. Python 환경 설정
conda create -n wcsmo python=3.9
conda activate wcsmo

# 3. 의존성 설치
pip install -r requirements.txt

# 4. CUDA 확인
nvidia-smi
```

### 데이터 전송 (필요시)

```bash
# 방법 A: GitHub LFS (대용량 파일용)
git lfs install
git lfs track "*.h5"
git lfs track "*.npy"

# 방법 B: 직접 다운로드 (Vast.ai 인스턴스에서)
# Google Drive, Dropbox, 또는 직접 업로드
```

---

## 빠른 시작 명령어

### GitHub 사용 (권장)

```bash
# 로컬
git add .
git commit -m "Ready for vast.ai"
git push origin master

# Vast.ai 인스턴스
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
conda activate wcsmo
python train_runner.py --config config_stage2_highres.yaml
```

### SSH 직접 전송

```bash
# 로컬 (WSL/Git Bash)
rsync -avz --exclude-from=.gitignore ./ root@ssh1.vast.ai:/root/sofar_deep/

# Vast.ai 인스턴스
cd /root/sofar_deep
conda activate wcsmo
python train_runner.py --config config_stage2_highres.yaml
```

---

## 주의사항

1. **대용량 데이터**: `R-DeepONet_Data/` (12GB+)는 GitHub에 올리지 마세요
   - `.gitignore`에 이미 제외됨
   - 필요시 별도로 전송 (rsync, scp, 또는 클라우드 스토리지)

2. **체크포인트**: `*.pt` 파일도 제외됨
   - 학습 후 필요시 다운로드

3. **환경 변수**: Vast.ai 인스턴스에서 환경 변수 설정 필요할 수 있음
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

4. **포트 포워딩**: Jupyter/MLflow 사용 시 Vast.ai 대시보드에서 포트 설정

---

## 문제 해결

### GitHub 인증 오류
```bash
# Personal Access Token 사용 (비공개 저장소)
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git
```

### SSH 연결 오류
```bash
# Vast.ai SSH 포트 확인
# 대시보드 → Instance → SSH Info
```

### 권한 오류
```bash
# Vast.ai 인스턴스에서
chmod +x train_runner.py
```

