# ArduScratch AI - Cloud Training Guide

## 🌩️ Cloud Training Options

AI Anda bisa training 24/7 di cloud tanpa bergantung laptop!

### Option 1: GitHub Actions (FREE) ⭐ RECOMMENDED

**Advantages:**
- ✅ Gratis unlimited
- ✅ Auto-restart setiap 6 jam
- ✅ Auto-commit progress
- ✅ Tidak perlu monitor

**Limitations:**
- ⏱️ Max 6 jam per session
- 🖥️ CPU only (slower)

**Setup:**

1. **Push ke GitHub:**
```bash
cd C:\Users\asus\public\ArduScratch
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ArduScratch.git
git push -u origin main
```

2. **Enable Actions:**
- Buka GitHub repo → Settings → Actions → General
- Enable "Read and write permissions"

3. **Start Training:**
- Go to Actions tab
- Click "Autonomous AI Training"
- Click "Run workflow"

**Auto-chain:** Workflow akan auto-restart setiap 6 jam sampai target loss tercapai!

---

### Option 2: Google Colab (FREE GPU) 🚀 FASTEST

**Advantages:**
- ✅ FREE GPU (Tesla T4)
- ✅ 10-15x lebih cepat dari CPU
- ✅ 12 jam per session
- ✅ Auto-save ke Google Drive

**Limitations:**
- ⏱️ Max 12 jam, lalu harus manual restart
- 📱 Perlu check setiap 12 jam

**Setup:**

1. **Upload ke Google Drive:**
   - Upload `data/dataset.bin` ke Drive (445 MB)
   - Upload `data/tokenizer/` ke Drive

2. **Open Colab:**
   - Open `colab_training.ipynb` di Google Colab
   - Runtime → Change runtime type → GPU

3. **Run cells** satu per satu

4. **Setiap 12 jam:**
   - Re-run notebook (progress auto-saved)

---

### Option 3: Railway / Render (PAID but 24/7)

**Advantages:**
- ✅ Jalan 24/7 non-stop
- ✅ Tidak perlu monitor
- ✅ Auto-restart on crash

**Cost:** ~$5-10/month

**Setup:**

1. **Railway.app:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway init
railway up
```

2. **Start training:**
```bash
railway run ./continuous_train.sh
```

---

## 📊 Training Progress Tracking

### Check Progress (from anywhere):

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/ArduScratch.git
cd ArduScratch

# Pull latest model
git pull

# Check metadata
cat models/latest/metadata.json
```

### Fetch Latest Model:

```bash
# On your local machine
cd C:\Users\asus\public\ArduScratch
git pull origin main
```

---

## ⚙️ Configuration

Edit `.github/workflows/train.yml`:

```yaml
# Train for 5 hours per session
timeout-minutes: 330

# Target loss
env:
  TARGET_LOSS: 0.5
```

Edit `scripts/autonomous_trainer.py`:

```python
CONFIG = {
    "target_loss": 0.5,           # Lower = smarter AI
    "checkpoint_every": 500,      # Save frequency
    "commit_every": 1000,         # Git push frequency
    "max_hours_per_session": 5,  # Cloud time limit
}
```

---

## 📈 Expected Timeline

| Platform | Speed | Time to Loss < 0.5 |
|----------|-------|-------------------|
| GitHub Actions (CPU) | 1x | ~150 hours (~25 sessions) |
| Google Colab (GPU) | 15x | ~10 hours (1 session) |
| Railway (CPU 24/7) | 1x | ~6 days continuous |

---

## 🎯 Recommended Strategy

**Best approach:**

1. **Use Google Colab for initial training** (0-12 hours)
   - Get from loss 5.0 → 2.0 quickly with GPU
   - Takes 1 Colab session

2. **Switch to GitHub Actions for refinement** (12+ hours)
   - Continue from loss 2.0 → 0.5
   - Auto-runs in background

3. **Fetch trained model:**
   ```bash
   git pull
   ```

---

## 🚨 Important Notes

1. **Dataset size:** GitHub has 100MB file limit
   - Use Git LFS: `git lfs track "*.bin"`
   - Or upload to Google Drive and download in Actions

2. **Auto-commit:** Model auto-pushes every 1000 steps
   - Your repo will have many commits
   - Can squash later if needed

3. **Rate limits:** GitHub Actions has 2000 mins/month on free
   - Unlimited for public repos!
   - Make your repo public for unlimited training

---

## 📱 Monitor Training

### GitHub:
- Go to Actions tab
- Watch live logs
- See commit history

### Telegram notification (optional):
```bash
# Add to autonomous_trainer.py
import requests
requests.post(
    "https://api.telegram.org/bot<TOKEN>/sendMessage",
    json={"chat_id": "<ID>", "text": f"Loss: {loss}"}
)
```

---

## ✅ Next Steps

1. Push ArduScratch ke GitHub
2. Enable GitHub Actions
3. Run workflow
4. Check back tomorrow - AI will be smarter! 🚀

**Laptop bisa sleep, AI tetap belajar di cloud!** ☁️
