# ArduScratch - AI Arduino Code Generator

AI buatan sendiri yang bisa generate kode Arduino dari nol! 🚀

## ✅ Status: BERJALAN!

AI ini sudah berhasil:
- ✓ Mengumpulkan **54,569 file Arduino** (17 juta baris kode)
- ✓ Melatih tokenizer ByteLevel BPE dengan vocab 8,000
- ✓ Memproses **233 juta token** menjadi dataset
- ✓ Melatih model GPT dari nol (training sedang berjalan)
- ✓ Build retrieval index dari 52,965 file
- ✓ Generate project Arduino (masih perlu training lebih lama)

## 📁 Struktur Project

```
ArduScratch/
├── data/
│   ├── corpus.txt          # 731 MB corpus dari semua file Arduino
│   ├── tokenizer/          # ByteLevel BPE tokenizer
│   ├── dataset.bin         # 445.8 MB binary tokens
│   └── index.json          # TF-IDF retrieval index
├── models/
│   └── latest/
│       └── model.pt        # Trained GPT model
├── scripts/
│   ├── collect_corpus.py   # Kumpulkan file Arduino
│   ├── train_tokenizer.py  # Train tokenizer
│   ├── prepare_dataset.py  # Convert corpus ke binary
│   ├── train_lm.py         # Train GPT model
│   ├── build_index.py      # Build retrieval index
│   └── generate_project.py # Generate Arduino code
├── specs/
│   └── example.txt         # Contoh spesifikasi project
└── out/
    └── LedBlink/           # Output generated code

```

## 🚀 Cara Pakai

### 1. Generate Project Baru

```bash
cd C:\Users\asus\public\ArduScratch
C:\Users\asus\public\.venv\Scripts\python.exe scripts\generate_project.py \
  --model models\latest \
  --tokenizer data\tokenizer \
  --index data\index.json \
  --spec specs\example.txt \
  --out out\MyProject
```

### 2. Continue Training (Biar Makin Pintar)

```bash
C:\Users\asus\public\.venv\Scripts\python.exe scripts\train_lm.py \
  --tokenizer data\tokenizer \
  --data data\dataset.bin \
  --out models\latest \
  --steps 1000 \
  --resume models\latest
```

### 3. Update Index (Kalau Ada File Baru)

```bash
C:\Users\asus\public\.venv\Scripts\python.exe scripts\build_index.py \
  --corpus "C:\Users\asus\AppData\Local\Arduino15;C:\Program Files (x86)\Arduino" \
  --out data\index.json
```

## 🎯 Spesifikasi

Buat file `.txt` dengan deskripsi project yang diinginkan:

```txt
Create a simple LED blink program for Arduino UNO.
The LED should blink every 1 second.
Use pin 13 for the LED.
```

## 📊 Dataset

- **54,569 file** dari:
  - `C:\Users\asus\AppData\Local\Arduino15` (libraries)
  - `C:\Program Files (x86)\Arduino` (examples & core)
- **17,146,329 baris kode**
- **233,737,477 token**
- **Vocab size: 8,000**

## 🧠 Model Architecture

- **Type**: GPT-style Transformer
- **Layers**: 6
- **Embedding**: 256
- **Heads**: 8
- **Context**: 512 tokens
- **Training**: Causal Language Modeling

## 🎓 Training Progress

- Initial training: 500 steps (loss: 9.17 → 4.98)
- Continue training: 1500 steps more (running...)
- Target: ~2000-5000 steps untuk hasil yang baik

## 💡 Tips

1. **Makin banyak training = makin pintar**: Resume training dengan `--resume models\latest`
2. **Update corpus**: Tambah file Arduino Anda sendiri ke corpus
3. **Tune temperature**: Edit `generate_project.py` untuk kreativitas (0.5-1.2)
4. **Check loss**: Loss < 3.0 biasanya sudah cukup baik

## 📝 License

Project ini dibuat untuk pembelajaran. Source code Arduino yang digunakan untuk training mengikuti lisensi masing-masing.

---

**Status**: AI ini 100% berjalan tanpa API eksternal! 🎉
