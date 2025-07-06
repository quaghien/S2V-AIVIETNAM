# S2V - Chuyển đổi Slide thành Video với Zalo AI TTS

## 🎯 Tổng quan
Ứng dụng chuyển đổi file PDF slides thành video tiếng Việt sử dụng AI. Phiên bản mới dùng **Zalo AI TTS** để tạo giọng nói tiếng Việt tự nhiên và xử lý đa luồng để tăng tốc độ.

## ⚡ Tính năng chính
- 🎤 **Zalo AI TTS**: Giọng nói tiếng Việt chất lượng cao
- 🚀 **Xử lý đa luồng**: Tạo audio song song, nhanh hơn 3-5 lần
- 🤖 **AI Enhancement**: GPT-4 Vision + Claude AI cải thiện nội dung
- 📹 **Video HD**: Xuất MP4 với độ phân giải 1920x1080
- 🎛️ **Tùy chỉnh linh hoạt**: Điều chỉnh batch size, số threads

## 🛠️ Cài đặt nhanh

### 1. Cài đặt thư viện
```bash
pip install -r requirements_zalo.txt
```

### 2. Thiết lập API Keys
Tạo file `.env`:
```bash
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
ZALO_API_KEY=your-zalo-api-key
```

### 3. Chạy ứng dụng
```bash
# Cơ bản
python cli_zalo.py input.pdf

# Nâng cao
python cli_zalo.py input.pdf --threads 6 --pdf-batch 5 -v
```

## 🎮 Các chế độ sử dụng

### 🏃‍♂️ Chế độ nhanh (Testing)
```bash
python cli_zalo.py input.pdf --quick
```
- PDF batch: 3 slides
- Threads: 2
- Thời gian: ~1-2 phút cho 10 slides

### 🛡️ Chế độ an toàn (Khuyến nghị)
```bash
python cli_zalo.py input.pdf --safe
```
- PDF batch: 3 slides  
- Threads: 4
- Thời gian: ~2-3 phút cho 10 slides

### 🏭 Chế độ production (Tối ưu)
```bash
python cli_zalo.py input.pdf --production
```
- PDF batch: 7 slides
- Threads: 8
- Thời gian: ~1-2 phút cho 10 slides

## 🔧 Workflow chi tiết

### Bước 1: Xử lý PDF
```
PDF → Hình ảnh HD (1920x1080) → GPT-4 Vision (batch 3-7 slides)
```

### Bước 2: Cải thiện nội dung
```
Mô tả thô → Claude AI → Nội dung giảng dạy chuyên nghiệp
```

### Bước 3: Tạo audio song song
```
Dịch tiếng Việt → Chia slides → Zalo AI TTS (đa luồng) → MP3 files
```

### Bước 4: Tạo video
```
Đồng bộ hình ảnh + audio → Render MP4 → Video hoàn chỉnh
```

## 🎤 Zalo AI TTS - Cách hoạt động

### API Call cho mỗi slide:
```bash
curl -s -H "apikey: YOUR_KEY" \
  --data-urlencode "input=Nội dung slide" \
  -d "encode_type=1" \
  -d "speaker_id=4" \
  -d "speed=1.0" \
  -X POST https://api.zalo.ai/v1/tts/synthesize
```

### Kết quả:
```json
{
  "error_code": 0,
  "data": {
    "url": "https://chunk-v3.tts.zalo.ai/secure/audio.mp3"
  }
}
```

## 📊 Tham số tùy chỉnh

| Tham số | Mô tả | Giá trị | Khuyến nghị |
|---------|--------|---------|-------------|
| `--pdf-batch` | Số slides/batch cho GPT-4 | 3-7 | 5 |
| `--threads` | Số luồng TTS song song | 2-8 | 4-6 |
| `--speed` | Tốc độ nói | 0.8-1.2 | 1.0 |
| `--speaker-id` | ID giọng nói | 1-10 | 4 |

## 🎯 Ví dụ sử dụng

### Presentation nhỏ (5-10 slides)
```bash
python cli_zalo.py slides.pdf --quick -v
```

### Presentation vừa (10-20 slides)
```bash
python cli_zalo.py slides.pdf --safe -o ./output
```

### Presentation lớn (20+ slides)
```bash
python cli_zalo.py slides.pdf --production --threads 8
```

### Tùy chỉnh chi tiết
```bash
python cli_zalo.py slides.pdf \
  --pdf-batch 4 \
  --threads 6 \
  --save-config my_config.json
```

## 📁 Kết quả đầu ra

### Cấu trúc thư mục:
```
output/
└── run_20250106_143022_abc123/
    ├── images/              # Hình ảnh slides
    │   ├── slide_1.png
    │   └── slide_2.png
    ├── audio/               # File audio
    │   ├── slide_1.mp3
    │   └── slide_2.mp3
    ├── descriptions.txt     # Mô tả gốc
    ├── translated_descriptions.txt
    └── final_video.mp4      # Video cuối cùng ✨
```

### Tính năng audio đặc biệt:
- `<<<sil#100>>>` → Tạo khoảng im lặng 100ms
- **CHỮ HOA** → Nhấn mạnh khi đọc
- Công thức toán → Đọc bằng tiếng Việt tự nhiên

## 🚨 Xử lý lỗi thường gặp

### Lỗi API Keys
```bash
❌ Missing API keys: ZALO_API_KEY
```
**Giải pháp**: Kiểm tra file `.env` có đầy đủ API keys

### Lỗi Memory
```bash
❌ Memory Error
```
**Giải pháp**: Giảm `--threads` xuống 2-4

### Lỗi Network
```bash
❌ Network timeout
```
**Giải pháp**: Giảm `--pdf-batch` xuống 3

### Test API Zalo
```bash
# Kiểm tra API hoạt động
curl -s -H "apikey: YOUR_KEY" \
  --data-urlencode "input=Test tiếng Việt" \
  -d "encode_type=1" -d "speaker_id=4" \
  -X POST https://api.zalo.ai/v1/tts/synthesize
```

## 📈 Hiệu suất

| Số slides | Chế độ | Thời gian | Threads |
|-----------|---------|-----------|---------|
| 5-10 | Quick | 1-2 phút | 2 |
| 10-20 | Safe | 2-4 phút | 4 |
| 20+ | Production | 3-6 phút | 8 |

## 🎨 Tính năng nâng cao

### Lưu cấu hình
```bash
python cli_zalo.py slides.pdf --save-config my_preset.json
```

### Sử dụng cấu hình đã lưu
```bash
python cli_zalo.py slides.pdf --config my_preset.json
```

### Debug chi tiết
```bash
python cli_zalo.py slides.pdf -v
```

## 🆚 So sánh với phiên bản cũ

| Tính năng | Phiên bản cũ | Zalo AI TTS |
|-----------|--------------|-------------|
| TTS Engine | Gemini | Zalo AI |
| Audio Quality | Tốt | Rất tốt |
| Tốc độ | Chậm | Nhanh 3-5x |
| Parallel Processing | Không | Có |
| Audio Splitting | Phức tạp | Đơn giản |
| Reliability | Trung bình | Cao |

## 🔮 Cải tiến tương lai
- [ ] Hỗ trợ nhiều giọng nói
- [ ] Tích hợp thêm TTS engines
- [ ] Cải thiện dịch thuật
- [ ] GUI interface
- [ ] Batch processing PDF

## 💡 Tips & Tricks

### Tăng tốc độ:
- Dùng `--production` cho presentations lớn
- Tăng `--threads` (không quá 8)
- Dùng SSD để lưu output

### Cải thiện chất lượng:
- Giảm `--pdf-batch` cho slides phức tạp
- Kiểm tra và chỉnh sửa translation trước khi TTS
- Sử dụng `speed=0.9` cho giọng nói chậm hơn

### Tiết kiệm API:
- Dùng `--quick` khi testing
- Kiểm tra PDF slides trước khi chạy
- Sử dụng cache cho presentations đã xử lý

---

**🎉 Chúc bạn tạo video thành công!** 