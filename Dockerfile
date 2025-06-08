# 選擇官方 Python 3.9 slim 映像
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 到容器
COPY requirements.txt .

# 安裝系統依賴（OpenCV 需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼到容器
COPY . .

# 建立上傳資料夾
RUN mkdir -p uploads templates

# 開放埠口
EXPOSE 5000

CMD ["python", "app.py"]