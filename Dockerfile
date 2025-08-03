# 1) استخدم صورة Python أساسية خفيفة
FROM python:3.12-slim

# 2) تثبيت مكتبات النظام المطلوبة
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 3) إعداد المجلد داخل الحاوية
WORKDIR /app
COPY . /app

# 4) تثبيت تبعيات Python
RUN pip install --no-cache-dir -r requirements.txt

# 5) كشف المنفذ الذي ستعمل عليه الـ API
EXPOSE 8000

# 6) أمر التشغيل الافتراضي
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
