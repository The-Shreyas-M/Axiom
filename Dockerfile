FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GRADIO_SHARE=false
EXPOSE 7860

CMD ["python", "app.py"]
