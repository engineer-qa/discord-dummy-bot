FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py people.json style_examples.txt ./
CMD ["python", "-u", "bot.py"]
