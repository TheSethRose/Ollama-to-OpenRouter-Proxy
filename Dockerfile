FROM python:3.11-slim

# Install uv (ultra fast Python package manager and runner)
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY requirements.txt ./
RUN uv pip install -r requirements.txt

COPY app ./app

EXPOSE 11434

CMD ["uv", "icorn", "app.main:app", "--host", "0.0.0.0", "--port", "11434"]
