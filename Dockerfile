FROM python:3.11-slim

# git is needed to install pybfs from its GitHub repo
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-interactive matplotlib backend; scratch dirs the app writes to at runtime
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# One worker with threads: background calibration/update jobs keep their progress
# state in process memory, so a second worker would not see it.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 300 app:app"]
