FROM python:3.12-slim

ENV PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY run_all.py .

RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "run_all.py"]