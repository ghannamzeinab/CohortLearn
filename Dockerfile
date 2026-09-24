FROM python:3.12-slim

# Reproducibility settings.
# The maths libraries pick CPU-specific code paths at runtime, which changes the
# last digits of floating-point results between machines. These settings force
# one generic code path and a single thread, so every machine does the same
# arithmetic in the same order.
ENV PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_CORETYPE=Nehalem \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_ICL AVX512_SPR"

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
