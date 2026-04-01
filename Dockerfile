FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY README.md .

ENTRYPOINT ["python", "src/run_task2.py"]
