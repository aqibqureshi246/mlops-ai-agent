FROM python:3.12.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements/inference.txt /app/requirements.txt
COPY requirements/constraints.txt /app/constraints.txt
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt && useradd --uid 10001 --create-home app
COPY api /app/api
COPY model /app/model
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
