FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8085

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8090", "--reload"]