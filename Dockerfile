FROM python:3.10
WORKDIR /ai_synth
COPY requirements.txt /ai_synth/
RUN pip install virtualenv
RUN python -m venv venv
ENV PATH="/ai_synth/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /storage/noyuzrad/ai_synth
CMD ["python", "src/main.py"]
