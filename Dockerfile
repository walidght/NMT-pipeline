FROM python:3.10-slim

# Hugging Face Spaces Security Requirement: Create a non-root user
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # We default this to None so Pydantic knows to look for the Hub ID in the cloud
    LOCAL_MODEL_PATH="" 

WORKDIR $HOME/app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

# Expose the required port for HF Spaces
EXPOSE 7860

# Start the FastAPI server using the module path
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]