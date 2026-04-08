FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install curl for healthcheck and cleanup in one layer
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security compliance
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy requirements and install as 'user'
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code with correct ownership
COPY --chown=user . .

# Set environment variables
ENV PORT=7860
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# Ensure python outputs log immediately to the console
ENV PYTHONUNBUFFERED=1 

# Switch to the non-root user
USER user

EXPOSE 7860

# Robust Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch with single worker (best for in-memory session dictionaries)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]