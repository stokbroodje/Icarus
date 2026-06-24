# Environment-only image: apt + pip. All code (icarus/, sim/) is bind-mounted by compose,
# so edits on the host land in the container without a rebuild.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # the tesserocr manylinux wheel bundles its own libtesseract but not the language
    # data — point it at the apt-installed tessdata (eng + nld)
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-nld \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY icarus/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir jupyter matplotlib
CMD ["python", "/app/sim/sim.py"]
