# Dockerfile — أداة RIPEMD-160 على GPU (self-test إلزامي عند كل تشغيل)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ripemd160.cu main.cu loop.sh ./

# sm_89 = معمارية Ada (RTX 4090). عدّل القيمة لو استهدفت GPU مختلفة
# (مثلاً sm_86 لأمبير RTX 3090، sm_90 لـHopper H100).
ARG CUDA_ARCH=sm_89
RUN nvcc -O3 -arch=${CUDA_ARCH} -o ripemd160_tool ripemd160.cu main.cu \
    && chmod +x /app/loop.sh

ENTRYPOINT ["/app/loop.sh"]
CMD ["f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8", "16777216"]

