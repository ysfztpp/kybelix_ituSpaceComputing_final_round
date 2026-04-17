FROM competition-base:pytorch2.5.1-cuda12.1-cudnn9

WORKDIR /workspace
COPY . /workspace
RUN chmod +x /workspace/run.sh

CMD ["./run.sh"]
