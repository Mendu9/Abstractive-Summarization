class Config:
    OFFLINE_DATA_DIR = "data/gov-report"
    REPORT_TYPES = ["crs", "gao"]
    DEFAULT_REPORT_TYPES = ["crs", "gao"]
    MAX_INPUT_LENGTH = 1024 
    MAX_TARGET_LENGTH = 512
    MODEL_NAME = "google/long-t5-tglobal-base"
    NUM_WORKERS = 4
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-4
    EPOCHS = 3
    GRAD_ACCUM = 1
    GPUS = 1
    FP16 = False
    OUTPUT_DIR = "outputs"
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
