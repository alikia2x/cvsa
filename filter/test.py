from labeling_system import LabelingSystem

DATABASE_PATH = "./data/main.db"
MODEL_PATH = "./filter/checkpoints/best_model_V3.11.pt"
OUTPUT_FILE = "./data/filter/real_test.jsonl"
BATCH_SIZE = 50

if __name__ == "__main__":
    labeling_system = LabelingSystem(
        mode='model_testing',
        database_path=DATABASE_PATH,
        output_file=OUTPUT_FILE,
        model_path=MODEL_PATH,
        batch_size=BATCH_SIZE
    )
    labeling_system.run()