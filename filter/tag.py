import argparse
from labeling_system import LabelingSystem

DEFAULT_OUTPUT = "./data/filter/labeled_data.jsonl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--startFrom", type=float)
    args = parser.parse_args()

    labeling_system = LabelingSystem(
        mode='labeling',
        output_file=args.output,
        start_from=args.startFrom,
        batch_size=50
    )
    labeling_system.run()