import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small MapleStory YOLO detector.")
    parser.add_argument(
        "--data", type=Path, default=Path("training_data/maple_three_class_v1/data.yaml")
    )
    parser.add_argument("--model", type=Path, default=Path(".yolo_runtime/yolo11n.pt"))
    parser.add_argument("--project", type=Path, default=Path("training_runs"))
    parser.add_argument("--name", default="maple_three_class_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def serializable_results(metrics):
    results = {}
    for key, value in metrics.results_dict.items():
        try:
            results[str(key)] = float(value)
        except (TypeError, ValueError):
            results[str(key)] = str(value)
    results["speed_ms_per_image"] = {
        str(key): float(value) for key, value in metrics.speed.items()
    }
    box = metrics.box
    results["per_class"] = {}
    names = metrics.names
    metric_rows = {
        int(class_id): row_index
        for row_index, class_id in enumerate(box.ap_class_index)
    }
    for class_id, class_name in names.items():
        row_index = metric_rows.get(int(class_id))
        if row_index is None:
            results["per_class"][str(class_name)] = {
                "instances": 0,
                "precision": None,
                "recall": None,
                "map50": None,
                "map50_95": None,
            }
        else:
            results["per_class"][str(class_name)] = {
                "instances": int(metrics.nt_per_class[int(class_id)]),
                "precision": float(box.p[row_index]),
                "recall": float(box.r[row_index]),
                "map50": float(box.ap50[row_index]),
                "map50_95": float(box.ap[row_index]),
            }
    return results


def main():
    args = parse_args()
    runtime = Path(".yolo_runtime").resolve()
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from ultralytics import YOLO

    data = args.data.resolve()
    model_path = args.model.resolve()
    project = args.project.resolve()
    model = YOLO(str(model_path))
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=12,
        seed=args.seed,
        deterministic=True,
        plots=True,
        verbose=True,
    )
    run_dir = project / args.name
    best_path = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(project),
        name=f"{args.name}_validation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    summary = {
        "best_model": str(best_path.resolve()),
        "data": str(data),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metrics": serializable_results(metrics),
    }
    summary_path = run_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
