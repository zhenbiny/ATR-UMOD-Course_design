from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CLASSES = [
    "car",
    "suv",
    "van",
    "bus",
    "freight_car",
    "truck",
    "motorcycle",
    "trailer",
    "tank_truck",
    "excavator",
    "crane",
]


@dataclass
class ObjectLabel:
    image_id: str
    cls: int
    points: np.ndarray


@dataclass
class Prediction:
    image_id: str
    cls: int
    conf: float
    points: np.ndarray


def parse_classes(text: str) -> tuple[list[str], dict[str, int]]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    return names, {name: i for i, name in enumerate(names)}


def norm_image_id(value: str) -> str:
    return Path(value).stem


def rect_to_points(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def iter_xml_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.glob("*.xml"))


def parse_xml_points(obj: ET.Element) -> np.ndarray | None:
    poly = obj.find("polygon")
    if poly is not None:
        coords: list[float] = []
        for i in range(1, 5):
            x = poly.findtext(f"x{i}")
            y = poly.findtext(f"y{i}")
            if x is None or y is None:
                return None
            coords.extend([float(x), float(y)])
        return np.array(coords, dtype=np.float32).reshape(4, 2)

    bnd = obj.find("bndbox")
    if bnd is not None:
        vals = [bnd.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")]
        if any(v is None for v in vals):
            return None
        return rect_to_points(*(float(v) for v in vals))

    return None


def load_ground_truth(path: Path, class_to_id: dict[str, int]) -> list[ObjectLabel]:
    labels: list[ObjectLabel] = []
    for xml_path in iter_xml_files(path):
        root = ET.parse(xml_path).getroot()
        image_id = norm_image_id(xml_path.stem)
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_to_id:
                continue
            points = parse_xml_points(obj)
            if points is not None:
                labels.append(ObjectLabel(image_id, class_to_id[name], points))
    return labels


def prediction_files(path: Path, class_to_id: dict[str, int]) -> list[tuple[Path, int | None]]:
    if path.is_file():
        return [(path, None)]
    return [(path / f"{name}.txt", cls_id) for name, cls_id in class_to_id.items() if (path / f"{name}.txt").exists()]


def load_predictions(path: Path, class_to_id: dict[str, int], conf_thres: float | None = None) -> list[Prediction]:
    predictions: list[Prediction] = []
    for txt_path, fixed_cls in prediction_files(path, class_to_id):
        for line_no, line in enumerate(txt_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if fixed_cls is None:
                if len(parts) in {11, 7} and parts[0] in class_to_id:
                    cls_id = class_to_id[parts[0]]
                    image_id, conf_text = parts[1], parts[2]
                    coord_text = parts[3:]
                elif len(parts) in {11, 7} and parts[0].isdigit():
                    cls_id = int(parts[0])
                    image_id, conf_text = parts[1], parts[2]
                    coord_text = parts[3:]
                else:
                    raise ValueError(f"Single prediction file requires class column at {txt_path}:{line_no}")
            else:
                cls_id = fixed_cls
                image_id, conf_text, *coord_text = parts

            conf = float(conf_text)
            if conf_thres is not None and conf < conf_thres:
                continue
            if len(coord_text) == 8:
                points = np.array([float(x) for x in coord_text], dtype=np.float32).reshape(4, 2)
            elif len(coord_text) == 4:
                points = rect_to_points(*(float(x) for x in coord_text))
            else:
                raise ValueError(f"Bad coordinate count at {txt_path}:{line_no}: {len(coord_text)}")
            predictions.append(Prediction(norm_image_id(image_id), cls_id, conf, points))
    return predictions


def image_set_from_gt(labels: list[ObjectLabel]) -> set[str]:
    return {x.image_id for x in labels}


def image_set_from_preds(predictions: list[Prediction]) -> set[str]:
    return {x.image_id for x in predictions}
