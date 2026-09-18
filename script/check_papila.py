"""Read-only checks for PAPILA, reserved as the external test dataset."""

import hashlib
import io
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from PIL import Image


DATA_DIR = Path(__file__).resolve().parents[1] / "data/raw"
ZIP_PATH = DATA_DIR / "PAPILA.zip"
# Official Figshare article 14798004, version 2, file 35013982.
EXPECTED_MD5 = "7d73fc5157d44b221a62ed5a4baf3779"
LABEL_NAMES = {0: "healthy", 1: "glaucoma", 2: "suspicious"}


def read_labels(archive, prefix):
    """Read both eye tables, preserving the original patient IDs and diagnoses."""
    records = []
    for eye in ("OD", "OS"):
        path = f"{prefix}/ClinicalData/patient_data_{eye.lower()}.xlsx"
        workbook = load_workbook(io.BytesIO(archive.read(path)), read_only=True, data_only=True)
        try:
            sheet = workbook.active
            if sheet["D2"].value != "Diagnosis" or sheet["A3"].value != "ID":
                raise ValueError(f"Unexpected spreadsheet headings: {path}")
            # The supplied workbooks have three heading rows. A is ID; D is diagnosis.
            for row in sheet.iter_rows(min_row=4, values_only=True):
                patient_id, diagnosis = row[0], row[3]
                if not isinstance(patient_id, str) or not re.fullmatch(r"#\d{3}", patient_id):
                    raise ValueError(f"Invalid patient ID in {path}: {patient_id!r}")
                records.append({"patient_id": patient_id, "eye": eye,
                                "image_id": f"RET{patient_id[1:]}{eye}", "diagnosis": diagnosis})
        finally:
            workbook.close()
    return records


def read_contour(archive, path):
    """Read the x and y positions that form an outline."""
    points = [tuple(map(float, line.split()))
              for line in archive.read(path).decode("utf-8-sig").splitlines() if line.strip()]
    if any(len(point) != 2 or not all(map(math.isfinite, point)) for point in points):
        raise ValueError("Each line needs two valid numbers")
    if len(set(points)) < 3:
        raise ValueError("The outline needs at least three different points")
    area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
               in zip(points, points[1:] + points[:1]))
    if area == 0:
        raise ValueError("The outline does not enclose any area")
    return points


def main():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Place the official PAPILA ZIP at {ZIP_PATH}")
    errors, warnings = [], []

    def check(message, problems, warning=False):
        print(f"  {message}: {len(problems)}")
        if problems:
            (warnings if warning else errors).append(message)
            print("    Examples (up to 12):", "; ".join(map(str, problems[:12])))

    print("PAPILA TEST DATA — checking photographs, labels and expert outlines")
    with ZIP_PATH.open("rb") as source:
        checksum = hashlib.file_digest(source, "md5").hexdigest()
    check("Download differs from the original file",
          [] if checksum == EXPECTED_MD5 else ["Download the original PAPILA ZIP again"])

    with ZipFile(ZIP_PATH) as archive:
        files = [n for n in archive.namelist() if not n.endswith("/")
                 and "__MACOSX" not in n and not Path(n).name.startswith(".")]
        check("Repeated ZIP filenames", [n for n, count in Counter(files).items() if count > 1])
        roots = {n.split("/")[0] for n in files}
        if len(roots) != 1:
            raise ValueError(f"Expected one main dataset folder, found {sorted(roots)}")
        prefix = roots.pop()
        photos = sorted(n for n in files if n.startswith(f"{prefix}/FundusImages/"))
        image_ids = [Path(n).stem for n in photos]
        check("Unexpected photograph count", [] if len(photos) == 488 else [len(photos)])
        check("Repeated photograph IDs", [key for key, count in Counter(image_ids).items() if count > 1])
        check("Unexpected photograph names", [Path(n).name for n in photos
              if not re.fullmatch(r"RET\d{3}(OD|OS)\.jpg", Path(n).name)])

        patients = defaultdict(list)
        for key in image_ids:
            if re.fullmatch(r"RET\d{3}(OD|OS)", key):
                patients[key[3:6]].append(key[6:])
        print(f"  Photographs: {len(photos)}; patients: {len(patients)}")
        check("Unexpected patient count", [] if len(patients) == 244 else [len(patients)])
        check("Patients without exactly one photograph per eye", [key for key, eyes in patients.items()
              if sorted(eyes) != ["OD", "OS"]])

        records = read_labels(archive, prefix)
        label_ids = [row["image_id"] for row in records]
        check("Repeated clinical records", [key for key, count in Counter(label_ids).items() if count > 1])
        check("Photographs without a clinical record", sorted(set(image_ids) - set(label_ids)))
        check("Clinical records without a photograph", sorted(set(label_ids) - set(image_ids)))
        check("Unknown or missing diagnosis labels", [row["image_id"] for row in records
              if row["diagnosis"] not in LABEL_NAMES])
        counts = Counter(row["diagnosis"] for row in records)
        print("  Original labels:", {f"{code} ({name})": counts[code] for code, name in LABEL_NAMES.items()})

        photo_hashes, sizes = {}, {}
        unreadable, duplicates, unusual_sizes = [], [], []
        for path in photos:
            key = Path(path).stem
            try:
                data = archive.read(path)
                with Image.open(io.BytesIO(data)) as photo:
                    photo.load()
                    sizes[key] = photo.size
                    if photo.size != (2576, 1934) or photo.mode != "RGB":
                        unusual_sizes.append(f"{key}: {photo.size}, {photo.mode}")
                digest = hashlib.sha256(data).hexdigest()
                if digest in photo_hashes:
                    duplicates.append(f"{key} = {photo_hashes[digest]}")
                photo_hashes[digest] = key
            except (OSError, ValueError, BadZipFile) as exc:
                unreadable.append(f"{key}: {exc}")
        check("Unreadable photographs", unreadable)
        check("Photographs with an unexpected size or colour format", unusual_sizes)
        check("Identical photograph files within PAPILA", duplicates, warning=True)

        contour_folder = f"{prefix}/ExpertsSegmentations/Contours/"
        expected = {f"{contour_folder}{key}_{part}_exp{expert}.txt": key
                    for key in image_ids for part in ("disc", "cup") for expert in (1, 2)}
        available = {n for n in files if n.startswith(contour_folder)}
        check("Missing expert outlines", [Path(n).name for n in sorted(set(expected) - available)])
        check("Extra outline files (left unchanged)", [Path(n).name for n in sorted(available - set(expected))],
              warning=True)
        bad_contours = []
        checked_contours = 0
        for path in sorted(set(expected) & available):
            try:
                points = read_contour(archive, path)
                key = expected[path]
                if key not in sizes:
                    raise ValueError("Cannot check the outline's position because its photograph did not open")
                width, height = sizes[key]
                if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
                    raise ValueError("The outline includes points outside the photograph")
                checked_contours += 1
            except (OSError, ValueError, BadZipFile) as exc:
                bad_contours.append(f"{Path(path).name}: {exc}")
        check("Problems in the required outline files", bad_contours)
        print(f"  Opened {len(sizes)} photographs; checked {checked_contours} required outlines.")

    # A basic leakage check, using exact file contents only. No images are removed.
    chaksu_path = DATA_DIR / "chaksu_train.zip"
    if chaksu_path.is_file():
        overlap = []
        with ZipFile(chaksu_path) as archive:
            training_photos = [path for path in archive.namelist()
                               if path.startswith("Train/1.0_Original_Fundus_Images/")
                               and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
                               and not Path(path).name.startswith(".")]
            check("Unexpected Chaksu training photograph count",
                  [] if len(training_photos) == 1009 else [len(training_photos)])
            print(f"  Comparing {len(training_photos)} Chaksu photographs with {len(sizes)} PAPILA photographs")
            for path in training_photos:
                digest = hashlib.sha256(archive.read(path)).hexdigest()
                if digest in photo_hashes:
                    overlap.append(f"{path} = {photo_hashes[digest]}")
        check("Identical photograph files shared with Chaksu training", overlap, warning=True)
    else:
        print("  Could not compare photographs with Chaksu: its training ZIP is missing.")

    print(f"\nResult: {len(errors)} failed checks; {len(warnings)} checks with warnings.")
    print("Keep PAPILA for final testing. Do not use it to train any part of the model.")
    print("Keep suspicious cases under their original label until we agree how to use them.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
