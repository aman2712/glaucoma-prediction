"""Read-only checks for the official Chaksu training ZIP. No training or extraction."""

import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageChops


ZIP_PATH = Path(__file__).resolve().parents[1] / "data/raw/chaksu_train.zip"
EXPECTED_IMAGES = {"Bosch": 104, "Forus": 95, "Remidio": 810}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LABELS = {"NORMAL", "GLAUCOMA SUSPECT"}
# A spelling error in the source: used only to check votes, never to rewrite data.
EXPERT_LABELS = {"NORMAL": "NORMAL", "GLAUCOMA SUSPECT": "GLAUCOMA SUSPECT",
                 "GLAUCOMA  SUSUPECT": "GLAUCOMA SUSPECT"}


def image_id(value):
    """Image101.jpg-Image101-1.jpg -> image101; also handles .png/.tif."""
    match = re.match(r"^(.+?)\.(?:jpe?g|png|tiff?|bmp)(?:-|$)", value.strip(), re.I)
    if not match:
        raise ValueError(f"Unrecognized image identifier: {value!r}")
    return match[1].casefold()


def read_rows(archive, path):
    return list(csv.DictReader(io.StringIO(archive.read(path).decode("utf-8-sig"))))


def main():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Place the training ZIP at {ZIP_PATH}")

    errors = []
    warnings = []
    hashes = {}
    decoded_images = 0
    decoded_masks = 0

    def check(message, problems, warning=False):
        """Print a count and a few examples, keeping the output easy to read."""
        print(f"  {message}: {len(problems)}")
        if problems:
            (warnings if warning else errors).append(message)
            print("    Examples (up to 6):", "; ".join(str(p) for p in problems[:6]))

    print("CHAKSU TRAINING DATA — checking photographs, labels and disc/cup masks")
    with ZipFile(ZIP_PATH) as archive:
        files = [
            n for n in archive.namelist()
            if not n.endswith("/") and "__MACOSX" not in n
            and not Path(n).name.startswith(".")
        ]
        check("Repeated ZIP filenames", [n for n, count in Counter(files).items() if count > 1])
        for camera, expected in EXPECTED_IMAGES.items():
            print(f"\n{camera}")
            originals = sorted(n for n in files if n.startswith(
                f"Train/1.0_Original_Fundus_Images/{camera}/"
            ) and Path(n).suffix.lower() in IMAGE_EXTENSIONS)
            ids = [image_id(Path(n).name) for n in originals]
            image_ids = set(ids)
            print(f"  Original images: {len(originals)} (expected {expected})")
            check("Unexpected image count", [] if len(originals) == expected else [len(originals)])
            check("Repeated image IDs", [key for key, count in Counter(ids).items() if count > 1])

            # These are the supplied majority-vote labels, not the measurement CSVs.
            rows = read_rows(archive, f"Train/6.0_Glaucoma_Decision/"
                             f"Glaucoma_Decision_Comparison_{camera}_majority.csv")
            label_ids = [image_id(row["Images"]) for row in rows]
            check("Repeated label IDs", [key for key, count in Counter(label_ids).items() if count > 1])
            check("Images without a label", sorted(image_ids - set(label_ids)))
            extra = sorted(set(label_ids) - image_ids)
            print(f"  Label rows without a training image (excluded from counts): {len(extra)}")
            if len(extra) != (41 if camera == "Bosch" else 0):
                warnings.append(f"{camera}: unexpected extra label rows")
            matched = [row for row in rows if image_id(row["Images"]) in image_ids]
            print("  Labels agreed on by most experts:", dict(Counter(row["Majority Decision"] for row in matched)))
            check("Unknown or blank majority labels", [row["Images"] for row in rows
                  if row["Majority Decision"] not in LABELS])
            bad_votes = []
            for row in rows:
                votes = [EXPERT_LABELS.get(row[f"Expert.{expert}"]) for expert in range(1, 6)]
                majority = Counter(votes).most_common(1)[0][0]
                if None in votes or majority != row["Majority Decision"]:
                    bad_votes.append(row["Images"])
            check("Unknown expert labels or disagreement with the recorded group decision", bad_votes)

            # Check that every expert's table and both mask filenames cover each image.
            # Only the majority masks are opened below; expert masks are checked for presence.
            for expert in range(1, 6):
                expert_rows = read_rows(archive, f"Train/6.0_Glaucoma_Decision/Expert {expert}/{camera}.csv")
                expert_ids = [image_id(row["Images"]) for row in expert_rows]
                if camera == "Forus" and expert == 4:
                    # These three supplied identifiers contain an extra label tag.
                    aliases = {"22glsusp": "22", "82glsusp": "82", "83glsusp": "83"}
                    print("  Expert 4 uses these names: 22GLSUSP for 22, 82GLSUSP for 82, 83GLSUSP for 83")
                    expert_ids = [aliases.get(key, key) for key in expert_ids]
                problems = sorted(image_ids - set(expert_ids))
                problems += [key for key, count in Counter(expert_ids).items() if count > 1]
                for part in ("Disc", "Cup"):
                    prefix = f"Train/3.0_Doctors_Annotations_Binary_OD_OC/Expert {expert}/{camera}/{part}/"
                    mask_ids = {Path(n).stem.casefold() for n in files if n.startswith(prefix)}
                    problems += [f"{key}: missing {part}" for key in sorted(image_ids - mask_ids)]
                check(f"Expert {expert} missing/repeated records or missing masks", problems)

            mask_paths = {}
            for part in ("Disc", "Cup"):
                prefix = f"Train/5.0_OD_OC_Mean_Median_Majority_STAPLE/{camera}/{part}/Majority/"
                paths = [n for n in files if n.startswith(prefix)]
                keys = [Path(n).stem.casefold() for n in paths]
                check(f"Repeated majority {part} mask IDs", [key for key, count in Counter(keys).items() if count > 1])
                check(f"Missing majority {part} masks", sorted(image_ids - set(keys)))
                mask_paths[part] = dict(zip(keys, paths))

            unreadable, bad_masks, outside_disc, duplicates = [], [], [], []
            sizes = Counter()
            for path, key in zip(originals, ids):
                try:
                    data = archive.read(path)
                    with Image.open(io.BytesIO(data)) as photo:
                        photo.load()  # Decode every photograph, not just its header.
                        size = photo.size
                        sizes[size] += 1
                    decoded_images += 1
                    digest = hashlib.sha256(data).hexdigest()
                    short_path = f"{camera}/{Path(path).name}"
                    if digest in hashes:
                        duplicates.append(f"{short_path} = {hashes[digest]}")
                    hashes[digest] = short_path
                except (OSError, ValueError, KeyError) as exc:
                    unreadable.append(f"{path}: {exc}")
                    continue

                masks = {}
                for part in ("Disc", "Cup"):
                    if key not in mask_paths[part]:
                        continue  # Already reported as missing above.
                    mask_path = mask_paths[part][key]
                    try:
                        with Image.open(io.BytesIO(archive.read(mask_path))) as mask:
                            mask.load()
                            decoded_masks += 1
                            if mask.mode != "L" or mask.size != size:
                                bad_masks.append(f"{key} {part}: unexpected image format or size ({mask.mode}, {mask.size})")
                                continue
                            histogram = mask.histogram()
                            if sum(histogram[1:255]) or not histogram[0] or not histogram[255]:
                                bad_masks.append(f"{key} {part}: empty mask or values other than black and white")
                                continue
                            masks[part] = mask.copy()
                    except (OSError, ValueError, KeyError) as exc:
                        unreadable.append(f"{mask_path}: {exc}")
                if len(masks) == 2 and ImageChops.subtract(masks["Cup"], masks["Disc"]).getbbox():
                    outside_disc.append(key)

            print("  Photo dimensions (width, height):", dict(sorted(sizes.items())))
            check("Unreadable photographs/majority masks", unreadable)
            check("Masks with a wrong size, empty area or unexpected colours", bad_masks)
            check("Identical photograph files", duplicates, warning=True)
            check("Majority cups extending outside the disc", outside_disc, warning=True)

    print(f"\nOpened {decoded_images} photographs and {decoded_masks} majority masks.")
    print(f"Result: {len(errors)} failed checks; {len(warnings)} checks with warnings.")
    print("Labels describe NORMAL / GLAUCOMA SUSPECT, not confirmed clinical diagnoses.")
    print("We do not yet know which Chaksu photographs belong to the same person.")
    print("PAPILA has a separate check script.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
