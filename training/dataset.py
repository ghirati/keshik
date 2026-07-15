from datasets import load_dataset
import os
from dotenv import load_dotenv
from PIL import Image
import argparse


load_dotenv()
hf_token = os.environ.get("HF_TOKEN")


def download_dataset(subset, grayscale=False):

    # name: train_quality
    # num_examples: 1196221
    # name: validation
    # num_examples: 18582
    # name: test
    # num_examples: 55763

    ds = load_dataset("Harvard-Edge/Wake-Vision",
                      streaming=True, split=subset, token=hf_token)

    # Every non-depicted image is saved, unfiltered by distance and uncapped —
    # class balance and any subsampling are handled at training time, not here.
    person_dir = f"data/{subset}/person"
    not_person_dir = f"data/{subset}/not_person"
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(not_person_dir, exist_ok=True)

    num_person = 0
    num_not_person = 0

    for idx, data in enumerate(ds):
        if idx % 1000 == 0:
            print(f"#person: {num_person}, #not_person: {num_not_person}")

        # Depicted images are not relevant in the project.
        if not data["depiction"]:

            # Squash-resize (independent x/y scale, no crop/letterbox), bilinear.
            # This is the reference implementation — firmware must replicate this
            # exact operation (not rely on the camera's built-in FRAMESIZE_96X96
            # scaling, whose crop/squash behavior is unverified). See TODO.
            img = data["image"].convert("L").resize(
                (96, 96), Image.Resampling.BILINEAR) if grayscale else data["image"].resize((96, 96), Image.Resampling.BILINEAR)

            if (data["person"]):
                # quality=90 in Pillow ≈ jpeg_quality=12 on the ESP32-S3 camera
                img.save(f"{person_dir}/{idx:08d}.jpg", quality=90)
                num_person += 1

            else:
                img.save(f"{not_person_dir}/{idx:08d}.jpg", quality=90)
                num_not_person += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True,
                        choices=["train_quality", "validation", "test"])
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()
    download_dataset(args.subset, grayscale=args.grayscale)
