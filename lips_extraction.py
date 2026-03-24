import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

PHYSICAL_PARTS = ["lefteye", "righteye", "nose", "leftmouth", "rightmouth"]


class LandmarkExtractor:
    RESULT_SIZE = 64

    def __init__(self, src_folder: str, df_landmarks: pd.DataFrame, n_workers: int = 8):
        self.src_folder = src_folder
        self.df_landmarks = df_landmarks.set_index("image_id")  # index for O(1) lookups
        self.n_workers = n_workers

    def show_image(self, image: np.ndarray, landmarks: pd.Series = None):
        image_tmp = image.copy()
        if landmarks is not None:
            for part in PHYSICAL_PARTS:
                x = landmarks[f"{part}_x"]
                y = landmarks[f"{part}_y"]
                cv2.circle(image_tmp, (x, y), 3, (0, 0, 255), -1)
        plt.imshow(image_tmp)
        plt.axis("off")
        plt.show()

    def get_croped_image(
        self, image_id: int, margin: float = 0.2, plot_images: bool = False
    ):
        img_name = f"{image_id:06d}.jpg"
        landmark = self.__get_landmark(img_name)
        image = self.__load_image(img_name)

        if plot_images:
            self.show_image(image, landmark)

        rotation, center, final_src_size = self.get_lips_rotation(landmark, margin)
        rotated_image = cv2.warpAffine(
            image, rotation, (image.shape[1], image.shape[0])
        )

        x, y = int(center[0]), int(center[1])
        half_size = int(final_src_size / 2)
        cropped_image = rotated_image[
            max(0, y - half_size) : min(rotated_image.shape[0], y + half_size),
            max(0, x - half_size) : min(rotated_image.shape[1], x + half_size),
        ]

        final_image = cv2.resize(cropped_image, (self.RESULT_SIZE, self.RESULT_SIZE))

        if plot_images:
            self.show_image(final_image)

        return final_image

    def __get_landmark(self, image_name: str) -> pd.Series:
        try:
            return self.df_landmarks.loc[image_name]
        except KeyError:
            raise ValueError(f"Image {image_name} not found in landmarks dataframe.")

    def __load_image(self, image_name: str) -> np.ndarray:
        image = cv2.imread(os.path.join(self.src_folder, image_name))
        if image is None:
            raise ValueError(f"Image {image_name} not found.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_lips_rotation(self, landmark: pd.Series, margin: float = 0.2):
        leftmouth_x, leftmouth_y = landmark["leftmouth_x"], landmark["leftmouth_y"]
        rightmouth_x, rightmouth_y = landmark["rightmouth_x"], landmark["rightmouth_y"]
        nose_x, nose_y = landmark["nose_x"], landmark["nose_y"]

        center_x = (leftmouth_x + rightmouth_x) / 2
        center_y = (leftmouth_y + rightmouth_y) / 2

        dist_leftmouth = np.linalg.norm(
            [leftmouth_x - center_x, leftmouth_y - center_y]
        )
        dist_rightmouth = np.linalg.norm(
            [rightmouth_x - center_x, rightmouth_y - center_y]
        )
        dist_nose = np.linalg.norm([nose_x - center_x, nose_y - center_y])

        final_src_size = (
            2 * max(dist_leftmouth, dist_rightmouth, dist_nose) * (1 + margin)
        )

        dx = rightmouth_x - leftmouth_x
        dy = rightmouth_y - leftmouth_y
        angle = np.degrees(np.arctan2(dy, dx))

        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        return rotation_matrix, (center_x, center_y), final_src_size

    def process_and_save(
        self, image_id: int, output_dir: str, margin: float = 0.2
    ) -> tuple[int, bool]:
        """Process a single image and save it. Returns (image_id, success)."""
        output_path = os.path.join(output_dir, f"{image_id:06d}.jpg")
        if os.path.exists(output_path):  # skip already processed
            return image_id, True
        try:
            cropped = self.get_croped_image(image_id, margin=margin)
            cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            return image_id, True
        except Exception as e:
            print(f"[WARN] image_id={image_id}: {e}")
            return image_id, False


def process_all_images(
    extractor: LandmarkExtractor,
    img_ids: list[int],
    output_dir: str,
    margin: float = 0.2,
) -> list[int]:
    """
    Process all images in img_ids, save crops to output_dir.
    Returns a list of failed image ids.
    """
    os.makedirs(output_dir, exist_ok=True)
    failed = []

    with ThreadPoolExecutor(max_workers=extractor.n_workers) as executor:
        futures = {
            executor.submit(
                extractor.process_and_save, img_id, output_dir, margin
            ): img_id
            for img_id in img_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Cropping images"
        ):
            img_id, success = future.result()
            if not success:
                failed.append(img_id)

    print(
        f"Done. {len(img_ids) - len(failed)}/{len(img_ids)} images saved to '{output_dir}/'."
    )
    if failed:
        print(f"Failed image ids: {failed}")
    return failed


def extract_and_save_lips(
    df_landmarks: pd.DataFrame,
    output_dir: str,
    src_folder: str,
    n_workers: int = 8,
    margin: float = 0.2,
):
    extractor = LandmarkExtractor(src_folder, df_landmarks, n_workers=n_workers)
    img_ids = df_landmarks["image_id"].apply(lambda x: int(x.split(".")[0])).tolist()
    failed_ids = process_all_images(
        extractor, img_ids, output_dir=output_dir, margin=margin
    )
    return failed_ids
