from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm
from typing import List
from pathlib import Path
from dataclasses import dataclass
import os
import numpy as np


@dataclass
class Point:
    x: int
    y: int


@dataclass
class IterRect:
    """
    Represent a rectangle in a coordinate system where x grows from left to right, but y grows from top to bottom
    """
    top_left: Point
    bottom_right: Point

    def __post_init__(self):
        assert self.top_left.x < self.bottom_right.x, "Top left corner x must be < bottom right corner x!"
        assert self.top_left.y < self.bottom_right.y, "Top left corner y must be < bottom right corner y!"

    @classmethod
    def init_from_coords(cls, top_left_x: int, top_left_y: int, bottom_right_x: int, bottom_right_y: int):
        return cls(Point(top_left_x, top_left_y),
                   Point(bottom_right_x, bottom_right_y))

    def len_x(self) -> int:
        return self.bottom_right.x - self.top_left.x

    def len_y(self) -> int:
        return self.top_left.y - self.bottom_right.y

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.top_left.x, self.top_left.y, self.bottom_right.x, self.bottom_right.y

    def n_rects_x(self, num_steps: int, step_width: int, include_self: bool = True) -> List["IterRect"]:
        rect_range = range(num_steps) if include_self else range(1, num_steps + 1)
        rects = [self.init_from_coords(
                    self.top_left.x + step_width * i,
                    self.top_left.y,
                    self.bottom_right.x + step_width * i,
                    self.bottom_right.y)
                 for i in rect_range]
        return rects

    def n_rects_y(self, num_steps: int, step_width: int, include_self: bool = True) -> List["IterRect"]:
        rect_range = range(num_steps) if include_self else range(1, num_steps + 1)
        rects = [self.init_from_coords(
                    self.top_left.x,
                    self.top_left.y + step_width * i,
                    self.bottom_right.x,
                    self.bottom_right.y + step_width * i)
                 for i in rect_range]
        return rects

    def get_all_crops(self, crops_per_row: List[int], step_width: Point) -> List["IterRect"]:
        iter_rects = []
        for y_index, crop in enumerate(self.n_rects_y(num_steps=len(crops_per_row), step_width=step_width.y)):
            new_iter_rects = crop.n_rects_x(num_steps=crops_per_row[y_index], step_width=step_width.x)
            iter_rects.extend(new_iter_rects)
        return iter_rects


@dataclass
class Screenshotter:
    pages: dict

    @staticmethod
    def save_im_grayscale(im: Image, path: Path) -> None:
        gray_scaled = im.convert('L')
        gray_scaled.save(path)

    def create_screenshots(self, crops: List[IterRect], filename_prefix: str, out_dir: Path) -> None:
        for page_index, page in tqdm(self.pages.items(), total=len(self.pages), desc="Creating screenshots..."):
            for crop_index, crop in enumerate(crops):
                cropped_page = page.crop(box=crop.to_tuple())
                filepath = out_dir / f"page-{page_index}_{filename_prefix}-{crop_index}.png"
                self.save_im_grayscale(cropped_page, filepath)


def create_screenshots(tab_dir: Path, fret_dir: Path, target_pdf_path: Path) -> None:
    excluded_page_indices = np.arange(29) * 18 + 24
    excluded_page_indices = np.append([*range(7)], excluded_page_indices).tolist()
    pages = convert_from_path(target_pdf_path, thread_count=4, grayscale=True, use_pdftocairo=True)
    page_dict = {page_index: page for page_index, page in enumerate(pages) if page_index not in excluded_page_indices}

    start_crop_tab = IterRect.init_from_coords(120, 620, 370, 920)
    start_crop_fret = IterRect.init_from_coords(336, 650, 400, 714)
    tab_crops = start_crop_tab.get_all_crops(crops_per_row=[2, 4, 4], step_width=Point(367, 375))
    fret_crops = start_crop_fret.get_all_crops(crops_per_row=[2, 4, 4], step_width=Point(367, 375))

    sc = Screenshotter(page_dict)
    sc.create_screenshots(tab_crops, filename_prefix="tab", out_dir=tab_dir)
    sc.create_screenshots(fret_crops, filename_prefix="fret", out_dir=fret_dir)


if __name__ == "__main__":
    parent_dir = Path(os.getcwd())
    root_dir = parent_dir.parent.parent
    data_dir = root_dir / "data"
    tab_dir = root_dir / "data" / "dataset" / "tabs"
    fret_dir = root_dir / "data" / "dataset" / "frets"
    target_pdf_path = root_dir / "data" / "Guitar Chords Galore.pdf"
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fret_dir, exist_ok=True)
    create_screenshots(tab_dir, fret_dir, target_pdf_path)
