"""
score_images_gui.py

Tkinter viewer for the Dart Sense pipeline over still images.

For each image in the sample folder it runs the same per-frame pipeline the
live app uses (YOLO detect -> homography -> transform -> score) and displays:

    [ left ]  the original image with detected darts + calibration points
    [ right ] the transformed board-plane view with board outline + scores

Navigate with Previous / Next (or the left/right arrow keys). The scored
visit (e.g. "T20 S5 D16 = 97") is shown beneath the images.

Usage:
    python score_images_gui.py
    python score_images_gui.py --images data/darts/images/small_sample
    python score_images_gui.py --weights weights.pt --conf 0.5
"""

import os
import argparse
import numpy as np
import cv2

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

from ultralytics import YOLO
from get_scores import GetScores  # reuse the app's scoring logic unchanged


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')


class ScoreViewer:
    def __init__(self, images_dir, weights, conf, display_size=560):
        self.images_dir = images_dir
        self.conf = conf
        self.display_size = display_size

        self.model = YOLO(weights)
        self.predict = GetScores(weights)

        self.files = sorted(f for f in os.listdir(images_dir) if f.endswith(IMG_EXTS))
        if not self.files:
            raise SystemExit(f"No images found in {images_dir}")
        self.index = 0

        self._build_ui()
        self.show_current()
        self.root.mainloop()

    # ---------- UI ----------
    def _build_ui(self):
        self.root = Tk()
        self.root.title("Dart Sense - sample image scorer")

        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.grid(row=0, column=0, sticky=(N, W, E, S))

        self.title_label = ttk.Label(self.frame, font=('Helvetica', 12, 'bold'))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 8))

        ttk.Label(self.frame, text="Original (detections)",
                  font=('Helvetica', 10)).grid(row=1, column=0)
        ttk.Label(self.frame, text="Transformed (board plane)",
                  font=('Helvetica', 10)).grid(row=1, column=1)

        self.left_canvas = Canvas(self.frame, width=self.display_size,
                                  height=self.display_size, highlightthickness=1,
                                  highlightbackground="#888")
        self.left_canvas.grid(row=2, column=0, padx=6)
        self.right_canvas = Canvas(self.frame, width=self.display_size,
                                   height=self.display_size, highlightthickness=1,
                                   highlightbackground="#888")
        self.right_canvas.grid(row=2, column=1, padx=6)

        self.score_label = ttk.Label(self.frame, font=('Helvetica', 14, 'bold'))
        self.score_label.grid(row=3, column=0, columnspan=2, pady=10)

        nav = ttk.Frame(self.frame)
        nav.grid(row=4, column=0, columnspan=2)
        ttk.Button(nav, text="< Previous", command=self.prev).grid(row=0, column=0, padx=6)
        self.counter_label = ttk.Label(nav, font=('Helvetica', 10))
        self.counter_label.grid(row=0, column=1, padx=12)
        ttk.Button(nav, text="Next >", command=self.next).grid(row=0, column=2, padx=6)

        self.root.bind('<Left>', lambda e: self.prev())
        self.root.bind('<Right>', lambda e: self.next())
        self.root.bind('<Escape>', lambda e: self.root.quit())

    # ---------- navigation ----------
    def next(self):
        self.index = (self.index + 1) % len(self.files)
        self.show_current()

    def prev(self):
        self.index = (self.index - 1) % len(self.files)
        self.show_current()

    # ---------- pipeline ----------
    def _centre_square_crop(self, img):
        h, w = img.shape[:2]
        size = min(h, w)
        x0, y0 = (w - size) // 2, (h - size) // 2
        return img[y0:y0 + size, x0:x0 + size], size

    def _draw_board_outline(self, img):
        ds = self.display_size
        radii = self.predict.scoring_radii * ds
        angles = np.append(self.predict.segment_angles, 81)
        outer, inner = radii[-1], radii[2]
        c = ds / 2
        for ang in angles:
            oa = outer * np.cos(np.deg2rad(ang)); oo = (outer**2 - oa**2) ** 0.5
            ia = inner * np.cos(np.deg2rad(ang)); io = (inner**2 - ia**2) ** 0.5
            if ang > 0:
                pts = [((c + oa, c + oo), (c + ia, c + io)),
                       ((c - oa, c - oo), (c - ia, c - io))]
            else:
                pts = [((c - oa, c + oo), (c - ia, c + io)),
                       ((c + oa, c - oo), (c + ia, c - io))]
            for p1, p2 in pts:
                cv2.line(img, tuple(np.round(p1).astype(int)),
                         tuple(np.round(p2).astype(int)), (255, 0, 0), 1)
        for r in np.round(radii).astype(int):
            cv2.circle(img, (int(c), int(c)), int(r), (255, 0, 0), 1)
        return img

    def _process(self, path):
        """Returns (original_bgr, board_bgr, visit_text)."""
        img = cv2.imread(path)
        if img is None:
            return None, None, "could not read image"

        crop, crop_size = self._centre_square_crop(img)
        ds = self.display_size

        result = self.model(crop, conf=self.conf, verbose=False)[0]
        calibration_coords, dart_coords = self.predict.process_yolo_output(result)

        # ----- left: original crop with raw detections -----
        original = cv2.resize(crop, (ds, ds))
        for c in calibration_coords:
            if np.all(c >= 0):
                x, y = int(c[0] * ds), int(c[1] * ds)
                cv2.circle(original, (x, y), 5, (255, 255, 255), 2)
        for d in dart_coords:
            x, y = int(float(d[0]) * ds), int(float(d[1]) * ds)
            cv2.circle(original, (x, y), 6, (0, 255, 255), 2)

        detected = np.count_nonzero(np.all(calibration_coords >= 0, axis=1))
        if detected < 4:
            return original, None, f"only {detected}/4 calibration points - cannot transform"

        # ----- right: board-plane transform + scores -----
        H_matrix = self.predict.find_homography(calibration_coords, crop_size)
        board_coords = self.predict.transform_to_boardplane(H_matrix[0], dart_coords, crop_size)
        darts, score = self.predict.score(np.array(board_coords))

        warped = cv2.warpPerspective(crop, H_matrix[0], (crop_size, crop_size))
        warped = cv2.resize(warped, (ds, ds))
        warped = self._draw_board_outline(warped)
        for (dx, dy), label in zip(board_coords, darts):
            px, py = int(round(dx * ds)), int(round(dy * ds))
            cv2.circle(warped, (px, py), 6, (0, 255, 255), 2)
            cv2.putText(warped, label, (px - 12, py + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        visit = ' '.join(darts) if darts else '(no darts detected)'
        text = f"{visit}  =  {score}"
        return original, warped, text

    def _to_photo(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    # ---------- render ----------
    def show_current(self):
        fname = self.files[self.index]
        path = os.path.join(self.images_dir, fname)
        original, board, text = self._process(path)

        self.title_label.configure(text=fname)
        self.counter_label.configure(text=f"{self.index + 1} / {len(self.files)}")
        self.score_label.configure(text=text)

        self.left_canvas.delete('all')
        self.right_canvas.delete('all')

        if original is not None:
            self._left_img = self._to_photo(original)  # keep ref
            self.left_canvas.create_image(0, 0, image=self._left_img, anchor=NW)
        if board is not None:
            self._right_img = self._to_photo(board)    # keep ref
            self.right_canvas.create_image(0, 0, image=self._right_img, anchor=NW)
        else:
            self.right_canvas.create_text(self.display_size / 2, self.display_size / 2,
                                          text="no transform", fill="#888")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='data/darts/images/small_sample')
    ap.add_argument('--weights', default='weights.pt')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--display-size', type=int, default=560)
    args = ap.parse_args()
    ScoreViewer(args.images, args.weights, args.conf, args.display_size)


if __name__ == '__main__':
    main()
