#!/usr/bin/env python3
"""
Script to create a video that alternates between images from a directory and gray frames.
Gray frames have RGB values of 127.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def get_image_files(directory):
    """Get all image files from directory, sorted by filename."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = []

    for file_path in Path(directory).iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    # Sort by filename
    image_files.sort(key=lambda x: x.name)
    return image_files


def create_gray_frame(width, height):
    """Create a gray frame with RGB values of 127."""
    gray_frame = np.full((height, width, 3), 127, dtype=np.uint8)
    return gray_frame


def resize_image_to_target(image, target_width, target_height):
    """Resize image and center-crop to exact target dimensions without padding."""
    # Get original dimensions
    orig_height, orig_width = image.shape[:2]

    # Calculate scaling factors for both width and height
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height

    # Use the larger scale to ensure the image covers the entire target area
    scale = max(scale_w, scale_h)

    # Calculate new dimensions after scaling
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # Resize image
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
    )

    # Center crop to exact target dimensions
    # Calculate crop coordinates
    x_start = (new_width - target_width) // 2
    y_start = (new_height - target_height) // 2

    # Perform center crop
    cropped = resized[
        y_start : y_start + target_height, x_start : x_start + target_width
    ]

    return cropped


def create_video(image_directory, output_path, fps=24, width=832, height=480):
    """Create video alternating between images and gray frames, plus a mask video."""

    # Get all image files
    image_files = get_image_files(image_directory)

    if not image_files:
        print(f"No image files found in {image_directory}")
        return False

    print(f"Found {len(image_files)} images")

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create mask video path
    mask_output_path = output_path.replace(".mp4", "_mask.mp4")
    mask_out = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False

    if not mask_out.isOpened():
        print(f"Error: Could not open video writer for {mask_output_path}")
        out.release()
        return False

    try:
        for i, image_file in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")

            # Load and process the image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Warning: Could not load {image_file}")
                continue

            # Resize image to fit target dimensions
            resized_image = resize_image_to_target(image, width, height)

            # Write the image frame
            out.write(resized_image)

            # Create and write black mask frame for image
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            mask_out.write(black_frame)

            # Create and write gray frame
            gray_frame = create_gray_frame(width, height)
            out.write(gray_frame)

            # Create and write white mask frame for gray frame
            white_frame = np.full((height, width, 3), 255, dtype=np.uint8)
            mask_out.write(white_frame)

        print(f"Video created successfully: {output_path}")
        print(f"Mask video created successfully: {mask_output_path}")
        return True

    except Exception as e:
        print(f"Error creating video: {e}")
        return False

    finally:
        out.release()
        mask_out.release()


def main():
    parser = argparse.ArgumentParser(
        description="Create a video that alternates between images and gray frames"
    )
    parser.add_argument("input_directory", help="Directory containing input images")
    parser.add_argument(
        "output_video", help="Output video file path (e.g., output.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for output video (default: 24)",
    )
    parser.add_argument(
        "--width", type=int, default=832, help="Video width in pixels (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (default: 1080)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_directory):
        print(f"Error: {args.input_directory} is not a valid directory")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the video
    success = create_video(
        args.input_directory, args.output_video, args.fps, args.width, args.height
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
