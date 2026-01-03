"""Video frame extraction and preprocessing."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FrameData:
    """Container for frame data with associated metadata."""
    frame: np.ndarray
    timestamp: float  # Timestamp in seconds
    frame_number: int  # Frame index in the video
    original_frame: np.ndarray  # Original frame before preprocessing


class VideoProcessor:
    """
    Video processor for frame extraction, preprocessing, and timestamp management.
    
    Features:
    - Configurable frame extraction rate (FPS)
    - Frame preprocessing (resize, normalize, contrast enhancement for OCR)
    - Timestamp tracking for temporal analysis
    """
    
    def __init__(
        self,
        video_path: str | Path,
        extraction_fps: float = 1.5,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        enhance_contrast: bool = True,
        normalize: bool = True
    ):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the video file
            extraction_fps: FPS at which to extract frames (default: 1.5)
            target_width: Target width for resizing (None = keep original)
            target_height: Target height for resizing (None = keep original)
            enhance_contrast: Whether to enhance contrast for OCR
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.extraction_fps = extraction_fps
        self.target_width = target_width
        self.target_height = target_height
        self.enhance_contrast = enhance_contrast
        self.normalize = normalize
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_fps: float = 0.0
        self.total_frames: int = 0
        self.duration: float = 0.0
        self.frame_interval: int = 0  # Number of frames to skip between extractions
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def open(self) -> None:
        """Open the video file and initialize metadata."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
        
        # Get video metadata
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0.0
        
        # Calculate frame interval for extraction
        if self.extraction_fps > 0 and self.video_fps > 0:
            self.frame_interval = max(1, int(self.video_fps / self.extraction_fps))
        else:
            self.frame_interval = 1
            
    def close(self) -> None:
        """Close the video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def get_video_info(self) -> dict:
        """
        Get video metadata information.
        
        Returns:
            Dictionary containing video metadata
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first.")
            
        return {
            "path": str(self.video_path),
            "fps": self.video_fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration,
            "duration_minutes": self.duration / 60.0,
            "extraction_fps": self.extraction_fps,
            "frame_interval": self.frame_interval,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for analysis and OCR.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame
        """
        processed = frame.copy()
        
        # Resize if target dimensions are specified
        if self.target_width is not None or self.target_height is not None:
            current_height, current_width = processed.shape[:2]
            
            # If only one dimension is specified, maintain aspect ratio
            if self.target_width is not None and self.target_height is None:
                scale = self.target_width / current_width
                target_height = int(current_height * scale)
                processed = cv2.resize(processed, (self.target_width, target_height))
            elif self.target_height is not None and self.target_width is None:
                scale = self.target_height / current_height
                target_width = int(current_width * scale)
                processed = cv2.resize(processed, (target_width, self.target_height))
            elif self.target_width is not None and self.target_height is not None:
                processed = cv2.resize(processed, (self.target_width, self.target_height))
        
        # Enhance contrast for better OCR (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        if self.enhance_contrast:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        # Normalize pixel values to [0, 1] if requested
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        else:
            # Ensure uint8 format if not normalizing
            processed = processed.astype(np.uint8)
            
        return processed
        
    def extract_frames(self) -> Iterator[FrameData]:
        """
        Extract frames from the video at the specified extraction rate.
        
        Yields:
            FrameData objects containing frame, timestamp, and metadata
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first or use context manager.")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        frame_number = 0
        extracted_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_number % self.frame_interval == 0:
                timestamp = frame_number / self.video_fps if self.video_fps > 0 else 0.0
                original_frame = frame.copy()
                processed_frame = self._preprocess_frame(frame)
                
                yield FrameData(
                    frame=processed_frame,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    original_frame=original_frame
                )
                
                extracted_count += 1
            
            frame_number += 1
            
    def extract_all_frames(self) -> List[FrameData]:
        """
        Extract all frames from the video and return as a list.
        
        Note: This loads all frames into memory. Use extract_frames() for
        memory-efficient iteration.
        
        Returns:
            List of FrameData objects
        """
        return list(self.extract_frames())
        
    def extract_frame_at_timestamp(self, timestamp: float) -> Optional[FrameData]:
        """
        Extract a single frame at the specified timestamp.
        
        Args:
            timestamp: Timestamp in seconds
            
        Returns:
            FrameData object or None if timestamp is out of bounds
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first or use context manager.")
        
        if timestamp < 0 or timestamp > self.duration:
            return None
        
        # Seek to the frame at the specified timestamp
        frame_number = int(timestamp * self.video_fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        original_frame = frame.copy()
        processed_frame = self._preprocess_frame(frame)
        
        return FrameData(
            frame=processed_frame,
            timestamp=timestamp,
            frame_number=frame_number,
            original_frame=original_frame
        )
        
    def extract_frames_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> Iterator[FrameData]:
        """
        Extract frames within a specific time range.
        
        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            
        Yields:
            FrameData objects within the specified time range
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open() first or use context manager.")
        
        if start_time < 0:
            start_time = 0.0
        if end_time > self.duration:
            end_time = self.duration
            
        start_frame = int(start_time * self.video_fps)
        end_frame = int(end_time * self.video_fps)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_number = start_frame
        
        while frame_number <= end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_number % self.frame_interval == 0:
                timestamp = frame_number / self.video_fps if self.video_fps > 0 else 0.0
                original_frame = frame.copy()
                processed_frame = self._preprocess_frame(frame)
                
                yield FrameData(
                    frame=processed_frame,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    original_frame=original_frame
                )
            
            frame_number += 1
