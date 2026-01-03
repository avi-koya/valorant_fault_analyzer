"""Tests for video_processor module."""

import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.video_processor import VideoProcessor, FrameData


class TestFrameData:
    """Tests for FrameData dataclass."""
    
    def test_frame_data_creation(self):
        """Test FrameData can be created with required fields."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        original_frame = np.ones((100, 100, 3), dtype=np.uint8)
        
        frame_data = FrameData(
            frame=frame,
            timestamp=1.5,
            frame_number=10,
            original_frame=original_frame
        )
        
        assert np.array_equal(frame_data.frame, frame)
        assert frame_data.timestamp == 1.5
        assert frame_data.frame_number == 10
        assert np.array_equal(frame_data.original_frame, original_frame)


class TestVideoProcessorInitialization:
    """Tests for VideoProcessor initialization."""
    
    def test_init_with_valid_path(self, tmp_path):
        """Test initialization with a valid path."""
        # Create a dummy video file
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(video_file, extraction_fps=2.0)
        
        assert processor.video_path == Path(video_file)
        assert processor.extraction_fps == 2.0
        assert processor.target_width is None
        assert processor.target_height is None
        assert processor.enhance_contrast is True
        assert processor.normalize is True
        assert processor.cap is None
    
    def test_init_with_string_path(self, tmp_path):
        """Test initialization accepts string path."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(str(video_file))
        
        assert processor.video_path == Path(video_file)
    
    def test_init_with_invalid_path(self):
        """Test initialization raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            VideoProcessor("nonexistent_video.mp4")
    
    def test_init_with_custom_parameters(self, tmp_path):
        """Test initialization with custom parameters."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(
            video_file,
            extraction_fps=5.0,
            target_width=640,
            target_height=480,
            enhance_contrast=False,
            normalize=False
        )
        
        assert processor.extraction_fps == 5.0
        assert processor.target_width == 640
        assert processor.target_height == 480
        assert processor.enhance_contrast is False
        assert processor.normalize is False


class TestVideoProcessorOpenClose:
    """Tests for video file opening and closing."""
    
    @patch('cv2.VideoCapture')
    def test_open_success(self, mock_video_capture, tmp_path):
        """Test successful video opening."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, extraction_fps=1.5)
        processor.open()
        
        assert processor.cap is not None
        assert processor.video_fps == 30.0
        assert processor.total_frames == 900
        assert processor.duration == 30.0  # 900 frames / 30 fps
        assert processor.frame_interval == 20  # 30 fps / 1.5 extraction_fps
    
    @patch('cv2.VideoCapture')
    def test_open_failure(self, mock_video_capture, tmp_path):
        """Test video opening failure."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        # Mock VideoCapture that fails to open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        
        with pytest.raises(ValueError, match="Failed to open video file"):
            processor.open()
    
    @patch('cv2.VideoCapture')
    def test_close(self, mock_video_capture, tmp_path):
        """Test video closing."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        processor.open()
        processor.close()
        
        mock_cap.release.assert_called_once()
        assert processor.cap is None
    
    def test_close_when_not_opened(self, tmp_path):
        """Test closing when video is not opened."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(video_file)
        processor.close()  # Should not raise error
        
        assert processor.cap is None


class TestVideoProcessorContextManager:
    """Tests for context manager functionality."""
    
    @patch('cv2.VideoCapture')
    def test_context_manager(self, mock_video_capture, tmp_path):
        """Test VideoProcessor as context manager."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        
        with processor:
            assert processor.cap is not None
        
        mock_cap.release.assert_called_once()
        assert processor.cap is None
    
    @patch('cv2.VideoCapture')
    def test_context_manager_exception_handling(self, mock_video_capture, tmp_path):
        """Test context manager properly closes on exception."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        
        with pytest.raises(ValueError):
            with processor:
                assert processor.cap is not None
                raise ValueError("Test exception")
        
        mock_cap.release.assert_called_once()
        assert processor.cap is None


class TestVideoProcessorGetVideoInfo:
    """Tests for get_video_info method."""
    
    @patch('cv2.VideoCapture')
    def test_get_video_info(self, mock_video_capture, tmp_path):
        """Test get_video_info returns correct metadata."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 1800,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, extraction_fps=2.0)
        processor.open()
        
        info = processor.get_video_info()
        
        assert info["path"] == str(video_file)
        assert info["fps"] == 30.0
        assert info["total_frames"] == 1800
        assert info["duration_seconds"] == 60.0
        assert info["duration_minutes"] == 1.0
        assert info["extraction_fps"] == 2.0
        assert info["frame_interval"] == 15  # 30 / 2
        assert info["width"] == 1920
        assert info["height"] == 1080
    
    def test_get_video_info_not_opened(self, tmp_path):
        """Test get_video_info raises error when video not opened."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(video_file)
        
        with pytest.raises(RuntimeError, match="Video not opened"):
            processor.get_video_info()


class TestVideoProcessorPreprocessing:
    """Tests for frame preprocessing."""
    
    @patch('cv2.VideoCapture')
    def test_preprocess_frame_no_resize(self, mock_video_capture, tmp_path):
        """Test preprocessing without resizing."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, normalize=False, enhance_contrast=False)
        processor.open()
        
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        processed = processor._preprocess_frame(frame)
        
        assert processed.shape == frame.shape
        assert processed.dtype == np.uint8
    
    @patch('cv2.VideoCapture')
    def test_preprocess_frame_resize_both_dimensions(self, mock_video_capture, tmp_path):
        """Test preprocessing with both width and height resizing."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(
            video_file,
            target_width=640,
            target_height=480,
            normalize=False,
            enhance_contrast=False
        )
        processor.open()
        
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        processed = processor._preprocess_frame(frame)
        
        assert processed.shape == (480, 640, 3)
    
    @patch('cv2.VideoCapture')
    def test_preprocess_frame_resize_width_only(self, mock_video_capture, tmp_path):
        """Test preprocessing with width-only resizing (maintains aspect ratio)."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(
            video_file,
            target_width=640,
            normalize=False,
            enhance_contrast=False
        )
        processor.open()
        
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        processed = processor._preprocess_frame(frame)
        
        # Aspect ratio should be maintained: 1920/1080 = 640/x -> x = 360
        assert processed.shape[1] == 640
        assert processed.shape[0] == 360
    
    @patch('cv2.VideoCapture')
    def test_preprocess_frame_normalize(self, mock_video_capture, tmp_path):
        """Test preprocessing with normalization."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, normalize=True, enhance_contrast=False)
        processor.open()
        
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        processed = processor._preprocess_frame(frame)
        
        assert processed.dtype == np.float32
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0
    
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    @patch('cv2.split')
    @patch('cv2.createCLAHE')
    @patch('cv2.merge')
    def test_preprocess_frame_enhance_contrast(
        self, mock_merge, mock_create_clahe, mock_split,
        mock_cvt_color, mock_video_capture, tmp_path
    ):
        """Test preprocessing with contrast enhancement."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        # Mock CLAHE
        mock_clahe = MagicMock()
        mock_l = np.zeros((100, 200), dtype=np.uint8)
        mock_clahe.apply.return_value = mock_l
        mock_create_clahe.return_value = mock_clahe
        
        # Mock color space conversions
        mock_cvt_color.side_effect = lambda img, code: img  # Return input as-is for simplicity
        mock_split.return_value = (mock_l, mock_l, mock_l)
        mock_merge.return_value = np.zeros((100, 200, 3), dtype=np.uint8)
        
        processor = VideoProcessor(video_file, enhance_contrast=True, normalize=False)
        processor.open()
        
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        processed = processor._preprocess_frame(frame)
        
        # Verify CLAHE was created and applied
        mock_create_clahe.assert_called_once_with(clipLimit=2.0, tileGridSize=(8, 8))
        mock_clahe.apply.assert_called_once()


class TestVideoProcessorExtractFrames:
    """Tests for frame extraction methods."""
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_basic(self, mock_video_capture, tmp_path):
        """Test basic frame extraction."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        # Create mock frames
        mock_frames = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: 5,
        }.get(prop, 0)
        
        # Mock read() to return frames sequentially
        read_count = [0]
        def mock_read():
            if read_count[0] < len(mock_frames):
                read_count[0] += 1
                return True, mock_frames[read_count[0] - 1]
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, extraction_fps=10.0, normalize=False, enhance_contrast=False)
        processor.open()
        
        frames = list(processor.extract_frames())
        
        # Should extract all frames when extraction_fps equals video_fps
        assert len(frames) == 5
        assert all(isinstance(f, FrameData) for f in frames)
        assert frames[0].frame_number == 0
        assert frames[0].timestamp == 0.0
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_with_interval(self, mock_video_capture, tmp_path):
        """Test frame extraction with frame interval."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_frames = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
        }.get(prop, 0)
        
        read_count = [0]
        def mock_read():
            if read_count[0] < len(mock_frames):
                read_count[0] += 1
                return True, mock_frames[read_count[0] - 1]
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_video_capture.return_value = mock_cap
        
        # Extract at 10 fps from 30 fps video (interval = 3)
        processor = VideoProcessor(video_file, extraction_fps=10.0, normalize=False, enhance_contrast=False)
        processor.open()
        
        frames = list(processor.extract_frames())
        
        # Should extract approximately 4 frames (frames 0, 3, 6, 9)
        # Actually, with frame_interval=3, we get frames at 0, 3, 6, 9 = 4 frames
        assert len(frames) == 4
        assert frames[0].frame_number == 0
        assert frames[1].frame_number == 3
        assert frames[2].frame_number == 6
        assert frames[3].frame_number == 9
    
    def test_extract_frames_not_opened(self, tmp_path):
        """Test extract_frames raises error when video not opened."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(video_file)
        
        with pytest.raises(RuntimeError, match="Video not opened"):
            list(processor.extract_frames())
    
    @patch('cv2.VideoCapture')
    def test_extract_all_frames(self, mock_video_capture, tmp_path):
        """Test extract_all_frames returns list."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_frames = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: 3,
        }.get(prop, 0)
        
        read_count = [0]
        def mock_read():
            if read_count[0] < len(mock_frames):
                read_count[0] += 1
                return True, mock_frames[read_count[0] - 1]
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, extraction_fps=10.0, normalize=False, enhance_contrast=False)
        processor.open()
        
        frames = processor.extract_all_frames()
        
        assert isinstance(frames, list)
        assert len(frames) == 3


class TestVideoProcessorExtractFrameAtTimestamp:
    """Tests for extract_frame_at_timestamp method."""
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_at_timestamp_valid(self, mock_video_capture, tmp_path):
        """Test extracting frame at valid timestamp."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,  # 30 seconds
        }.get(prop, 0)
        mock_cap.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, normalize=False, enhance_contrast=False)
        processor.open()
        
        frame_data = processor.extract_frame_at_timestamp(10.0)
        
        assert frame_data is not None
        assert isinstance(frame_data, FrameData)
        assert frame_data.timestamp == 10.0
        assert frame_data.frame_number == 300  # 10 seconds * 30 fps
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 300)
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_at_timestamp_out_of_bounds_negative(self, mock_video_capture, tmp_path):
        """Test extracting frame at negative timestamp."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        processor.open()
        
        frame_data = processor.extract_frame_at_timestamp(-1.0)
        
        assert frame_data is None
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_at_timestamp_out_of_bounds_too_large(self, mock_video_capture, tmp_path):
        """Test extracting frame at timestamp beyond video duration."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,  # 30 seconds
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        processor.open()
        
        frame_data = processor.extract_frame_at_timestamp(100.0)
        
        assert frame_data is None
    
    @patch('cv2.VideoCapture')
    def test_extract_frame_at_timestamp_read_fails(self, mock_video_capture, tmp_path):
        """Test extracting frame when read fails."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        processor.open()
        
        frame_data = processor.extract_frame_at_timestamp(10.0)
        
        assert frame_data is None
    
    def test_extract_frame_at_timestamp_not_opened(self, tmp_path):
        """Test extract_frame_at_timestamp raises error when video not opened."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(video_file)
        
        with pytest.raises(RuntimeError, match="Video not opened"):
            processor.extract_frame_at_timestamp(10.0)


class TestVideoProcessorExtractFramesInRange:
    """Tests for extract_frames_in_range method."""
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_in_range_valid(self, mock_video_capture, tmp_path):
        """Test extracting frames in valid time range."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_frames = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,  # 30 seconds
        }.get(prop, 0)
        
        read_count = [0]
        def mock_read():
            if read_count[0] < len(mock_frames):
                read_count[0] += 1
                return True, mock_frames[read_count[0] - 1]
            return False, None
        
        mock_cap.read.side_effect = mock_read
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, extraction_fps=30.0, normalize=False, enhance_contrast=False)
        processor.open()
        
        # Extract frames from 10 to 12 seconds
        frames = list(processor.extract_frames_in_range(10.0, 12.0))
        
        # Should seek to frame 300 (10 seconds * 30 fps)
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 300)
        assert all(isinstance(f, FrameData) for f in frames)
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_in_range_negative_start(self, mock_video_capture, tmp_path):
        """Test extracting frames with negative start time (should clamp to 0)."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        processor.open()
        
        frames = list(processor.extract_frames_in_range(-5.0, 10.0))
        
        # Should clamp start_time to 0, so seek to frame 0
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 0)
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_in_range_end_too_large(self, mock_video_capture, tmp_path):
        """Test extracting frames with end time beyond duration (should clamp)."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,  # 30 seconds
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        processor.open()
        
        frames = list(processor.extract_frames_in_range(10.0, 100.0))
        
        # Should clamp end_time to duration (30 seconds), so end_frame = 900
        call_args = mock_cap.set.call_args_list
        # First call should be to set position to start_frame
        assert call_args[0][0][1] == 300  # 10 seconds * 30 fps
    
    def test_extract_frames_in_range_not_opened(self, tmp_path):
        """Test extract_frames_in_range raises error when video not opened."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        processor = VideoProcessor(video_file)
        
        with pytest.raises(RuntimeError, match="Video not opened"):
            list(processor.extract_frames_in_range(10.0, 20.0))


class TestVideoProcessorEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    @patch('cv2.VideoCapture')
    def test_zero_fps_handling(self, mock_video_capture, tmp_path):
        """Test handling of video with zero FPS."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 0.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file)
        processor.open()
        
        assert processor.video_fps == 0.0
        assert processor.duration == 0.0
        assert processor.frame_interval == 1
    
    @patch('cv2.VideoCapture')
    def test_zero_extraction_fps_handling(self, mock_video_capture, tmp_path):
        """Test handling of zero extraction FPS."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor(video_file, extraction_fps=0.0)
        processor.open()
        
        # Should use frame_interval of 1 when extraction_fps is 0
        assert processor.frame_interval == 1
