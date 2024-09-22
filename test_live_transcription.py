import pytest
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock
from live_transcription import (
    setup_logging,
    save_audio_chunk,
    process_live_chunk,
    transcribe_live,
    list_audio_devices,
    select_audio_device,
    callback
)

@pytest.fixture
def audio_data():
    return np.random.rand(16000, 2) * 2 - 1  # Sample stereo audio chunk

@pytest.fixture
def transcription_output():
    return "test_output.txt"

@pytest.fixture
def mapping_file():
    return "test_mapping.jsonl"

@pytest.fixture
def setup_environment(transcription_output, mapping_file):
    # Ensure clean state before tests
    if os.path.exists(transcription_output):
        os.remove(transcription_output)
    if os.path.exists(mapping_file):
        os.remove(mapping_file)
    yield
    # Cleanup after tests
    if os.path.exists(transcription_output):
        os.remove(transcription_output)
    if os.path.exists(mapping_file):
        os.remove(mapping_file)

def test_setup_logging_with_file(tmp_path):
    log_file = tmp_path / "test_log.log"
    setup_logging(True, str(log_file))
    assert os.path.exists(log_file)


def test_save_audio_chunk_success(audio_data):
    filename = "test_audio_chunk.wav"
    save_audio_chunk(filename, audio_data, 16000)
    assert os.path.exists(filename)
    os.remove(filename)

def test_save_audio_chunk_invalid_path():
    # Attempt to save to an invalid directory path
    invalid_path = "/invalid_path/test_audio_chunk.wav"
    audio_data = np.random.rand(16000, 2) * 2 - 1  # Sample stereo audio chunk

    with patch("live_transcription.logging.error") as mock_logging_error:
        save_audio_chunk(invalid_path, audio_data, 16000)
        # Assert that logging.error was called
        mock_logging_error.assert_called_once()

@pytest.mark.asyncio
@patch("live_transcription.Pipeline.from_pretrained")
@patch("live_transcription.pipeline")
async def test_process_live_chunk_empty_speaker_map(mock_pipeline, mock_diarization_pipeline, audio_data, transcription_output, mapping_file):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": "Test transcription"}
    mock_pipeline.return_value = mock_pipe

    mock_diarization = MagicMock()
    mock_diarization_pipeline.return_value = mock_diarization

    transcription = []
    speaker_map = {}
    speaker_counter = [0]

    await process_live_chunk(
        mock_pipe,
        mock_diarization_pipeline,
        audio_data,
        transcription,
        transcription_output,
        "training_data",
        mapping_file,
        16000,
        16000,
        speaker_map,
        speaker_counter
    )

    assert len(transcription) > 0
    assert "Test transcription" in transcription[0]

@pytest.mark.asyncio
@patch("live_transcription.Pipeline.from_pretrained")
@patch("live_transcription.pipeline")
async def test_process_live_chunk_handles_diarization_error(mock_pipeline, mock_diarization_pipeline, audio_data):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": "Hello world"}
    mock_pipeline.return_value = mock_pipe

    # Simulate an error during diarization
    mock_diarization_pipeline.side_effect = Exception("Diarization Error")

    transcription = []
    speaker_map = {}
    speaker_counter = [1]

    with pytest.raises(Exception):
        await process_live_chunk(
            mock_pipe,
            mock_diarization_pipeline,
            audio_data,
            transcription,
            None,
            None,
            None,
            16000,
            16000,
            speaker_map,
            speaker_counter
        )

@patch("live_transcription.sd.query_devices")
def test_list_audio_devices_handles_no_devices(mock_query_devices):
    mock_query_devices.return_value = []
    list_audio_devices()

def test_select_audio_device_invalid_name():
    device = select_audio_device("Invalid Device Name")
    assert device is None

@pytest.mark.parametrize("device_id, expected_name", [
    (0, "Device 1"),
    (1, "Device 2"),
    (2, None)
])
@patch("live_transcription.sd.query_devices")
def test_select_audio_device_by_id(mock_query_devices, device_id, expected_name):
    mock_query_devices.return_value = [
        {"name": "Device 1", "max_input_channels": 1, "hostapi": 0},
        {"name": "Device 2", "max_input_channels": 2, "hostapi": 1},
    ]
    device = select_audio_device(str(device_id))
    assert device == (device_id if device_id < len(mock_query_devices.return_value) else None)

@pytest.mark.asyncio
@patch("live_transcription.Pipeline.from_pretrained")
@patch("live_transcription.pipeline")
async def test_process_live_chunk_empty_audio(mock_pipeline, mock_diarization_pipeline):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": ""}
    mock_pipeline.return_value = mock_pipe
    mock_diarization = MagicMock()
    mock_diarization_pipeline.return_value = mock_diarization

    transcription = []
    await process_live_chunk(
        mock_pipe,
        mock_diarization_pipeline,
        np.array([]),  # Empty audio chunk
        transcription,
        None,
        None,
        None,
        16000,
        16000,
        {},
        [1]
    )

    assert len(transcription) == 0

@pytest.mark.asyncio
@patch("live_transcription.Pipeline.from_pretrained")
@patch("live_transcription.pipeline")
async def test_transcribe_live_handles_keyboard_interrupt(mock_pipeline, mock_diarization_pipeline, setup_environment):
    with patch('builtins.input', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            await transcribe_live(
                model_id="openai/whisper-large-v4",
                verbose=True,
                log_file=None,
                output_file=None,
                debug_audio_dir=None,
                training_data_dir=None,
                mapping_file=None,
                samplerate=16000,
                block_duration=0.5,
                threshold=0.01,
                max_silence_duration=2.0,
                device=None,
                target_samplerate=16000,
                enable_diarization=True
            )


@pytest.mark.asyncio
@patch("live_transcription.Pipeline.from_pretrained")
@patch("live_transcription.pipeline")
async def test_transcribe_live_without_diarization(mock_pipeline, mock_diarization_pipeline, setup_environment):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": "Test without diarization"}
    mock_pipeline.return_value = mock_pipe

    await transcribe_live(
        model_id="openai/whisper-tiny",
        verbose=True,
        log_file=None,
        output_file="test_output_no_diarization.txt",
        debug_audio_dir=None,
        training_data_dir=None,
        mapping_file=None,
        samplerate=16000,
        block_duration=0.5,
        threshold=0.01,
        max_silence_duration=2.0,
        device=None,
        target_samplerate=16000,
        enable_diarization=False
    )

    assert os.path.exists("test_output_no_diarization.txt")
    with open("test_output_no_diarization.txt", "r") as f:
        content = f.read()
    assert "Test without diarization" in content
    os.remove("test_output_no_diarization.txt")

def test_save_audio_chunk_mono_audio():
    filename = "test_audio_chunk_mono.wav"
    audio_data = np.random.rand(16000) * 2 - 1  # Mono audio chunk
    save_audio_chunk(filename, audio_data, 16000)
    assert os.path.exists(filename)
    os.remove(filename)

def test_save_audio_chunk_different_samplerate():
    filename = "test_audio_chunk_different_rate.wav"
    audio_data = np.random.rand(16000, 2) * 2 - 1
    save_audio_chunk(filename, audio_data, 44100)  # Non-standard samplerate
    assert os.path.exists(filename)
    os.remove(filename)

@patch("live_transcription.sd.query_devices")
def test_list_audio_devices_all_zero_input_channels(mock_query_devices):
    mock_query_devices.return_value = [
        {"name": "Device 1", "max_input_channels": 0, "hostapi": 0},
        {"name": "Device 2", "max_input_channels": 0, "hostapi": 1},
    ]
    list_audio_devices()
    # This test doesn't assert since we're mainly ensuring it runs without crashing

@pytest.mark.asyncio
@patch("live_transcription.pipeline")
async def test_process_live_chunk_without_diarization_pipeline(mock_pipeline, audio_data):
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"text": "Transcription without diarization"}
    mock_pipeline.return_value = mock_pipe

    transcription = []
    await process_live_chunk(
        mock_pipe,
        None,  # No diarization pipeline
        audio_data,
        transcription,
        None,
        None,
        None,
        16000,
        16000,
        {},
        [1]
    )

    assert len(transcription) > 0
    assert "Transcription without diarization" in transcription[0]

@patch("live_transcription.asyncio.run_coroutine_threadsafe")
def test_audio_callback_with_status(mock_run_coroutine_threadsafe, audio_data):
    # Mock parameters
    status = MagicMock()
    queue = MagicMock()
    loop = MagicMock()

    callback(audio_data, 16000, None, status, queue, loop, 16000, 16000)
    mock_run_coroutine_threadsafe.assert_called_once()
