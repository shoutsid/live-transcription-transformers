#!/usr/bin/env python3
"""
Live Transcription Script Using Hugging Face Transformers Pipeline

Features:
- Uses 'openai/whisper-large-v3' model.
- Excludes 'language' and 'task' arguments as per user request.
- Includes a '--verbose' command-line option to set logging levels.
- Includes a '--log_file' command-line option to save logs to a file.
- Adds a '--list_devices' command-line option to list available audio input devices.
- Optionally saves audio chunks for debugging purposes.
- Implements audio buffering to capture full-length speech.
- Saves live transcription to a specified text file.
- Allows configuration of buffer settings and debug audio directory.
- Supports selection of audio input device.
- Handles stereo audio inputs by converting them to mono.
- Implements resampling of audio to match target sampling rate.
- Optionally maps each transcription to its corresponding audio file for training data.
"""

import argparse
import asyncio
import logging
import wave
import torch
from transformers import pipeline
import sounddevice as sd
import numpy as np
import os
import sys
from datetime import datetime
from scipy.signal import resample
import json

def setup_logging(verbose: bool, log_file: str = None):
    """
    Sets up logging with console and optional file output.

    Args:
        verbose (bool): If True, set logging level to DEBUG, else INFO.
        log_file (str, optional): Path to a log file. Defaults to None.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )

def save_audio_chunk(filename: str, audio: np.ndarray, samplerate: int):
    """
    Saves an audio chunk as a WAV file for debugging purposes.

    Args:
        filename (str): Name of the WAV file.
        audio (np.ndarray): Audio data.
        samplerate (int): Sampling rate of the audio.
    """
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # Convert float32 to int16 for WAV format
        audio_int16 = np.int16(audio * 32767)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(samplerate)
            wf.writeframes(audio_int16.tobytes())
        logging.debug(f"Saved audio chunk to {filename}")
    except Exception as e:
        logging.error(f"Failed to save audio chunk: {e}")

async def process_live_chunk(pipe, audio_chunk: np.ndarray, transcription: list, output_file: str,
                            training_data_dir: str, mapping_file: str,
                            target_samplerate: int, samplerate: int):
    """
    Processes and transcribes a chunk of audio data.

    Args:
        pipe (pipeline): The Hugging Face transcription pipeline.
        audio_chunk (np.ndarray): The audio data to transcribe.
        transcription (list): The list to append transcription results.
        output_file (str): Path to save the transcription text.
        training_data_dir (str): Directory to save training audio files.
        mapping_file (str): Path to the JSONL mapping file.
        target_samplerate (int): The sampling rate expected by the pipeline.
        samplerate (int): The original sampling rate of the captured audio.
    """
    try:
        logging.debug(f"Original audio_chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")

        # Check if resampling is needed
        if samplerate != target_samplerate:
            num_samples = int(len(audio_chunk) * target_samplerate / samplerate)
            audio_chunk = resample(audio_chunk, num_samples)
            logging.debug(f"Resampled audio_chunk to {target_samplerate} Hz with shape: {audio_chunk.shape}")

        # Ensure audio is a 1D array
        if audio_chunk.ndim != 1:
            logging.error(f"Audio chunk has {audio_chunk.ndim} dimensions, expected 1.")
            return

        # Transcribe the audio chunk
        result = pipe(audio_chunk)
        text = result.get("text", "").strip()
        if text:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            transcription_entry = f"[{timestamp}] {text}\n"
            transcription.append(transcription_entry)
            print(f"\n[{timestamp}] {text}", end=' ', flush=True)
            if output_file:
                with open(output_file, "a") as f:
                    f.write(transcription_entry)

            # **NEW: Save audio for training and map it to the transcription**
            if training_data_dir and mapping_file:
                # Generate a unique filename based on timestamp
                audio_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                audio_filename = os.path.join(training_data_dir, f"audio_{audio_timestamp}.wav")
                save_audio_chunk(audio_filename, audio_chunk, samplerate)

                # Create a mapping entry
                mapping_entry = {
                    "audio_file": audio_filename,
                    "transcription": text,
                    "timestamp": timestamp
                }

                # Append the mapping to the JSONL file
                try:
                    dir_name = os.path.dirname(mapping_file)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    with open(mapping_file, "a") as mf:
                        mf.write(json.dumps(mapping_entry) + "\n")
                    logging.debug(f"Mapped transcription to audio file: {audio_filename}")
                except Exception as e:
                    logging.error(f"Failed to write mapping entry: {e}")

    except Exception as e:
        logging.error(f"Error during transcription: {e}")

async def transcribe_live(model_id: str, verbose: bool, log_file: str, output_file: str,
                         debug_audio_dir: str, training_data_dir: str, mapping_file: str,
                         samplerate: int, block_duration: float, threshold: float,
                         max_silence_duration: float, device: int,
                         target_samplerate: int):
    """
    Performs live transcription using the specified Whisper model.

    Args:
        model_id (str): The Hugging Face model ID to use for transcription.
        verbose (bool): If True, enables DEBUG logging.
        log_file (str): Path to save the log file.
        output_file (str): Path to save the transcription text.
        debug_audio_dir (str): Directory to save debug audio chunks.
        training_data_dir (str): Directory to save training audio files.
        mapping_file (str): Path to the JSONL mapping file.
        samplerate (int): Sampling rate for audio capture.
        block_duration (float): Duration of each audio block in seconds.
        threshold (float): Amplitude threshold for detecting speech.
        max_silence_duration (float): Seconds of silence to consider end of speech.
        device (int): Audio input device ID.
        target_samplerate (int): The sampling rate expected by the pipeline.
    """
    setup_logging(verbose, log_file)
    logging.info("Initializing Whisper model...")

    # Initialize the pipeline without 'language' and 'task' arguments
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        logging.error(f"Failed to initialize pipeline with model '{model_id}': {e}")
        return

    logging.info("Whisper model loaded successfully.")

    transcription = []
    queue = asyncio.Queue()
    buffer = []
    buffer_duration = 0  # Duration in seconds
    consecutive_silence_blocks = 0
    max_consecutive_silence_blocks = int(max_silence_duration / block_duration)

    # Obtain the current event loop
    loop = asyncio.get_running_loop()

    async def consumer():
        nonlocal buffer, buffer_duration, consecutive_silence_blocks
        while True:
            audio_chunk = await queue.get()
            if audio_chunk is None:
                break

            # Compute maximum amplitude for the chunk
            max_amplitude = np.max(np.abs(audio_chunk))
            logging.debug(f"Max amplitude in chunk: {max_amplitude:.4f}")

            if max_amplitude > threshold:
                buffer.append(audio_chunk)
                buffer_duration += block_duration
                consecutive_silence_blocks = 0
                logging.debug(f"Speech detected. Buffer duration: {buffer_duration:.2f}s")
            else:
                consecutive_silence_blocks += 1
                logging.debug(f"Silence detected. Consecutive silence blocks: {consecutive_silence_blocks}")
                if consecutive_silence_blocks >= max_consecutive_silence_blocks and buffer:
                    logging.debug(f"Silence sustained for {max_silence_duration}s. Processing buffer.")
                    # Concatenate all audio chunks in the buffer
                    full_audio = np.concatenate(buffer)
                    # Process the full_audio
                    await process_live_chunk(
                        pipe,
                        full_audio,
                        transcription,
                        output_file,
                        training_data_dir,
                        mapping_file,
                        target_samplerate,
                        samplerate
                    )
                    # Save the full audio for debugging
                    if debug_audio_dir:
                        if len(buffer) <= 5:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(debug_audio_dir, f"full_audio_{timestamp}.wav")
                            save_audio_chunk(filename, full_audio, samplerate)
                    # Reset the buffer
                    buffer = []
                    buffer_duration = 0
                    consecutive_silence_blocks = 0

            queue.task_done()

    def callback(indata, frames, time_info, status, queue, loop, samplerate, target_samplerate):
        if status:
            logging.warning(f"Sounddevice status: {status}")
        try:
            audio = indata.copy()
            # Check if audio has more than one channel
            if audio.ndim > 1 and audio.shape[1] > 1:
                # Convert stereo to mono by averaging channels
                audio = np.mean(audio, axis=1)
                logging.debug("Converted stereo audio to mono.")
            elif audio.ndim > 1 and audio.shape[1] == 1:
                # Flatten to 1D array
                audio = audio.flatten()
                logging.debug("Flattened single-channel audio to 1D array.")
            elif audio.ndim == 1:
                # Already mono
                logging.debug("Audio is already mono.")
            else:
                logging.error(f"Unexpected audio shape: {audio.shape}")
                return
            # Enqueue audio chunk
            asyncio.run_coroutine_threadsafe(queue.put(audio), loop)
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")

    try:
        consumer_task = asyncio.create_task(consumer())
        logging.info("Starting live transcription. Press Ctrl+C to stop.")
        with sd.InputStream(callback=lambda indata, frames, time_info, status: 
                            callback(indata, frames, time_info, status, queue, loop, samplerate, target_samplerate),
                           channels=1, samplerate=samplerate,
                           dtype='float32', blocksize=int(samplerate * block_duration),
                           device=device):
            while True:
                await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Live transcription stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        await queue.put(None)
        await consumer_task
        logging.info("Transcription session ended.")
        if output_file:
            logging.info(f"Transcription saved to {output_file}")
        if mapping_file:
            logging.info(f"Audio-Transcription mappings saved to {mapping_file}")

def list_audio_devices():
    """
    Lists all available audio input devices.
    """
    devices = sd.query_devices()
    print("Available audio input devices:")
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{idx}: {device['name']} - {device['hostapi']}")

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Live Transcription Script Using Hugging Face Transformers Pipeline")
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to save the log file.'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='openai/whisper-large-v3',
        help='Hugging Face model ID to use for transcription.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='live_transcription.txt',
        help='Path to save the transcription text.'
    )
    parser.add_argument(
        '--debug_audio_dir',
        type=str,
        default=None,
        help='Directory to save debug audio chunks. If not provided, debug audio will not be saved.'
    )
    # **NEW: Arguments for training data mapping (made optional)**
    parser.add_argument(
        '--training_data_dir',
        type=str,
        default=None,
        help='Directory to save training audio files. If not provided, training audio will not be saved.'
    )
    parser.add_argument(
        '--mapping_file',
        type=str,
        default=None,
        help='Path to save the audio-transcription mappings in JSON Lines format. Requires --training_data_dir to be set.'
    )
    parser.add_argument(
        '--samplerate',
        type=int,
        default=16000,
        help='Sampling rate for audio capture. Default is 16000 Hz, recommended for Whisper models.'
    )
    parser.add_argument(
        '--block_duration',
        type=float,
        default=0.5,
        help='Duration of each audio block in seconds.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.02,
        help='Amplitude threshold for detecting speech.'
    )
    parser.add_argument(
        '--max_silence_duration',
        type=float,
        default=1.0,
        help='Seconds of silence to consider end of speech.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Audio input device ID or substring.'
    )
    parser.add_argument(
        '--list_devices',
        action='store_true',
        help='List all available audio input devices and exit.'
    )
    parser.add_argument(
        '--target_samplerate',
        type=int,
        default=16000,
        help='Target sampling rate for transcription. Default is 16000 Hz, recommended for Whisper models.'
    )
    return parser.parse_args()

def select_audio_device(device_arg):
    """
    Selects the audio input device based on user input.

    Args:
        device_arg (str): Device ID or substring.

    Returns:
        int or None: Selected device ID or None for default.
    """
    if device_arg is None:
        return None
    try:
        devices = sd.query_devices()
        if device_arg.isdigit():
            device_id = int(device_arg)
            if device_id < len(devices):
                logging.debug(f"Selected audio device ID: {device_id} - {devices[device_id]['name']}")
                return device_id
            else:
                logging.warning(f"Audio device ID '{device_id}' is out of range. Using default device.")
                return None
        else:
            for idx, device in enumerate(devices):
                if device_arg.lower() in device['name'].lower():
                    logging.debug(f"Selected audio device '{device_arg}' with ID: {idx}")
                    return idx
            logging.warning(f"Audio device '{device_arg}' not found. Using default device.")
            return None
    except Exception as e:
        logging.error(f"Error selecting audio device: {e}")
        return None

if __name__ == "__main__":
    args = parse_arguments()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # Setup a basic logger before selecting the device to capture any early warnings
    setup_logging(args.verbose, args.log_file)

    device = select_audio_device(args.device)
    try:
        asyncio.run(transcribe_live(
            model_id=args.model_id,
            verbose=args.verbose,
            log_file=args.log_file,
            output_file=args.output_file,
            debug_audio_dir=args.debug_audio_dir,
            training_data_dir=args.training_data_dir,  # Now optional
            mapping_file=args.mapping_file,            # Now optional
            samplerate=args.samplerate,
            block_duration=args.block_duration,
            threshold=args.threshold,
            max_silence_duration=args.max_silence_duration,
            device=device,
            target_samplerate=args.target_samplerate  # Passing the target_samplerate
        ))
    except Exception as e:
        logging.error(f"Failed to start live transcription: {e}")
