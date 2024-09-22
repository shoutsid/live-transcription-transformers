# Live Transcription with Hugging Face Transformers

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Stars](https://img.shields.io/github/stars/shoutsid/live-transcription-transformers?style=social)

Transform your spoken words into text in real-time with our **Live Transcription Script** powered by Hugging Face's Transformers and the state-of-the-art `openai/whisper-large-v3` model. Whether you're conducting interviews, lectures, or need accessibility features, this Python script provides a seamless and efficient solution for live speech-to-text conversion.

## 🚀 Features

- **High-Accuracy Transcription**: Leverages the `openai/whisper-large-v3` model for precise and reliable speech recognition.
- **Real-Time Processing**: Captures and transcribes audio live, displaying results instantly in the console.
- **Flexible Logging**:
  - **Verbose Mode**: Activate detailed DEBUG logs for in-depth monitoring.
  - **Log File Support**: Save logs to a specified file for later review.
- **Device Management**:
  - **List Audio Devices**: Easily view all available audio input devices.
  - **Select Preferred Device**: Choose your desired microphone or audio input source.
- **Advanced Audio Handling**:
  - **Stereo to Mono Conversion**: Automatically converts multi-channel audio to mono for consistency.
  - **Resampling**: Adjusts audio sampling rates to match model requirements.
  - **Audio Buffering**: Efficiently captures complete speech segments by buffering audio chunks.
  - **Debug Audio Saving**: Optionally save audio chunks for troubleshooting and analysis.
- **Transcription Output**:
  - **Console Display**: View transcriptions in real-time directly in your terminal.
  - **File Export**: Automatically save transcriptions to a designated text file.
- **Training Data Collection** *(Optional)*:
  - **Save Audio for Training**: Store audio files alongside their transcriptions to build custom datasets.
  - **JSONL Mapping**: Maintain a structured mapping of audio files to their corresponding transcriptions.

## 🌟 Why Choose This Script?

Whether you're building applications that require live speech recognition, enhancing accessibility for users, or collecting training data for custom models, this script offers a robust and extensible foundation. Its integration with Hugging Face Transformers ensures you're utilizing cutting-edge technology that's both scalable and adaptable to various use cases.

## 📦 Installation

### 🛠 Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from the [official website](https://www.python.org/downloads/).

### 📥 Dependencies

Install the necessary Python packages using `pip`:

```bash
pip install transformers sounddevice numpy scipy torch
```

**Note**: The `sounddevice` library requires appropriate backend support for audio capture on your operating system. Ensure your microphone is properly connected and recognized.

## 📝 Usage

Run the script using Python:

```bash
python live_transcription.py [OPTIONS]
```

### 🎛 Command-Line Arguments

| Argument                        | Description                                                                                                                                                           | Default                   |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| `--verbose`                     | Enable verbose (DEBUG) logging to get detailed insights during execution.                                                                                           | `False`                   |
| `--log_file LOG_FILE`           | Specify a file path to save logs for future reference and debugging.                                                                                               | `None`                    |
| `--model_id MODEL_ID`           | Choose a Hugging Face model ID for transcription. Defaults to `openai/whisper-large-v3` for optimal accuracy.                                                       | `openai/whisper-large-v3` |
| `--output_file OUTPUT`          | Define the path to save the transcription text output.                                                                                                              | `live_transcription.txt`  |
| `--debug_audio_dir DIR`         | Set a directory to save audio chunks for debugging purposes. If not provided, audio chunks won't be saved.                                                          | `None`                    |
| `--training_data_dir DIR`       | Designate a directory to save audio files for training custom models.                                                                                               | `None`                    |
| `--mapping_file FILE`           | Provide a path to save audio-transcription mappings in JSON Lines format. Requires `--training_data_dir` to be set.                                                 | `None`                    |
| `--samplerate RATE`             | Set the sampling rate for audio capture. `16000` Hz is recommended for Whisper models.                                                                                | `16000`                   |
| `--block_duration SEC`          | Specify the duration of each audio block in seconds for processing.                                                                                                 | `0.5`                     |
| `--threshold THRESH`            | Define the amplitude threshold to detect speech presence. Adjust based on ambient noise levels.                                                                       | `0.02`                    |
| `--max_silence_duration SEC`    | Set the duration of silence (in seconds) to determine the end of a speech segment.                                                                                  | `1.0`                     |
| `--device DEVICE`               | Select the audio input device by ID or name substring. Use `--list_devices` to view available options.                                                                 | `None` (default device)   |
| `--list_devices`                | Display all available audio input devices and exit.                                                                                                                | `False`                   |
| `--target_samplerate RATE`      | Specify the target sampling rate for transcription. `16000` Hz is recommended for optimal compatibility with Whisper models.                                         | `16000`                   |

### 📚 Examples

1. **Basic Live Transcription**:
   
   Transcribe live audio and save the output to `live_transcription.txt`.
   
   ```bash
   python live_transcription.py
   ```

2. **Enable Detailed Logging and Save Logs**:
   
   Activate verbose logging and save logs to `transcription.log`.
   
   ```bash
   python live_transcription.py --verbose --log_file transcription.log
   ```

3. **List Available Audio Input Devices**:
   
   View all audio input devices connected to your system.
   
   ```bash
   python live_transcription.py --list_devices
   ```

4. **Select a Specific Audio Input Device**:
   
   After identifying your device ID (e.g., `2`), select it for transcription.
   
   ```bash
   python live_transcription.py --device 2
   ```

5. **Save Debug Audio Chunks**:
   
   Store audio chunks in the `./debug_audio` directory for troubleshooting.
   
   ```bash
   python live_transcription.py --debug_audio_dir ./debug_audio
   ```

6. **Collect Training Data with Audio-Transcription Mapping**:
   
   Save audio files for training and maintain a mapping in `mappings.jsonl`.
   
   ```bash
   python live_transcription.py --training_data_dir ./training_audio --mapping_file ./mappings.jsonl
   ```

## 📌 Notes

- **Audio Device Selection**: Use the `--list_devices` option to identify the correct device ID or name substring for your audio input device.
  
- **Sampling Rate**: The default sampling rate is set to `16000` Hz, which is ideal for Whisper models. Adjust only if necessary based on your specific requirements.
  
- **Interrupting the Script**: Press `Ctrl+C` at any time to gracefully stop the live transcription session.

## 🛠 Troubleshooting

- **No Audio Detected**:
  - Ensure the correct audio input device is selected.
  - Verify that your microphone is properly connected and functioning.
  
- **High CPU/GPU Usage**:
  - Transcription models are resource-intensive. Ensure your system meets the necessary hardware requirements.
  - Consider using a GPU for enhanced performance and reduced processing time.
  
- **Dependency Issues**:
  - Confirm all required packages are installed.
  - Use a virtual environment to manage dependencies and avoid conflicts.
  
- **Audio Quality Issues**:
  - Adjust the `--threshold` and `--samplerate` parameters to better suit your environment and hardware setup.

## 📖 Contributing

Contributions are welcome! Whether it's reporting bugs, suggesting features, or improving documentation, your input helps make this project better.

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/YourFeature`
3. **Commit Your Changes**: `git commit -m 'Add some feature'`
4. **Push to the Branch**: `git push origin feature/YourFeature`
5. **Open a Pull Request**

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it as per the license terms.

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for providing powerful NLP tools.
- [OpenAI Whisper](https://github.com/openai/whisper) for advanced speech recognition capabilities.
- [SoundDevice](https://github.com/spatialaudio/python-sounddevice) for seamless audio capture in Python.
- [Scipy](https://www.scipy.org/) and [NumPy](https://numpy.org/) for scientific computing essentials.

## 📫 Contact

For any inquiries, suggestions, or support, feel free to [open an issue](https://github.com/shoutsid/live-transcription-transformers/issues) on GitHub.

---

Embark on a seamless journey from speech to text with our Live Transcription Script, harnessing the latest in machine learning and audio processing technologies!