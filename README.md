# Audio AI: Deep Neural Network Architectures for Audio Processing Tasks
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A repository hosting deep neural network architectures trained for audio based tasks. These include TTS, ASR, etc. The main purpose of this repository is to serve as a practitioner's guide to building an end-to-end industry grade deep learning model training pipeline.

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Audio](#understanding-audio)
    - [Sampling Rate](#sampling-rate)
    - [Time & Frequency Domain](#time-&-frequency-domain)
    - [Transformations](#transformations)
3. [Contributing](#contributing)
4. [License](#license)

## Introduction
The increasing research in the field of deep learning has led to the development of many state-of-the-art deep neural network architectures for audio processing tasks. However, the implementation of these architectures is often not straightforward and requires a lot of effort. There has not been many attempts to provide a comprehensive guide to building an end-to-end industry grade deep learning model training pipeline for audio processing tasks. This repository aims to fill this gap by providing a practitioner's guide to building an end-to-end industry grade deep learning model training pipeline for audio processing tasks. The repository also hosts deep neural network architectures trained for audio processing tasks. These include wav2vec, ASR, etc. 

## Understanding Audio
Working with audio data for deep learning is quite complex when compared to textual and visual data. Compared to various data formats like images, videos and text, audio takes many different forms of representations. Audio is just *vibration*. Vibrating materials cause particles to oscillate which in turn produces audio. It travels through a medium and is usually air but it can also be any object or material. They can be represented by waves of varying pressure levels. 

Audio wave is composed of different features such as [amplitude](https://en.wikipedia.org/wiki/Amplitude), [frequency](https://en.wikipedia.org/wiki/Frequency), etc. Audio can be represented in time and frequency domains. They both use different features to represent audio. The frequency domain representation is a popular representation adopted in many audio processing pipelines to build deep learning models. But before we dive into the preprocessing audio, we first need to understand the characteristics of audio. 
```math
\Large{y(t)\ =\ Asin(2{\pi}ft\ +\ {\varphi})}
```
The above equation represents a sinusoidal wave representation of audio where,
- 'A' is the amplitude (i.e. height of the wave)
- 'f' is the frequency (i.e. no. of cycles completed per second)
- 't' is the time
- '$`\phi`$' is the phase (i.e. amount to shift right or left)

### Sampling Rate
Audio is a continuous wave form that can be represented by any real number. But, when storing audio digitally, we often have to [quantize](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) (limit the range of numbers we use to represent it) it to some range say 16-bit, 24-bit etc. To achieve this we sample and quantize audio waves while recording them.

Sampling is the process of recording the amplitude of the sound wave at a given time for a number of time step. It is represented with the unit 'Hz' (Hertz). The sampling rate in which the CDs store audio is 44100 Hz. Higher sampling rate means lesser distortions in audio. The recommended sampling rate for an audio wave is determined by the [Nyquist Rate](https://en.wikipedia.org/wiki/Nyquist_rate).

[Bit-depth](https://en.wikipedia.org/wiki/Audio_bit_depth) is the number if bits of information present in each sample. It signifies how representative it can be of the *variations in the amplitude* of the audio wave. For example, the amplitudes values of a 16-bit recorded audio wave will range from *-32,768 to 32, 767* whereas the same audio wave recorded with 24-bit will have it's amplitude values range between *−8,388,608 to 8,388,607*
### Time & Frequency Domain
Audio wave is represented usually along the axes of time domain where the x-axis denotes the time and y-axis denotes the amplitude of the wave and that instant. Audio wave can also be represented in frequency domain where the x-axis denotes frequency and y-axis denotes the amplitude. This conversion from time domain representation to frequency domain can be done using [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform). 

Fourier Transform tries to find *sinusoidal waves* of different frequencies that highly *correlate* with the audio wave. In essence, the summation of these sinusoidal waves reproduce the original audio wave. However, in practice we use Discrete Fourier Transform in digital applications.

Since, frequency is a important characteristic of the an audio wave we often use this representation to work with audio data. 

### Transformations
There are certain transformation that can be applied to the audio wave other than the Fourier transform which are useful to prepare the audio data to train deep learning models.

#### Short Time Fourier Transform
[Short Time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) divides the audio-wave into multiple overlapping frames and applies Fourier transform to each frame individually. 

#### Mel Spectrograms
Since we perceive sound in log scale, we try to express frequency and amplitude in [Mel scale](https://en.wikipedia.org/wiki/Mel_scale). The transformed signal (wave) is then represented through a [spectrogram](https://en.wikipedia.org/wiki/Spectrogram).

## Contributing
Contributions to this repository are very much welcomed! Since I'm a beginner myself, I'm sure there are many things that can be improved. Please refer to the [CONTRIBUTING](CONTRIBUTING.md) file for more details.

## License
This repository is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for more details.