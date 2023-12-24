<!-- # Emotion Disentangling Module - PyTorch Implementation -->

This is an unofficial PyTorch implementation of [**Cross-Speaker Emotion Disentangling and Transfer for End-to-End Speech Synthesis**](https://arxiv.org/abs/2109.06733). Feel free to use/modify the code.

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

# Implementation Issues

- As outlined in the paper, I replaced the location-sensitive attention mechanism in Tacotron 2 with the attention mechanism from Tacotron.
- I've got a few questions about integrating speaker embeddings. Here I have followed the approach in [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).
- The parameter settings are consistent with those in the original paper.

Please inform me if you find any mistakes in this repo, or any useful tips to train a more emotional TTS model. Additionally, I would greatly appreciate it if the paper's author could share the original code.

# References
- [Cross-Speaker Emotion Disentangling and Transfer for End-to-End Speech Synthesis](https://arxiv.org/abs/2109.06733), Tao Li, *et al*.
- [Tacotron2 implementation](https://github.com/NVIDIA/tacotron2)
- [GST implementation](https://github.com/KinglittleQ/GST-Tacotron)
- [Mellotron](https://github.com/NVIDIA/mellotron)

# Citation
```
@ARTICLE{9747987,
  author={Li, Tao and Wang, Xinsheng and Xie, Qicong and Wang, Zhichao and Xie, Lei},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Cross-Speaker Emotion Disentangling and Transfer for End-to-End Speech Synthesis}, 
  year={2022},
  volume={30},
  number={},
  pages={1448-1460},
  doi={10.1109/TASLP.2022.3164181}
}
```