# Dataset Attribution & Provenance

## MIMII Dataset 1.0

This project uses the original **MIMII (Malfunctioning Industrial Machine Investigation and Inspection) Dataset 1.0**, published by Hitachi, Ltd.

- **Title:** MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection
- **Authors:** Harsh Purohit, Ryo Tanabe, Kenji Nishida, Takashi Endo, Yuki Nikaido, Kota Yamamoto, Yohei Kawaguchi
- **Institution:** Hitachi, Ltd.
- **Publication Date:** August 28, 2019
- **DOI / Record:** [10.5281/zenodo.3384388](https://zenodo.org/records/3384388)
- **Paper Citation:** Harsh Purohit et al., *"MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,"* arXiv preprint arXiv:1909.09347, 2019.
- **License:** Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/))

### Machine Scope for Stetho

- **Machine Category:** `fan`
- **SNR Condition:** 0 dB (and -6 dB / 6 dB where specified)
- **Machine IDs:** `id_00`, `id_02`, `id_04`, `id_06`
- **Audio Specifications:**
  - Sample Rate: 16,000 Hz
  - Bit Depth: 16-bit PCM
  - Channels: 8 channels (circular microphone array)
  - Duration per clip: ~10 seconds

### Compliance & Redistribution Notice

In accordance with CC BY-SA 4.0:
1. Raw multi-gigabyte audio dataset files are excluded from Git version control via `.gitignore`.
2. Derived public artifacts and code retain clear dataset attribution and license notifications.
3. User recordings collected for local demo validation are kept strictly separate from MIMII benchmark data.
