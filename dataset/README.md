# dataset

**`dataset_toiletflush_16k.zip`** (~38 MB) — 16 kHz mono ~8 s WAVs (`positives/` flush, `negatives/` not flush). Upload to Colab with `colab/colab_train_flush.ipynb`.

GitHub’s **website** “Upload files” rejects anything over **25 MB**. This zip is ~38 MB, so do **not** drag it into the browser. Use one of:

- **[GitHub Desktop](https://desktop.github.com/)** (or `git push`): git allows files up to 100 MB. Commit the whole `ToiletFlush` folder including the zip.
- **Release:** create a Release on the repo and attach `dataset_toiletflush_16k.zip` as a binary. Point Colab at that download if the zip is not in `dataset/`.

The zip root is `dataset_toiletflush_16k/`. Unzip **once**; do not unzip into a folder that is already named `dataset_toiletflush_16k`. To rebuild: zip the `dataset_toiletflush_16k` folder so `positives/` sits directly inside it.

The zip root is `dataset_toiletflush_16k/`. Unzip **once**; do not unzip into a folder that is already named `dataset_toiletflush_16k`. To rebuild: zip the `dataset_toiletflush_16k` folder so `positives/` sits directly inside it.

## License

- **Bathroom recordings** (names like `toiletflush_*`, `fannoise_*`, `shower_*`, `tapwater_*`): recorded in the author’s bathroom with the Arduino Nano 33 BLE Sense microphone, [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
- **Freesound:** filenames starting `Freesound.org__` are from [Freesound](https://freesound.org/), published there as [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Files were clipped and converted to 16 kHz mono WAV. Id and username are in the filename (`https://freesound.org/s/<id>/`).
