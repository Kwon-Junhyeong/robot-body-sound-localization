# Robot Body Sound Localization

Finds the azimuth and elevation of a speaker with four microphones on a robot body. The body scatters the sound before it reaches the microphones, so free-field methods and networks trained without that scattering lose accuracy on the robot. Here the microphone responses are computed from the body's shape and matched to the recording, with no training. Research intern work at the KIST AI and Robotics Institute, Humanoid Research Center (2026).

![System](docs/system.png)

## No training: responses computed from the body's shape

`bem/h_table_generator_gpu.py` solves the sound pressure on the body surface with the boundary element method on a GPU, once per body, and stores one response per direction. A new body or microphone layout needs only a new table.

## Matching with a confidence gate

`doa/matcher.py` scores every direction in the table against the phase-only cross-spectra of the four microphones. It answers only when two scores pass, the phase closure of the four microphones and the prominence of the peak, and abstains otherwise.

## Conformal prediction

`doa/conformal.py` turns the score map into a region that holds the true direction with probability 1 - alpha over answers. Each task sets its own risk level alpha. When the noise or the room changes, adaptive conformal prediction updates alpha after each answer from whether the true direction fell outside the region. A narrow region is passed on as an answer and a wide one is withheld, since wrong directions gather in wide regions: withholding the widest 20% cut answers more than 5 degrees off from 12% to 2.6%.

## Results

On 4,000 simulated reverberant recordings of 0.5 s, 87.2% of the directions are within 5 degrees, against 50.6% with a free-field table. With 1.0 s of audio the mean error is 1.23 degrees. On a 3D-printed test body, a source at 90 degrees is found at 89 degrees.

![Accuracy on simulated recordings](docs/accuracy.png)

## Run

```sh
pip install -r requirements.txt
python bem/h_table_generator_gpu.py              # set the body and microphones at the top; writes h_table.npz
python locate.py h_table.npz RECORDING.wav       # 16 kHz, 4 channels
```
