# BOM Radar — Next Steps

## Warp consistency loss for flow model

The flow model underestimates motion speed (visible as a bright trailing-edge band in the difference images). Fix: add a **warp consistency loss** that directly supervises the estimated flow vectors — warp frame t by the predicted flow and compare with frame t+1, penalising flow fields that don't match actual frame-to-frame displacement. This targets the root cause rather than just penalising the final output.

## GAN discriminator to reduce blurriness

All models produce blurry "averaged" predictions because weighted MSE inherently favours safe, smooth outputs over sharp but possibly misplaced ones. Code for a PatchGAN discriminator is already written in `train_gan.py` but was never run. Try adversarial training on top of the ConvGRU generator — the discriminator should penalise blurry predictions and encourage realistic sharp structure.

## Continuously archive FTP PNGs to grow the training set

The training archive covers Dec 2018 – Feb 2021. Set up a launchd job to fetch new frames from `ftp://ftp.bom.gov.au/anon/gen/radar/` every 5 minutes and save them to `~/projects/BOM/pngs/`. At ~20KB per frame this is ~6MB/day, ~2GB/year. The capture script already exists at `/tmp/bom-png-archive/capture.sh` — move it into the project and create a plist in `~/.config/launchd/`.
