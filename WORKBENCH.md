# 4DAnyone Workbench

The browser GUI runs upstream 4DAnyone-Turbo and its dynamic Gaussian trainer.
Each job reconstructs one 121-frame segment starting at the selected time. It
produces a dynamic NPZ, an orbit MP4, and a resumable trainer checkpoint.

## RunPod

1. Select the image `ghcr.io/pareekshit28/4danyone-workbench:latest` and leave
   the container command at its default. Choose a single CUDA GPU. A 48 GB
   RTX 6000 Ada gives headroom over upstream's reported 22 GB inference peak.
   For a private GHCR package, select a RunPod container registry credential
   with your GitHub username and a classic token with `read:packages`.
2. Mount persistent storage at `/workspace`. The image stores models, jobs
   and checkpoints in `/workspace/4danyone-workbench`. The public weights
   alone take about 20 GiB. Leave additional room for each job's generated
   views and extracted training frames.
3. Expose HTTP port `8000` and set the `WORKBENCH_PASSWORD` environment
   variable. Open port 8000 from the pod's Connect menu and sign in as
   `workbench` with that password.
4. Accept the [SMPL-X license](https://smpl-x.is.tue.mpg.de/), download its
   official archive, and install the neutral NPZ in the GUI. If the proxy
   rejects the upload, copy the archive to the pod and run:

   ```bash
   cd /opt/4danyone
   python scripts/download_smplx.py /workspace/models_smplx_v1_1.zip --model_dir /workspace/4danyone-workbench/models
   ```

5. Upload a video with one clearly visible person. Choose a start time that
   leaves at least 121 usable frames. The GUI accepts files up to 90 MiB.
   Set **Views** and **Training steps** independently. The defaults are
   6 views and 10,000 steps. Views must be a multiple of 6. Public model weights download on the
   first job. Follow the job log, play the orbit, and download the NPZ.
   **Download ZIP** bundles the NPZ with the matching segment of the source
   audio as an M4A when the upload has an audio track.

One GPU job runs at a time. If the pod stops, restart it with the same
storage and choose **Resume job**. This reuses completed multiview generation
and the trainer's latest checkpoint when one is available.

## Interactive playback

For free-viewpoint playback, start the upstream viewer in the pod and
forward local port 7861 over SSH. The job ID is in the NPZ download URL.

```bash
cd /opt/4danyone
python reconstruct.py view --model /workspace/4danyone-workbench/jobs/JOB_ID/reconstruction/checkpoints/latest.npz
```

Open `http://127.0.0.1:7861/` through the tunnel and select the scene.
See the [upstream viewer guide](docs/4dgs.md) for more options.

## Existing Linux installations

With the upstream CUDA dependencies and GVHMR submodule installed, run:

```bash
python -m pip install -r workbench/requirements.txt
python -m uvicorn workbench.server:app --host 127.0.0.1 --port 8000
```
