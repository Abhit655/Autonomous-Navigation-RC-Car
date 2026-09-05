# Superseded scripts

These are earlier inference experiments, retained for reference only. They are **not**
part of the deployed system and are not maintained.

| Script | What it was |
|---|---|
| `gst_trt_live_inference.py` | First working GStreamer + TensorRT pipeline with an on-screen overlay window. Requires a display, so it cannot run over SSH on a headless vehicle. |
| `jetson_csi_live_inference.py` | Earlier CSI capture variant. |
| `realtime_csi.py` | Earlier real-time capture experiment. |
| `camera_inference.py` | Single-camera inference test used during bring-up. |

The deployed system is `jetson_deploy/autonomous_drive.py`, which runs headless, streams to
a browser instead of opening a window, and closes the control loop through to the vehicle.
