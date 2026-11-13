# Failure Log

What broke, why, and how it was fixed. Newest first within each phase.

---

## Phase 1: COLMAP poses (Colab, Nov 2026)

### F2.2: On Kaggle "T4 x2", gsplat silently trained on both GPUs
- **Symptom:** bicycle finished (PSNR 25.63, above the paper), then garden `default --packed` died with `CUDA error: an illegal memory access` from the NCCL process-group watchdog (`ProcessGroupNCCL`, `ProcessExitedException ... SIGABRT`). The log shows two worker process IDs.
- **Cause:** `simple_trainer.py` starts one process per visible GPU. In multi-GPU mode it splits the initial points between GPUs (`points[world_rank::world_size]`), gives each GPU its own strategy (so `cap-max 1M` means up to 2M in total), uses one image per GPU per step (batch 2), and reports `num_GS` and writes the `.ply` per GPU. The bicycle score came from a different, bigger setup than every other row, and the exported model was half of it. The garden crash is in the multi-GPU path, not out of memory, so it says nothing about F2.1.
- **Fix:** the Kaggle notebook sets `CUDA_VISIBLE_DEVICES=0` before anything touches the GPU. Rerun on one T4: bicycle mcmc 1M **25.22 / 0.763 / 0.225** in 33 min (paper 25.25 / 0.771 / 0.205), so the earlier 25.63 came from the bigger 2-GPU setup. Garden default finished too (F2.1).
- **Lesson:** "more GPUs" isn't free speed: check how many processes the trainer starts, and whether the metrics and the exported file cover the whole model. A result that beats the paper on the first try is a reason to check the setup, not to celebrate.

### F2.1: The `default` strategy runs out of GPU memory on a T4 (garden, 1/4 res, 30k steps)
- **Symptom:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.23 GiB` at step ~11,825, inside `DefaultStrategy._grow_gs` → `split`. The T4 has 14.56 GiB; 9.58 GiB was allocated by PyTorch and **4.19 GiB reserved but unused** (fragmentation).
- **Cause:** the default (ADC) strategy keeps splitting Gaussians until step 15k. It had 5.38M at 7k and kept growing, and each split briefly needs a full copy of the optimizer state. It also slowed down: 14.8 → 3.0 steps/s, so the full 30k run would have taken ~2.5 h even with enough memory.
- **Kept:** the 7k scores (PSNR 26.34, SSIM 0.833, LPIPS 0.154 with 5.38M Gaussians). For comparison, MCMC 1M at 7k: PSNR 25.93 with 1M Gaussians.
- **Fix / decision:** on a free T4 we use MCMC with a cap. It reached PSNR 26.99 at 30k in 35 min with 3.4 GB peak. Default 30k needs an L4/A100. The notebook now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (it reduces fragmentation but won't make 5M+ Gaussians fit on a T4), and keeps a failed run's folder instead of deleting it.
- **Resolved (Kaggle, `02c`, one T4):** with `expandable_segments:True` **and `--packed`**, default reached 30k: **7,786,777 Gaussians, PSNR 27.22 / SSIM 0.865 / LPIPS 0.104** (paper 27.41 / 0.868 / 0.103), **11.3 GB peak**, 158 min. So it was never a hard 16 GB limit: the Colab crash came from fragmentation (4.19 GB reserved but unused) and the unpacked mode. MCMC 1M stays our default: 0.23 dB lower, but 1/8 the Gaussians, 1/4.5 the time, and a file small enough for the web.
- **Lesson:** a Gaussian cap isn't just about file size: on a 16 GB GPU it decides whether the run finishes at all. Watch the steps/s line: a steady drop means the Gaussian count is growing fast.

### F1.6: gsplat's built-in fisheye handling doesn't fit Wayve's cameras (caught before running)
- **Where:** `examples/datasets/colmap.py` in gsplat v1.5.3, the `OPENCV_FISHEYE` branch.
- **Problem:** it applies the four fisheye coefficients as plain radial distortion on pinhole coordinates, skipping the `θ = atan(r)` step of the real fisheye model. It also centres the map on `width // 2` instead of the camera's optical centre.
- **Evidence (local test on scene_078):** Wayve's 3D points reproject to their stored pixels with a median error of 0.52-0.75 px using the proper fisheye model, and **83 px** (90th percentile 374 px) with gsplat's version.
- **Fix:** undistort with `cv2.fisheye` (the same model as COLMAP's `OPENCV_FISHEYE`) into pinhole images, and write matching `PINHOLE` cameras. After undistortion the error is 0.28-0.40 px at half resolution.
- **Lesson:** check a library's camera model against the data before training. A reprojection test takes seconds; a misaligned training run wastes a GPU session.

### F1.5: About 13 minutes of GPU time lost before the first training step
- **Symptom:** in the phone-video run, training reached step 2,543 at 14:37 on the progress bar, but that many steps only take about a minute at the speed it was running. Total run 18.7 min, of which roughly 4-5 min was real training.
- **Likely cause:** gsplat loads its CUDA code at the first render call (`_make_lazy_cuda_func` in `gsplat/cuda/_wrapper.py`) and compiles it there if no compiled package is installed. Caching on Drive with `TORCH_EXTENSIONS_DIR` probably didn't help: that build uses file timestamps, and a fresh Colab machine reinstalls the sources with new ones. The garden notebook's first run also took about 8 min longer than its second. Not measured directly.
- **Fix attempt:** build a wheel once with `pip wheel --no-build-isolation`, keep it on Drive for each GPU and torch version, and install it in later sessions.
- **Status: not fixed.** On the Wayve run the wheel build failed (`Building wheel for gsplat ... exit code: 1`, no output because of `-q`), so it fell back to compiling at training start: `CUDA extension has been set up successfully in 885.08 seconds` (14.8 min). The cause is still unknown. One possibility: gsplat sets `MAX_JOBS=10` by default, and Colab has only 2 CPUs and ~12 GB RAM, so parallel `nvcc` jobs may run out of memory. Not confirmed.
- **Second fix (Phase 2 notebook onward):** drop the wheel. Instead, keep the library the JIT build produces (`gsplat_cuda.so`) on Drive under `cache/gsplat_jit/<gsplat, torch, Python, GPU arch>/`. A small patch to `gsplat/cuda/_backend.py` loads it with torch's own `_import_module_from_library` before trying anything else, and falls back to the normal JIT compile if the file doesn't load. Tested locally: the patch applies to the real v1.5.3 file, only once, and the result is valid Python. To be confirmed on Colab: the second session should print `✅ gsplat CUDA loaded from the Drive cache`.
- **First Colab result (Phase 2, first session):** compiled once in 13.9 min and saved to Drive. Reloading from the cached file took **2 s**, and the training run used it: the first progress line came at 2 min with no compile pause. 
- **✅ Confirmed fixed (next session):** `✅ gsplat CUDA loaded from the Drive cache in 3s (gsplat1.5.3_torch2.14.1+cu130_py313_sm75)`. The ~14 min compile now happens once per torch version, not once per session. Colab has meanwhile updated torch from 2.11.0 (Phase 0) to 2.14.1; the cache key includes the torch version, so a future update just triggers one more compile.
- **Lesson:** read the time per step, not just the total. Limited GPU quota makes this worth checking.

### F1.4: `colmap image_undistorter failed (exit -6)` on the phone video
- **Cause:** most likely the output path's parent folder (`/content/work/data`) didn't exist. The garden notebook had created it when downloading the dataset, but the phone-video notebook hadn't. Inferred from the fix working; COLMAP's own error message wasn't captured.
- **Fix:** `os.makedirs(SCENE_DIR, exist_ok=True)` before running the undistorter. Training and export then completed.
- **Lesson:** a negative exit code means the process was killed or aborted (-6 is SIGABRT), so read the tool's own log. Don't rely on side effects from an earlier notebook.

### F1.3: Our poses seemed to beat the official poses by 1.77 dB (invalid comparison)
- **Symptom:** after fixing F1.2, official poses scored PSNR 25.98 and ours 27.75, even though the two sets of poses differed by only 0.012° (median rotation).
- **Cause:** the runs were scored against different images. The official run used the original `images_4` JPGs. Our run used images resampled and cropped by `colmap image_undistorter`, with our own calibrated camera.
- **Test:** train official poses on our undistorted images with our camera, so the poses are the only difference. Result: official 27.73, ours 27.75 (+0.03 dB before rounding).
- **Conclusion:** poses are equivalent. The 1.77 dB came from the images and camera model together. This test doesn't separate those two effects.
- **Lesson:** change one variable at a time, and only compare PSNR across runs scored against the same test images.

### F1.2: Runs silently trained at a much lower resolution (notebook bug)
- **Symptom:** official poses scored PSNR 30.02 at "1/4 res, 7k steps", well above the ~27.4 published for 30k steps. Our poses scored 27.75, a gap of -2.27 dB that looked like bad poses.
- **Cause:** to avoid downloading full-res images, the notebooks symlinked `images → images_4` (or `images_8`) and passed `--data_factor 4` (or `8`). In gsplat v1.5.3 (`examples/datasets/colmap.py`, lines 193-196), `factor > 1` with JPGs makes the loader **ignore `images_N/` and downscale `images/` by N** into `images_N_png/`. The images were shrunk twice:
  - Phase 0: 1/8 × 1/8 ≈ 1/64 (about 81×52 px), reported as "1/8".
  - Phase 1 official: 1/4 × 1/4 = 1/16 (about 324×210 px), reported as "1/4".
  - Phase 1 ours: `--data_factor 1`, so it was correct all along (about 1297×840 px).
- **Why nothing crashed:** the loader rescales the intrinsics to match the real image size, so geometry stayed correct and training looked normal. Lower resolution is easier, so the scores looked *better*.
- **Fix:** train already-small `images/` folders with `--data_factor 1`. `train()` now compares the render size with the input size and stops on a mismatch.
- **Lesson:** a result that beats published numbers needs investigating, not celebrating. Verify what the code actually ran on. The -2.27 dB gap said nothing about pose quality.

### F1.1: `colmap: error while loading shared libraries: libfaiss.so`
- **Where:** first COLMAP install via micromamba (`colmap=*=*cuda* glomap`)
- **Symptoms, in order:**
  1. The environment installed, but `libfaiss.so` was missing, so COLMAP couldn't start. The notebook misreported this as "CPU-only", because its check only searched the banner for "with CUDA".
  2. Installing plain `libfaiss` got COLMAP running, but the environment ended up on **COLMAP 3.8 without CUDA**. Most likely the solver swapped the GPU build for an old CPU one to satisfy the request (inferred, not confirmed).
  3. `colmap=*=*cuda* libfaiss=*=*cuda* glomap` installed, but failed at startup with `undefined symbol`.
- **Fix:** `colmap=*=*cuda* libfaiss=*=*cuda*` without the `glomap` package → **COLMAP 4.2.1 with CUDA**. The notebook now pins `colmap=4.2.1=*cuda*`.
- **Lesson:** a successful install doesn't prove the tool works. Run the binary and check for the capability you need, and treat a failed start as an error, not a fallback.

---

## Phase 0: Setup & Hello Splat (Colab, Nov 2026)

Environment: Google Colab, T4 GPU, **Python 3.13**, pip 26.2.

### F0.5: `OverflowError` when importing `pycolmap` under numpy 2
- **Where:** `import pycolmap` (rmbrualla fork, used by gsplat's COLMAP loader)
- **Cause:** `pycolmap/scene_manager.py` calls `np.uint64(-1)` as an "invalid ID" sentinel. numpy 2 no longer wraps negative values into unsigned types, so it raises `OverflowError`.
- **Fix:** patch the installed file: `np.uint64(-1)` → `np.uint64(18446744073709551615)` (= 2⁶⁴−1, the same value numpy 1 produced).
- **Lesson:** removing the `numpy<2` pin (F0.3) is what exposed this. Unpinning one dependency can break another package that relied on the old behaviour.

### F0.4: `ModuleNotFoundError: No module named 'datasets.colmap'`
- **Where:** starting `gsplat/examples/simple_trainer.py`
- **Cause:** Colab ships Hugging Face's `datasets` package. gsplat's `examples/datasets/` folder has no `__init__.py`, so Python treats it as a namespace package, and a regular installed package (HF `datasets`) takes priority over a namespace package.
- **Fix:** `touch gsplat/examples/datasets/__init__.py` (fallback: `pip uninstall -y datasets`).
- **Lesson:** watch for local folder names that shadow, or get shadowed by, preinstalled packages.

### F0.3: numpy version conflicts on Python 3.13
- **Symptom:** pip resolver errors: `jaxlib/ml-dtypes require numpy>=2.0, but you have numpy 1.26.4` and `numba requires numpy<2.5, but you have 2.5.4`.
- **Cause:** gsplat's example requirements pin `numpy<2`, which conflicts with Colab's Python 3.13 package set.
- **Fix:** drop the numpy pin from requirements and install `numpy>=2.1,<2.5`, then **restart the runtime**.

### F0.2: pip tries to clone a nonexistent repo (`...fused-ssim@<sha>git+https://.../fused-bilagrid`)
- **Cause:** bug in gsplat's `examples/requirements.txt` for this version: the `fused-ssim` and `fused-bilagrid` URLs are on **one line** with no newline between them.
- **Fix:** split them with a regex and install each one separately:
  `grep -oE 'git\+https://github\.com/[^@ ]+@[0-9a-f]{40}' requirements.txt`

### F0.1: `Failed to build 'git+https://github.com/rahul-goel/fused-ssim@...' when getting requirements to build wheel`
- **Cause:** `fused-ssim` (and `fused-bilagrid`) import `torch` in `setup.py`. pip's default build isolation creates a clean environment without torch.
- **Fix:** `pip install --no-build-isolation <url>`, after installing `setuptools wheel ninja`. Setting `TORCH_CUDA_ARCH_LIST=7.5` (T4) speeds up the compile.

---

## Template

### F<phase>.<n>: <short symptom>
- **Where:**
- **Cause:**
- **Fix:**
- **Lesson:**
