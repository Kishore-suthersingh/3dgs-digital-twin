# Filming a Scene for 3D Reconstruction

COLMAP works out where the camera was by tracking the same fixed points across many frames. Everything below helps it do that. The notebook for this is `notebooks/01b_phone_video.ipynb`.

## Pick a scene
**Good first choices:** a statue, bench, parked vehicle, tree, or small courtyard with plenty of surface detail, ideally something you can walk all the way around.

| Works well | Causes trouble |
|---|---|
| Still objects | People, cars, or anything else moving |
| Rough, detailed surfaces (brick, bark, stone) | Plain walls, clear sky |
| Overcast or soft, even light | Harsh shadows, changing sun |
| Matte materials | Glass, mirrors, water, shiny metal, polished floors |
| Solid shapes | Thin, see-through structures (wicker, fences, railings) |
| Asymmetric subjects | Symmetric subjects and repeating patterns, which can look identical from different sides |

**Privacy and permission:** film where you're allowed to, and avoid identifiable people, license plates, and private property. Keep the video private if you aren't sure.

## Camera settings
- **Lock focus and exposure.** On most phones, press and hold on the subject until you see a lock indicator (AE/AF lock). If your phone can't lock them, keep the light as even as you can.
- **Turn HDR video off.** It can make frames look washed out.
- **Use 1080p or 4K at 30 fps,** in the phone's normal main lens.
- **Don't zoom, and don't switch lenses.** The reconstruction assumes one camera with fixed settings.
- **Hold the orientation** you started with. Don't rotate between portrait and landscape.

## How to move
1. **Keep the subject in frame and move slowly.** A slow walk is about right. Frames should be sharp if you pause the video.
2. **Move around the subject, not just the phone.** Turning on the spot gives COLMAP no depth information.
3. **Make two full loops:** one at chest height, and one higher or lower while angled towards the subject.
4. **End where you started.** That overlap lets COLMAP close the loop and correct drift.
5. **Avoid sudden turns.** Each frame should share most of its view with the next one.

**Length:** about 1-3 minutes.

## Before you upload
Watch the video through and check for:
- [ ] Sharp frames with little motion blur
- [ ] Nothing moving through the scene
- [ ] No sudden changes in brightness
- [ ] The whole subject covered from all sides you want to view later
- [ ] It's the **original file from the phone**, not a copy sent through a messaging app, which can shrink and heavily compress it

Then upload it to Google Drive as `MyDrive/3dgs-digital-twin/videos/scene1.mp4`.

## If COLMAP struggles
| What you see | Likely cause | What to try |
|---|---|---|
| Few frames registered | Too much blur, or too little texture | Film again more slowly, or pick a more detailed subject |
| Several separate models | Parts of the path don't overlap | Set `MATCHER = 'exhaustive'`, then film again with slower turns if needed |
| Doubled or curved geometry | Loop didn't close, or repeating patterns confused matching | Set `MATCHER = 'exhaustive'`, end the video where you started |
| Jumps in the camera path | Wrong matches, or sudden movement | Check the frames around the jump |
| Ghosting in the splat | Something moved during filming | Film again without the moving object |
| Blotchy brightness in the splat | Exposure changed while filming | Lock exposure and film again |

After changing `MATCHER`, delete `results/phase1b_<scene>_sfm/sparse_colmap` on Drive so COLMAP runs again.
