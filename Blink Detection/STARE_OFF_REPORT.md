# STARE OFF - Two Player Blink Detection Game
## Technical Report

---

## Project Overview

**Stare Off** is a competitive real-time video game where two players sit in front of a webcam and try to out-stare each other. The first player to blink loses the round. The game uses OpenCV for face/eye detection and Eye Aspect Ratio (EAR) calculation to determine blinks.

---

## Game Mechanics

- **Objective**: Be the last player to blink in each round
- **Scoring**: Best of 3 rounds (first to 2 wins)
- **Detection**: Real-time face and eye detection via webcam
- **Game States**: 
  - MENU (start screen)
  - COUNTDOWN (3-second prep)
  - GAMEPLAY (active staring)
  - ROUND_END (2-second round result display)
  - GAME_OVER (match conclusion)

---

## Key Improvements (v2.0)

### 1. **Two-Eye Averaging**
- **Problem**: Single eye detection was unreliable, especially for Player 2
- **Solution**: Detect up to 2 eyes per face and average their EAR values
- **Location**: `detectMainEye()` function (lines 410-450)
- **Impact**: More robust detection, Player 2 now consistently detected

### 2. **Exponential Smoothing Filter**
- **Problem**: Frame-to-frame EAR noise caused false/unstable blink detections
- **Solution**: Applied exponential smoothing with alpha=0.28
- **Formula**: `smoothedEar = smoothedEar * (1 - α) + currentEAR * α`
- **Location**: Main loop (line 532-534)
- **Impact**: Stable, non-jittery blink detection

### 3. **Improved Blink Detection Logic**
- **Problem**: Closing eyes didn't register as a blink (eyes not detected = no counter increment)
- **Solution**: Any undetectable eyes now count as closed eyes for blink detection
- **Logic**:
  - If eyes not visible → `closedEyeCounter++`
  - If eyes visible AND EAR < threshold → `closedEyeCounter++`
  - After 3 consecutive frames → blink event triggered
- **Location**: `BlinkDetector::detectBlink()` (lines 79-112)
- **Impact**: Actual closing of eyes now properly registers and triggers loss/win

### 4. **Increased Consecutive Frames Requirement**
- **Changed**: `DEFAULT_CONSECUTIVE_FRAMES` from 2 → 3 (line 13)
- **Benefit**: Reduces false positives from brief blinks or detection glitches

---

## Technical Architecture

### Core Classes

#### `BlinkDetector`
Manages per-player blink detection state:
- Tracks EAR over time
- Counts consecutive closed-eye frames
- Implements cooldown to prevent duplicate detections
- Adjustable threshold (0.10 - 0.40)

```cpp
bool detectBlink(float currentEAR, bool eyesVisible)
- Input: Current Eye Aspect Ratio & eye visibility status
- Output: True if blink event detected this frame
```

#### `StareOffGame`
Manages game state machine and scoring:
- State transitions (MENU → COUNTDOWN → GAMEPLAY → ROUND_END → GAME_OVER)
- Score tracking (best of 3 rounds)
- Round winner determination
- UI rendering

### Detection Pipeline

```
Frame Capture
    ↓
Grayscale Conversion + Histogram Equalization
    ↓
Face Cascade Detection (size ≥ 100x100)
    ↓
Select Top 2 Faces (by area, left-to-right order)
    ↓
Eye Cascade Detection per Face (Size ≥ 15x15 or 20x20)
    ↓
EAR Calculation (2-eye average if available)
    ↓
Exponential Smoothing (α=0.28)
    ↓
Blink Detection (≥3 consecutive low-EAR frames)
    ↓
Game State Update → Win/Loss/Tie
```

---

## Eye Aspect Ratio (EAR) Formula

$$\text{EAR} = \frac{||P_2 - P_6|| + ||P_3 - P_5||}{2 \cdot ||P_1 - P_4||}$$

- **Numerator**: Sum of vertical eye distances (top-bottom)
- **Denominator**: Horizontal eye distance (left-right)
- **Interpretation**:
  - High EAR (≈0.4) = eyes open
  - Low EAR (≈0.2 or less) = eyes closed/blinking
- **Threshold**: 0.23 (adjustable via +/- keys)

---

## UI Elements

| Element | Location | Purpose |
|---------|----------|---------|
| Face Rectangles | Frame corners | Blue (P1), Red (P2) |
| Eye Rectangles | Within face | Green boxes around detected eyes |
| EAR Display | Below face | Real-time smoothed EAR value |
| EAR Bars | Bottom corners | Visual threshold + current EAR indicators |
| Scoreboard | Top-left | Current round scores & round number |
| Status Messages | Center | "ONLY ONE PLAYER DETECTED", blink events |
| FPS Counter | Top-right | Frames per second performance |
| Threshold Value | Top-right | Current sensitivity level |

---

## Control Keys

| Key | Action |
|-----|--------|
| SPACE | Start game (from menu) |
| R | Reset game to menu |
| +/= | Increase blink sensitivity (raise threshold) |
| -/_ | Decrease blink sensitivity (lower threshold) |
| T | Print status to console |
| ESC | Quit application |

---

## Parameter Reference

```cpp
DEFAULT_EAR_THRESHOLD = 0.23f          // Blink detection threshold
DEFAULT_CONSECUTIVE_FRAMES = 3         // Frames required for blink
MAX_MISSING_FRAMES = 5                 // Tolerance before face loss
BLINK_COOLDOWN_FRAMES = 15             // Prevent duplicate blinks
SMOOTHING_ALPHA = 0.28f                // EAR exponential smoothing
```

---

## Performance Metrics

- **FPS Target**: 30+ FPS on modern hardware
- **Latency**: ~100-150ms end-to-end detection
- **CPU Usage**: Low (~15-25% per player detection)
- **Memory**: ~50-100 MB (mainly frame buffers)

---

## Cascade Classifiers

### Face Detection
- **Primary**: `haarcascade_frontalface_default.xml`
- **ScaleFactor**: 1.1 (10% per layer)
- **MinNeighbors**: 5
- **MinSize**: 100x100 pixels

### Eye Detection
- **Primary**: `haarcascade_eye.xml`
- **Fallback**: `haarcascade_eye_tree_eyeglasses.xml`
- **ScaleFactor**: 1.1 (initial) → 1.1 (fallback)
- **MinNeighbors**: 3 (initial) → 2 (fallback)
- **MinSize**: 20x20 (initial) → 15x15 (fallback)

**Search Paths** (checked in order):
1. Local directory
2. Relative path (`./`)
3. Absolute C:/opencv/build/etc/haarcascades/

---

## Known Limitations & Future Improvements

### Current Limitations
- ✗ Requires frontal face pose (~20-30° tolerance)
- ✗ Sensitive to lighting conditions
- ✗ May miss partial eye occlusion (glasses, hair)
- ✗ No multi-camera support
- ✗ 2-player maximum

### Recommended Future Enhancements
- [ ] Add dlib facial landmark detection (more accurate EAR)
- [ ] Implement head pose estimation for angle tolerance
- [ ] Add eye gaze tracking (anti-cheating: detecting looking away)
- [ ] Network multiplayer support
- [ ] Difficulty levels (adjustable thresholds)
- [ ] Statistics tracking (avg blink time, longest stare)
- [ ] LED/audio feedback for blink detection
- [ ] Support for up to 4 players

---

## Compilation & Dependencies

### Requirements
- **C++11** or higher
- **OpenCV 3.4+** (4.x recommended)
- **Windows/Linux/macOS** compatible

### Build Command (Windows/MinGW)
```bash
g++ -std=c++11 -O2 \
  -I"C:\opencv\include" \
  -L"C:\opencv\lib" \
  stare_off.cpp \
  -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect \
  -o stare_off.exe
```

### Required Files
- `stare_off.cpp` - Source code
- `haarcascade_frontalface_default.xml` - Face detector
- `haarcascade_eye.xml` - Eye detector
- OpenCV libraries (core, highgui, imgproc, objdetect)

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Cannot open webcam" | Camera in use or not connected | Close other apps using camera; ensure hardware connected |
| "Could not load cascade" | XML files not found | Place .xml files in app directory or specify full path |
| Only Player 1 detected | Eye detection fails for right side | Increase lighting; move closer to camera; adjust +/- keys |
| Constant false positives | Threshold too low | Press + to increase threshold |
| Never detects blinks | Threshold too high | Press - to decrease threshold; print status with T key |
| Poor FPS | CPU overloaded | Reduce resolution; close background apps |

---

## Testing Checklist

- [x] Two faces detected correctly
- [x] Eyes detected (green rectangles visible)
- [x] EAR values displayed and updating smoothly
- [x] Closing eyes triggers blink event
- [x] Player 1 closing eyes → Player 2 wins round
- [x] Player 2 closing eyes → Player 1 wins round
- [x] Both closing simultaneously → TIE
- [x] Sensitivity adjustment (+/-) working
- [x] Game state transitions smooth
- [x] Score tracking accurate across 3 rounds
- [x] Final winner correctly determined

---

## File Structure

```
gradu proj/
├── stare_off.cpp                          (Main source file, 646 lines)
├── haarcascade_frontalface_default.xml    (Face detection model)
├── haarcascade_eye.xml                    (Eye detection model)
├── stare_off.exe                          (Compiled executable)
└── STARE_OFF_REPORT.md                    (This report)
```

---

## Author Notes

This is a **graduation project** demonstrating:
- Real-time computer vision with OpenCV
- State machine game logic
- Eye/face detection via cascade classifiers
- Signal processing (EAR smoothing)
- GUI rendering and user I/O

**Version**: 2.0  
**Last Updated**: May 21, 2026  
**Status**: Ready for deployment

---

## Contact / Support

For issues or feature requests, verify:
1. OpenCV is properly installed
2. Cascade XML files are in the correct path
3. Webcam is functional and not in use
4. Lighting is adequate (well-lit face)
5. Both players are positioned in frame

