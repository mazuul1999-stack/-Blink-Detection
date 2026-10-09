#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace std::chrono;

static const float DEFAULT_EAR_THRESHOLD = 0.23f;
static const int DEFAULT_CONSECUTIVE_FRAMES = 3;
static const int MAX_MISSING_FRAMES = 5;
static const int BLINK_COOLDOWN_FRAMES = 15;

float calculateEAR(const vector<Point2f>& eyePoints) {
    if (eyePoints.size() != 6)
        return 0.4f;

    double vertical1 = norm(eyePoints[1] - eyePoints[5]);
    double vertical2 = norm(eyePoints[2] - eyePoints[4]);
    double horizontal = norm(eyePoints[0] - eyePoints[3]);

    if (horizontal < 1e-5)
        return 0.4f;

    return static_cast<float>((vertical1 + vertical2) / (2.0 * horizontal));
}

vector<Point2f> getEyePoints(const Rect& face, const Rect& eyeRect) {
    vector<Point2f> eyePoints;

    float x = static_cast<float>(face.x + eyeRect.x);
    float y = static_cast<float>(face.y + eyeRect.y);
    float w = static_cast<float>(eyeRect.width);
    float h = static_cast<float>(eyeRect.height);

    float topOffset = h * 0.25f;
    float bottomOffset = h * 0.75f;
    float cornerOffset = w * 0.1f;

    Point2f leftCorner(x + cornerOffset, y + h / 2.0f);
    Point2f rightCorner(x + w - cornerOffset, y + h / 2.0f);

    Point2f topLeft(x + w * 0.2f, y + topOffset);
    Point2f topRight(x + w * 0.8f, y + topOffset);
    Point2f bottomRight(x + w * 0.8f, y + bottomOffset);
    Point2f bottomLeft(x + w * 0.2f, y + bottomOffset);

    eyePoints.push_back(leftCorner);
    eyePoints.push_back(topLeft);
    eyePoints.push_back(topRight);
    eyePoints.push_back(rightCorner);
    eyePoints.push_back(bottomRight);
    eyePoints.push_back(bottomLeft);

    return eyePoints;
}

void fillRoundedRect(Mat& frame, const Rect& rect, const Scalar& color, int radius, int alpha) {
    Mat overlay;
    frame.copyTo(overlay);
    Scalar fillColor = color;

    rectangle(overlay, Rect(rect.x + radius, rect.y, rect.width - 2 * radius, rect.height), fillColor, FILLED);
    rectangle(overlay, Rect(rect.x, rect.y + radius, rect.width, rect.height - 2 * radius), fillColor, FILLED);

    circle(overlay, Point(rect.x + radius, rect.y + radius), radius, fillColor, FILLED);
    circle(overlay, Point(rect.x + rect.width - radius - 1, rect.y + radius), radius, fillColor, FILLED);
    circle(overlay, Point(rect.x + radius, rect.y + rect.height - radius - 1), radius, fillColor, FILLED);
    circle(overlay, Point(rect.x + rect.width - radius - 1, rect.y + rect.height - radius - 1), radius, fillColor, FILLED);

    addWeighted(overlay, alpha / 255.0, frame, 1.0 - alpha / 255.0, 0, frame);
}

void drawPanel(Mat& frame, const Rect& rect, const Scalar& bgColor, const Scalar& borderColor, int radius, int alpha) {
    fillRoundedRect(frame, rect, bgColor, radius, alpha);
    rectangle(frame, rect, borderColor, 2, LINE_AA);
}

class BlinkDetector {
private:
    float earThreshold;
    int consecutiveFrames;
    int closedEyeCounter;
    int missingEyeFrames;
    int blinkCooldown;
    bool isBlinking;
    int totalBlinks;
    float lastEAR;

public:
    BlinkDetector()
        : earThreshold(DEFAULT_EAR_THRESHOLD), consecutiveFrames(DEFAULT_CONSECUTIVE_FRAMES),
          closedEyeCounter(0), missingEyeFrames(0), blinkCooldown(0), isBlinking(false),
          totalBlinks(0), lastEAR(0.4f) {
    }

    bool detectBlink(float currentEAR, bool facePresent, bool eyesVisible) {
        lastEAR = currentEAR;
        bool blinkEvent = false;

        if (blinkCooldown > 0)
            blinkCooldown--;

        if (!facePresent) {
            // No face detected - do not count as blink.
            missingEyeFrames = 0;
            closedEyeCounter = 0;
            isBlinking = false;
        } else if (!eyesVisible) {
            // Eyes not visible but face is present: count as closed eye detection.
            missingEyeFrames++;
            closedEyeCounter++;
        } else {
            // Eyes visible - use raw EAR threshold to decide closure.
            missingEyeFrames = 0;
            if (currentEAR < earThreshold) {
                closedEyeCounter++;
            } else {
                closedEyeCounter = 0;
                isBlinking = false;
            }
        }

        // Reset if prolonged loss of eye detection while face is present.
        if (facePresent && missingEyeFrames > MAX_MISSING_FRAMES) {
            closedEyeCounter = 0;
            isBlinking = false;
        }

        if (closedEyeCounter >= consecutiveFrames && !isBlinking && blinkCooldown == 0) {
            isBlinking = true;
            blinkEvent = true;
            totalBlinks++;
            blinkCooldown = BLINK_COOLDOWN_FRAMES;
        }

        return blinkEvent;
    }

    int getTotalBlinks() const { return totalBlinks; }
    float getCurrentEAR() const { return lastEAR; }
    float getThreshold() const { return earThreshold; }

    void resetBlinks() {
        closedEyeCounter = 0;
        missingEyeFrames = 0;
        blinkCooldown = 0;
        isBlinking = false;
        totalBlinks = 0;
        lastEAR = 0.4f;
    }

    void setThreshold(float threshold) {
        earThreshold = threshold;
        if (earThreshold < 0.10f)
            earThreshold = 0.10f;
        if (earThreshold > 0.40f)
            earThreshold = 0.40f;
    }
};

class StareOffGame {
private:
    enum GameState {
        MENU,
        COUNTDOWN,
        GAMEPLAY,
        ROUND_END,
        GAME_OVER
    } currentState;

    int player1Score;
    int player2Score;
    int currentRound;
    int totalRounds;
    int roundWinner;
    string winnerName;
    time_point<steady_clock> stateStartTime;
    int countdownValue;
    bool blinkDetectedThisRound;

public:
    StareOffGame()
        : currentState(MENU), player1Score(0), player2Score(0), currentRound(1),
          totalRounds(3), roundWinner(0), winnerName(""), countdownValue(3),
          blinkDetectedThisRound(false) {
    }

    void update(bool player1Blink, bool player2Blink, bool player1Visible, bool player2Visible, int& gameBlinkCount) {
        gameBlinkCount = 0;

        switch (currentState) {
        case MENU:
            break;
        case COUNTDOWN:
            updateCountdown();
            break;
        case GAMEPLAY:
            updateGameplay(player1Blink, player2Blink, player1Visible, player2Visible, gameBlinkCount);
            break;
        case ROUND_END:
            updateRoundEnd();
            break;
        case GAME_OVER:
            updateGameOver();
            break;
        }
    }

    void updateCountdown() {
        auto now = steady_clock::now();
        int elapsed = static_cast<int>(duration_cast<seconds>(now - stateStartTime).count());
        int newCount = max(0, 3 - elapsed);

        if (newCount != countdownValue) {
            countdownValue = newCount;
        }

        if (elapsed >= 3) {
            currentState = GAMEPLAY;
            stateStartTime = steady_clock::now();
            blinkDetectedThisRound = false;
        }
    }

    void updateGameplay(bool player1Blink, bool player2Blink, bool player1Visible, bool player2Visible, int& gameBlinkCount) {
        if (blinkDetectedThisRound)
            return;

        bool blinkEventDetected = player1Blink || player2Blink;
        if (!blinkEventDetected && !player1Visible && !player2Visible)
            return;

        if (player1Blink && !player2Blink) {
            blinkDetectedThisRound = true;
            gameBlinkCount = 1;
            player2Score++;
            roundWinner = 2;
            winnerName = "PLAYER 2";
            currentState = ROUND_END;
            stateStartTime = steady_clock::now();
        } else if (player2Blink && !player1Blink) {
            blinkDetectedThisRound = true;
            gameBlinkCount = 1;
            player1Score++;
            roundWinner = 1;
            winnerName = "PLAYER 1";
            currentState = ROUND_END;
            stateStartTime = steady_clock::now();
        } else if (player1Blink && player2Blink) {
            blinkDetectedThisRound = true;
            gameBlinkCount = 2;
            roundWinner = 0;
            winnerName = "TIE";
            currentState = ROUND_END;
            stateStartTime = steady_clock::now();
        }
    }

    void updateRoundEnd() {
        auto now = steady_clock::now();
        int elapsed = static_cast<int>(duration_cast<seconds>(now - stateStartTime).count());

        if (elapsed < 2)
            return;

        if (currentRound >= totalRounds) {
            currentState = GAME_OVER;
            if (player1Score > player2Score) {
                winnerName = "PLAYER 1";
            } else if (player2Score > player1Score) {
                winnerName = "PLAYER 2";
            } else {
                winnerName = "DRAW";
            }
            stateStartTime = steady_clock::now();
        } else {
            currentRound++;
            countdownValue = 3;
            currentState = COUNTDOWN;
            stateStartTime = steady_clock::now();
        }
    }

    void updateGameOver() {
        auto now = steady_clock::now();
        int elapsed = static_cast<int>(duration_cast<seconds>(now - stateStartTime).count());
        if (elapsed >= 4) {
            resetGame();
        }
    }

    void startGame() {
        resetGame();
        currentState = COUNTDOWN;
        countdownValue = 3;
        stateStartTime = steady_clock::now();
    }

    void resetGame() {
        player1Score = 0;
        player2Score = 0;
        currentRound = 1;
        roundWinner = 0;
        winnerName.clear();
        currentState = MENU;
        countdownValue = 3;
        blinkDetectedThisRound = false;
    }

    void draw(Mat& frame, float player1EAR, float player2EAR, bool player1Visible, bool player2Visible, float fps) {
        switch (currentState) {
        case MENU:
            drawMenu(frame);
            break;
        case COUNTDOWN:
            drawCountdown(frame);
            break;
        case GAMEPLAY:
            drawGameplay(frame, player1EAR, player2EAR, player1Visible, player2Visible, fps);
            break;
        case ROUND_END:
            drawRoundEnd(frame);
            break;
        case GAME_OVER:
            drawGameOver(frame);
            break;
        }
        drawScoreboard(frame);
    }

    void drawMenu(Mat& frame) {
        drawPanel(frame, Rect(80, 80, frame.cols - 160, frame.rows - 160), Scalar(20, 30, 50), Scalar(150, 190, 255), 22, 220);
        putText(frame, "STARE OFF", Point(frame.cols / 2 - 165, 140), FONT_HERSHEY_DUPLEX, 2.4, Scalar(255, 220, 100), 5, LINE_AA);
        putText(frame, "Keep your eyes open longer than your opponent", Point(frame.cols / 2 - 355, 220), FONT_HERSHEY_SIMPLEX, 0.85, Scalar(210, 210, 240), 1, LINE_AA);

        drawPanel(frame, Rect(frame.cols / 2 - 240, frame.rows / 2 - 30, 480, 120), Scalar(10, 10, 25), Scalar(0, 190, 255), 18, 190);
        putText(frame, "Press SPACE to start", Point(frame.cols / 2 - 120, frame.rows / 2 + 20), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2, LINE_AA);
        putText(frame, "R = reset   |   +/- = sensitivity   |   T = status", Point(frame.cols / 2 - 260, frame.rows / 2 + 58), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(190, 190, 255), 1, LINE_AA);
    }

    void drawCountdown(Mat& frame) {
        drawPanel(frame, Rect(frame.cols / 2 - 200, frame.rows / 2 - 160, 400, 320), Scalar(12, 18, 40), Scalar(0, 210, 220), 22, 210);
        putText(frame, "GET READY", Point(frame.cols / 2 - 120, frame.rows / 2 - 40), FONT_HERSHEY_DUPLEX, 1.4, Scalar(200, 230, 255), 2, LINE_AA);
        putText(frame, to_string(countdownValue), Point(frame.cols / 2 - 35, frame.rows / 2 + 70), FONT_HERSHEY_DUPLEX, 5.0, Scalar(0, 240, 255), 10, LINE_AA);
        putText(frame, "ROUND " + to_string(currentRound), Point(frame.cols / 2 - 90, frame.rows / 2 + 170), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(220, 220, 255), 2, LINE_AA);
    }

    void drawGameplay(Mat& frame, float player1EAR, float player2EAR, bool player1Visible, bool player2Visible, float fps) {
        Rect topBar(12, 12, frame.cols - 24, 96);
        drawPanel(frame, topBar, Scalar(20, 30, 55), Scalar(160, 220, 255), 18, 210);
        putText(frame, "STARE OFF", Point(30, 60), FONT_HERSHEY_DUPLEX, 1.2, Scalar(255, 220, 120), 2, LINE_AA);
        putText(frame, "ROUND " + to_string(currentRound) + "  |  P1 " + to_string(player1Score) + " - " + to_string(player2Score) + " P2", Point(frame.cols / 2 - 130, 60), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(220, 220, 255), 1, LINE_AA);
        putText(frame, "Sensitivity: " + to_string(blinkDetector1.getThreshold()).substr(0, 4), Point(frame.cols - 260, 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(190, 240, 255), 1, LINE_AA);
        putText(frame, "FPS " + to_string(static_cast<int>(fps)), Point(frame.cols - 260, 70), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(190, 240, 255), 1, LINE_AA);

        Rect p1Panel(24, 120, frame.cols / 2 - 42, 150);
        Rect p2Panel(frame.cols / 2 + 18, 120, frame.cols / 2 - 42, 150);
        drawPanel(frame, p1Panel, Scalar(30, 40, 70), Scalar(0, 190, 255), 18, 190);
        drawPanel(frame, p2Panel, Scalar(30, 40, 70), Scalar(0, 190, 255), 18, 190);

        putText(frame, "PLAYER 1", Point(p1Panel.x + 18, p1Panel.y + 35), FONT_HERSHEY_DUPLEX, 0.9, Scalar(180, 220, 255), 2, LINE_AA);
        putText(frame, "PLAYER 2", Point(p2Panel.x + 18, p2Panel.y + 35), FONT_HERSHEY_DUPLEX, 0.9, Scalar(180, 220, 255), 2, LINE_AA);

        putText(frame, "EAR " + to_string(player1EAR).substr(0, 5), Point(p1Panel.x + 18, p1Panel.y + 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(220, 220, 255), 1, LINE_AA);
        putText(frame, "EAR " + to_string(player2EAR).substr(0, 5), Point(p2Panel.x + 18, p2Panel.y + 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(220, 220, 255), 1, LINE_AA);

        putText(frame, player1Visible ? "EYES VISIBLE" : "EYES HIDDEN", Point(p1Panel.x + 18, p1Panel.y + 118), FONT_HERSHEY_SIMPLEX, 0.65, player1Visible ? Scalar(0, 255, 150) : Scalar(220, 130, 80), 1, LINE_AA);
        putText(frame, player2Visible ? "EYES VISIBLE" : "EYES HIDDEN", Point(p2Panel.x + 18, p2Panel.y + 118), FONT_HERSHEY_SIMPLEX, 0.65, player2Visible ? Scalar(0, 255, 150) : Scalar(220, 130, 80), 1, LINE_AA);

        Rect statusBar(20, frame.rows - 70, frame.cols - 40, 50);
        drawPanel(frame, statusBar, Scalar(10, 18, 35), Scalar(100, 180, 255), 16, 200);
        putText(frame, "Keep your eyes open. First blink loses.", Point(statusBar.x + 18, statusBar.y + 34), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(210, 230, 255), 1, LINE_AA);
    }

    void drawRoundEnd(Mat& frame) {
        drawPanel(frame, Rect(frame.cols / 2 - 260, frame.rows / 2 - 120, 520, 240), Scalar(10, 18, 35), Scalar(100, 255, 200), 22, 220);
        string result = (winnerName == "TIE") ? "ROUND " + to_string(currentRound) + " IS A TIE" : winnerName + " WINS ROUND " + to_string(currentRound);
        putText(frame, result, Point(frame.cols / 2 - 220, frame.rows / 2 - 10), FONT_HERSHEY_DUPLEX, 1.4, Scalar(255, 240, 200), 3, LINE_AA);
        putText(frame, "Next round begins soon...", Point(frame.cols / 2 - 170, frame.rows / 2 + 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(220, 220, 255), 1, LINE_AA);
    }

    void drawGameOver(Mat& frame) {
        drawPanel(frame, Rect(frame.cols / 2 - 260, frame.rows / 2 - 140, 520, 280), Scalar(12, 18, 35), Scalar(160, 230, 240), 22, 220);
        putText(frame, "MATCH OVER", Point(frame.cols / 2 - 140, frame.rows / 2 - 30), FONT_HERSHEY_DUPLEX, 1.6, Scalar(255, 220, 160), 4, LINE_AA);
        putText(frame, "WINNER: " + winnerName, Point(frame.cols / 2 - 145, frame.rows / 2 + 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(220, 220, 255), 2, LINE_AA);
        putText(frame, "FINAL SCORE: P1 " + to_string(player1Score) + " - " + to_string(player2Score) + " P2", Point(frame.cols / 2 - 165, frame.rows / 2 + 80), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(220, 220, 255), 2, LINE_AA);
        putText(frame, "Press R to restart", Point(frame.cols / 2 - 120, frame.rows / 2 + 130), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(180, 255, 220), 1, LINE_AA);
    }

    void drawScoreboard(Mat& frame) {
        Rect panel(18, 76, 280, 120);
        drawPanel(frame, panel, Scalar(16, 24, 42), Scalar(120, 190, 235), 18, 210);
        putText(frame, "SCORE", Point(panel.x + 22, panel.y + 32), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(210, 220, 255), 2, LINE_AA);
        putText(frame, "P1: " + to_string(player1Score), Point(panel.x + 22, panel.y + 65), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(220, 220, 255), 1, LINE_AA);
        putText(frame, "P2: " + to_string(player2Score), Point(panel.x + 22, panel.y + 95), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(220, 220, 255), 1, LINE_AA);
    }

    bool isInMenu() const { return currentState == MENU; }
    bool isGameActive() const { return currentState == GAMEPLAY; }
};

vector<Rect> getTopTwoFaces(vector<Rect> faces) {
    sort(faces.begin(), faces.end(), [](const Rect& a, const Rect& b) {
        return a.area() > b.area();
    });
    if (faces.size() > 2)
        faces.resize(2);
    sort(faces.begin(), faces.end(), [](const Rect& a, const Rect& b) {
        return a.x < b.x;
    });
    return faces;
}

bool detectMainEye(const Mat& gray, const Rect& face, CascadeClassifier& eyeCascade, Rect& mainEye, float& earValue) {
    Mat faceROI = gray(face);
    vector<Rect> eyes;
    eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 3, CASCADE_SCALE_IMAGE, Size(20, 20));

    if (eyes.empty()) {
        eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, CASCADE_SCALE_IMAGE, Size(15, 15));
    }

    if (eyes.empty())
        return false;

    // Find up to two largest eye rectangles
    sort(eyes.begin(), eyes.end(), [](const Rect& a, const Rect& b) { return a.area() > b.area(); });
    Rect eyeA = eyes[0];
    vector<float> ears;
    // Calculate EAR for the largest eye
    {
        vector<Point2f> eyePoints = getEyePoints(face, eyeA);
        ears.push_back(calculateEAR(eyePoints));
    }
    // If there's a second eye, include it in the average
    if (eyes.size() >= 2) {
        Rect eyeB = eyes[1];
        vector<Point2f> eyePointsB = getEyePoints(face, eyeB);
        ears.push_back(calculateEAR(eyePointsB));
    }

    // Use the largest eye as the mainEye for drawing
    mainEye = eyeA;

    // Average EARs
    float sum = 0.0f;
    for (float v : ears) sum += v;
    earValue = sum / static_cast<float>(ears.size());
    return true;
}

bool tryLoadCascade(CascadeClassifier& cascade, const vector<string>& paths) {
    for (const string& path : paths) {
        if (cascade.load(path))
            return true;
    }
    return false;
}

int main() {
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;

    vector<string> facePaths = {
        "haarcascade_frontalface_default.xml",
        "./haarcascade_frontalface_default.xml",
        "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml"
    };
    vector<string> eyePaths = {
        "haarcascade_eye.xml",
        "./haarcascade_eye.xml",
        "haarcascade_eye_tree_eyeglasses.xml",
        "./haarcascade_eye_tree_eyeglasses.xml",
        "C:/opencv/build/etc/haarcascades/haarcascade_eye.xml",
        "C:/opencv/build/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    };

    cout << "Loading cascades...\n";
    if (!tryLoadCascade(faceCascade, facePaths)) {
        cerr << "ERROR: Could not load face cascade. Please place haarcascade_frontalface_default.xml in the app folder." << endl;
        return -1;
    }
    if (!tryLoadCascade(eyeCascade, eyePaths)) {
        cerr << "ERROR: Could not load eye cascade. Please place haarcascade_eye.xml in the app folder." << endl;
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cap.open(1);
    }
    if (!cap.isOpened()) {
        cap.open(2);
    }
    if (!cap.isOpened()) {
        cerr << "ERROR: Cannot open webcam. Make sure camera is connected and not used by another app." << endl;
        return -1;
    }

    cout << "Webcam opened successfully." << endl;

    Mat frame, gray;
    BlinkDetector blinkDetector1;
    BlinkDetector blinkDetector2;
    StareOffGame game;

    float earValue1 = 0.4f;
    float earValue2 = 0.4f;
    bool eyesDetected1 = false;
    bool eyesDetected2 = false;
    int gameBlinkCount = 0;

    // Smoothed EAR values to reduce noise
    float smoothedEar1 = earValue1;
    float smoothedEar2 = earValue2;

    int frameCount = 0;
    double fps = 0.0;
    auto lastTime = steady_clock::now();

    cout << "=== STARE OFF - 2 PLAYER VERSION ===" << endl;
    cout << "Player 1 = left face, Player 2 = right face" << endl;
    cout << "Press SPACE to start, R to reset, +/- to adjust sensitivity, ESC to quit." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Warning: empty frame received." << endl;
            continue;
        }

        frameCount++;
        if (frameCount >= 30) {
            auto currentTime = steady_clock::now();
            double elapsedMs = duration_cast<milliseconds>(currentTime - lastTime).count();
            if (elapsedMs > 0)
                fps = frameCount * 1000.0 / elapsedMs;
            frameCount = 0;
            lastTime = currentTime;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, CASCADE_SCALE_IMAGE, Size(100, 100));
        vector<Rect> selectedFaces = getTopTwoFaces(faces);

        earValue1 = 0.4f;
        earValue2 = 0.4f;
        eyesDetected1 = false;
        eyesDetected2 = false;

        bool player1Blink = false;
        bool player2Blink = false;
        bool player1FacePresent = false;
        bool player2FacePresent = false;
        Rect face1, face2;

        if (selectedFaces.size() == 1) {
            int centerX = selectedFaces[0].x + selectedFaces[0].width / 2;
            if (centerX < frame.cols / 2) {
                player1FacePresent = true;
                face1 = selectedFaces[0];
            } else {
                player2FacePresent = true;
                face2 = selectedFaces[0];
            }
        } else if (selectedFaces.size() >= 2) {
            player1FacePresent = true;
            player2FacePresent = true;
            face1 = selectedFaces[0];
            face2 = selectedFaces[1];
        }

        if (player1FacePresent) {
            rectangle(frame, face1, Scalar(255, 0, 0), 2);
            putText(frame, "PLAYER 1", Point(face1.x, max(20, face1.y - 10)), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);

            Rect mainEye1;
            eyesDetected1 = detectMainEye(gray, face1, eyeCascade, mainEye1, earValue1);
            if (eyesDetected1) {
                Rect eyeRect1(face1.x + mainEye1.x, face1.y + mainEye1.y, mainEye1.width, mainEye1.height);
                rectangle(frame, eyeRect1, Scalar(0, 255, 0), 2);
                putText(frame, "EAR1: " + to_string(smoothedEar1).substr(0, 5), Point(face1.x, face1.y + face1.height + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            }
        }

        if (player2FacePresent) {
            rectangle(frame, face2, Scalar(0, 0, 255), 2);
            putText(frame, "PLAYER 2", Point(face2.x, max(20, face2.y - 10)), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

            Rect mainEye2;
            eyesDetected2 = detectMainEye(gray, face2, eyeCascade, mainEye2, earValue2);
            if (eyesDetected2) {
                Rect eyeRect2(face2.x + mainEye2.x, face2.y + mainEye2.y, mainEye2.width, mainEye2.height);
                rectangle(frame, eyeRect2, Scalar(0, 255, 0), 2);
                putText(frame, "EAR2: " + to_string(smoothedEar2).substr(0, 5), Point(face2.x, face2.y + face2.height + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            }
        }

        // Apply light exponential smoothing for display only
        const float alpha = 0.28f;
        smoothedEar1 = smoothedEar1 * (1.0f - alpha) + earValue1 * alpha;
        smoothedEar2 = smoothedEar2 * (1.0f - alpha) + earValue2 * alpha;

        player1Blink = blinkDetector1.detectBlink(earValue1, player1FacePresent, eyesDetected1);
        player2Blink = blinkDetector2.detectBlink(earValue2, player2FacePresent, eyesDetected2);

        game.update(player1Blink, player2Blink, eyesDetected1, eyesDetected2, gameBlinkCount);
        game.draw(frame, smoothedEar1, smoothedEar2, eyesDetected1, eyesDetected2, static_cast<float>(fps));

        string fpsText = "FPS: " + to_string(static_cast<int>(fps));
        string thresholdText = "Threshold: " + to_string(blinkDetector1.getThreshold()).substr(0, 4);
        putText(frame, fpsText, Point(frame.cols - 110, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        putText(frame, thresholdText, Point(frame.cols - 140, 55), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 200), 1);

        if (selectedFaces.empty()) {
            putText(frame, "NO FACE DETECTED", Point(frame.cols / 2 - 120, frame.rows / 2), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        } else if (selectedFaces.size() == 1) {
            putText(frame, "ONLY ONE PLAYER DETECTED", Point(frame.cols / 2 - 180, frame.rows / 2), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 165, 255), 2);
        }

        if (game.isGameActive()) {
            if (player1Blink && !player2Blink) {
                putText(frame, "PLAYER 1 BLINKED FIRST!", Point(frame.cols / 2 - 190, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            } else if (player2Blink && !player1Blink) {
                putText(frame, "PLAYER 2 BLINKED FIRST!", Point(frame.cols / 2 - 190, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            } else if (player1Blink && player2Blink) {
                putText(frame, "BOTH PLAYERS BLINKED!", Point(frame.cols / 2 - 170, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }
        }

        int barWidth = 120;
        int barHeight = 12;
        int p1BarX = 20;
        int p1BarY = frame.rows - 60;
        rectangle(frame, Rect(p1BarX, p1BarY, barWidth, barHeight), Scalar(50, 50, 50), -1);
        int p1EarBarWidth = min(barWidth, max(0, static_cast<int>(smoothedEar1 * barWidth / 0.5f)));
        Scalar p1BarColor = smoothedEar1 < blinkDetector1.getThreshold() ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
        rectangle(frame, Rect(p1BarX, p1BarY, p1EarBarWidth, barHeight), p1BarColor, -1);
        int p1ThresholdX = p1BarX + min(barWidth, max(0, static_cast<int>(blinkDetector1.getThreshold() * barWidth / 0.5f)));
        line(frame, Point(p1ThresholdX, p1BarY - 2), Point(p1ThresholdX, p1BarY + barHeight + 2), Scalar(255, 255, 0), 2);
        putText(frame, "P1", Point(p1BarX, p1BarY - 6), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        int p2BarX = frame.cols - 150;
        int p2BarY = frame.rows - 60;
        rectangle(frame, Rect(p2BarX, p2BarY, barWidth, barHeight), Scalar(50, 50, 50), -1);
        int p2EarBarWidth = min(barWidth, max(0, static_cast<int>(smoothedEar2 * barWidth / 0.5f)));
        Scalar p2BarColor = smoothedEar2 < blinkDetector2.getThreshold() ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
        rectangle(frame, Rect(p2BarX, p2BarY, p2EarBarWidth, barHeight), p2BarColor, -1);
        int p2ThresholdX = p2BarX + min(barWidth, max(0, static_cast<int>(blinkDetector2.getThreshold() * barWidth / 0.5f)));
        line(frame, Point(p2ThresholdX, p2BarY - 2), Point(p2ThresholdX, p2BarY + barHeight + 2), Scalar(255, 255, 0), 2);
        putText(frame, "P2", Point(p2BarX, p2BarY - 6), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

        imshow("STARE OFF - Ultimate Blink Challenge", frame);
        char key = static_cast<char>(waitKey(10));
        if (key == 27)
            break;

        if (key == ' ' && game.isInMenu()) {
            game.startGame();
            blinkDetector1.resetBlinks();
            blinkDetector2.resetBlinks();
        }
        if (key == 'r' || key == 'R') {
            game.resetGame();
            blinkDetector1.resetBlinks();
            blinkDetector2.resetBlinks();
        }
        if (key == '+' || key == '=') {
            float newThreshold = blinkDetector1.getThreshold() + 0.01f;
            blinkDetector1.setThreshold(newThreshold);
            blinkDetector2.setThreshold(newThreshold);
        }
        if (key == '-' || key == '_') {
            float newThreshold = blinkDetector1.getThreshold() - 0.01f;
            blinkDetector1.setThreshold(newThreshold);
            blinkDetector2.setThreshold(newThreshold);
        }
        if (key == 't' || key == 'T') {
            cout << "\n=== STARE OFF STATUS ===" << endl;
            cout << "P1 EAR: " << smoothedEar1 << endl;
            cout << "P2 EAR: " << smoothedEar2 << endl;
            cout << "Threshold: " << blinkDetector1.getThreshold() << endl;
            cout << "FPS: " << static_cast<int>(fps) << endl;
            cout << "P1 Eyes Detected: " << (eyesDetected1 ? "YES" : "NO") << endl;
            cout << "P2 Eyes Detected: " << (eyesDetected2 ? "YES" : "NO") << endl;
            cout << "Game State: " << (game.isGameActive() ? "ACTIVE" : "MENU") << endl;
            cout << "=======================\n" << endl;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
