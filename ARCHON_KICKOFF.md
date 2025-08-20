# Archon OS Agent Kickoff: AI Boxing Trainer

This document provides a comprehensive overview of the AI Boxing Trainer project, including a codebase audit, knowledge base resources, and a prioritized, AI-executable task list.

## 1. Codebase Audit

The AI Boxing Trainer is a Python application designed to capture and classify boxing punches using computer vision. The core logic is encapsulated within the `ai_trainer` module, supported by various scripts for data collection, training, and testing.

### Key Modules and Components:

| File/Module | Description | Dependencies |
| :--- | :--- | :--- |
| `main.py` | Main application entry point for real-time punch classification. | `ai_trainer` |
| `individual_punch_recorder.py` | Script for collecting labeled training data for individual punches. | `opencv-python` |
| `ai_trainer/` | Core module containing the AI logic. | `mediapipe`, `numpy` |
| `ai_trainer/pose_tracker.py` | Handles pose estimation using MediaPipe BlazePose. | `mediapipe` |
| `ai_trainer/punch_classifier.py`| Classifies punches based on pose data. | `numpy` |
| `ai_trainer/form_analyzer.py` | Scores punching form against ideal joint angles. | `numpy` |
| `ai_trainer/utils.py` | Utility functions used across the `ai_trainer` module. | |
| `requirements.txt` | Lists all Python dependencies for the project. | `pip` |

### Potential Blockers:

*   **Model Training Pipeline:** The process for training the `PunchClassifier` is not explicitly defined. A dedicated training script is needed.
*   **Pre-trained Models:** It's unclear if any pre-trained models are included or where they should be stored.
*   **Configuration:** No centralized configuration for parameters like camera resolution, model paths, or training settings.

## 2. Knowledge Base Injection

The following resources provide essential context for the AI agent. They should be linked to the relevant tasks and modules.

| Resource | Type | Description | Relevant To |
| :--- | :--- | :--- | :--- |
| `README.md` | Documentation | Project overview, setup, and data collection process. | All tasks |
| `HEAVY_BAG_ENHANCEMENT.md`| Documentation | Details on the heavy bag mode implementation. | `Implement Heavy Bag Mode` task |
| `combos/` | Data | JSON files defining boxing combinations. | `Add TTS Combination Callouts` task |
| `data/` | Data | Directory for storing training data and models. | `PunchClassifier` training |

## 3. AI-Executable Task Roadmap

The following tasks have been reviewed, optimized, and prioritized for execution by an AI agent.

### Sprint 1: Core Model Development

| Task ID | Title | Description | Inputs | Expected Output | Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ac0ea167` | Implement PunchClassifier Module | Create a `PunchClassifier` module in `ai_trainer/punch_classifier.py` that uses 3D landmark data to distinguish between jab, cross, hook, and uppercut punches with >90% accuracy. | 3D pose landmark data from `PoseTracker`. | A classified punch type (jab, cross, hook, uppercut) and a confidence score. | `895f07ae` | `todo` |
| `d37711d8` | Implement FormAnalyzer Module | Create a `FormAnalyzer` module in `ai_trainer/form_analyzer.py` that scores punches based on deviations from ideal joint angles, providing real-time form scores. | Classified punch type and 3D pose data. | A form score (e.g., A-F grade or percentage) and actionable feedback. | `ac0ea167` | `todo` |
| `e2b55723` | Create Testing and Validation Suite | Develop a comprehensive testing suite to validate punch classification accuracy (>90%), performance, and reliability. | Labeled test data. | A test report summarizing accuracy, precision, recall, and performance metrics. | `ac0ea167` | `todo` |

### Sprint 2: Feature Enhancements

| Task ID | Title | Description | Inputs | Expected Output | Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `6a17ab72` | Create Real-time Punch Counter UI | Implement an on-screen, real-time counter for each punch type that updates immediately upon detection. | Classified punch events. | A UI display with updated punch counts. | `ac0ea167` | `todo` |
| `7d3911ed` | Implement Timed Training Rounds | Create timed rounds (e.g., 3 minutes) that track total punches and average form score, displaying a summary at the end. | Punch and form data. | An end-of-round performance summary. | `d37711d8` | `todo` |
| `70e0f6e5` | Add TTS Combination Callouts | Implement TTS to call out boxing combinations and track user performance against them. | Pre-set combinations from `combos/`. | Spoken combination callouts and performance feedback. | `pyttsx3` | `todo` |

### Sprint 3: User Experience and Robustness

| Task ID | Title | Description | Inputs | Expected Output | Dependencies | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `29c21b85` | Implement Heavy Bag Mode | Create a heavy bag mode that adjusts tracking for angled positions, maintaining accuracy when one side of the body is obscured. | `HEAVY_BAG_ENHANCEMENT.md` | Robust tracking for users at an angle to the camera. | `895f07ae` | `todo` |
| `0f1d1797` | Add Camera Setup Guidance | Implement on-screen guidance to help users position their camera for optimal tracking. | Camera feed. | Visual indicators for distance, angle, and lighting. | | `todo` |
| `0fa04c38` | Optimize for Performance | Optimize the application to maintain >30 FPS and <50ms latency on consumer hardware. | Performance profiling data. | An optimized application meeting performance targets. | | `todo` |

## 4. Agent Kickoff Overview

**Objective:** Your primary goal is to complete the tasks outlined in the roadmap to build a fully functional AI Boxing Trainer.

**Getting Started:**

1.  **Begin with Sprint 1.** Start with task `ac0ea167: Implement PunchClassifier Module`.
2.  **Consult the Knowledge Base.** Refer to the linked resources for each task.
3.  **Follow the Dependencies.** Ensure tasks are completed in the specified order.
4.  **Update Task Status.** As you complete tasks, update their status in Archon OS.

This structured approach will ensure efficient and effective project completion.
