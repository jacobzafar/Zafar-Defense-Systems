# Zafar Defense Systems AB

Initial R&D repository for AI-based drone detection, target tracking, control logic, and operator interface development.

This repository is the starting point for building a modular system focused on detecting and tracking drones from video input. The project is currently in the early research and prototype phase and is being developed with speed, clarity, and iteration in mind.

## Purpose

The purpose of this repository is to organize the technical and product development work for a first MVP.

The initial MVP is scoped to demonstrate the following:

- Video input from file or camera
- Drone detection in the image
- Target tracking across frames
- Simple UI for visualizing output
- Simple event logging

The MVP does **not** include:

- Jamming
- Any kinetic or effect system
- Advanced autonomy
- Multi-sensor fusion
- Production-grade military hardening
- Integration with external defense platforms

## Current Focus

The current focus is to build the fastest possible end-to-end prototype using a pragmatic mix of:

- custom integration logic,
- custom testing and documentation,
- internal iteration based on real demo milestones.

The working principle is simple: build a functioning pipeline first, then improve precision, robustness, hardware integration, and operational value over time.

## Repository Structure

```text
.
├── research/   Background research, open-source evaluation, notes, competitor review
├── detector/   Drone detection code and experiments
├── tracker/    Target tracking code and experiments
├── ui/         Operator interface and visualization layer
├── control/    Control logic, signal mapping, and hardware/simulator integration
└── docs/       MVP docs, architecture, planning, and internal documentation
```

## Folder Overview

### `research/`
Contains technical research, source reviews, open-source shortlist, dataset notes, and competitor or market observations.

### `detector/`
Contains experiments and code related to detecting drones in image or video streams.

### `tracker/`
Contains logic and experiments for tracking a detected target over time and maintaining target identity between frames.

### `ui/`
Contains the user interface layer, including basic operator views, detection overlays, and logging output.

### `control/`
Contains control logic that maps detection/tracking output into signals for simulation, turret movement, or later hardware integration.

### `docs/`
Contains internal documents such as the MVP definition, architecture notes, sprint plans, and roadmap material.

## MVP Definition

The first 4-week MVP is considered complete when the system can:

1. Run a video through the pipeline without crashing
2. Detect a drone in the frame
3. Track the target across multiple frames
4. Show the result in a simple UI
5. Save a basic log of detection/tracking events
6. Produce a short demo that shows the full workflow

## Working Principles

This project follows a few simple rules:

- Use the fastest path to a working prototype
- Prefer simple solutions over perfect solutions in the early phase
- Keep architecture modular from the start
- Document decisions as they are made
- Avoid unnecessary complexity in the first iterations

## Development Approach

The project is currently organized around a modular pipeline:

**Video input → Detection → Tracking → Control output → UI → Logging**

This modular structure makes it easier to swap components later without rebuilding the entire system.

Examples of likely future iterations:

- better small-object detection,
- more stable tracking,
- control integration with a physical platform,
- more robust field testing,
- improved operator interface,
- support for additional sensors.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/jacobzafar/Zafar-Defense-Systems-AB.git
cd Zafar-Defense-Systems-AB
```

Create the project structure if needed:

```bash
mkdir -p research detector tracker ui control docs
```

Add placeholder files if you want empty folders to remain visible in Git:

```bash
touch research/.gitkeep detector/.gitkeep tracker/.gitkeep ui/.gitkeep control/.gitkeep docs/.gitkeep
```

## Recommended First Files

Suggested initial files for a clean start:

- `docs/MVP.md`
- `docs/ARCHITECTURE.md`
- `research/open-source.md`
- `research/datasets.md`
- `detector/README.md`
- `tracker/README.md`
- `ui/README.md`
- `control/README.md`

## Documentation

Project documentation should live in the `docs/` folder and be kept lightweight, current, and practical.

Examples:

- MVP scope
- Architecture diagrams
- Weekly plans
- Demo criteria
- Test notes
- Risks and assumptions

## Open Source and Licensing

This repository may incorporate or build upon open-source tools, models, or code where licensing allows it. 

## Status

Current stage: **Early prototype / MVP planning**

Current objective: **Build a working end-to-end demonstrator as fast as possible**

## Maintainers

- Jacob Zafar

## Notes

This repository is intended as a technical development base for early-stage R&D and prototyping. The contents will evolve as the project matures from MVP to more robust product iterations.
