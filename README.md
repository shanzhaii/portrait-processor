# portrait-processor
Python scripts to process portaits by removing background and scaling + centering according to eye position

# full photo workflow
1. face_detector.py
   * import photo
   * detect eye position
   * translate and scale to move eye to desired location
2. transparent_creator
   * remove background
