# Raspberry Pi Shader Camera
##### CS 445 Final Project: Meghan Reynolds, Kimberlee Phan, & Rachel Muzzelo


### Hardware Specifications ###
- Raspberry Pi 4 Model B (4 GB)
- Raspberry Pi Camera Module v2
  
### Code Quick Start (Raspbian) ###
1. Clone this repository:
    ```
    git clone https://github.com/meghanreynolds/raspberry-pi-shader-camera.git
    ```
2. Change lines 176 and 318 of ShaderCamera.py to your absolute path to the corresponding fonts
3. Run the Shader Camera:
   ```
   python3 ShaderCamera.py <filter_code>
   ```
   The filter codes are:
   - bp: Blueprint filter
   - night: Nightvision filter
   - retro: Retro Videogame filter
   - sketch: Sketchbook filter
   - toon: Cartoon filter
   - vhs: VHS Tape filter
   - water: Watercolor filter
   - none: Unfiltered video feed
   - help: Print a list of filter codes

### Test Data ###
To access test data, go to: [https://drive.google.com/drive/folders/1Io4CAaOvoPNScr7LzSYsAzUaSdonWQ3H?usp=drive_link](https://drive.google.com/drive/folders/1Io4CAaOvoPNScr7LzSYsAzUaSdonWQ3H?usp=drive_link)
