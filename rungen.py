import subprocess

def run_inference(driven_audio, source_image, enhancer=True):
    # command = ["python", "inference.py", "--driven_audio", driven_audio, "--source_image", source_image, "--enhancer", enhancer]
    # command = ["python3", "inference.py", "--driven_audio", driven_audio, "--source_image", source_image]
    command = ["python3", "inference_test.py", "--driven_audio", driven_audio, "--source_image", source_image]
    subprocess.run(command)

# Replace "<audio.wav>", "<video.mp4 or picture.png>", and "gfpgan" with actual file paths and enhancer name.
run_inference("./examples/driven_audio/bus_chinese.wav", "./examples/source_image/asian_4.png")
