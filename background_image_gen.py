import sys
from image_gen import generate_image

if __name__ == "__main__":
    prompt = sys.argv[1]
    output_file = sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 35
    try:
        path, seed = generate_image(prompt, output_file, steps=steps)
        with open("image_status.txt", "w") as f:
            f.write(f"done | seed={seed} | steps={steps} | prompt={prompt}")
    except Exception as e:
        with open("image_status.txt", "w") as f:
            f.write(f"FAILED: {e}")