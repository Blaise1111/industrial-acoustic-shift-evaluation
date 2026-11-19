"""Place the four machine-specific figure pairs side by side at native resolution."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIGS = Path(__file__).resolve().parents[1] / "figs"


def main():
    output = FIGS / "comparison"
    output.mkdir(exist_ok=True)
    valve = sorted((FIGS / "valve").glob("fig[1-4]_*.png"))
    fan = sorted((FIGS / "fan").glob("fig[1-4]_*.png"))
    if len(valve) != 4 or [p.name for p in valve] != [p.name for p in fan]:
        raise ValueError("Run the four figures for both valve and fan before comparing")
    header = 130
    font = ImageFont.load_default(size=68)
    for valve_path, fan_path in zip(valve, fan):
        with Image.open(valve_path) as left, Image.open(fan_path) as right:
            canvas = Image.new("RGB", (left.width + right.width, header + max(left.height, right.height)),
                               "white")
            canvas.paste(left, (0, header))
            canvas.paste(right, (left.width, header))
            draw = ImageDraw.Draw(canvas)
            draw.text((left.width // 2, header // 2), "Valve | intermittent sound",
                      anchor="mm", fill="#26384a", font=font)
            draw.text((left.width + right.width // 2, header // 2), "Fan | continuous sound",
                      anchor="mm", fill="#26384a", font=font)
            canvas.save(output / valve_path.name, dpi=(300, 300))


if __name__ == "__main__":
    main()
