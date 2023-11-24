from PIL import Image, ImageDraw, ImageFont
import numpy as np
import datetime

# TODO: Maybe add actual timestamp as input?
# def vhs_filter(frame, timestamp):
def vhs_filter(frame):
    # Convert the image to numpy array
    np_image = np.array(frame)

    # Adding scanlines: Skip every other line
    np_image[::2, :, :] = np_image[::2, :, :] * 0.3

    # Adjusting color balance: Slightly increase red, decrease blue
    np_image[:, :, 0] = np_image[:, :, 0] * 1.1  # Red channel
    np_image[:, :, 2] = np_image[:, :, 2] * 0.9  # Blue channel

    # Adding noise
    noise = np.random.normal(0, 25, np_image.shape)
    np_image = np.clip(np_image + noise, 0, 255)

    # Converting back to PIL Image
    vhs_image = Image.fromarray(np_image.astype('uint8'), 'RGB')

    # Adding timestamp
    draw = ImageDraw.Draw(vhs_image)
    # Using a larger font size and loading a custom font if available
    font_size = 20
    font = ImageFont.truetype("fonts/VCR_OSD_MONO.ttf", 20)

    # Formatting the timestamp text
    text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Positioning the text at the bottom left corner
    text_x = 10
    text_y = vhs_image.size[1] - font_size - 10  # Adjust the Y position
    draw.text((text_x, text_y), text, (255, 255, 255), font=font)

    # Adding the recording label
    label_text="REC"
    rec_x = 10
    rec_y = 10
    # Drawing a red dot
    draw.ellipse([(rec_x, rec_y), (rec_x+10, rec_y+10)], fill=(255, 0, 0))
    # Drawing the label text next to the red dot
    draw.text((rec_x + 15, rec_y - 5), label_text, (255, 255, 255), font=font)

    return vhs_image

