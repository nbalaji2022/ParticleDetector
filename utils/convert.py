from PIL import Image

# Open the TIFF image
try:
    tiff_image = Image.open("/Users/balaji.karrupuswamy/Downloads/Unet-CNN-main/datasets/images/val/val_0.tif")
    # Convert the image to RGB if it's not already
    if tiff_image.mode != "RGB":
        jpeg_image = tiff_image.convert("RGB")
    else:
         jpeg_image = tiff_image
    # Save the image as JPEG
    jpeg_image.save("/Users/balaji.karrupuswamy/Downloads/Unet-CNN-main/datasets/images/val/val_0.jpg", format="JPEG")
    print("Image converted successfully!")
except FileNotFoundError:
    print("Error: TIFF image not found.")
except Exception as e:
    print(f"An error occurred: {e}")