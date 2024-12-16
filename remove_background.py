import cv2
import numpy as np
import torch
from torchvision import transforms, models

# Load pre-trained DeepLabV3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(device)
model.eval()

# Global variables for rectangle
rect_start = None
rect_end = None
cropped_image = None
object_mask = None
img_resized = None
scale_x = 1
scale_y = 1

# Function to resize image to fit screen while maintaining aspect ratio
def resize_to_fit_screen(image):
    global scale_x, scale_y
    screen_width = 1920  # Replace with your screen width
    screen_height = 1080  # Replace with your screen height
    
    img_height, img_width = image.shape[:2]
    aspect_ratio = img_width / img_height
    
    # Calculate the new width and height to maintain aspect ratio
    if img_width > img_height:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
        if new_height > screen_height:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        if new_width > screen_width:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
    
    # Calculate the scaling factors
    scale_x = new_width / img_width
    scale_y = new_height / img_height

    # Resize the image
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, cropped_image, img_resized
    
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and rect_start:
        rect_end = (x, y)
        img_copy = img_resized.copy()  # Use resized image size for showing
        cv2.rectangle(img_copy, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow('Select Object', img_copy)  # Show resized image
    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)
        # cv2.rectangle(img_resized, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow('Select Object', img_resized)  # Show resized image
        
        # Map the rectangle coordinates back to the resized image
        x1, y1 = rect_start
        x2, y2 = rect_end
        x1 = int(x1 / scale_x)
        y1 = int(y1 / scale_y)
        x2 = int(x2 / scale_x)
        y2 = int(y2 / scale_y)

        # Crop the selected region from the original (non-resized) image
        cropped_image = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]


# Load image
image_path = 'bear1.jpg'  # Replace with your image path
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found.")
    exit()

# Resize image to fit screen (before showing it)
img_resized = resize_to_fit_screen(img)

# Show the resized image
cv2.imshow('Select Object', img_resized)  # Show resized image
cv2.setMouseCallback('Select Object', draw_rectangle)  # Set the callback for the mouse
cv2.waitKey(0)  # Wait for the user to select the object

if cropped_image is not None:
    # Proceed with background removal and further processing
    # Preprocess the cropped image for the model
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Resize for the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(cropped_image).unsqueeze(0).to(device)

    # Get segmentation mask from the model
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()

    # Resize mask to match the size of the cropped image
    mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask to remove background
    mask = (mask > 0).astype(np.uint8) * 255  # Binary mask
    object_mask = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    # Create a transparent image with alpha channel for extracted object
    result_with_alpha = cv2.cvtColor(object_mask, cv2.COLOR_BGR2BGRA)
    result_with_alpha[:, :, 3] = mask  # Add alpha channel (use mask as transparency)

    # Write text on the resized image where the object was located
    font = cv2.FONT_HERSHEY_TRIPLEX
    text = "WHITE BEAR"
    text_position = (rect_start[0]-200 , rect_start[1] + 200)  # Text position near the object

    cv2.putText(img_resized, text, text_position, font, 6, (0, 0, 0), 5, cv2.LINE_AA)

    # Overlay the extracted object back on the resized image
    img_with_text = img_resized.copy()
    alpha_mask = result_with_alpha[:, :, 3] / 255.0  # Normalize alpha mask

    # Resize the object and alpha mask to match the selected region
    x1, y1 = rect_start
    x2, y2 = rect_end
    overlay_region = img_with_text[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

    # Resize the extracted object to the selected region size
    object_resized = cv2.resize(result_with_alpha, (overlay_region.shape[1], overlay_region.shape[0]))

    # Resize the alpha mask to the selected region size
    alpha_mask_resized = cv2.resize(alpha_mask, (overlay_region.shape[1], overlay_region.shape[0]))

    # Now, apply the alpha blending
    for c in range(0, 3):  # Loop through RGB channels
        overlay_region[:, :, c] = (overlay_region[:, :, c] * (1 - alpha_mask_resized)) + \
                                  (object_resized[:, :, c] * alpha_mask_resized)

    # Save and display result
    cv2.imwrite('final_result.png', img_with_text)
    cv2.imshow('Final Result', img_with_text)  # Display the final result
    cv2.waitKey(0)

cv2.destroyAllWindows()
