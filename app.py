import cv2
import os
from PIL import Image
import google.generativeai as genai

# Set up Generative AI model
#secret_key = os.getenv(
genai.configure(api_key="AIzaSyB1CZDZJf-wifmwFoBTNZhw9ruZNxlzlN4")
frame_description=""
thumbnail_description=""

def get_gemini_response(input_text, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

def thumbnail_desc(thumbnail_path):
    image = Image.open(thumbnail_path)
    input_prompt = f"Description for thumbnail {thumbnail_path}: "
    response = get_gemini_response(input_prompt, image)
    global thumbnail_description
    # Print the generated text
    thumbnail_description=response
    print(f"Text for {thumbnail_path}: {response}")
    # global frame_description
    # frame_description=frame_description+"\n"+text_response
    return 

def convert_videos_to_images_with_transcription(input_folder, output_folder, frame_interval=1):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        # Check if the file is a video
        if os.path.isfile(file_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_name = os.path.splitext(file_name)[0]
            video_output_folder = os.path.join(output_folder, video_name)
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)
            
            # Open the video file
            video_capture = cv2.VideoCapture(file_path)
            success, frame = video_capture.read()
            count = 0

            # Read each frame and process it
            while success:
                if count % frame_interval == 0:
                    # Save the frame as an image
                    image_path = os.path.join(video_output_folder, f"frame_{count:04d}.jpg")
                    cv2.imwrite(image_path, frame)

                    # Load the image and generate text description
                    image = Image.open(image_path)
                    input_prompt = f"Description for frame {count:04d}: "
                    response = get_gemini_response(input_prompt, image)

                    # Print the generated text
                    print(f"Text for {image_path}: {response}")
                    text_response= (f"Text for {image_path}: {response}")
                    global frame_description
                    frame_description=frame_description+"\n"+text_response
                success, frame = video_capture.read()  # Read next frame
                count += 1

            # Release the video capture object
            video_capture.release()
            

# Provide the path to the input video folder and output image folder
thumbnail_path='./input/thumbnails/thumb2.jpg'
input_folder = './input/test'
output_folder = './output'
frame_interval = 200  # Set the interval between frames (e.g., every 200th frame)

# Call the function to convert all videos in the input folder to images with text generation
convert_videos_to_images_with_transcription(input_folder, output_folder, frame_interval)
thumbnail_desc(thumbnail_path)

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

# safety_settings = [
#   {
#     "category": "HARM_CATEGORY_HARASSMENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_HATE_SPEECH",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
# ]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=None)

convo = model.start_chat(history=[

])
msg="these are frame description of each frame from a video, summarize this to understand the video content in paragraph \n "+ frame_description
convo.send_message(msg)
video_summary=convo.last.text

detector="compare the thumbnail description and video summary to just state whether thumbnail act as a clickbait for the video.if there is more than 90 percent similarity between thumbnail and any frame then consider it as not clickbait \n thumbnail description:" +thumbnail_description+"\n video summary:"+frame_description
convo.send_message(detector)
print(convo.last.text)
