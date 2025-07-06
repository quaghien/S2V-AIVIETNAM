import os
import base64
import re
import requests
import openai
from moviepy import AudioFileClip, ImageSequenceClip, concatenate_videoclips
from pathlib import Path
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import natsort
import anthropic
import time
from openai import OpenAI
from google import genai
from google.genai import types
import wave
import uuid
from datetime import datetime
import io
import json
from urllib.parse import quote
import subprocess

class ZaloGPTProcessor:
    def __init__(self, openai_api_key, anthropic_api_key, gemini_api_key, zalo_api_key):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        self.zalo_api_key = zalo_api_key
        
        # Initialize API clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Initialize Gemini client
        os.environ['GEMINI_API_KEY'] = gemini_api_key
        self.gemini_client = genai.Client()

    def images_from_folder(self, folder_path):
        """Reads all images from a folder and sorts them."""
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                       file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return natsort.natsorted(image_files)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_base64_image_content(self, filenames):
        image_content = []
        for filename in filenames:
            base64_image = self.encode_image(filename)
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                },
            })
        return image_content

    def process_response(self, json_response):
        content = json_response['choices'][0]['message']['content']
        slides = re.split(r'(#slide\d+#)', content)[1:]

        slide_dict = {}
        for i in range(0, len(slides), 2):
            slide_number = int(re.findall(r'\d+', slides[i])[0])
            slide_text = slides[i + 1].strip()
            slide_dict[slide_number] = slide_text

        return slide_dict

    def send_batch_request(self, image_files, start_slide, previous_response_text="", is_first_batch=True):
        """Sends a batch of image files to the API and returns the response."""
        image_content = self.create_base64_image_content(image_files)
        slide_tags = [f"#slide{start_slide + i}#" for i in range(len(image_files))]

        if is_first_batch:
            prompt_text = (
                previous_response_text + " "
                "Please read the content of these slides carefully and take on the role of a professor to give a lecture. I need you to understand the meaning of each slide thoroughly and explain them with smooth transitions between the content, rather than just reading the existing text. Please help me achieve this. Since I need to use this later, please divide the content with the tags " + ", ".join(
                slide_tags) + " for easy reference. Make sure the explanations meaningful, if which part you think important for the lecture, explain detail. Note, only use the tags " + ", ".join(
                slide_tags) + " and do not include any other text."
            )
        else:
            prompt_text = (
                previous_response_text + " "
                "Please continue reading the content of these slides carefully and take on the role of a professor to give a lecture. (Do not perform greetings) I need you to understand the meaning of each slide thoroughly and explain them with smooth transitions between the content, rather than just reading the existing text. Please help me achieve this. Since I need to use this later, please divide the content with the tags " + ", ".join(
                slide_tags) + " for easy reference. Make sure the explanations meaningful, if which part you think important for the lecture. Note, only use the tags " + ", ".join(
                slide_tags) + " and do not include any other text. Please preserve content of the slide, do not change the content of the slide, respect the tag. Don't get another content from another slide, just use the content of the slide. Read slide by slide and give corresponding slide tag. Please TEACH the slide which you process, don't skip any content"
            )

        text_content = {
            "type": "text",
            "text": prompt_text,
        }

        messages = [
            {
                "role": "user",
                "content": [
                    text_content,
                    *image_content
                ]
            }
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=3000
            )
            
            response_dict = {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }
                ]
            }
            
            print("Response received successfully")
            return response_dict
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            raise

    def process_pdf_to_descriptions(self, pdf_path, output_folder, batch_size=3):
        """Process PDF to descriptions with configurable batch size."""
        print(f"📄 Processing PDF with batch size: {batch_size}")
        
        image_folder = os.path.join(output_folder, 'images')
        image_files = self.pdf_to_images(pdf_path, image_folder)
        start_slide = 1
        all_descriptions = {}
        previous_response_text = ""
        is_first_batch = True

        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}: slides {start_slide}-{start_slide + len(batch_files) - 1}")
            
            response = self.send_batch_request(batch_files, start_slide, previous_response_text, is_first_batch)
            slide_dict = self.process_response(response)

            previous_response_text = response['choices'][0]['message']['content']
            is_first_batch = False

            all_descriptions.update(slide_dict)
            start_slide += batch_size

        descriptions = [all_descriptions[key] for key in sorted(all_descriptions.keys())]
        descriptions_file = os.path.join(output_folder, "descriptions.txt")
        self.save_descriptions(descriptions, descriptions_file)
        return descriptions_file, image_files

    def pdf_to_images(self, pdf_path, output_folder):
        """Converts a PDF into images with minimum 1920x1080 resolution."""
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count

        os.makedirs(output_folder, exist_ok=True)

        image_paths = []
        min_width, min_height = 1920, 1080
        
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            
            # Get original page dimensions
            page_rect = page.rect
            original_width = page_rect.width
            original_height = page_rect.height
            
            # Calculate zoom to ensure minimum resolution
            zoom_x = min_width / original_width
            zoom_y = min_height / original_height
            zoom = max(zoom_x, zoom_y, 2.0)  # At least 2x zoom for quality
            
            mat = fitz.Matrix(zoom, zoom)
            
            # Get high resolution pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image for resizing
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Ensure minimum dimensions while maintaining aspect ratio
            current_width, current_height = img.size
            
            if current_width < min_width or current_height < min_height:
                scale_x = min_width / current_width
                scale_y = min_height / current_height
                scale = max(scale_x, scale_y)
                
                new_width = int(current_width * scale)
                new_height = int(current_height * scale)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the high-resolution image
            image_path = os.path.join(output_folder, f"slide_{page_num + 1}.png")
            img.save(image_path, "PNG", optimize=True, quality=95)
            image_paths.append(image_path)
            
            print(f"Slide {page_num + 1}: {img.size[0]}x{img.size[1]} pixels")
            
        return image_paths

    # def zalo_text_to_speech(self, text, output_path, speaker_id=4, speed=1.0):
    #     """Convert text to speech using Zalo AI API."""
    #     try:
    #         print(f"🎤 Calling Zalo AI TTS...")
    #         print(f"🎤 Text length: {len(text)} characters")
    #         print(f"🎤 Output path: {output_path}")
            
    #         # Ensure output directory exists
    #         output_dir = os.path.dirname(output_path)
    #         os.makedirs(output_dir, exist_ok=True)
    #         print(f"📁 Ensured directory exists: {output_dir}")
            
    #         # Step 1: Call API using requests (avoid shell escaping issues)
    #         headers = {
    #             'apikey': self.zalo_api_key,
    #             'Content-Type': 'application/x-www-form-urlencoded'
    #         }
            
    #         data = {
    #             'input': text,
    #             'encode_type': 0,  # WAV format
    #             'speaker_id': speaker_id,
    #             'speed': speed
    #         }
            
    #         print(f"🔗 Calling Zalo API...")
    #         response = requests.post('https://api.zalo.ai/v1/tts/synthesize', headers=headers, data=data)
            
    #         if response.status_code != 200:
    #             print(f"❌ API Error: {response.status_code} - {response.text}")
    #             return False
            
    #         response_json = response.json()
            
    #         if 'data' not in response_json or 'url' not in response_json['data']:
    #             print(f"❌ Invalid response format: {response_json}")
    #             return False
            
    #         audio_url = response_json['data']['url']
    #         print(f"✅ Got audio URL: {audio_url}")
            
    #         # Step 2: Download using multiple methods
    #         print(f"📥 Downloading audio...")
            
    #         # Method 1: Try curl with more options
    #         print(f"🔄 Trying curl download...")
    #         download_result = subprocess.run([
    #             'curl', '-L', '-s', '-o', output_path, 
    #             '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    #             '--connect-timeout', '30',
    #             '--max-time', '60',
    #             '--retry', '3',
    #             '--retry-delay', '1',
    #             audio_url
    #         ], capture_output=True, text=True, timeout=90)
            
    #         # Check if curl worked
    #         if download_result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    #             print(f"✅ Curl download successful")
    #         else:
    #             print(f"⚠️ Curl failed: {download_result.stderr}")
    #             print(f"🔄 Trying requests download...")
                
    #             # Method 2: Try requests with headers
    #             try:
    #                 headers = {
    #                     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    #                     'Accept': 'audio/wav,audio/*,*/*',
    #                     'Accept-Language': 'en-US,en;q=0.9',
    #                     'Connection': 'keep-alive'
    #                 }
                    
    #                 download_response = requests.get(audio_url, headers=headers, timeout=60, stream=True)
                    
    #                 if download_response.status_code == 200:
    #                     with open(output_path, 'wb') as f:
    #                         for chunk in download_response.iter_content(chunk_size=8192):
    #                             if chunk:
    #                                 f.write(chunk)
    #                     print(f"✅ Requests download successful")
    #                 else:
    #                     print(f"❌ Requests download failed: {download_response.status_code}")
    #                     return False
                        
    #             except Exception as e:
    #                 print(f"❌ Requests download error: {str(e)}")
    #                 return False
            
    #         # Final check
    #         if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
    #             print(f"❌ Downloaded file is empty or doesn't exist")
    #             return False
            
    #         print(f"✅ Audio saved: {output_path} (WAV format, {os.path.getsize(output_path)} bytes)")
    #         return True
            
    #     except subprocess.TimeoutExpired:
    #         print(f"❌ Download timeout (30s)")
    #         return False
    #     except Exception as e:
    #         print(f"❌ Error in Zalo TTS: {str(e)}")
    #         return False

    def zalo_text_to_speech(self, text, output_path, speaker_id=4, speed=1.0):
        """
        Chuyển đổi văn bản thành giọng nói bằng Zalo AI API với cơ chế thử lại.
        """
        print("--- Zalo TTS Process Started ---")
        print(f"🎤 Text length: {len(text)} characters")
        print(f"🎤 Output path: {output_path}")

        # --- Step 1: Gọi API để lấy URL ---
        try:
            # Đảm bảo thư mục đầu ra tồn tại
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            api_url = 'https://api.zalo.ai/v1/tts/synthesize'
            headers = {
                'apikey': self.zalo_api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            data = {
                'input': text,
                'speaker_id': speaker_id,
                'speed': speed
            }

            print("🔗 Calling Zalo API to get audio URL...")
            response = requests.post(api_url, headers=headers, data=data, timeout=60)

            if response.status_code != 200:
                print(f"❌ API Error: Status {response.status_code} - {response.text}")
                return False

            response_json = response.json()
            if response_json.get("error_code") != 0:
                print(f"❌ API Logic Error: {response_json.get('error_message')}")
                return False

            audio_url = response_json.get("data", {}).get("url")
            if not audio_url:
                print(f"❌ Invalid API response: Could not find audio URL. Response: {response_json}")
                return False
            
            print(f"✅ Got audio URL: {audio_url}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling Zalo API: {e}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred during API call: {e}")
            return False

        # --- Step 2: Tải file âm thanh với cơ chế thử lại ---
        print("📥 Starting audio download...")
        max_retries = 5
        retry_delay_seconds = 2 # Thời gian chờ giữa các lần thử

        for attempt in range(max_retries):
            print(f"🔄 Attempt {attempt + 1} of {max_retries}...")
            try:
                download_headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                download_response = requests.get(audio_url, headers=download_headers, timeout=90, stream=True)

                if download_response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"✅ Download successful! File saved to {output_path}")
                        print("--- Zalo TTS Process Finished ---")
                        return True
                    else:
                        print("⚠️ Download seemed successful, but the file is empty. Retrying...")

                elif download_response.status_code == 404:
                    print("- File not found (404). Server might still be processing. Retrying...")
                
                else:
                    print(f"- Unexpected status code: {download_response.status_code}. Retrying...")

            except requests.exceptions.RequestException as e:
                print(f"❌ A network error occurred during download: {e}. Retrying...")
            
            # Đợi trước khi thử lại (không đợi ở lần cuối cùng)
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)

        print("❌ Download failed after all attempts.")
        print("--- Zalo TTS Process Finished with Error ---")
        return False

    def save_descriptions(self, descriptions, file_path):
        """Saves the descriptions to a text file with XML tags for each slide."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, desc in enumerate(descriptions, 1):
                f.write(f"<slide_{i}>\n{desc}\n</slide_{i}>\n\n")

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to extract from individual slide XML tags
        slide_matches = re.findall(r'<slide_(\d+)>(.*?)</slide_\d+>', content, re.DOTALL)
        if slide_matches:
            # Reconstruct content with #slide# format
            reconstructed = ""
            for slide_num, slide_content in slide_matches:
                reconstructed += f"#slide{slide_num}#\n{slide_content.strip()}\n\n"
            return reconstructed.strip()
        
        # Try old text format
        xml_match = re.search(r'<text>(.*?)</text>', content, re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()
        
        # Fallback to old format if no XML tags found
        return content.strip('full_content = """').strip('"""')

    def write_file(self, file_path, content):
        with open(file_path, 'w', encoding='utf-8') as f:
            # Parse content and write with individual slide XML tags
            slides = re.findall(r'#slide(\d+)#(.*?)(?=#slide\d+#|\Z)', content, re.DOTALL)
            for slide_num, slide_content in slides:
                f.write(f"<slide_{slide_num}>\n{slide_content.strip()}\n</slide_{slide_num}>\n\n")

    def process_with_claude(self, descriptions_file, output_folder):
        full_content = self.read_file(descriptions_file)
        total_slides = len(re.findall(r'#slide\d+#', full_content))

        # Dynamically determine batch sizes
        batch_sizes = []
        remaining_slides = total_slides
        while remaining_slides > 0:
            batch_size = min(10, remaining_slides)
            batch_sizes.append(batch_size)
            remaining_slides -= batch_size

        start = 1
        for i, batch_size in enumerate(batch_sizes, 1):
            end = min(start + batch_size - 1, total_slides)
            print(f"Processing batch {i} (slides {start}-{end})")

            processed_batch = self.process_batch(self.anthropic_client, full_content, start, end, total_slides)

            # Extract the text content from the TextBlock object
            if isinstance(processed_batch, list) and len(processed_batch) > 0 and hasattr(processed_batch[0], 'text'):
                processed_batch = processed_batch[0].text
            elif not isinstance(processed_batch, str):
                processed_batch = str(processed_batch)

            full_content = self.replace_batch(full_content, processed_batch, start, end)
            start = end + 1

        final_context_file = os.path.join(output_folder, "final-context.txt")
        self.write_file(final_context_file, full_content)
        return final_context_file

    def process_batch(self, client, full_content, start, end, total_slides):
        start_time = time.time()
        slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', full_content, re.DOTALL)
        batch = slides[start - 1:end]
        batch_content = '\n'.join(batch)

        prompt = self.create_prompt(batch_content, start, end, total_slides)
        end_time = time.time()
        print("EXECUTION TIME: ",end_time-start_time)
        
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        # Return the text content directly
        if isinstance(message.content, list) and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            content = message.content[0].text
        else:
            content = str(message.content)
        
        # Extract from individual slide XML tags if present
        slide_matches = re.findall(r'<slide_(\d+)>(.*?)</slide_\d+>', content, re.DOTALL)
        if slide_matches:
            # Reconstruct content with #slide# format
            reconstructed = ""
            for slide_num, slide_content in slide_matches:
                reconstructed += f"#slide{slide_num}#\n{slide_content.strip()}\n\n"
            return reconstructed.strip()
        
        # Extract from old text format if present
        xml_match = re.search(r'<text>(.*?)</text>', content, re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()
        
        return content

    def create_prompt(self, batch_content, start, end, total_slides):
        prompt = f"""{batch_content}
Please read the content of these slides carefully and assume the role of a knowledgeable and engaging professor delivering a comprehensive and captivating lecture. Your goal is to deeply understand the meaning and context of each slide, explaining them in a manner that is both thorough and engaging. Rather than merely reading the existing text, provide insightful and detailed explanations, ensuring smooth and natural transitions between the content.
Create seamless and logical transitions that connect each slide to the overall theme of the lecture. Use phrases like, "This concept will be further explored in upcoming slides," or "Keep this idea in mind, as it will be crucial later on," to link different sections and maintain coherence throughout the lecture. Additionally, incorporate natural lecturer comments and anecdotes to make the lecture feel more authentic, relatable, and engaging.
You'll smoothly transition from the concepts covered in slides 1-{start-1} to the advanced topics in slides {start}-{total_slides}. As you know from the first ten slides, [summarize key points from slides 1-{start-1}], it's essential to build upon these foundations to fully grasp the ideas presented in the subsequent slides.

⚠️  CRITICAL: If you want to add silence markers for pauses, use <<<sil#100>>> but:
- NEVER put non-space characters directly before or after <<<sil#100>>>
- WRONG: "<<<sil#100>>>-" or "-<<<sil#100>>>" (will be read as Vietnamese)
- CORRECT: "word <<<sil#100>>> next word" (with spaces before and after)
- For sentence breaks: "sentence. <<<sil#100>>> Next sentence"

IMPORTANT: Please wrap each slide in individual XML tags <slide_1></slide_1>, <slide_2></slide_2>, etc. for easy parsing."""
        return prompt

    def replace_batch(self, full_content, processed_batch, start, end):
        slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', full_content, re.DOTALL)
        new_slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', processed_batch, re.DOTALL)

        for i, new_slide in enumerate(new_slides):
            slide_number = start + i
            if slide_number <= end and slide_number - 1 < len(slides):
                full_content = full_content.replace(slides[slide_number - 1], new_slide)

        return full_content

    def translate_to_vietnamese(self, descriptions_file, output_folder):
        """Translates the descriptions to Vietnamese using Gemini API."""
        print("🌐 Starting translation to Vietnamese using Gemini...")
        
        full_content = self.read_file(descriptions_file)
        
        try:
            # Create translation prompt
            translation_prompt = f'''Dịch toàn bộ nội dung sau sang tiếng Việt, giữ nguyên format với các thẻ #slide#.

QUAN TRỌNG:
- Dịch nội dung sang tiếng Việt tự nhiên, phù hợp cho bài giảng
- Khi gặp công thức toán học, hãy tạo lại chính xác cách đọc công thức thành văn đọc tiếng Việt
  * Ví dụ: "Â_i = (r_i - μ)/σ" → "Â i bằng r i trừ miu, tất cả chia sigma"
- Tự động nhận diện các nơi cần ngắt hơi (im lặng), và thêm ' <<<sil#100>>> ' vào những nơi đó
- Rút gọn nội dung nhưng giữ đủ ý nghĩa quan trọng
- Đảm bảo nội dung phù hợp cho Text-to-Speech (TTS)
- BẮT BUỘC: Wrap từng slide riêng biệt trong XML tags <slide_1></slide_1>, <slide_2></slide_2>, v.v.

⚠️  LỖI NGHIÊM TRỌNG CẦN TRÁNH KHI THÊM NHỊP NGỪNG:
- KHÔNG ĐƯỢC có ký tự không phải khoảng trắng trước hoặc sau <<<sil#100>>>
- VÍ DỤ SAI: "<<<sil#100>>>-" hoặc "-<<<sil#100>>>" sẽ bị đọc như tiếng Việt
- VÍ DỤ ĐÚNG: "từ đầu <<<sil#100>>> từ tiếp theo" (có khoảng trắng trước và sau)
- Nếu cần ngắt câu, dùng: "câu trước. <<<sil#100>>> Câu sau"

Nội dung cần dịch:
{full_content}

Vui lòng trả về kết quả trong format:
<slide_1>
[nội dung dịch slide 1]
</slide_1>

<slide_2>
[nội dung dịch slide 2]
</slide_2>
...'''
            
            # Call Gemini API
            print("🤖 Calling Gemini API for translation...")
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=translation_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            
            if not response.text:
                print("❌ Empty response from Gemini API")
                return self.simple_translate_to_vietnamese(descriptions_file, output_folder)
            
            translated_content = response.text
            
            # Save translated content
            translated_file = os.path.join(output_folder, "translated_descriptions.txt")
            with open(translated_file, "w", encoding="utf-8") as file:
                file.write(translated_content)
            
            print(f"✅ Vietnamese translation completed successfully!")
            print(f"📁 Saved to: {translated_file}")
            
            return translated_file
            
        except Exception as e:
            print(f"❌ Error during Gemini translation: {str(e)}")
            print("🔄 Falling back to simple translation...")
            return self.simple_translate_to_vietnamese(descriptions_file, output_folder)

    def simple_translate_to_vietnamese(self, descriptions_file, output_folder):
        """Simple translation fallback - just copy the file for now."""
        print("⚠️  Using simple translation fallback...")
        
        full_content = self.read_file(descriptions_file)
        
        # Extract slides and wrap in individual XML tags
        slides = re.findall(r'#slide(\d+)#(.*?)(?=#slide\d+#|\Z)', full_content, re.DOTALL)
        
        translated_file = os.path.join(output_folder, "translated_descriptions.txt")
        with open(translated_file, "w", encoding="utf-8") as file:
            for slide_num, slide_content in slides:
                file.write(f"<slide_{slide_num}>\n{slide_content.strip()}\n</slide_{slide_num}>\n\n")
        
        print(f"✅ Simple translation saved to: {translated_file}")
        return translated_file

    def generate_vietnamese_audio_sequential(self, final_context_file, output_folder):
        """Generate Vietnamese audio from context file using sequential processing."""
        with open(final_context_file, 'r', encoding='utf-8') as f:
            final_context = f.read()

        # Translate to Vietnamese
        print("🌐 Starting translation process...")
        translated_file = self.translate_to_vietnamese(final_context_file, output_folder)
        
        # Extract Vietnamese descriptions
        print("📝 Extracting Vietnamese descriptions...")
        with open(translated_file, 'r', encoding='utf-8') as f:
            translated_content = f.read()
        vietnamese_descriptions = self.extract_slide_descriptions(translated_content)

        # Prepare audio generation
        audio_folder = os.path.join(output_folder, 'audio')
        os.makedirs(audio_folder, exist_ok=True)
        
        print(f"🎤 Starting sequential TTS generation...")
        print(f"📊 Processing {len(vietnamese_descriptions)} slides...")
        print(f"🎵 Audio format: WAV (encode_type=0)")
        
        # Process slides sequentially
        audio_files = []
        successful_slides = 0
        failed_slides = 0
        
        for i, description in enumerate(vietnamese_descriptions):
            slide_number = i + 1
            output_path = os.path.join(audio_folder, f'slide_{slide_number}.wav')
            
            print(f"🎤 Processing slide {slide_number}/{len(vietnamese_descriptions)}...")
            
            success = self.zalo_text_to_speech(description, output_path)
            
            if success:
                successful_slides += 1
                print(f"✅ Slide {slide_number} completed successfully")
            else:
                failed_slides += 1
                print(f"❌ Slide {slide_number} failed")
                # Create fallback audio
                self.create_silent_audio(output_path, duration=5.0)
            
            audio_files.append(output_path)
        
        print(f"\n📊 TTS Generation Summary:")
        print(f"   ✅ Successful: {successful_slides}")
        print(f"   ❌ Failed: {failed_slides}")
        print(f"   📁 Total files: {len(audio_files)}")
        
        return audio_files, vietnamese_descriptions, translated_file

    def create_silent_audio(self, filename, duration=5.0, rate=44100):
        """Creates a silent audio file as fallback."""
        import numpy as np
        from scipy.io import wavfile
        
        try:
            # Create silent audio data
            samples = int(duration * rate)
            silent_data = np.zeros(samples, dtype=np.int16)
            
            # Save as WAV
            wavfile.write(filename, rate, silent_data)
            
            print(f"🔇 Created silent audio: {filename}")
            
        except Exception as e:
            print(f"❌ Error creating silent audio: {str(e)}")
            # Create minimal file
            with open(filename, 'wb') as f:
                f.write(b'')

    def extract_slide_descriptions(self, final_context):
        """Extract slide descriptions from final context with XML tags."""
        # First try to extract from individual slide XML tags
        slide_matches = re.findall(r'<slide_(\d+)>(.*?)</slide_\d+>', final_context, re.DOTALL)
        if slide_matches:
            # Sort by slide number and return descriptions
            slide_matches.sort(key=lambda x: int(x[0]))
            return [desc.strip() for _, desc in slide_matches]
        
        # Try old text format
        xml_match = re.search(r'<text>(.*?)</text>', final_context, re.DOTALL)
        if xml_match:
            content = xml_match.group(1).strip()
        else:
            content = final_context
        
        # Then extract slide descriptions using #slide# format
        slide_descriptions = re.findall(r'#(?:slide|Trình)\s*\d+#(.*?)(?=#(?:slide|Trình)\s*\d+#|\Z)', content, re.DOTALL)
        return [desc.strip() for desc in slide_descriptions]

    def create_video_from_context(self, final_context_file, image_files, output_folder):
        """Create video from context with sequential TTS processing."""
        print("🎤 Starting Vietnamese audio generation...")
        audio_files, vietnamese_descriptions, translated_file = self.generate_vietnamese_audio_sequential(
            final_context_file, output_folder
        )
        
        # Create video with Vietnamese audio using GPU acceleration
        video_path = os.path.join(output_folder, "final_video.mp4")
        durations = self.create_video(image_files, audio_files, video_path)
        
        return video_path, vietnamese_descriptions, durations

    def create_video(self, image_files, audio_files, output_file, fps=24):
        """Tạo video từ hình ảnh và file âm thanh WAV."""
        clips = []
        durations = []
        
        print(f"🎬 Creating video from {len(image_files)} images and {len(audio_files)} WAV audio files...")
        
        # Phần xử lý clip vẫn giữ nguyên
        for i, (image_file, audio_file) in enumerate(zip(image_files, audio_files)):
            print(f"📽️  Processing clip {i+1}/{len(image_files)}...")
            try:
                audio = AudioFileClip(audio_file)
                duration = audio.duration
                durations.append(duration)
                
                # Mở ảnh và chuyển thành numpy array
                with Image.open(image_file) as img:
                    img_array = np.array(img)

                img_clip = ImageSequenceClip([img_array], durations=[duration])
                img_clip = img_clip.with_audio(audio)
                img_clip = img_clip.with_fps(fps)
                clips.append(img_clip)
                
                print(f"   ✅ Clip {i+1}: {duration:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Error processing clip {i+1}: {str(e)}")
                # Tạo clip dự phòng nếu có lỗi
                with Image.open(image_file) as img:
                    img_array = np.array(img)
                fallback_duration = 5.0
                img_clip = ImageSequenceClip([img_array], durations=[fallback_duration])
                img_clip = img_clip.set_fps(fps)
                clips.append(img_clip)
                durations.append(fallback_duration)

        if clips:
            final_clip = concatenate_videoclips(clips, method="compose")
            
            # --- THAY ĐỔI QUAN TRỌNG Ở ĐÂY ---
            # Cấu hình GPU acceleration ĐÚNG
            gpu_params = {
                # Chọn codec mã hóa bằng GPU NVIDIA.
                'codec': 'h264_nvenc', 
                'audio_codec': 'aac',
                'fps': fps,
                # Preset 'fast' được hỗ trợ bởi NVENC
                'preset': 'fast', 
                'ffmpeg_params': [
                    # XÓA '-hwaccel' và '-crf'
                    '-movflags', '+faststart', # Giữ lại để xem video online tốt hơn
                ]
            }
            
            # Cấu hình CPU để dự phòng
            cpu_params = {
                'codec': 'libx264',
                'audio_codec': 'aac',
                'fps': fps,
                'preset': 'fast',
                'threads': os.cpu_count() # Tận dụng tất cả các luồng CPU
            }
            
            try:
                print("🚀 Attempting video creation with GPU acceleration...")
                final_clip.write_videofile(output_file, **gpu_params)
                print("✅ Video created successfully with GPU acceleration!")
            except Exception as e:
                print(f"⚠️ GPU acceleration failed: {e}")
                print("🔄 Falling back to CPU encoding...")
                final_clip.write_videofile(output_file, **cpu_params)
                print("✅ Video created successfully with CPU!")
        else:
            print("❌ No clips to concatenate")

        return durations

    def create_random_output_folder(self, base_output_folder):
        """Create a random subfolder for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        folder_name = f"run_{timestamp}_{random_id}"
        
        full_output_path = os.path.join(base_output_folder, folder_name)
        os.makedirs(full_output_path, exist_ok=True)
        
        print(f"📁 Created output folder: {full_output_path}")
        return full_output_path

    def process_pdf_to_video(self, pdf_path, output_folder, pdf_batch_size=3):
        """Complete workflow: Process PDF to video with sequential TTS processing."""
        print("🚀 Starting PDF to Video conversion workflow")
        print(f"📄 PDF batch size: {pdf_batch_size}")
        print(f"🎤 TTS processing: Sequential (1 thread)")
        print(f"🎬 Video encoding: GPU accelerated")
        print("=" * 60)
        
        # Step 1: Process PDF to descriptions
        print("Step 1: Processing PDF to descriptions...")
        descriptions_file, image_files = self.process_pdf_to_descriptions(pdf_path, output_folder, pdf_batch_size)
        print(f"✅ Found {len(image_files)} slides")
        
        # Step 2: Process with Claude
        print("\nStep 2: Processing with Claude...")
        final_context_file = self.process_with_claude(descriptions_file, output_folder)
        
        # Step 3: Create video with sequential TTS
        print(f"\nStep 3: Creating video with sequential TTS...")
        video_path, vietnamese_descriptions, durations = self.create_video_from_context(
            final_context_file, image_files, output_folder
        )
        
        print("\n" + "=" * 60)
        print("🎉 Workflow completed successfully!")
        print(f"📊 Total slides: {len(image_files)}")
        print(f"🎤 TTS processing: Sequential")
        print(f"⏱️  Total video duration: {sum(durations):.2f}s")
        print(f"📁 Output folder: {output_folder}")
        print(f"🎥 Final video: {video_path}")
        
        return video_path, vietnamese_descriptions, durations

def main():
    from config import Config
    
    # Load configuration
    config = Config()
    
    # Get API keys from config
    openai_api_key, anthropic_api_key, gemini_api_key, zalo_api_key = config.get_api_keys()
    
    # Validate API keys
    if not openai_api_key or not anthropic_api_key or not gemini_api_key or not zalo_api_key:
        print("❌ Error: API keys not found! Please check your .env file.")
        return
        
    processor = ZaloGPTProcessor(openai_api_key, anthropic_api_key, gemini_api_key, zalo_api_key)

    pdf_path = '/Users/twang/Downloads/Week 1 - Summary copy.pdf'
    base_output_folder = "/Users/twang/PycharmProjects/transition_test/[AIVIETNAM]"
    
    output_folder = processor.create_random_output_folder(base_output_folder)
    print(f"🔄 Processing PDF: {pdf_path}")
    print(f"📂 Output directory: {output_folder}")

    pdf_batch_size = 5
    
    print(f"⚙️  Configuration:")
    print(f"   📄 PDF batch size: {pdf_batch_size}")
    print(f"   🎤 TTS processing: Sequential (1 thread)")
    print(f"   🎬 Video encoding: GPU accelerated")
    print()

    start_time = time.time()
    
    # Process PDF to video
    video_path, vietnamese_descriptions, durations = processor.process_pdf_to_video(
        pdf_path, output_folder, pdf_batch_size
    )
    
    end_time = time.time()
    
    print(f"\n⏱️  Total processing time: {(end_time - start_time):.2f} seconds")
    print(f"📊 Number of slides: {len(vietnamese_descriptions)}")
    print(f"🎵 Total video duration: {sum(durations):.2f} seconds")
    print(f"🎥 Final video created at: {video_path}")
    print(f"\n✅ Processing completed!")
    print(f"📁 All files saved in: {output_folder}")

if __name__ == "__main__":
    main() 