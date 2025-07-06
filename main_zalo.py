import os
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"

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

import moviepy.config as mpycfg
print("🔍 Using FFmpeg from:", mpycfg.FFMPEG_BINARY)

class TimeTracker:
    """Class to track timing for different processing steps"""
    
    def __init__(self):
        self.times = {}
        self.step_counts = {}
        self.current_step_start = {}
    
    def start_step(self, step_name):
        """Start timing a step"""
        self.current_step_start[step_name] = time.time()
        if step_name not in self.times:
            self.times[step_name] = 0
            self.step_counts[step_name] = 0
    
    def end_step(self, step_name):
        """End timing a step and add to total"""
        if step_name in self.current_step_start:
            duration = time.time() - self.current_step_start[step_name]
            self.times[step_name] += duration
            self.step_counts[step_name] += 1
            del self.current_step_start[step_name]
            return duration
        return 0
    
    def get_summary(self):
        """Get timing summary"""
        summary = {}
        for step, total_time in self.times.items():
            count = self.step_counts[step]
            summary[step] = {
                'total_time': total_time,
                'count': count,
                'avg_time': total_time / count if count > 0 else 0
            }
        return summary
    
    def print_summary(self):
        """Print detailed timing summary"""
        print("\n" + "=" * 80)
        print("⏱️  DETAILED TIMING ANALYSIS")
        print("=" * 80)
        
        summary = self.get_summary()
        total_time = sum(data['total_time'] for data in summary.values())
        
        for step, data in summary.items():
            percentage = (data['total_time'] / total_time * 100) if total_time > 0 else 0
            print(f"🔄 {step}:")
            print(f"   ⏰ Total: {data['total_time']:.2f}s ({percentage:.1f}%)")
            print(f"   🔢 Count: {data['count']}")
            print(f"   📊 Average: {data['avg_time']:.2f}s per operation")
            print()
        
        print(f"🏁 TOTAL PROCESSING TIME: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print("=" * 80)

class ZaloGPTProcessor:
    def __init__(self, openai_api_key, anthropic_api_key, gemini_api_key, zalo_api_key):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        self.zalo_api_key = zalo_api_key
        
        # Initialize time tracker
        self.timer = TimeTracker()
        
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

    def process_response(self, json_response, expected_start_slide, expected_count):
        """Process GPT response using position-based logic instead of tag numbers."""
        content = json_response['choices'][0]['message']['content']
        
        # Extract all slide content in order of appearance
        slide_pattern = r'#slide\d+#(.*?)(?=#slide\d+#|\Z)'
        slide_contents = re.findall(slide_pattern, content, re.DOTALL)
        
        # Map to correct slide numbers based on position, not tag numbers
        slide_dict = {}
        for i, slide_content in enumerate(slide_contents):
            correct_slide_number = expected_start_slide + i
            slide_dict[correct_slide_number] = slide_content.strip()
            
            # Stop if we have enough slides
            if len(slide_dict) >= expected_count:
                break
        
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

        gpt_model = "gpt-4o-mini"
        print(f"Using model: {gpt_model}")
        try:
            response = self.openai_client.chat.completions.create(
                model=gpt_model,
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
            
            return response_dict
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            raise

    def process_pdf_to_descriptions(self, pdf_path, output_folder, batch_size=3):
        """Process PDF to descriptions with configurable batch size."""
        # Start timing for entire Step 1
        self.timer.start_step("Step 1: PDF Processing (Convert + GPT Analysis)")
        
        print(f"📄 Processing PDF with batch size: {batch_size}")
        
        image_folder = os.path.join(output_folder, 'images')
        image_files = self.pdf_to_images(pdf_path, image_folder)
        
        start_slide = 1
        all_descriptions = {}
        previous_response_text = ""
        is_first_batch = True

        print()
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            print(f"🤖 Processing batch {i//batch_size + 1}: slides {start_slide}-{start_slide + len(batch_files) - 1}")
            
            response = self.send_batch_request(batch_files, start_slide, previous_response_text, is_first_batch)
            slide_dict = self.process_response(response, start_slide, len(batch_files))

            previous_response_text = response['choices'][0]['message']['content']
            is_first_batch = False

            all_descriptions.update(slide_dict)
            start_slide += batch_size

        # Sort and validate descriptions by slide number
        sorted_keys = sorted(all_descriptions.keys())
        descriptions = [all_descriptions[key] for key in sorted_keys]
        
        print(f"📋 Organizing content: Found {len(descriptions)} slides")
        print(f"📋 Slide range: {min(sorted_keys)} to {max(sorted_keys)}")
        
        # Create properly formatted content for next steps
        formatted_content = ""
        for i, (slide_num, desc) in enumerate(zip(sorted_keys, descriptions)):
            formatted_content += f"#slide{slide_num}#\n{desc}\n\n"
        
        descriptions_file = os.path.join(output_folder, "descriptions.txt")
        with open(descriptions_file, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        # End timing for Step 1
        step1_duration = self.timer.end_step("Step 1: PDF Processing (Convert + GPT Analysis)")
        print(f"✅ Step 1 completed in {step1_duration:.2f}s - Content organized and saved")
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

        print(f"PDF to Images Conversion completed in {self.timer.end_step('0. PDF to Images Conversion'):.2f}s")
            
        return image_paths

    def zalo_text_to_speech(self, text, output_path, speaker_id=4, speed=1.0):
        """
        Chuyển đổi văn bản thành giọng nói bằng Zalo AI API với cơ chế thử lại.
        """

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

            response = requests.post(api_url, headers=headers, data=data, timeout=60)

            if response.status_code != 200:
                print(f"❌ Zalo TTS API error: HTTP {response.status_code}")
                return False

            response_json = response.json()
            if response_json.get("error_code") != 0:
                error_message = response_json.get("error_message", "Unknown error")
                print(f"❌ Zalo TTS API error: {error_message} (code: {response_json.get('error_code')})")
                return False

            audio_url = response_json.get("data", {}).get("url")
            if not audio_url:
                print(f"❌ Zalo TTS API error: No audio URL returned")
                return False

        except Exception as e:
            print(f"❌ Zalo TTS API connection error: {str(e)}")
            return False

        # --- Step 2: Tải file âm thanh với cơ chế thử lại ---
        max_retries = 5
        retry_delay_seconds = 2

        for attempt in range(max_retries):
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
                        return True
                
                # Log download error if not 200
                if attempt == max_retries - 1:  # Only print on last attempt
                    print(f"❌ Zalo TTS download error: HTTP {download_response.status_code}")

                # Đợi trước khi thử lại (không đợi ở lần cuối cùng)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_seconds)

            except Exception as e:
                if attempt == max_retries - 1:  # Only print on last attempt
                    print(f"❌ Zalo TTS download error: {str(e)}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_seconds)

        print(f"❌ Zalo TTS failed after {max_retries} attempts: {os.path.basename(output_path)}")
        return False

    def sort_slide_content(self, content):
        """Sort slide content by slide number to maintain logical order."""
        # Extract all slides with their numbers
        slide_pattern = r'(#slide(\d+)#.*?)(?=#slide\d+#|\Z)'
        matches = re.findall(slide_pattern, content, re.DOTALL)
        
        if not matches:
            return content
        
        # Sort by slide number
        sorted_slides = sorted(matches, key=lambda x: int(x[1]))
        
        # Reconstruct content in correct order
        sorted_content = ""
        for slide_content, slide_num in sorted_slides:
            sorted_content += slide_content + "\n"
        
        return sorted_content.strip()

    def sort_image_files(self, image_files):
        """Sort image files by slide number."""
        def extract_slide_number(filename):
            # Extract number from filename like 'slide_1.png' or '/path/slide_1.png'
            match = re.search(r'slide_(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        sorted_files = sorted(image_files, key=extract_slide_number)
        return sorted_files

    def sort_audio_files(self, audio_files):
        """Sort audio files by slide number."""
        def extract_slide_number(filename):
            # Extract number from filename like 'slide_1.wav' or '/path/slide_1.wav'
            match = re.search(r'slide_(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        sorted_files = sorted(audio_files, key=extract_slide_number)
        return sorted_files

    def save_descriptions(self, descriptions, file_path):
        """Saves the descriptions to a text file with XML tags for each slide."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, desc in enumerate(descriptions, 1):
                f.write(f"<slide_{i}>\n{desc}\n</slide_{i}>\n\n")

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to extract from individual slide XML tags (position-based)
        xml_contents = re.findall(r'<slide_\d+>(.*?)</slide_\d+>', content, re.DOTALL)
        if xml_contents:
            # Reconstruct content with sequential #slide# tags based on position
            reconstructed = ""
            for i, slide_content in enumerate(xml_contents, 1):
                reconstructed += f"#slide{i}#\n{slide_content.strip()}\n\n"
            return reconstructed.strip()
        
        # Try old text format
        xml_match = re.search(r'<text>(.*?)</text>', content, re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()
        
        # Fallback to old format if no XML tags found
        return content.strip('full_content = """').strip('"""')

    def write_file(self, file_path, content):
        with open(file_path, 'w', encoding='utf-8') as f:
            # Extract slide contents by position, ignoring tag numbers
            slide_contents = re.findall(r'#slide\d+#(.*?)(?=#slide\d+#|\Z)', content, re.DOTALL)
            
            # Write with sequential numbers based on position
            for i, slide_content in enumerate(slide_contents, 1):
                f.write(f"<slide_{i}>\n{slide_content.strip()}\n</slide_{i}>\n\n")

    def process_with_claude(self, descriptions_file, output_folder):
        # Start timing for Step 2
        self.timer.start_step("Step 2: Claude Content Enhancement")
        
        full_content = self.read_file(descriptions_file)
        
        # Sort content by slide number before processing
        full_content = self.sort_slide_content(full_content)
        total_slides = len(re.findall(r'#slide\d+#', full_content))
        
        print(f"📋 Content organized: {total_slides} slides ready for Claude enhancement")

        # Dynamically determine batch sizes
        batch_sizes = []
        remaining_slides = total_slides
        while remaining_slides > 0:
            batch_size = min(10, remaining_slides)
            batch_sizes.append(batch_size)
            remaining_slides -= batch_size

        print()
        start = 1
        for i, batch_size in enumerate(batch_sizes, 1):
            end = min(start + batch_size - 1, total_slides)           
            print(f"Claude processing batch {i} (slides {start}-{end})")

            processed_batch = self.process_batch(self.anthropic_client, full_content, start, end, total_slides)

            # Extract the text content from the TextBlock object
            if isinstance(processed_batch, list) and len(processed_batch) > 0 and hasattr(processed_batch[0], 'text'):
                processed_batch = processed_batch[0].text
            elif not isinstance(processed_batch, str):
                processed_batch = str(processed_batch)

            full_content = self.replace_batch(full_content, processed_batch, start, end)
            
            # Sort content after each batch to maintain order
            full_content = self.sort_slide_content(full_content)
            start = end + 1

        # Final sort before saving
        full_content = self.sort_slide_content(full_content)
        
        # End timing for Step 2
        step2_duration = self.timer.end_step("Step 2: Claude Content Enhancement")
        print(f"✅ Step 2 completed in {step2_duration:.2f}s - Content enhanced by Claude")
        
        final_context_file = os.path.join(output_folder, "final-context.txt")
        self.write_file(final_context_file, full_content)
        return final_context_file

    def process_batch(self, client, full_content, start, end, total_slides):
        slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', full_content, re.DOTALL)
        batch = slides[start - 1:end]
        batch_content = '\n'.join(batch)

        prompt = self.create_prompt(batch_content, start, end, total_slides)

        claude_model = "claude-sonnet-4-20250514"
        print(f"Using model: {claude_model}")
        message = client.messages.create(
            model=claude_model,
            max_tokens=8000,
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

IMPORTANT: Please wrap each slide in individual XML tags in order: <slide_1></slide_1>, <slide_2></slide_2>, <slide_3></slide_3>, etc. The numbers should match the order of slides you process, starting from slide_1 for the first slide in this batch."""
        return prompt

    def replace_batch(self, full_content, processed_batch, start, end):
        """Replace batch content using position-based logic."""
        # Get original slides in order
        slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', full_content, re.DOTALL)
        
        # Extract new slide contents (ignore tag numbers, use position)
        new_slide_contents = re.findall(r'#slide\d+#(.*?)(?=#slide\d+#|\Z)', processed_batch, re.DOTALL | re.IGNORECASE)
        if not new_slide_contents:
            # Fallback: try XML format
            new_slide_contents = re.findall(r'<slide_\d+>(.*?)</slide_\d+>', processed_batch, re.DOTALL | re.IGNORECASE)
        
        # Replace slides by position, not by tag numbers
        for i, new_content in enumerate(new_slide_contents):
            slide_position = start + i - 1  # Convert to 0-based index
            if slide_position < len(slides) and (start + i) <= end:
                # Create new slide with correct number
                correct_slide_number = start + i
                new_slide = f"#slide{correct_slide_number}#\n{new_content.strip()}\n"
                full_content = full_content.replace(slides[slide_position], new_slide)

        return full_content

    def translate_to_vietnamese(self, descriptions_file, output_folder):
        """Translates the descriptions to Vietnamese using Gemini API."""
        # Note: Timing for Step 3 will be handled in generate_vietnamese_audio_sequential
        
        full_content = self.read_file(descriptions_file)
        
        # Sort content before translation to ensure logical order
        full_content = self.sort_slide_content(full_content)
        total_slides = len(re.findall(r'#slide\d+#', full_content))
        print(f"📋 Content organized for translation: {total_slides} slides")
        
        try:
            # Create translation prompt
            translation_prompt = f'''Dịch toàn bộ nội dung sau sang tiếng Việt, giữ nguyên format với các thẻ #slide#.

QUAN TRỌNG:
- Dịch nội dung sang tiếng Việt tự nhiên, phù hợp cho bài giảng
- Khi gặp công thức toán học, hãy tạo lại chính xác cách đọc công thức thành văn đọc tiếng Việt ngắn gọn
  * Ví dụ: "Â_i = (r_i - μ)/σ" → "Â mũ i bằng hiệu của r i và miu, tất cả chia sigma"
  * Ví dụ: "MSE = 1/n * sum_(i=1)^n (y_i - y_hat_i)^2" → "MSE bằng một phần n, nhân với tổng sigma của mở ngoặc y i trừ y mũ i đóng ngoặc, tất cả bình phương, trong đó i chạy từ một đến n"
- Tự động nhận diện các nơi cần ngắt hơi (im lặng), và thêm ' <<<sil#100>>> ' vào những nơi đó
- Rút gọn nội dung nhưng giữ đủ ý nghĩa quan trọng
- Đảm bảo nội dung phù hợp cho Text-to-Speech (TTS)
- BẮT BUỘC: Wrap từng slide riêng biệt trong XML tags theo thứ tự: <slide_1></slide_1>, <slide_2></slide_2>, <slide_3></slide_3>, v.v. Đánh số theo thứ tự xuất hiện từ 1 đến N

LỖI NGHIÊM TRỌNG CẦN TRÁNH KHI THÊM NHỊP NGỪNG:
- KHÔNG ĐƯỢC có ký tự không phải khoảng trắng trước hoặc sau <<<sil#100>>>
- VÍ DỤ SAI: "<<<sil#100>>>-" hoặc "-<<<sil#100>>>" sẽ bị đọc như tiếng Việt
- VÍ DỤ ĐÚNG: "từ đầu <<<sil#100>>> từ tiếp theo" (có khoảng trắng trước và sau)
- Nếu cần ngắt câu, dùng: "câu trước. <<<sil#100>>> Câu sau"

Nội dung cần dịch:
{full_content}

Vui lòng trả về kết quả trong format (đánh số tuần tự từ 1):
<slide_1>
[nội dung dịch slide đầu tiên]
</slide_1>

<slide_2>
[nội dung dịch slide thứ hai]
</slide_2>

<slide_3>
[nội dung dịch slide thứ ba]
</slide_3>
...và tiếp tục cho đến hết.'''
            
            # Call Gemini API
            gemini_model = "gemini-2.5-flash"
            print(f"Using model: {gemini_model}")
            response = self.gemini_client.models.generate_content(
                model=gemini_model,
                contents=translation_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            
            if not response.text:
                print("❌ Empty response from Gemini API")
                return None
            
            translated_content = response.text
            
            # Save translated content
            translated_file = os.path.join(output_folder, "translated_descriptions.txt")
            with open(translated_file, "w", encoding="utf-8") as file:
                file.write(translated_content)
            
            return translated_file
            
        except Exception as e:
            print(f"❌ Error during Gemini translation: {str(e)}")
            return None


    def generate_vietnamese_audio_sequential(self, final_context_file, output_folder):
        """Generate Vietnamese audio from context file using sequential processing."""
        with open(final_context_file, 'r', encoding='utf-8') as f:
            final_context = f.read()

        # Step 3: Translate to Vietnamese
        print("\nStep 3: Processing translate to Vietnamese...")
        self.timer.start_step("Step 3: Gemini Vietnamese Translation")
        
        translated_file = self.translate_to_vietnamese(final_context_file, output_folder)
        if not translated_file:
            print("❌ Translation failed")
            return [], [], None
        
        step3_duration = self.timer.end_step("Step 3: Gemini Vietnamese Translation")
        print(f"✅ Step 3 completed in {step3_duration:.2f}s - Content translated to Vietnamese")
        
        # Extract Vietnamese descriptions and ensure proper order
        with open(translated_file, 'r', encoding='utf-8') as f:
            translated_content = f.read()
        vietnamese_descriptions = self.extract_slide_descriptions(translated_content)
        
        print(f"📋 Audio generation order: {len(vietnamese_descriptions)} slides organized")

        # Prepare audio generation
        audio_folder = os.path.join(output_folder, 'audio')
        os.makedirs(audio_folder, exist_ok=True)
        
        # Process slides sequentially
        print("\nStep 4: Processing TTS...")
        self.timer.start_step("Step 4: Zalo AI TTS Processing")        
        audio_files = []
        successful_slides = 0
        failed_slides = 0
        
        total_slides = len(vietnamese_descriptions)
        print(f"🎤 Generating audio for {total_slides} slides...")
        
        for i, description in enumerate(vietnamese_descriptions):
            slide_number = i + 1
            output_path = os.path.join(audio_folder, f'slide_{slide_number}.wav')
            
            # Progress bar
            progress = (i + 1) / total_slides * 100
            bar_length = 30
            filled_length = int(bar_length * (i + 1) // total_slides)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r🎤 Progress: [{bar}] {progress:.1f}% ({i + 1}/{total_slides})', end='', flush=True)
            
            success = self.zalo_text_to_speech(description, output_path)
            
            if success:
                successful_slides += 1
            else:
                failed_slides += 1
                # Create fallback audio
                self.create_silent_audio(output_path, duration=5.0)
            
            audio_files.append(output_path)
        
        print()  # New line after progress bar
        
        # End TTS timing for Step 4
        step4_duration = self.timer.end_step("Step 4: Zalo AI TTS Processing")
        
        print(f"✅ Step 4 completed in {step4_duration:.2f}s - TTS: {successful_slides} success, {failed_slides} failed")
        
        return audio_files, vietnamese_descriptions, translated_file

    def create_silent_audio(self, filename, duration=5.0, rate=44100):
        """Creates a silent audio file as fallback."""
        try:
            import numpy as np
            from scipy.io import wavfile
            
            # Create silent audio data
            samples = int(duration * rate)
            silent_data = np.zeros(samples, dtype=np.int16)
            
            # Save as WAV
            wavfile.write(filename, rate, silent_data)
            
        except:
            # Create minimal file
            with open(filename, 'wb') as f:
                f.write(b'')

    def extract_slide_descriptions(self, final_context):
        """Extract slide descriptions using position-based logic, ignoring tag numbers."""
        
        # Method 1: Try XML format first (position-based)
        xml_slides = re.findall(r'<slide_\d+>(.*?)</slide_\d+>', final_context, re.DOTALL)
        if xml_slides:
            descriptions = [desc.strip() for desc in xml_slides]
            return descriptions
        
        # Method 2: Try #slide# format (position-based)
        slide_contents = re.findall(r'#(?:slide|Trình)\s*\d+#(.*?)(?=#(?:slide|Trình)\s*\d+#|\Z)', final_context, re.DOTALL)
        if slide_contents:
            descriptions = [desc.strip() for desc in slide_contents]
            return descriptions
        
        # Method 3: Try old text format
        xml_match = re.search(r'<text>(.*?)</text>', final_context, re.DOTALL)
        if xml_match:
            content = xml_match.group(1).strip()
            slide_contents = re.findall(r'#(?:slide|Trình)\s*\d+#(.*?)(?=#(?:slide|Trình)\s*\d+#|\Z)', content, re.DOTALL)
            if slide_contents:
                descriptions = [desc.strip() for desc in slide_contents]
                return descriptions
        
        # Fallback: return empty if no pattern matches
        print(f"⚠️ Could not extract slides from content")
        return []

    def create_video_from_context(self, final_context_file, image_files, output_folder):
        """Create video from context with sequential TTS processing."""
        audio_files, vietnamese_descriptions, translated_file = self.generate_vietnamese_audio_sequential(
            final_context_file, output_folder
        )
        
        # Ensure files are in correct order for video creation
        print("\nStep 5: Processing video creation...")
        self.timer.start_step("Step 5: Video Creation & Encoding")
        
        sorted_image_files = self.sort_image_files(image_files)
        sorted_audio_files = self.sort_audio_files(audio_files)
        
        # Validate file counts match
        if len(sorted_image_files) != len(sorted_audio_files):
            print(f"⚠️ File count mismatch: {len(sorted_image_files)} images vs {len(sorted_audio_files)} audio files")
        
        # Create video with Vietnamese audio using GPU acceleration
        video_path = os.path.join(output_folder, "final_video.mp4")
        durations = self.create_video(sorted_image_files, sorted_audio_files, video_path)
        
        step5_duration = self.timer.end_step("Step 5: Video Creation & Encoding")
        print(f"✅ Step 5 completed in {step5_duration:.2f}s")
        
        return video_path, vietnamese_descriptions, durations

    def create_video(self, image_files, audio_files, output_file, fps=24):
        """Tạo video từ hình ảnh và file âm thanh WAV."""
        clips = []
        durations = []
        
        total_clips = len(image_files)
        successful_clips = 0
        failed_clips = 0
        
        print(f"🎬 Creating video from {total_clips} clips...")
        
        # Phần xử lý clip với progress bar
        for i, (image_file, audio_file) in enumerate(zip(image_files, audio_files)):
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
                
                successful_clips += 1
                
            except Exception as e:
                failed_clips += 1
                # Tạo clip dự phòng nếu có lỗi
                with Image.open(image_file) as img:
                    img_array = np.array(img)
                fallback_duration = 5.0
                img_clip = ImageSequenceClip([img_array], durations=[fallback_duration])
                img_clip = img_clip.with_fps(fps)
                clips.append(img_clip)
                durations.append(fallback_duration)
            
            # Progress bar sau khi xử lý (không bị gián đoạn bởi exception)
            progress = (i + 1) / total_clips * 100
            bar_length = 30
            filled_length = int(bar_length * (i + 1) // total_clips)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r🎬 Progress: [{bar}] {progress:.1f}% ({i + 1}/{total_clips})', end='', flush=True)
        
        print()  # New line after progress bar

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
                print("🚀 Creating video with GPU acceleration...")
                final_clip.write_videofile(output_file, **gpu_params)
                print(f"✅ Video created successfully!")
            except Exception as e:
                print(f"⚠️ GPU failed, using CPU...")
                final_clip.write_videofile(output_file, **cpu_params)
                print(f"✅ Video created successfully!")
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
        print("=" * 60)
        
        # Step 1: Process PDF to descriptions
        print("\nStep 1: Processing PDF to descriptions...")
        descriptions_file, image_files = self.process_pdf_to_descriptions(pdf_path, output_folder, pdf_batch_size)
        print(f"✅ Found {len(image_files)} slides")
        
        # Step 2: Process with Claude
        print("\nStep 2: Processing with Claude...")
        final_context_file = self.process_with_claude(descriptions_file, output_folder)
        
        video_path, vietnamese_descriptions, durations = self.create_video_from_context(
            final_context_file, image_files, output_folder
        )
        
        print("\n" + "=" * 60)
        print("🎉 Workflow completed successfully!")
        print(f"📊 Total slides: {len(image_files)}")
        print(f"🎤 TTS processing: Sequential")
        print(f"⏱️  Total video duration: {sum(durations):.2f}s")
        print(f"📋 Content organization: All slides processed in logical sequence")
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
    
    # Print detailed timing analysis
    processor.timer.print_summary()

if __name__ == "__main__":
    main() 