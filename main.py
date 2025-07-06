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
class GPTProcessor:
    def __init__(self, openai_api_key, anthropic_api_key, gemini_api_key):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        openai.api_key = openai_api_key
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.gemini_client = genai.Client(api_key=gemini_api_key)

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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        payload = {
            "model": "gpt-4.1-mini",
            "messages": messages,
            "max_tokens": 3000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print("Response JSON:", response.json())
        return response.json()

    def process_pdf_to_descriptions(self, pdf_path, output_folder, batch_size=3):
        image_folder = os.path.join(output_folder, 'images')
        image_files = self.pdf_to_images(pdf_path, image_folder)
        start_slide = 1
        all_descriptions = {}
        previous_response_text = ""
        is_first_batch = True

        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
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

            print("Type of processed_batch:", type(processed_batch))
            print("Content of processed_batch:", processed_batch)

            # Extract the text content from the TextBlock object
            if isinstance(processed_batch, list) and len(processed_batch) > 0 and hasattr(processed_batch[0], 'text'):
                processed_batch = processed_batch[0].text
            elif not isinstance(processed_batch, str):
                processed_batch = str(processed_batch)

            print("Type of processed_batch after conversion:", type(processed_batch))
            print("Content of processed_batch after conversion:", processed_batch)

            full_content = self.replace_batch(full_content, processed_batch, start, end)

            start = end + 1

        final_context_file = os.path.join(output_folder, "final-context.txt")
        self.write_file(final_context_file, full_content)
        return final_context_file

    def create_video_from_context(self, final_context_file, image_files, output_folder, tts_batch_size=1):
        """
        Create video from context with configurable TTS batch size.
        
        Args:
            final_context_file: Path to final context file
            image_files: List of image file paths
            output_folder: Output folder path
            tts_batch_size: Number of slides to process in one TTS call (1-5 recommended)
        """
        with open(final_context_file, 'r', encoding='utf-8') as f:
            final_context = f.read()

        slide_descriptions = self.extract_slide_descriptions(final_context)

        # Translate to Vietnamese
        print("🌐 Starting translation process...")
        translated_file = self.translate_to_vietnamese(final_context_file, output_folder)
        
        # Extract Vietnamese descriptions with tags kept
        with open(translated_file, 'r', encoding='utf-8') as f:
            translated_content = f.read()
        vietnamese_descriptions = self.extract_slide_descriptions(translated_content, keep_tags=True)

        # Generate Vietnamese audio with configurable batch size (with tags)
        audio_folder = os.path.join(output_folder, 'audio')
        audio_files = self.text_to_speech_vietnamese_batch(vietnamese_descriptions, audio_folder, tts_batch_size)
        
        # Create video with Vietnamese audio
        video_path = os.path.join(output_folder, "final_video.mp4")
        durations = self.create_video(image_files, audio_files, video_path)
        
        return video_path, vietnamese_descriptions, durations

    def generate_vietnamese_audio(self, final_context_file, output_folder, tts_batch_size=1):
        """
        Generate Vietnamese audio from context file.
        
        Args:
            final_context_file: Path to final context file
            output_folder: Output folder path
            tts_batch_size: Number of slides to process in one TTS call (1-5 recommended)
            
        Returns:
            tuple: (audio_files, vietnamese_descriptions, translated_file)
        """
        with open(final_context_file, 'r', encoding='utf-8') as f:
            final_context = f.read()

        slide_descriptions = self.extract_slide_descriptions(final_context)

        # Translate to Vietnamese
        print("🌐 Starting translation process...")
        translated_file = self.translate_to_vietnamese(final_context_file, output_folder)
        
        # Extract Vietnamese descriptions with tags kept
        print("📝 DEBUG - Reading translated file (generate_vietnamese_audio)...")
        with open(translated_file, 'r', encoding='utf-8') as f:
            translated_content = f.read()
        print("📝 DEBUG - Translated file content (first 100 chars):")
        print(repr(translated_content[:100]))
        print()
        
        vietnamese_descriptions = self.extract_slide_descriptions(translated_content, keep_tags=True)

        # Generate Vietnamese audio with configurable batch size (with tags)
        audio_folder = os.path.join(output_folder, 'audio')
        audio_files = self.text_to_speech_vietnamese_batch(vietnamese_descriptions, audio_folder, tts_batch_size)
        
        print(f"🎤 Vietnamese audio generation completed!")
        print(f"📁 Audio files saved in: {audio_folder}")
        print(f"📊 Generated {len(audio_files)} audio files")
        
        return audio_files, vietnamese_descriptions, translated_file

    def create_video_with_audio(self, image_files, audio_files, output_folder):
        """
        Create video from images and audio files.
        
        Args:
            image_files: List of image file paths
            audio_files: List of audio file paths
            output_folder: Output folder path
            
        Returns:
            tuple: (video_path, durations)
        """
        print("🎬 Starting video creation...")
        video_path = os.path.join(output_folder, "final_video.mp4")
        durations = self.create_video(image_files, audio_files, video_path)
        
        print(f"🎥 Video created successfully at: {video_path}")
        print(f"🎵 Total video duration: {sum(durations):.2f} seconds")
        
        return video_path, durations

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
                # Calculate scale to meet minimum requirements
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

    def wave_file(self, filename, pcm, channels=1, rate=24000, sample_width=2):
        """Helper function to save wave file."""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)

    def text_to_speech_with_openai(self, descriptions, output_dir="audio_files"):
        """Converts text descriptions to speech using OpenAI TTS."""
        os.makedirs(output_dir, exist_ok=True)
        audio_files = []
        
        # Initialize OpenAI client
        client = OpenAI(api_key=self.openai_api_key)
        
        for i, description in enumerate(descriptions):
            speech_file_path = Path(output_dir) / f"slide_{i + 1}.mp3"
            style = """
                Đặc tính giọng nói: Rõ ràng, dễ hiểu, thân thiện; thể hiện sự chuyên nghiệp nhưng không cứng nhắc, như một giảng viên tận tâm đang truyền đạt kiến thức.
                Giọng điệu: Thân thiện, chuyên nghiệp nhưng không đơn điệu; mang tính biểu cảm nhẹ nhàng và tự nhiên, tạo cảm giác như đang được một người thầy giàu kinh nghiệm hướng dẫn một cách kiên nhẫn.
                Nhịp độ: Tốc độ bình thường.
                Cảm xúc: Thể hiện sự nhiệt tình trong giảng dạy, chân thành và quan tâm đến người học; tạo cảm giác động viên mà không gây áp lực, khuyến khích người nghe tích cực tham gia.
                Phát âm: Rõ ràng, chuẩn mực, phù hợp với người Việt; nhấn nhá từng từ quan trọng một cách tự nhiên, không cường điệu quá mức nhưng đủ để làm nổi bật những điểm chính. 
                Khoảng nghỉ: Sử dụng khoảng lặng có chủ đích sau các khái niệm quan trọng và giữa các phần để người nghe tiếp thu.
                """
            # style = """
            #     Đặc tính giọng nói (Voice Affect): Rõ ràng, dễ hiểu, thân thiện; thể hiện sự chuyên nghiệp nhưng không cứng nhắc, như một giảng viên tận tâm đang truyền đạt kiến thức. Giọng nói có độ từ, ấm áp nhưng vẫn giữ được tính học thuật, tạo môi trường học tập thoải mái và hiệu quả.

            #     Giọng điệu (Tone): Thân thiện, chuyên nghiệp nhưng không đơn điệu; mang tính biểu cảm nhẹ nhàng và tự nhiên, tạo cảm giác như đang được một người thầy giàu kinh nghiệm hướng dẫn một cách kiên nhẫn. Giọng điệu có sự biến thiên nhẹ nhàng giữa nghiêm túc khi truyền đạt kiến thức và gần gũi khi tương tác với người học.

            #     Nhịp độ (Pacing): Tốc độ vừa phải (khoảng 1.25x), không quá nhanh hay vội vàng; nhịp độ cân đối giúp người nghe dễ theo dõi và ghi nhớ thông tin, tạo không gian để tiếp thu kiến thức. Đọc có đoạn nhanh, đoạn chậm, tự nhiên - nhanh lên khi giải thích các khái niệm quen thuộc, chậm lại khi trình bày nội dung phức tạp hoặc quan trọng.

            #     Cảm xúc (Emotion): Thể hiện sự nhiệt tình trong giảng dạy, chân thành và quan tâm đến người học; tạo cảm giác động viên mà không gây áp lực, khuyến khích người nghe tích cực tham gia. Có sự hào hứng có kiểm soát khi giới thiệu các khái niệm mới, thể hiện sự kiên nhẫn và khuyến khích khi giải thích các phần khó.

            #     Phát âm (Pronunciation): Rõ ràng, chuẩn mực, phù hợp với người Việt; nhấn nhá từng từ quan trọng một cách tự nhiên, không cường điệu quá mức nhưng đủ để làm nổi bật những điểm chính. Phát âm các thuật ngữ chuyên môn cẩn thận và có thể lặp lại để người nghe ghi nhớ.

            #     Khoảng lặng (Pauses): Sử dụng khoảng lặng có chủ đích sau các khái niệm quan trọng và giữa các phần để người nghe có thời gian tiếp thu và suy ngẫm. Khoảng lặng ngắn (1-2 giây) sau định nghĩa, khoảng lặng dài hơn (2-3 giây) khi chuyển đổi giữa các chủ đề lớn.

            #     Ngữ điệu (Intonation):  Ngữ điệu có sự biến thiên nhẹ nhàng để tạo nhịp điệu cho bài giảng, tránh tình trạng đơn điệu.

            #     Ấn tượng (Impressions): Tạo ấn tượng của một giảng viên có kinh nghiệm, am hiểu sâu về chuyên môn nhưng vẫn approachable và quan tâm đến việc học của sinh viên. Giọng nói thể hiện sự tự tin trong kiến thức nhưng không kiêu ngạo.

            #     """
            # Create TTS with OpenAI
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=description,
                # instructions=f"{style}"
            ) as response:
                response.stream_to_file(speech_file_path)
            
            audio_files.append(str(speech_file_path))
            
        return audio_files

    def create_video(self, image_files, audio_files, output_file, fps=24):
        clips = []
        durations = []  # Save the duration for each slide
        for image_file, audio_file in zip(image_files, audio_files):
            audio = AudioFileClip(audio_file)
            duration = audio.duration
            durations.append(duration)
            img = Image.open(image_file)
            img_array = np.array(img)
            img_clip = ImageSequenceClip([img_array], durations=[duration])
            img_clip = img_clip.with_audio(audio)
            img_clip = img_clip.with_fps(fps)
            clips.append(img_clip)

        if clips:
            final_clip = concatenate_videoclips(clips, method="compose")
            
            # GPU acceleration settings for Mac (MPS) and other platforms
            gpu_params = {
                'codec': 'libx264',
                'audio_codec': 'aac',
                'fps': fps,
                'preset': 'fast',  # Faster encoding
                'ffmpeg_params': [
                    '-hwaccel', 'auto',  # Auto hardware acceleration
                    '-c:v', 'h264_videotoolbox',  # Mac hardware encoder
                    '-allow_sw', '1',  # Allow software fallback
                    '-crf', '23',  # Good quality/speed balance
                    '-movflags', '+faststart'  # Better streaming
                ]
            }
            
            # Try GPU acceleration, fallback to CPU if failed
            try:
                print("Attempting video creation with GPU acceleration...")
                final_clip.write_videofile(output_file, **gpu_params)
                print("✅ Video created successfully with GPU acceleration!")
            except Exception as e:
                print(f"⚠️ GPU acceleration failed: {e}")
                print("Falling back to CPU encoding...")
                # Fallback to standard CPU encoding
                final_clip.write_videofile(
                    output_file, 
                    codec="libx264", 
                    audio_codec="aac", 
                    fps=fps,
                    preset='fast'
                )
                print("✅ Video created successfully with CPU!")
        else:
            print("No clips to concatenate")

        return durations

    def save_descriptions(self, descriptions, file_path):
        """Saves the descriptions to a text file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('full_content = """\n')
            for i, desc in enumerate(descriptions, 1):
                f.write(f"#slide{i}#\n{desc}\n\n")
            f.write('"""')

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content.strip('full_content = """').strip('"""')

    def write_file(self, file_path, content):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('full_content = """\n' + content + '\n"""')

    def create_prompt(self, batch_content, start, end, total_slides):
        prompt = f"""{batch_content}
Please read the content of these slides carefully and assume the role of a knowledgeable and engaging professor delivering a comprehensive and captivating lecture. Your goal is to deeply understand the meaning and context of each slide, explaining them in a manner that is both thorough and engaging. Rather than merely reading the existing text, provide insightful and detailed explanations, ensuring smooth and natural transitions between the content.
Create seamless and logical transitions that connect each slide to the overall theme of the lecture. Use phrases like, "This concept will be further explored in upcoming slides," or "Keep this idea in mind, as it will be crucial later on," to link different sections and maintain coherence throughout the lecture. Additionally, incorporate natural lecturer comments and anecdotes to make the lecture feel more authentic, relatable, and engaging.
You'll smoothly transition from the concepts covered in slides 1-{start-1} to the advanced topics in slides {start}-{total_slides}. As you know from the first ten slides, [summarize key points from slides 1-{start-1}], it's essential to build upon these foundations to fully grasp the ideas presented in the subsequent slides.
To recap, we've discussed [briefly mention a few key points from slides 1-{start-1}]. Building on this, you will now explore how [mention the new concepts in slides {start}-{total_slides}] expand and improve your understanding of [content in slides 1-{start-1}]. Additionally, as highlighted earlier, [mention another key point from slides 1-{start-1}], which serves as a critical link to the upcoming sections.
By connecting these concepts, you'll gain a comprehensive understanding of the subject matter. This approach ensures that you can seamlessly integrate the knowledge from the initial slides with the more advanced topics to follow.
Now, let's delve into the details from slide {start} to slide {end}.
Please read the content of these slides carefully and assume the role of a knowledgeable and engaging professor delivering a comprehensive and captivating lecture. Your goal is to deeply understand the meaning and context of each slide, explaining them in a manner that is both thorough and engaging. Rather than merely reading the existing text, provide insightful and detailed explanations, ensuring smooth and natural transitions between the content.
Create seamless and logical transitions that connect each slide to the overall theme of the lecture. Use some phrases to link different sections and maintain coherence throughout the lecture. Additionally, incorporate natural lecturer comments and anecdotes to make the lecture feel more authentic, relatable, and engaging.
You'll smoothly transition from the concepts covered in slides 1-{start-1} to the advanced topics in slides {start}-{total_slides}. As you know from the first ten slides, [summarize key points from slides 1-{start-1}], it's essential to build upon these foundations to fully grasp the ideas presented in the subsequent slides.
To recap, we've discussed [briefly mention a few key points from slides 1-{start-1}]. Building on this, you will now explore how [mention the new concepts in slides {start}-{total_slides}] expand and improve your understanding of [content in slides 1-{start-1}]. Additionally, as highlighted earlier, [mention another key point from slides 1 to {start-1}], which serves as a critical link to the upcoming sections.
By connecting these concepts, you'll gain a comprehensive understanding of the subject matter. This approach ensures that you can seamlessly integrate the knowledge from the initial slides with the more advanced topics to follow.
DO IT FROM SLIDE {start} to {end}, don't greet, just continue the presentations. Using some questionn such as, you have known about the [content in previous slide ]. Mention the future and the past is crucial and must have
You should hold the tag , #slide# format for each slide.Make Lecture short, focus on slide that have important information. You don't just list idea, ensure the lecture is smooth, natural, and engaging.
emphasize of the speech is identify by the caps of the word, the !!! the ??? the capitalism, the -. Ensure the lecture is fully cover, make the atmosphere positive. Make Lecture short, focus on slide that have important information. You don't just list idea, ensure the lecture is smooth, natural, and engaging. Please make content short, focus on important information. But respect eachs slide content. Don't add something like *draws a conceptual map on an imaginary whiteboard*, or 
*raises an eyebrow*, just CAPLOCS,!!!,?? and content of the slide"""
        return prompt

    def replace_batch(self, full_content, processed_batch, start, end):
        slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', full_content, re.DOTALL)
        new_slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', processed_batch, re.DOTALL)

        for i, new_slide in enumerate(new_slides):
            slide_number = start + i
            if slide_number <= end and slide_number - 1 < len(slides):
                full_content = full_content.replace(slides[slide_number - 1], new_slide)

        return full_content

    def process_batch(self, client, full_content, start, end, total_slides):
        start_time = time.time()
        slides = re.findall(r'(#slide\d+#.*?(?=#slide\d+#|\Z))', full_content, re.DOTALL)
        batch = slides[start - 1:end]
        batch_content = '\n'.join(batch)

        prompt = self.create_prompt(batch_content, start, end, total_slides)
        end_time = time.time()
        print("EXECUTION TIME: ",end_time-start_time)
        start_time = time.time()
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
        end_time = time.time()
        print("CLAUDE RESPONSE TIME: ",end_time-start_time)
        print("Type of message.content:", type(message.content))
        print("Content of message:", message.content)

        # Return the text content directly
        if isinstance(message.content, list) and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            return message.content[0].text
        else:
            return str(message.content)

    def extract_slide_descriptions(self, final_context, keep_tags=False):
        print(f"📝 DEBUG - extract_slide_descriptions called with keep_tags={keep_tags}")
        print("📝 DEBUG - Input content (first 100 chars):")
        print(repr(final_context[:100]))
        
        if keep_tags:
            # Handle both #slide# and #Trình# patterns and KEEP the tags
            slide_descriptions = re.findall(r'(#(?:slide|Trình)\s*\d+#.*?)(?=#(?:slide|Trình)\s*\d+#|\Z)', final_context, re.DOTALL)
            print(f"📝 DEBUG - Found {len(slide_descriptions)} descriptions with tags")
            for i, desc in enumerate(slide_descriptions[:3]):  # Show first 3
                print(f"📝 DEBUG - Desc {i+1} (first 100 chars): {repr(desc[:100])}")
            return [desc.strip() for desc in slide_descriptions]
        else:
            # Handle both #slide# and #Trình# patterns and REMOVE the tags (original behavior)
            slide_descriptions = re.findall(r'#(?:slide|Trình)\s*\d+#(.*?)(?=#(?:slide|Trình)\s*\d+#|\Z)', final_context, re.DOTALL)
            print(f"📝 DEBUG - Found {len(slide_descriptions)} descriptions without tags")
            for i, desc in enumerate(slide_descriptions[:3]):  # Show first 3
                print(f"📝 DEBUG - Desc {i+1} (first 100 chars): {repr(desc[:100])}")
            return [desc.strip() for desc in slide_descriptions]

    def translate_to_vietnamese(self, descriptions_file, output_folder):
        """Translates the descriptions to Vietnamese using Gemini API."""
        full_content = self.read_file(descriptions_file)
        
        print("🌐 Translating content to Vietnamese...")
        print("📝 DEBUG - Original content (first 100 chars):")
        print(repr(full_content[:100]))
        print()
        
        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=f'''Dịch toàn bộ sang tiếng việt, trả đúng format y như cũ, rút gọn nội dung, không thay đổi nội dung slide. 
                QUAN TRỌNG:
                - Khi gặp công thức toán học, hãy tạo lại chính xác cách đọc công thức thành văn đọc tiếng Việt
                ***Ví dụ: "Â_i = (r_i - μ)/σ" → "Â i bằng r i trừ miu, tất cả chia sigma"***
                - Tự động nhận diện các nơi cần ngắt hơi (im lặng), và thêm ' <<<sil#100>>> ' vào những nơi đó.
                Nội dung slide: {full_content}''',
        )
        
        translated_content = response.text
        print("📝 DEBUG - Translated content BEFORE tag replacement (first 100 chars):")
        print(repr(translated_content[:100]))
        print()
        
        # Replace #slide X# with #Trình X#
        print("🔄 Replacing slide tags with Vietnamese format...")
        original_tags = re.findall(r'#slide\d+#', translated_content)
        print(f"📝 DEBUG - Found original tags: {original_tags}")
        
        translated_content = re.sub(r'#slide(\d+)#', r'#Trình \1#', translated_content)
        
        new_tags = re.findall(r'#Trình\s*\d+#', translated_content)
        print(f"📝 DEBUG - New tags after replacement: {new_tags}")
        print("📝 DEBUG - Translated content AFTER tag replacement (first 100 chars):")
        print(repr(translated_content[:100]))
        print()
        
        # Save translated content
        translated_file = os.path.join(output_folder, "translated_descriptions.txt")
        with open(translated_file, "w", encoding="utf-8") as file:
            file.write(translated_content)
        
        print(f"✅ Vietnamese translation saved to: {translated_file}")
        return translated_file

    def text_to_speech_vietnamese_batch(self, descriptions, output_dir="audio", tts_batch_size=1):
        """Converts Vietnamese text descriptions to speech using Gemini TTS with smart batching and splitting.
        
        Args:
            descriptions: List of slide descriptions
            output_dir: Output directory for audio files
            tts_batch_size: Number of slides to process in one TTS call (1-5 recommended)
        
        Returns:
            List of individual audio files for each slide
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎤 Converting Vietnamese text to speech (batch size: {tts_batch_size})...")
        print(f"📝 DEBUG - Received {len(descriptions)} descriptions for TTS")
        
        if tts_batch_size == 1:
            # Single slide processing - use original method
            return self._tts_single_slide(descriptions, output_dir)
        else:
            # Batch processing with transcription splitting
            return self._tts_batch_with_splitting(descriptions, output_dir, tts_batch_size)

    def _tts_single_slide(self, descriptions, output_dir):
        """Process slides one by one (original method)."""
        audio_files = []
        
        for i, description in enumerate(descriptions):
            print(f"Processing slide {i + 1}/{len(descriptions)}")
            
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=f"Bạn là một chuyên gia lĩnh vực AI, hãy đọc bằng tiếng việt nhưng bạn phải đọc các từ viết tắt, thuật ngữ chuyên ngành bằng tiếng anh. Đoạn bạn cần đọc là: {description}",
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Charon',
                                )
                            )
                        ),
                    )
                )
                
                data = response.candidates[0].content.parts[0].inline_data.data
                file_name = os.path.join(output_dir, f'slide_{i + 1}.wav')
                self.wave_file(file_name, data)
                audio_files.append(file_name)
                
            except Exception as e:
                print(f"Error generating audio for slide {i + 1}: {e}")
                fallback_file = os.path.join(output_dir, f'slide_{i + 1}.wav')
                self.create_silent_audio(fallback_file, duration=5.0)
                audio_files.append(fallback_file)
        
        return audio_files

    def _tts_batch_with_splitting(self, descriptions, output_dir, tts_batch_size):
        """Process slides in batches and split using transcription."""
        batch_audio_files = []
        batch_info = []
        
        # Step 1: Create batch audio files
        print("🎤 Step 1: Creating batch audio files...")
        for batch_start in range(0, len(descriptions), tts_batch_size):
            batch_end = min(batch_start + tts_batch_size, len(descriptions))
            batch_descriptions = descriptions[batch_start:batch_end]
            
            print(f"Creating batch audio for slides {batch_start + 1}-{batch_end}/{len(descriptions)}")
            
            try:
                # Combine descriptions for this batch
                combined_content = "\n\n".join(batch_descriptions)
                
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=f"Bạn là một chuyên gia lĩnh vực AI, hãy đọc bằng tiếng việt nhưng bạn phải đọc các từ viết tắt, thuật ngữ chuyên ngành bằng tiếng anh. Đoạn bạn cần đọc là: {description}",
                    config=types.GenerateContentConfig( 
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Charon',
                                )
                            )
                        ),
                    )
                )
                
                data = response.candidates[0].content.parts[0].inline_data.data
                batch_file_name = os.path.join(output_dir, f'batch_{batch_start + 1}_to_{batch_end}.wav')
                self.wave_file(batch_file_name, data)
                
                batch_audio_files.append(batch_file_name)
                batch_info.append({
                    'file': batch_file_name,
                    'start_slide': batch_start + 1,
                    'end_slide': batch_end,
                    'slide_count': batch_end - batch_start
                })
                
                print(f"✅ Created batch file: {batch_file_name}")
                
            except Exception as e:
                print(f"❌ Error creating batch audio for slides {batch_start + 1}-{batch_end}: {e}")
                # Create fallback batch file
                batch_file_name = os.path.join(output_dir, f'batch_{batch_start + 1}_to_{batch_end}.wav')
                self.create_silent_audio(batch_file_name, duration=10.0 * (batch_end - batch_start))
                batch_audio_files.append(batch_file_name)
                batch_info.append({
                    'file': batch_file_name,
                    'start_slide': batch_start + 1,
                    'end_slide': batch_end,
                    'slide_count': batch_end - batch_start
                })
        
        # Step 2: Split each batch file into individual slides
        print("\n🔪 Step 2: Splitting batch files into individual slides...")
        all_slide_files = []
        
        for batch in batch_info:
            print(f"Processing batch: {batch['file']}")
            
            try:
                # Create temporary directory for this batch
                batch_output_dir = os.path.join(output_dir, f"temp_batch_{batch['start_slide']}_to_{batch['end_slide']}")
                
                # Split the batch audio file
                segment_files = self.transcribe_and_split_audio(batch['file'], batch_output_dir)
                
                # Rename and move segments to match slide numbers
                for i, segment_file in enumerate(segment_files[:batch['slide_count']]):
                    slide_num = batch['start_slide'] + i
                    final_slide_file = os.path.join(output_dir, f'slide_{slide_num}.wav')
                    
                    # Copy segment to final location
                    import shutil
                    shutil.copy2(segment_file, final_slide_file)
                    all_slide_files.append(final_slide_file)
                    
                    print(f"✅ Created slide audio: slide_{slide_num}.wav")
                
                # Clean up temporary directory
                import shutil
                shutil.rmtree(batch_output_dir, ignore_errors=True)
                
            except Exception as e:
                print(f"❌ Error splitting batch {batch['file']}: {e}")
                # Create fallback individual files
                for i in range(batch['slide_count']):
                    slide_num = batch['start_slide'] + i
                    fallback_file = os.path.join(output_dir, f'slide_{slide_num}.wav')
                    self.create_silent_audio(fallback_file, duration=5.0)
                    all_slide_files.append(fallback_file)
        
        # Step 3: Clean up batch files
        print("\n🧹 Step 3: Cleaning up batch files...")
        for batch_file in batch_audio_files:
            try:
                os.remove(batch_file)
                print(f"🗑️  Removed batch file: {os.path.basename(batch_file)}")
            except:
                pass
        
        print(f"✅ Created {len(all_slide_files)} individual slide audio files")
        return sorted(all_slide_files)  # Ensure correct order

    def create_silent_audio(self, filename, duration=5.0, rate=24000):
        """Creates a silent audio file as fallback."""
        samples = int(duration * rate)
        silent_data = b'\x00\x00' * samples  # 16-bit silence
        self.wave_file(filename, silent_data, rate=rate)

    def transcribe_and_split_audio(self, audio_file_path, output_dir="output_segments"):
        """
        Transcribe audio file and split into segments based on 'Trình X' markers.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            output_dir: Output directory for audio segments
            
        Returns:
            List of segment file paths
        """
        from pydub import AudioSegment
        import os
        
        print(f"🎤 Transcribing audio: {audio_file_path}")
        
        # Initialize OpenAI client for transcription
        client = OpenAI(api_key=self.openai_api_key)
        
        # Transcribe audio with word-level timestamps
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )

        print("📝 Transcription completed!")
        print(f"Text: {transcription.text}...")

        # Find presentation segments
        segments = self._find_presentation_segments(transcription.words)
        
        if not segments:
            print("⚠️ No 'Trình X' markers found in transcription.")
            return []

        print(f"\n🔍 Found {len(segments)} segments:")
        for i, segment in enumerate(segments):
            duration = segment['end'] - segment['start']
            print(f"  Segment {i + 1}: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {duration:.2f}s)")

        # Split audio into segments
        segment_files = self._split_audio(audio_file_path, segments, output_dir)
        
        print(f"✅ Audio split into {len(segment_files)} segments!")
        return segment_files

    def _find_presentation_segments(self, words):
        """Find segments based on 'Trình X' or 'Chình X' markers in transcription."""
        segments = []
        presentation_markers = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Find pattern "Trình [số]" or "Chình [số]" - more flexible matching
            word_text = word.word.lower().strip()
            # Accept both "trình" and "chình" (case insensitive)
            if (word_text in ['trình', 'chình'] or 'trình' in word_text or 'chình' in word_text):
                # Look for number in next few words
                for j in range(i + 1, min(i + 4, len(words))):
                    next_word = words[j].word.strip()
                    if next_word.isdigit():
                        # Found marker
                        presentation_markers.append({
                            'number': int(next_word),
                            'start': word.start,
                            'end': words[j].end
                        })
                        
                        print(f"🔍 Found '{word.word} {next_word}' from {word.start:.2f}s to {words[j].end:.2f}s")
                        break
                
                i += 1
            else:
                i += 1
        
        # Create segments based on markers
        print(f"📊 Found {len(presentation_markers)} presentation markers")
        
        if presentation_markers:
            # Sort markers by number to ensure correct order
            presentation_markers.sort(key=lambda x: x['number'])
            
            for i, marker in enumerate(presentation_markers):
                # Main segment: from after current marker to before next marker (or end of file)
                start_time = marker['end'] + 0.1  # Add small buffer after marker (reduced from 0.5)
                
                if i + 1 < len(presentation_markers):
                    end_time = presentation_markers[i + 1]['start'] - 0.1  # Buffer before next marker (reduced from 0.5)
                else:
                    # Last segment - use end of audio
                    end_time = words[-1].end if words else start_time + 30  
                
                # Ensure minimum segment length
                if end_time - start_time < 1.0:
                    end_time = start_time + 5.0  # Minimum 5 seconds
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'label': f'slide_{marker["number"]}'  # Use slide_ prefix for consistency
                })
                
                print(f"📝 Segment {marker['number']}: {start_time:.2f}s - {end_time:.2f}s ({end_time-start_time:.2f}s)")
        
        return segments

    def _split_audio(self, audio_path, segments, output_dir):
        """Split audio file into segments."""
        from pydub import AudioSegment
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        segment_files = []
        
        for i, segment in enumerate(segments):
            start_ms = int(segment['start'] * 1000)  # Convert to milliseconds
            end_ms = int(segment['end'] * 1000)
            
            # Extract segment
            segment_audio = audio[start_ms:end_ms]
            
            # Save segment with label name
            output_path = os.path.join(output_dir, f"{segment['label']}.wav")
            segment_audio.export(output_path, format="wav")
            segment_files.append(output_path)
            
            print(f"💾 Saved: {output_path}")
            
            # Print duration info
            duration = segment['end'] - segment['start']
            print(f"   Duration: {duration:.2f}s ({start_ms}ms - {end_ms}ms)")
        
        return segment_files

    def create_random_output_folder(self, base_output_folder):
        """Create a random subfolder for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        folder_name = f"run_{timestamp}_{random_id}"
        
        full_output_path = os.path.join(base_output_folder, folder_name)
        os.makedirs(full_output_path, exist_ok=True)
        
        print(f"📁 Created output folder: {full_output_path}")
        return full_output_path

    def test_workflow_with_batch_splitting(self, pdf_path, output_folder, pdf_batch_size=3, tts_batch_size=5):
        """
        Test the complete workflow with batch TTS and audio splitting.
        
        Example:
        - 9 slides with tts_batch_size=5 will create:
          * Batch 1: slides 1-5 → batch_1_to_5.wav → split into slide_1.wav, slide_2.wav, ..., slide_5.wav
          * Batch 2: slides 6-9 → batch_6_to_9.wav → split into slide_6.wav, slide_7.wav, slide_8.wav, slide_9.wav
        - Final result: 9 individual slide audio files + 9 images → final video
        """
        print("🚀 Starting batch TTS + audio splitting workflow test")
        print(f"📄 PDF batch size: {pdf_batch_size}")
        print(f"🎤 TTS batch size: {tts_batch_size}")
        print("=" * 60)
        
        # Step 1: Process PDF
        print("Step 1: Processing PDF to descriptions...")
        descriptions_file, image_files = self.process_pdf_to_descriptions(pdf_path, output_folder, pdf_batch_size)
        print(f"✅ Found {len(image_files)} slides")
        
        # Step 2: Process with Claude
        print("\nStep 2: Processing with Claude...")
        final_context_file = self.process_with_claude(descriptions_file, output_folder)
        
        # Step 3: Generate audio with batch splitting
        print(f"\nStep 3: Generating Vietnamese audio (batch size: {tts_batch_size})...")
        audio_files, vietnamese_descriptions, translated_file = self.generate_vietnamese_audio(
            final_context_file, output_folder, tts_batch_size
        )
        
        print(f"✅ Generated {len(audio_files)} individual audio files")
        
        # Step 4: Verify audio files
        print("\nStep 4: Verifying audio files...")
        missing_files = []
        for i, audio_file in enumerate(audio_files):
            if not os.path.exists(audio_file):
                missing_files.append(f"slide_{i+1}.wav")
            else:
                # Get audio duration
                try:
                    audio_clip = AudioFileClip(audio_file)
                    duration = audio_clip.duration
                    audio_clip.close()
                    print(f"✅ slide_{i+1}.wav: {duration:.2f}s")
                except:
                    print(f"⚠️  slide_{i+1}.wav: Could not read duration")
        
        if missing_files:
            print(f"❌ Missing audio files: {missing_files}")
            return None, None, None
        
        # Step 5: Create video
        print(f"\nStep 5: Creating final video...")
        video_path, durations = self.create_video_with_audio(image_files, audio_files, output_folder)
        
        print("\n" + "=" * 60)
        print("🎉 Workflow completed successfully!")
        print(f"📊 Total slides: {len(image_files)}")
        print(f"🎤 TTS batch size: {tts_batch_size}")
        print(f"⏱️  Total video duration: {sum(durations):.2f}s")
        print(f"📁 Output folder: {output_folder}")
        print(f"🎥 Final video: {video_path}")
        
        return video_path, audio_files, durations

def main():
    from config import Config
    
    # Load configuration
    config = Config()
    
    # Get API keys from config
    openai_api_key, anthropic_api_key, gemini_api_key = config.get_api_keys()
    
    # Validate API keys
    if not openai_api_key or not anthropic_api_key or not gemini_api_key:
        print("❌ Error: API keys not found! Please check your .env file.")
        return
        
    processor = GPTProcessor(openai_api_key, anthropic_api_key, gemini_api_key)

    pdf_path = '/Users/twang/Downloads/Week 1 - Summary copy.pdf'
    
    base_output_folder = "/Users/twang/PycharmProjects/transition_test/[AIVIETNAM]"
    
    # Create random subfolder for this run
    output_folder = processor.create_random_output_folder(base_output_folder)

    print(f"🔄 Processing PDF: {pdf_path}")
    print(f"📂 Output directory: {output_folder}")

    # Configuration - you can modify these values
    pdf_batch_size = 5  # Number of slides to process with GPT-4 at once
    tts_batch_size = 5  # Number of slides to send to TTS at once (>1 enables batch splitting)
    use_batch_splitting = True  # Use new batch TTS + audio splitting workflow
    
    print(f"⚙️  Configuration:")
    print(f"   📄 PDF processing batch size: {pdf_batch_size}")
    print(f"   🎤 TTS batch size: {tts_batch_size}")
    print(f"   🔪 Use batch splitting: {'Yes' if use_batch_splitting else 'No'}")
    print()

    if use_batch_splitting and tts_batch_size > 1:
        # Use new workflow with batch TTS and audio splitting
        print("🚀 Using NEW workflow: Batch TTS + Audio Splitting")
        print("   This creates more natural audio by processing in batches,")
        print("   then uses transcription to split into individual slides.")
        print()
        
        start_time = time.time()
        video_path, audio_files, durations = processor.test_workflow_with_batch_splitting(
            pdf_path, output_folder, pdf_batch_size, tts_batch_size
        )
        end_time = time.time()
        
        print(f"\n⏱️  Total processing time: {(end_time - start_time):.2f} seconds")
        
    else:
        # Use original workflow
        print("🔄 Using ORIGINAL workflow: Individual slide processing")
        print()
        
        start_time = time.time()
        
        # Step 1: Process PDF to descriptions with configurable batch_size
        descriptions_file, image_files = processor.process_pdf_to_descriptions(pdf_path, output_folder, pdf_batch_size)
        print(f"📝 Descriptions saved to: {descriptions_file}")

        # Step 2: Process with Claude
        final_context_file = processor.process_with_claude(descriptions_file, output_folder)
        print(f"🤖 Final context saved to: {final_context_file}")

        # Step 3: Generate Vietnamese audio
        print("🎤 Starting Vietnamese audio generation...")
        audio_files, vietnamese_descriptions, translated_file = processor.generate_vietnamese_audio(
            final_context_file, output_folder, tts_batch_size=1  # Force single slide for original workflow
        )
        
        print(f"📊 Number of slides: {len(vietnamese_descriptions)}")
        print(f"🇻🇳 Vietnamese translation saved to: {translated_file}")

        # Step 4: Create video
        print("\n🎬 Starting video creation...")
        video_path, durations = processor.create_video_with_audio(
            image_files, audio_files, output_folder
        )
        
        end_time = time.time()
        
        print(f"⏱️  Total processing time: {(end_time - start_time):.2f} seconds")
        print(f"🎵 Total video duration: {sum(durations):.2f} seconds")
        print(f"🎥 Final video created at: {video_path}")
        
    print(f"\n✅ Processing completed!")
    print(f"📁 All files saved in: {output_folder}")

if __name__ == "__main__":
    main()