import os
from typing import Dict, Any
import json
from dotenv import load_dotenv

class Config:
    """Configuration class for S2V (Slides to Video) application"""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # API Keys from environment variables
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.zalo_api_key = os.getenv('ZALO_API_KEY', '')
        
        # Default file paths
        self.default_pdf_path = ""
        self.default_output_folder = "./output"
        
        # Processing settings
        self.pdf_batch_size = 5
        self.max_threads = 4
        self.use_batch_splitting = True
        
        # Video settings
        self.video_fps = 24
        self.audio_rate = 24000
        
        # Load from config file if exists
        self.load_from_file()
    
    def load_from_file(self, config_file="config.json"):
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        
                print(f"✅ Configuration loaded from {config_file}")
            except Exception as e:
                print(f"⚠️ Error loading config file: {e}")
    
    def save_to_file(self, config_file="config.json"):
        """Save current configuration to JSON file"""
        config_data = {
            'default_pdf_path': self.default_pdf_path,
            'default_output_folder': self.default_output_folder,
            'pdf_batch_size': self.pdf_batch_size,
            'max_threads': self.max_threads,
            'use_batch_splitting': self.use_batch_splitting,
            'video_fps': self.video_fps,
            'audio_rate': self.audio_rate
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Error saving config file: {e}")
    
    def get_api_keys(self):
        """Get all API keys"""
        return self.openai_api_key, self.anthropic_api_key, self.gemini_api_key, self.zalo_api_key
    
    def validate_api_keys(self):
        """Validate that required API keys are present"""
        missing_keys = []
        
        if not self.openai_api_key:
            missing_keys.append('OPENAI_API_KEY')
        if not self.anthropic_api_key:
            missing_keys.append('ANTHROPIC_API_KEY')
        if not self.gemini_api_key:
            missing_keys.append('GEMINI_API_KEY')
        if not self.zalo_api_key:
            missing_keys.append('ZALO_API_KEY')
            
        return missing_keys
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'pdf_path': self.default_pdf_path,
            'output_folder': self.default_output_folder,
            'pdf_batch_size': self.pdf_batch_size,
            'max_threads': self.max_threads,
            'use_batch_splitting': self.use_batch_splitting,
            'video_fps': self.video_fps,
            'audio_rate': self.audio_rate
        } 