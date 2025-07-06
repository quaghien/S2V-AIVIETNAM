#!/usr/bin/env python3
"""
S2V (Slides to Video) - Command Line Interface with Zalo AI TTS
AI-Powered PDF to Vietnamese Video Converter
Sequential TTS Processing + GPU-Accelerated Video Encoding
"""

import argparse
import os
import sys
import time
from pathlib import Path
from main_zalo import ZaloGPTProcessor
from config import Config

def validate_pdf_path(pdf_path):
    """Validate PDF file path"""
    if not os.path.exists(pdf_path):
        raise argparse.ArgumentTypeError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.lower().endswith('.pdf'):
        raise argparse.ArgumentTypeError(f"File is not a PDF: {pdf_path}")
    return pdf_path

def validate_positive_int(value):
    """Validate positive integer"""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"Invalid positive integer: {value}")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}")

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="S2V - Convert PDF presentations to Vietnamese video with Zalo AI TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli_zalo.py input.pdf
  
  # Custom output folder
  python cli_zalo.py input.pdf -o /path/to/output
  
  # Adjust batch size
  python cli_zalo.py input.pdf --pdf-batch 3
  
  # Verbose output
  python cli_zalo.py input.pdf -v
  
  # Quick mode (small batches for testing)
  python cli_zalo.py input.pdf --quick
  
  # Production mode (optimized settings)
  python cli_zalo.py input.pdf --production
        """
    )
    
    # Required arguments
    parser.add_argument(
        'pdf_path',
        type=validate_pdf_path,
        help='Path to the PDF file to convert'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output',
        help='Output folder for generated files (default: ./output)'
    )
    
    parser.add_argument(
        '--pdf-batch',
        type=validate_positive_int,
        default=5,
        help='Number of slides to process with VLM at once (default: 5, recommended: 3-7)'
    )
    
    parser.add_argument(
        '--threads',
        type=validate_positive_int,
        default=4,
        help='[DEPRECATED] Thread count (no longer used - TTS is now sequential)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Preset modes
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: Small batches for testing (PDF=3)'
    )
    
    preset_group.add_argument(
        '--production',
        action='store_true',
        help='Production mode: Optimized settings (PDF=7)'
    )
    
    preset_group.add_argument(
        '--safe',
        action='store_true',
        help='Safe mode: Conservative settings (PDF=3)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Load configuration from JSON file'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to JSON file'
    )
    
    return parser

def apply_preset(args):
    """Apply preset configurations"""
    if args.quick:
        args.pdf_batch = 3
        args.threads = 2  # Keep for compatibility but not used
        if args.verbose:
            print("🚀 Quick mode: PDF=3 (TTS: Sequential)")
    
    elif args.production:
        args.pdf_batch = 7
        args.threads = 8  # Keep for compatibility but not used
        if args.verbose:
            print("🏭 Production mode: PDF=7 (TTS: Sequential)")
    
    elif args.safe:
        args.pdf_batch = 3
        args.threads = 4  # Keep for compatibility but not used
        if args.verbose:
            print("🛡️ Safe mode: PDF=3 (TTS: Sequential)")

def print_summary(args):
    """Print configuration summary"""
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"📄 PDF File:            {args.pdf_path}")
    print(f"📁 Output Folder:       {args.output}")
    print(f"📊 PDF Batch Size:      {args.pdf_batch}")
    print(f"📢 Verbose Mode:        {'Enabled' if args.verbose else 'Disabled'}")
    print(f"🎤 TTS Engine:          Zalo AI")
    print(f"🎤 TTS Processing:      Sequential (1 thread)")
    print(f"🎬 Video Encoding:      GPU Accelerated")
    print(f"🔄 Workflow:            Individual Slide Processing")
    print("=" * 50)

def main():
    """Main CLI function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Apply presets
    apply_preset(args)
    
    # Load configuration
    config = Config()
    if args.config:
        config.load_from_file(args.config)
        if args.verbose:
            print(f"✅ Configuration loaded from {args.config}")
    
    # Override config with CLI arguments
    config.default_pdf_path = args.pdf_path
    config.default_output_folder = args.output
    config.pdf_batch_size = args.pdf_batch
    config.max_threads = args.threads
    
    # Save configuration if specified
    if args.save_config:
        config.save_to_file(args.save_config)
        if args.verbose:
            print(f"✅ Configuration saved to {args.save_config}")
    
    # Print banner
    if args.verbose:
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║                    S2V - SLIDES TO VIDEO                  ║")
        print("║                  AI-Powered PDF to Video                  ║")
        print("║                  (Vietnamese Audio - Zalo AI)             ║")
        print("╚═══════════════════════════════════════════════════════════╝")
    
    # Print configuration summary
    print_summary(args)
    
    # Validate API keys
    missing_keys = config.validate_api_keys()
    if missing_keys:
        print(f"\n❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please set these environment variables in your .env file:")
        for key in missing_keys:
            print(f"   {key}=your_api_key_here")
        return 1
    
    # Confirm execution (skip in non-interactive mode)
    if sys.stdin.isatty():  # Only ask if running interactively
        try:
            confirm = input("\n✅ Proceed with conversion? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Conversion cancelled.")
                return 1
        except KeyboardInterrupt:
            print("\n❌ Conversion cancelled.")
            return 1
    else:
        print("\n🚀 Starting conversion in non-interactive mode...")
    
    try:
        # Initialize processor
        if args.verbose:
            print("\n🔧 Initializing AI processor...")
        
        processor = ZaloGPTProcessor(*config.get_api_keys())
        
        # Create output folder with timestamp
        output_folder = processor.create_random_output_folder(config.default_output_folder)
        
        if args.verbose:
            print(f"📁 Output folder created: {output_folder}")
        
        print(f"\n🎬 Starting PDF to Video conversion...")
        start_time = time.time()
        
        # Process PDF to video with sequential TTS
        video_path, vietnamese_descriptions, durations = processor.process_pdf_to_video(
            config.default_pdf_path,
            output_folder,
            config.pdf_batch_size
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Print results
        print("\n" + "=" * 60)
        print("🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Total processing time: {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
        print(f"🎥 Video file:           {video_path}")
        print(f"📁 Output folder:        {output_folder}")
        print(f"⏱️  Video duration:       {sum(durations):.2f} seconds")
        print(f"📄 Slides processed:     {len(durations)}")
        print(f"🎤 Processing Mode:      Sequential TTS + GPU Video")
        
        # Performance info
        if args.verbose:
            avg_per_slide = processing_time / len(durations)
            print(f"⚡ Avg per slide:        {avg_per_slide:.2f} seconds")
            print(f"🎤 TTS Engine:          Zalo AI")
        
        print("\n✅ You can find all generated files in the output folder above.")
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Conversion interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        if args.verbose:
            import traceback
            print("\n🔍 Detailed error information:")
            traceback.print_exc()
        
        print("\n💡 Troubleshooting tips:")
        print("   - Check your internet connection")
        print("   - Verify API keys are valid (OpenAI, Anthropic, Zalo)")
        print("   - Ensure PDF file is readable")
        print("   - Check available disk space")
        print("   - Try reducing PDF batch size if issues occur")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 