#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Entry point for Pictionary Live

Clean architecture with camera integration for hand and mouse modes.
"""

import os
# Reduce TensorFlow logs and disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Verify and install dependencies (silent)
from dependencies import main as check_dependencies
check_dependencies()


def main():
    """Main function - launches PyQt6 application with clean camera integration."""
    parser = argparse.ArgumentParser(
        description="Pictionary Live - Draw in the air with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--ia-dir", type=str, default="./IA", 
                       help="Path to model directory")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera ID (default: 0)")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug logging")

    args = parser.parse_args()

    try:
        # Import clean integration layer
        from app_integration import PictionaryApp
        
        # Create and run application
        app = PictionaryApp(
            ia_dir=args.ia_dir,
            camera_id=args.camera,
            debug=args.debug
        )
        
        sys.exit(app.run())
        
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install dependencies: pip install -r src/requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication terminated")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
