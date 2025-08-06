#!/usr/bin/env python3
"""Test basic imports for Phase 2 components."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing Phase 2 imports...")
print(f"Python path: {sys.path[:3]}")

try:
    print("\n1. Testing pixel shuffle import...")
    from models.pixel_shuffle_3d import PixelShuffle3D
    print("✅ PixelShuffle3D imported successfully")
except Exception as e:
    print(f"❌ Failed to import PixelShuffle3D: {e}")

try:
    print("\n2. Testing task encoding import...")
    from models.task_encoding import TaskEncodingModule
    print("✅ TaskEncodingModule imported successfully")
except Exception as e:
    print(f"❌ Failed to import TaskEncodingModule: {e}")

try:
    print("\n3. Testing encoder import...")
    from models.encoder_3d import Encoder3D
    print("✅ Encoder3D imported successfully")
except Exception as e:
    print(f"❌ Failed to import Encoder3D: {e}")

try:
    print("\n4. Testing decoder import...")
    from models.decoder_3d import QueryBasedDecoder
    print("✅ QueryBasedDecoder imported successfully")
except Exception as e:
    print(f"❌ Failed to import QueryBasedDecoder: {e}")

try:
    print("\n5. Testing fixed decoder import...")
    from models.decoder_3d_fixed import QueryBasedDecoderFixed
    print("✅ QueryBasedDecoderFixed imported successfully")
except Exception as e:
    print(f"❌ Failed to import QueryBasedDecoderFixed: {e}")

try:
    print("\n6. Testing IRIS model import...")
    from models.iris_model import IRISModel
    print("✅ IRISModel imported successfully")
except Exception as e:
    print(f"❌ Failed to import IRISModel: {e}")

try:
    print("\n7. Testing fixed IRIS model import...")
    from models.iris_model_fixed import IRISModelFixed
    print("✅ IRISModelFixed imported successfully")
except Exception as e:
    print(f"❌ Failed to import IRISModelFixed: {e}")

print("\nImport test complete!")