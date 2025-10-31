#!/usr/bin/env python3
"""
Simple Video Processor with CLIP ReID
A straightforward script to process videos with text-based person tracking
No Jupyter notebook required!
"""

import argparse
import sys
from pathlib import Path
from standalone_clip_reid_notebook import StandaloneCLIPReID

def main():
    """
    Main function to process video with CLIP ReID
    """
    parser = argparse.ArgumentParser(
        description="Process video with CLIP-based person tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_video_processor.py input.mp4 "person wearing red shirt"
  python simple_video_processor.py input.mp4 "woman in blue dress" --output tracked_output.mp4
  python simple_video_processor.py input.mp4 "man with black jacket" --clip-threshold 0.3 --display
        """
    )
    
    # Required arguments
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("text_query", help="Text description of target person(s)")
    
    # Optional arguments
    parser.add_argument("--output", "-o", default="tracked_output.mp4",
                       help="Output video path (default: tracked_output.mp4)")
    parser.add_argument("--clip-threshold", type=float, default=0.25,
                       help="CLIP matching threshold (default: 0.25)")
    parser.add_argument("--reid-threshold", type=float, default=0.7,
                       help="ReID matching threshold (default: 0.7)")
    parser.add_argument("--max-disappeared", type=int, default=10,
                       help="Max frames before track removal (default: 10)")
    parser.add_argument("--display", action="store_true",
                       help="Display video during processing")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                       help="Device to use (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.video_path).exists():
        print(f"❌ Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    print("🚀 CLIP ReID Video Processor")
    print("=" * 50)
    print(f"📹 Input video: {args.video_path}")
    print(f"🎯 Text query: '{args.text_query}'")
    print(f"💾 Output video: {args.output}")
    print(f"🔧 CLIP threshold: {args.clip_threshold}")
    print(f"🔧 ReID threshold: {args.reid_threshold}")
    print(f"🔧 Max disappeared: {args.max_disappeared}")
    print(f"🖥️  Display during processing: {args.display}")
    print()
    
    try:
        # Initialize processor
        print("🔄 Initializing CLIP ReID processor...")
        processor = StandaloneCLIPReID(device=args.device)
        
        # Configure parameters
        processor.clip_threshold = args.clip_threshold
        processor.reid_threshold = args.reid_threshold
        processor.max_disappeared = args.max_disappeared
        
        print(f"✅ Processor initialized on {processor.device}")
        print()
        
        # Process video
        print("🎬 Starting video processing...")
        stats = processor.process_video(
            video_path=args.video_path,
            text_query=args.text_query,
            output_path=args.output,
            display=args.display
        )
        
        # Display results
        print("\n🎉 Processing completed successfully!")
        print("=" * 50)
        print("📊 RESULTS:")
        print(f"   🎬 Frames processed: {stats['processed_frames']:,}")
        print(f"   ⏱️  Processing time: {stats['processing_time']:.2f} seconds")
        print(f"   🚀 Average FPS: {stats['fps']:.2f}")
        print(f"   👥 Total detections: {stats['total_detections']:,}")
        print(f"   🎯 CLIP matches: {stats['clip_matches']:,}")
        print(f"   🆔 Unique tracks: {stats['unique_tracks']}")
        
        if stats['total_detections'] > 0:
            match_rate = (stats['clip_matches'] / stats['total_detections']) * 100
            print(f"   📈 CLIP match rate: {match_rate:.1f}%")
        
        print(f"\n💾 Output saved to: {args.output}")
        
        # Performance assessment
        if stats['fps'] > 15:
            print("✅ Performance: Excellent for real-time processing")
        elif stats['fps'] > 5:
            print("⚠️  Performance: Good for offline processing")
        else:
            print("🐌 Performance: Consider using GPU or reducing video resolution")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

def batch_process():
    """
    Batch process multiple videos or queries
    """
    print("🔄 Batch Processing Mode")
    print("=" * 30)
    
    # Example batch configuration
    batch_config = [
        {
            "video": "video1.mp4",
            "query": "person wearing red shirt",
            "output": "output1.mp4"
        },
        {
            "video": "video2.mp4", 
            "query": "woman in blue dress",
            "output": "output2.mp4"
        },
        {
            "video": "video3.mp4",
            "query": "man with black jacket",
            "output": "output3.mp4"
        }
    ]
    
    processor = StandaloneCLIPReID()
    
    for i, config in enumerate(batch_config, 1):
        print(f"\n🎬 Processing batch {i}/{len(batch_config)}")
        print(f"   Video: {config['video']}")
        print(f"   Query: '{config['query']}'")
        
        if not Path(config['video']).exists():
            print(f"   ❌ Skipping: Video not found")
            continue
        
        try:
            # Reset processor for each video
            processor.person_tracks = {}
            processor.next_track_id = 1
            processor.feature_memory.clear()
            processor.track_history.clear()
            
            stats = processor.process_video(
                video_path=config['video'],
                text_query=config['query'],
                output_path=config['output'],
                display=False
            )
            
            print(f"   ✅ Completed: {stats['clip_matches']} matches, {stats['unique_tracks']} tracks")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n🎉 Batch processing completed!")

if __name__ == "__main__":
    # Check if running in batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_process()
    else:
        main()