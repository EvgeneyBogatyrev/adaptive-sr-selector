import argparse
from model import SRSelector


def parse_args():
    parser = argparse.ArgumentParser(prog='SR selector')
    parser.add_argument('video_path', help='Path to video')
    parser.add_argument('-f', '--frame_step', default=1, type=int, help='Step to process frames with')
    parser.add_argument('--tmp_dir', default="./tmp", help='Folder where to store tmp data')
    parser.add_argument('--no_split_scenes', action='store_true', default=True, help='Use scene detector or not')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    selector = SRSelector(frame_step=args.frame_step, split_to_scenes=not args.no_split_scenes, tmp_dir=args.tmp_dir)
    print(selector(args.video_path))
    
if __name__ == "__main__":
    main()

