import argparse

def parse_args_path():
    parser = argparse.ArgumentParser(description="Image Navigator")
    parser.add_argument("path", type=str, help="Path de la imagen")
    args = parser.parse_args()
    return args.path