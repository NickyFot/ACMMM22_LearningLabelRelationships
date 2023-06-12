import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    for root, dirs, filenames in os.walk(sys.argv[1]):
        for filename in filenames:
            filename = os.path.join(root, filename)
            if not filename.endswith('.zip'):
                continue
            os.system("7za x " + filename + f' -aoa -o{sys.argv[2]}')
