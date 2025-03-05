# なんか消えたconfigファイルを復元するためのプログラム
from util import (
    setup_config,
)

if __name__ == "__main__":
    cfg = setup_config()
    print("setup")