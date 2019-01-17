from AlbumInfoParser import AlbumInfoParser
from AlbumInfoParser import AlbumInfo


ALBUM_INFO_PATH = "../../../album_info/albumDetail.txt"
ALBUM_ID_PATH = "../../../album_info/albumList.txt"


if __name__ == "__main__":
    parser = AlbumInfoParser(ALBUM_INFO_PATH, ALBUM_ID_PATH)

