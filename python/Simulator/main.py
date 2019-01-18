from AlbumInfoParser import AlbumInfoParser
from AlbumInfoParser import AlbumInfo


# all constants defined here
# data file path
ALBUM_INFO_PATH = "../../../album_info/albumDetail.txt"
ALBUM_ID_PATH = "../../../album_info/albumList.txt"
# the words upper limit
VOCABULARY_SIZE = 5000


if __name__ == "__main__":
    parser = AlbumInfoParser(ALBUM_INFO_PATH, ALBUM_ID_PATH)


