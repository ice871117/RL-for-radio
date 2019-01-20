"""A parser for Album information in nextradio.
Album info is organized in json:
{"album":{"album_cover":"http://imgcache.qq.com/fm/photo/album/rmid_album_360/i/G/000127VY05pliG.jpg?time=1508857753","album_desc":"楼房大小的血睛鬃毛狮，力大无穷的紫睛金毛猿，毁天灭地的九头蛇皇，携带着毁灭雷电的恐怖雷龙……","album_id":"rd000127VY05pliG","album_is_charge":0,"album_is_collected":0,"album_name":"盘龙1（上译忠臣版）","album_owner":{"anchor_album_num":0,"anchor_create_timestamp":1445250251,"anchor_desc":"暂无介绍","anchor_fans_num":1682,"anchor_gender":-1,"anchor_id":"1838970","anchor_logo":"http://imgcache.qq.com/fm/photo/singer/rmid_singer_240/S/e/002QC9532OpfSe.jpg?time=1433474289","anchor_name":"上译忠臣","anchor_show_num":982},"album_update_timestamp":1436120191,"category":[{"category_id":"39092","category_name":"有声小说","level":1},{"category_id":"38954","category_name":"玄幻奇幻","level":2}],"is_serial":0,"show_num":250,"show_sort_type":1},"category_list":[{"category_id":"39092","category_name":"有声小说","level":1},{"category_id":"38954","category_name":"玄幻奇幻","level":2}],"h5_url":"http://fm.qzone.qq.com/luobo/radio?_wv=4097&aid=rd000127VY05pliG&jumpapp=0","is_lost":0,"ret":0,"schema":"nextradio://a/albumdetail?albumid=rd000127VY05pliG"}

"""
import re
import jieba
import json
import logging


class AlbumInfo:
    # static re pattern which will remove any punctuation in the target string
    __invalid_char_re = re.compile(r"[,.-\\，。《》<>\"“”、\(\)（）!！\?；：;:\[\]【】#\s·—]")

    def __init__(self, album_id='', name='', category1='', category2='', anchor='', album_desc=''):
        """this constructor might be time consuming, for it will perform string processing and word cutting"""
        # album id must already be stripped
        self.album_id = album_id
        # a pure name field without punctuation
        self.name = AlbumInfo.remove_invalid_char(name)
        # a list contains all words in name
        # e.g. '郭德纲相声选' will be turned into ['郭德纲', '相声', '选']
        self.name_cut = AlbumInfo.cut_to_words(self.name)
        # category1 and category2 must already be stripped
        self.category1 = AlbumInfo.remove_invalid_char(category1)
        self.category1_cut = AlbumInfo.cut_to_words(self.category1)
        self.category2 = AlbumInfo.remove_invalid_char(category2)
        self.category2_cut = AlbumInfo.cut_to_words(self.category2)
        # anchor must already be stripped
        self.anchor = AlbumInfo.remove_invalid_char(anchor)
        self.anchor_cut = AlbumInfo.cut_to_words(self.anchor)
        # album_desc must already be stripped
        self.album_desc = AlbumInfo.remove_invalid_char(album_desc)
        self.album_desc_cut = AlbumInfo.cut_to_words(self.album_desc)

    @staticmethod
    def remove_invalid_char(name):
        stripped = name.strip()
        return re.sub(AlbumInfo.__invalid_char_re, '', stripped)

    @staticmethod
    def cut_to_words(sentence):
        return list(jieba.cut(sentence))

    def is_valid(self):
        return len(self.name) > 0


class AlbumInfoParser:

    BOUND = '##'

    def __init__(self, album_info_path, album_id_path):
        # 1st process album info
        self.albums = []
        with open(album_info_path, 'r') as album_info_file:
            for line in album_info_file.readlines():
                try:
                    json_obj = json.loads(line)
                    if json_obj.get('album') is None:
                        self.albums.append(AlbumInfo(album_id=json_obj.get('album_id')))
                        # print("no valid info for " + str(json_obj))
                        continue
                    album_name = json_obj.get('album').get('album_name').strip()
                    album_id = json_obj.get('album').get('album_id').strip()
                    anchor = json_obj.get('album').get('album_owner').get('anchor_name').strip()
                    category_list = json_obj.get('category_list')
                    category1 = category_list[0].get('category_name').strip() if len(category_list) > 0 else ''
                    category2 = category_list[1].get('category_name').strip() if len(category_list) > 1 else ''
                    desc = json_obj.get('album').get('album_desc').strip()
                    self.albums.append(AlbumInfo(album_id, album_name, category1, category2, anchor))
                except BaseException as e:
                    self.albums.append(AlbumInfo(album_id='exception'))
                    logging.exception(e)

        # 2nd process album_ids
        self.album_ids = []
        with open(album_id_path, 'r') as album_id_file:
            i = 0
            for line in album_id_file.readlines():
                album_id = line.strip()
                self.album_ids.append(album_id)
                self.albums[i].album_id = album_id
                i += 1

        # build vocabulary
        self.vocabulary = []
        self.build_vocabulary()
        print("------- AlbumInfoParser init() done -------")

    def build_vocabulary(self):
        for album in self.albums:
            if album.is_valid():
                self.add_to_vocabulary(album.name_cut)
                self.add_to_vocabulary(album.category1_cut)
                self.add_to_vocabulary(album.category2_cut)
                self.add_to_vocabulary(album.album_desc_cut)

    def add_to_vocabulary(self, word_cut_list):
        for word in word_cut_list:
            self.vocabulary.append(word)
        self.vocabulary.append(AlbumInfoParser.BOUND)

    def release_vocabulary(self):
        del self.vocabulary


