"""split: context split to different fine-grained type

article (str):
    all lines fatten to a string
    e.g.: 'hello world! hello python!'
paragraphs (List[str]):
    all original lines, each itme in list is a str line
    e.g.: ['hello world!', 'hello python!']
paragraph (str)
    one paragraph can be an article
    e.g.: 'hello world!'
chunked_paragraphs (List[str]):
    each line has the same length as paragraphs as possibly, each itme in list is a str line
    e.g.: ['hello world! hello python!']
segments (List[List[str]]):
    all lines after cut, each itme in list is a cut word list
    e.g.: [['hello', 'world!'], ['hello', 'python!']]
segment (List[str])
    one segment can be a paragraphs
    e.g.: ['hello', 'world!']
word (str)
    items of segment
    e.g.: 'hello'
chunked_segments (List[List[str]]):
    each line has the same length as segments as possibly, each itme in list is a cut word list
    e.g.: [['hello', 'world!', 'hello', 'python!']]
"""
import re
import unicodedata
from typing import List
from tqdm import tqdm
from utils.excluded.charset_dict import utf8_int_dict
from utils.visualize import TextVisualize


class ToArticle:
    def __init__(self, sep='\n'):
        self.sep = sep

    def article_from_paragraphs(self, paragraphs: List[str]) -> str:
        return self.sep.join(paragraphs)


class ToParagraphs:
    def __init__(self, sep='\n', keep_sep=False):
        self.sep = sep
        self.keep_sep = keep_sep

    def from_article(self, article: str) -> List[str]:
        if self.keep_sep:
            paragraphs = [line + self.sep for line in article.split(self.sep)]
        else:
            paragraphs = article.split(self.sep)

        return paragraphs

    def from_segments(self, segments: List[List[str]]) -> List[str]:
        return [self.sep.join(s) for s in segments]

    def from_segments_with_zh_en_mix(self, segments: List[List[str]]) -> List[str]:
        """['你', '好', 'hello', 'world', '你', '好'] -> '你好 hello world 你好'"""
        paragraphs = []
        for segment in segments:
            _segment = []
            tmp = ''
            for s in segment:
                if s.isascii():
                    if tmp:
                        _segment.append(tmp)
                    _segment.append(s)
                    tmp = ''
                else:
                    tmp += s

            if tmp:
                _segment.append(tmp)

            paragraph = ' '.join(_segment)
            paragraphs.append(paragraph)
        return paragraphs


class ToSegment:
    """
    Usage:
        >>> text = 'hello [MASK]!'
        >>> ToSegment().from_paragraph(text)
        ['hello', '[', 'MASK', ']', '!']

        >>> ToSegment(sep='').from_paragraph(text)
        ['h', 'e', 'l', 'l', 'o', ' ', '[', 'M', 'A', 'S', 'K', ']', '!']

        >>> ToSegment(sp_tokens=['[MASK]']).from_paragraph(text)
        ['hello', '[MASK]', '!']

        >>> from .cleaner import Lower
        >>> ToSegment(cleaner=Lower().from_paragraph).from_paragraph(text)
        >>> ['hello', '[', 'mask', ']', '!']
    """

    def __init__(
            self,
            sep=None, sep_pattern=None,
            sp_tokens=(), cleaner=None, deep_split_funcs=(),
            is_split_punctuation=True, is_word_piece=False, vocab=None,
            **kwargs
    ):
        """

        Args:
            sep: split seq symbol, for `shallow_split()`
            sep_pattern (str or re.Pattern): re pattern seq, not work while `sep` is set, for `shallow_split()`
            sp_tokens: for `from_paragraph_with_sp_tokens()`
            cleaner (Callable): a func that accepted one parameter of text
            is_split_punctuation: for `deep_split()`
            is_word_piece: for `deep_split()`
            vocab: for `from_word_by_word_piece()`
        """
        sp_pattern = []
        for s in sp_tokens:
            for a in '\\[]{}.*?|':
                s = s.replace(a, '\\' + a)
            sp_pattern.append(s)
        self.sp_tokens = sp_tokens
        self.sp_pattern = re.compile('|'.join(sp_pattern))

        self.sep = sep
        self.sep_pattern = sep_pattern
        self.cleaner = cleaner
        self.vocab = vocab

        self.deep_split_funcs = []
        if is_split_punctuation:
            self.deep_split_funcs.append(self.from_paragraph_with_punctuation)
        if is_word_piece:
            self.deep_split_funcs.append(self.from_word_by_word_piece)
        self.deep_split_funcs.extend(deep_split_funcs)

    def from_paragraph(self, paragraph):
        if self.sp_tokens:
            _segment = self.from_paragraph_with_sp_tokens(paragraph)
        else:
            _segment = [paragraph]

        segment = []
        for text in _segment:
            if text in self.sp_tokens:
                segment.append(text)
            else:
                if self.cleaner:
                    text = self.cleaner(text)
                seg = self.shallow_split(text)
                seg = self.deep_split(seg)
                segment += seg
        return segment

    def from_paragraph_with_sp_tokens(self, paragraph):
        """
        >>> text = 'hello [MASK]!'
        >>> ToSegment(sp_tokens=['[MASK]']).from_paragraph_with_sp_tokens(text)
        ['hello', '[MASK]', '!']
        """
        segment = []
        while True:
            r = self.sp_pattern.search(paragraph)
            if r:
                span = r.span()
                if span[0] == span[1]:
                    break
                segment.append(paragraph[: span[0]])
                segment.append(paragraph[span[0]: span[1]])
                paragraph = paragraph[span[1]:]
            else:
                break

        if paragraph:
            segment.append(paragraph)

        return segment

    def shallow_split(self, paragraph):
        """base on sep token or sep pattern

        >>> text = 'hello world1.hello world2!hello world3!'

        >>> ToSegment(sep='!').shallow_split(text)
        ['hello world1.hello world2', 'hello world3', '']

        >>> ToSegment(sep_pattern=r'.*?[\.!]').shallow_split(text)
        ['hello world1.', 'hello world2!', 'hello world3!']
        """
        if self.sep == '':
            segment = list(paragraph)
        elif self.sep is not None:
            segment = paragraph.split(self.sep)
        elif self.sep_pattern is not None:
            if hasattr(self.sep_pattern, 'findall'):
                segment = self.sep_pattern.findall(paragraph)
            else:
                segment = re.findall(self.sep_pattern, paragraph)
        else:
            segment = paragraph.split(self.sep)
        return segment

    def deep_split(self, segment):
        def split(segment, func):
            _segment = []
            for seg in segment:
                _segment += func(seg)
            return _segment

        for func in self.deep_split_funcs:
            segment = split(segment, func)

        return segment

    @staticmethod
    def _is_punctuation(char):
        """Checks whether `char` is a punctuation character.
        treat all non-letter/number ASCII as punctuation"""
        cp = ord(char)
        for span in utf8_int_dict['en_pr']:
            if span[0] <= cp <= span[1]:
                return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def from_paragraph_with_punctuation(self, paragraph):
        """'hello,world' -> ['hello', ',', 'world']"""
        output = []
        # for text in seg:
        chars = list(paragraph)
        i = 0
        start_new_word = True
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def from_word_by_word_piece(self, word: str):
        """'uncased' -> ['un', '##cased']"""
        chars = list(word)
        is_bad = False
        start = 0
        segment = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            segment.append(cur_substr)
            start = end

        if is_bad:
            segment = [word]

        return segment

    def from_segment_by_word_piece_v2(self, segment: List[str], timestamp):
        """['un@@', 'cased'] -> ['uncased']"""
        groups = []
        cur = []
        for i, t in enumerate(segment):
            cur.append(i)

            if not t.endswith('@@'):
                groups.append(cur)
                cur = []

        assert not cur, f'the last word is not allow to be end with "@@" which is {cur}, please check about it'

        new_segment = []
        new_timestamps = []
        for group in groups:
            s = ''
            t = []
            for idx in group:
                ss = segment[idx]
                if ss.endswith('@@'):
                    ss = ss[:-2]
                s += ss
                if t:
                    t[-1] = timestamp[idx][-1]
                else:
                    t = timestamp[idx]
            new_segment.append(s)
            new_timestamps.append(t)

        return new_segment, new_timestamps

    def from_segment_by_merge_single_char(self, segment: List[str], timestamp):
        """['n', 'l', 'p', 'is', 'a', 'shorthand', 'word'] -> ['NLP', 'is', 'a', 'shorthand', 'word']"""
        groups = []
        cur = []
        flag = False
        for i, s in enumerate(segment):
            if len(s) == 1 and s.isalpha() and s.isascii():  # single en char
                if flag:
                    cur.append(i)
                else:
                    if cur:
                        groups.append(cur)
                    cur = [i]
                flag = True
            else:
                if cur:
                    groups.append(cur)
                cur = [i]
                flag = False

        if cur:
            groups.append(cur)

        new_segment = []
        new_timestamps = []
        for group in groups:
            s = ''
            t = []
            for idx in group:
                ss = segment[idx]
                s += ss
                if t:
                    t[-1] = timestamp[idx][-1]
                else:
                    t = timestamp[idx]
            if len(group) > 1:
                s = s.upper()
            new_segment.append(s)
            new_timestamps.append(t)

        return new_segment, new_timestamps


class ToSegments:
    def __init__(self, verbose=False, **segment_kwargs):
        self.verbose = verbose
        self.to_segment = ToSegment(**segment_kwargs)

    def from_paragraphs(self, paragraphs: List[str]) -> List[List[str]]:
        """see also cleaner
        Usage:
            >>> ToSegments().from_paragraphs(['hello world!'])
            [['hello', 'world', '!']]
        """
        segments = []
        if self.verbose:
            paragraphs = tqdm(paragraphs, desc=TextVisualize.highlight_str('Split paragraphs'))

        for line in paragraphs:
            seg = self.to_segment.from_paragraph(line)
            segments.append(seg)

        return segments

    def from_paragraphs_by_jieba(self, paragraphs: List[str]) -> List[List[str]]:
        """see also cleaner
        Usage:
            >>> ToSegments(is_split_punctuation=False).from_paragraphs_by_jieba(['你好 世界！'])
            [['你好', ' ', '世界', '！']]
        """
        import jieba

        if self.to_segment.cleaner:
            paragraphs = map(self.to_segment.cleaner, paragraphs)
        segments = map(jieba.lcut, paragraphs)
        segments = map(self.to_segment.deep_split, segments)

        return list(segments)
