import re
import unicodedata
from typing import List, Optional
from functools import partial


class Base:
    def from_article(self, article: str):
        return self.from_paragraph(article)

    def from_paragraph(self, paragraph: str):
        raise NotImplemented

    def from_paragraphs(self, paragraphs: List[str]):
        return map(self.from_paragraph, paragraphs)

    def from_segment(self, segment: List[str]):
        return self.from_paragraphs(segment)

    def from_segments(self, segments: List[List[str]]):
        return map(self.from_segment, segments)


class FilterBlank(Base):
    """
    Usage:
        >>> FilterBlank().from_paragraph('hello world!')
        'helloworld!'
    """

    def from_paragraph(self, paragraph: str):
        return ''.join(paragraph.split())


class FileterDuplicateBlank(Base):
    """
    Usage:
        >>> FileterDuplicateBlank().from_paragraph('  hello  world!  ')
        'hello world!'
        >>> FileterDuplicateBlank().from_paragraph('你 好')
        '你好'
    """

    def __init__(self, blank_chars=' \t'):
        self.blank_chars = blank_chars

    def from_paragraph(self, paragraph: str):
        new_paragraph = []
        for i, c in enumerate(paragraph):
            if len(new_paragraph) == 0:
                if c not in self.blank_chars:
                    new_paragraph.append(c)
            elif i == len(paragraph) - 1:
                if c not in self.blank_chars:
                    new_paragraph.append(c)
            elif c in self.blank_chars:
                if new_paragraph[-1].isascii() and new_paragraph[-1] not in self.blank_chars:
                    new_paragraph.append(c)
            else:
                new_paragraph.append(c)
        new_paragraph = "".join(new_paragraph).strip()
        return new_paragraph


class FilterShort(Base):
    """
    >>> FilterShort(5).from_paragraph('hello world!')
    ''
    >>> FilterShort(20).from_paragraph('hello world!')
    'hello world!'
    """

    def __init__(self, min_len):
        self.min_len = min_len

    def from_paragraph(self, paragraph: str):
        return paragraph if len(paragraph) >= self.min_len else ''

    def from_paragraphs(self, paragraphs: List[str]):
        return [p for p in paragraphs if len(p) >= self.min_len]


class FilterPattern(Base):
    """
    Usage:
        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> FilterPattern(utf8_pattern_dict['en_pr']).from_paragraph('hello world!')
        'hello world'

        >>> FilterPattern(utf8_pattern_dict['cjk_pr'], utf8_pattern_dict['en_pr_double']).from_paragraph('你好 世界！')
        '你好 世界'
    """

    def __init__(self, *pattern: str or re.Pattern):
        patterns = []
        for p in pattern:
            if isinstance(p, str):
                patterns.append(p)
            else:
                patterns.append(p.pattern)
        self.pattern = re.compile('|'.join(patterns))

    def from_paragraph(self, paragraph: str):
        return re.sub(self.pattern, '', paragraph)


class KeepPattern(Base):
    """
    Usage:
        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> KeepPattern(utf8_pattern_dict['en'], re.compile(' ')).from_paragraph('hello world!你好 世界！')
        'hello world '

        >>> KeepPattern(utf8_pattern_dict['zh']).from_paragraph('hello world!你好 世界！')
        '你好世界'
    """

    def __init__(self, *pattern: str or re.Pattern):
        patterns = []
        for p in pattern:
            if isinstance(p, str):
                patterns.append(p)
            else:
                patterns.append(p.pattern)
        self.pattern = re.compile('|'.join(patterns))

    def from_paragraph(self, paragraph: str):
        return ''.join(re.findall(self.pattern, paragraph))


class Lower(Base):
    def from_paragraph(self, paragraph: str):
        return paragraph.lower()


class StripAccents(Base):
    """
    Usage:
        >>> StripAccents().from_paragraph('ü')
        'u'
    """

    def from_paragraph(self, paragraph: str):
        paragraph = unicodedata.normalize("NFD", paragraph)
        output = []
        for char in paragraph:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)


class Normalize(Base):
    model: Optional

    def from_paragraph(self, paragraph: str):
        paragraph = self.model.normalize(paragraph)
        return paragraph


class NormalizeZh(Normalize):
    """convert number and punctuation to Chineese

    >>> NormalizeZh().from_paragraph('2024年7月1日')
    '二零二四年七月一日'
    >>> NormalizeZh().from_paragraph('3.14')
    '三点一四'
    >>> NormalizeZh().from_paragraph('100元')
    '一百元'
    >>> NormalizeZh().from_paragraph('unet-transformer')
    'unet减transformer'
    """

    def __init__(self, corner_marks_mapping=None):
        from tn.chinese.normalizer import Normalizer  # pip install WeTextProcessing
        self.model = Normalizer()


class NormalizeEn(Normalize):
    """convert number to English
    >>> NormalizeEn().from_paragraph('20%')
    'twenty percent'
    """

    def __init__(self, corner_marks_mapping=None):
        from tn.english.normalizer import Normalizer  # pip install WeTextProcessing
        self.model = Normalizer()


class NormalizeNum(Base):
    def from_paragraph(self, paragraph: str):
        import inflect
        inflect_parser = inflect.engine()

        new_paragraph = []
        while True:
            r = re.search('\d+', paragraph)
            if r:
                span = r.span()
                s, e = span

                new_paragraph.append(paragraph[: s])
                new_paragraph.append(self._covert(paragraph[s: e]))

                paragraph = paragraph[e:]
            else:
                break

        new_paragraph.append(paragraph)
        return ''.join(new_paragraph)

    def _covert(self, num):
        raise NotImplemented


class Num2Zh(NormalizeNum):
    """convert Arabic numerals to Chinese"""

    def __init__(self, mode='low'):
        from cn2an import An2Cn   # pip install cn2an
        self.model = An2Cn()
        self.mode = mode

    def _covert(self, num):
        return self.model.an2cn(num, self.mode)


class Num2En(NormalizeNum):
    """convert Arabic numerals to English"""

    def __init__(self):
        import inflect
        self.model = inflect.engine()

    def _covert(self, num):
        return self.model.number_to_words(num)
