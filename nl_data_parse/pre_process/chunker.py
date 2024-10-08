"""chunk: merge short sequences to one long sequence

paragraphs (List[str]):
    all original lines, each itme in list is a str line
    e.g.: ['hello world!', 'hello python!']
chunked_paragraphs (List[str]):
    each line has the same length as paragraphs as possibly, each itme in list is a str line
    e.g.: ['hello world! hello python!']
segments (List[List[str]]):
    all lines after cut, each itme in list is a cut word list
    e.g.: [['hello', 'world!'], ['hello', 'python!']]
chunked_segments (List[List[str]]):
    each line has the same length as segments as possibly, each itme in list is a cut word list
    e.g.: [['hello', 'world!', 'hello', 'python!']]
"""
import re
from typing import List

import numpy as np
from tqdm import tqdm

from utils.visualize import TextVisualize


class ToChunkedParagraphs:
    """chunk without dropping any context

    Usage:
        >>> ToChunkedParagraphs(max_length=5).from_paragraphs(['abcdefghijklmn'])
        ['abcde', 'fghij', 'klmn']

        >>> ToChunkedParagraphs(max_length=5).from_paragraphs(['abc', 'def', 'ghi', 'jk', 'lmn'])
        ['abcde', 'fghij', 'klmn']

        >>> ToChunkedParagraphs(max_length=5, min_length=3).from_paragraphs(['abc', 'def', 'ghi', 'jk', 'lmn'])
        ['abc', 'def', 'ghi', 'jklmn']

        >>> ToChunkedParagraphs(max_length=5).from_paragraphs(['abc,def.ghijk,lmn.'])
        ['abc,', 'def.', 'ghijk', ',lmn.']
    """

    full_stop_rx = re.compile(r'.*[。\.!?！？]', re.DOTALL)
    half_stop_rx = re.compile(r'.*[\];；,，、》）}]', re.DOTALL)
    newline_stop_rx = re.compile(r'.+\n', re.DOTALL)
    max_length = 512
    min_length = None

    def __init__(self, **kwargs):
        """

        Args:
            max_length:
                chunked will be stopped before len(seq) < max_length
            min_length:
                chunked stopped after len(seq) > min_length
        """
        self.__dict__.update(kwargs)

    def from_paragraphs(self, paragraphs: List[str]):
        """
        ['abc', 'def', 'ghi', 'jk', 'lmn'] -> ['abcde', 'fghij', 'klmn']
        """
        chunked_paragraphs = []
        chunk = ''

        for p in paragraphs:
            chunk += p

            if len(chunk) > self.max_length:
                chunks = self.from_paragraph(chunk)
                chunked_paragraphs.extend(chunks[:-1])
                chunk = chunks[-1]
            elif self.min_length and len(chunk) >= self.min_length:
                chunked_paragraphs.append(chunk)
                chunk = ''

        if chunk:
            chunked_paragraphs.append(chunk)

        return chunked_paragraphs

    def from_paragraph(self, paragraph: str) -> List[str]:
        """
        'abcdefghijklmn' -> ['abcde', 'fghij', 'klmn']
        """
        chunked_paragraphs = []
        rest = paragraph

        while True:
            if len(rest) <= self.max_length:
                if rest:
                    chunked_paragraphs.append(rest)
                break

            tail = rest[self.max_length:]

            left_f, right_f, is_matched_f = self.truncate_by_stop_symbol(rest[:self.max_length], self.full_stop_rx)
            left_n, right_n, is_matched_n = self.truncate_by_stop_symbol(rest[:self.max_length], self.newline_stop_rx)

            if is_matched_f and is_matched_n:
                if len(left_f) >= len(left_n):
                    sect, rest = left_f, right_f + tail
                else:
                    sect, rest = left_n, right_n + tail
            elif is_matched_f:
                sect, rest = left_f, right_f + tail
            elif is_matched_n:
                sect, rest = left_n, right_n + tail
            else:
                left_h, right_h, is_matched_h = self.truncate_by_stop_symbol(rest[:self.max_length], self.half_stop_rx)
                if is_matched_h:
                    sect, rest = left_h, right_h + tail
                else:
                    sect, rest = rest[:self.max_length], rest[self.max_length:]

            chunked_paragraphs.append(sect)

        return chunked_paragraphs

    @staticmethod
    def truncate_by_stop_symbol(line, pattern: re.Pattern) -> tuple:
        m = re.match(pattern, line)

        if m:
            left = line[:m.span()[1]]
            right = line[m.span()[1]:]
            is_matched = True
        else:
            left, right = line, ''
            is_matched = False

        return left, right, is_matched


class RandomToChunkedParagraphs(ToChunkedParagraphs):
    min_choices = 2
    max_choices = None

    def from_paragraphs(self, paragraphs: List[str]):
        """split paragraphs in pieces randomly, and then chunk them one by one"""
        n = len(paragraphs)
        max_choices = self.max_choices or n
        max_choices = min(max_choices, n)
        min_choices = min(self.min_choices, max_choices - 1)
        choices = np.random.randint(min_choices, max_choices)
        idxes = np.random.choice(range(n), size=choices, replace=False)
        idxes = np.sort(idxes)
        idxes = np.append(idxes, n)

        s = 0
        chunks = []
        for i in idxes:
            _chunk = paragraphs[s: i]
            chunks.extend(super().from_paragraphs(_chunk))
            s = i

        return chunks

    def from_paragraph(self, paragraph: str) -> List[str]:
        raise NotImplementedError


class ToChunkedSegments:
    """chunk without dropping any context"""

    full_stop_tokens = set('。.!?！？')
    half_stop_tokens = set('];；,，、》）}')
    newline_stop_tokens = set('\n')
    max_length = 512
    min_length = None
    verbose = False

    def __init__(self, **kwargs):
        """see also `ToChunkedParagraphs`"""
        self.__dict__.update(kwargs)

    def from_segments(self, segments: List[List[str]]):
        chunked_segments = []
        chunked_segment = []

        if self.verbose:
            segments = tqdm(segments, desc=TextVisualize.highlight_str('Chunk segments'))

        for p in segments:
            chunked_segment += p

            if len(chunked_segment) > self.max_length:
                chunks = self.from_segment(chunked_segment)
                chunked_segments.extend(chunks[:-1])
                chunked_segment = chunks[-1]
            elif self.min_length and len(chunked_segment) >= self.min_length:
                chunked_segments.append(chunked_segment)
                chunked_segment = []

        if chunked_segment:
            chunked_segments.append(chunked_segment)

        return chunked_segments

    def from_segment(self, segment):
        chunked_segments = []
        rest = segment

        while True:
            if len(rest) <= self.max_length:
                if rest:
                    chunked_segments.append(rest)
                break

            keep = rest[:self.max_length]
            tail = rest[self.max_length:]

            left_f, right_f, is_matched_f = self.truncate_by_stop_token(keep, self.full_stop_tokens)
            left_n, right_n, is_matched_n = self.truncate_by_stop_token(keep, self.newline_stop_tokens)

            if is_matched_f and is_matched_n:
                if len(left_f) >= len(left_n):
                    sect, rest = left_f, right_f + tail
                else:
                    sect, rest = left_n, right_n + tail
            elif is_matched_f:
                sect, rest = left_f, right_f + tail
            elif is_matched_n:
                sect, rest = left_n, right_n + tail
            else:
                left_h, right_h, is_matched_h = self.truncate_by_stop_token(keep, self.half_stop_tokens)
                if is_matched_h:
                    sect, rest = left_h, right_h + tail
                else:
                    sect, rest = keep, tail

            chunked_segments.append(sect)

        return chunked_segments

    @staticmethod
    def truncate_by_stop_token(segment, token: set) -> tuple:
        is_matched = False
        left, right = segment, []

        for i, s in enumerate(segment):
            if s in token:
                left = segment[:i + 1]
                right = segment[i + 1:]
                is_matched = True

        return left, right, is_matched


class RandomToChunkedSegments(ToChunkedSegments):
    min_choices = 2
    max_choices = None

    def from_segments(self, segments: List[List[str]]):
        """split segments in pieces randomly, and then chunk them one by one"""
        n = len(segments)
        max_choices = self.max_choices or n
        max_choices = min(max_choices, n)
        choices = np.random.randint(self.min_choices, max_choices)
        idxes = np.random.choice(range(n), size=choices, replace=False)
        idxes = np.sort(idxes)
        idxes = np.append(idxes, n)

        s = 0
        chunks = []
        for i in idxes:
            _chunk = segments[s: i]
            chunks.extend(super().from_segments(_chunk))
            s = i

        return chunks

    def from_segment(self, segment):
        raise NotImplementedError


class WindowsToChunkedSegments(ToChunkedSegments):
    """chunk by window sliding, it will have duplicated context"""

    def from_segments(self, segments):
        pass


class ToChunkedParagraph:
    """chunk by dropping some context"""

    max_length: int = 512

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def from_segment(self, segment):
        raise NotImplementedError

    def from_paragraph(self, paragraph):
        raise NotImplementedError


class HeadToChunkedParagraph(ToChunkedParagraph):
    def from_segment(self, segment):
        chunked_paragraph = ''

        for seg in segment:
            if len(chunked_paragraph + seg) > self.max_length:
                break

            chunked_paragraph += seg

        return chunked_paragraph


class TailToChunkedParagraph(HeadToChunkedParagraph):
    def from_segment(self, segment):
        return super().from_segment(segment[::-1])


class MiddleToChunkedParagraph(ToChunkedParagraph):
    def from_segment(self, segment):
        n = len(segment)
        mid_n = n // 2
        word_n = [len(s) for s in segment]

        flag = 'r'
        r, l = 0, 1
        pl = 0
        keep = set()
        while True:
            if flag == 'r':
                i = mid_n + r
                a = word_n[i]
                if pl + a > self.max_length:
                    break
                else:
                    keep.add(i)
                    r += 1
                    flag = 'l'
            else:
                i = mid_n - l
                a = word_n[i]
                if pl + a > self.max_length:
                    break
                else:
                    keep.add(i)
                    l += 1
                    flag = 'r'

            pl += a

        chunked_paragraph = ''.join([segment[i] for i in sorted(keep)])
        return chunked_paragraph


class TwoPiecesToChunkedParagraph(ToChunkedParagraph):
    """truncate for 2 pieces: head + tail"""
    ratios = (0.5, 0.5)

    def from_segment(self, segment):
        if sum(len(s) for s in segment) <= self.max_length:
            return ''.join(segment)

        max_seq = [int(self.max_length * r) for r in self.ratios]
        chunked_paragraph = ''
        chunked_paragraph += HeadToChunkedParagraph(max_length=max_seq[0]).from_segment(segment)
        chunked_paragraph += TailToChunkedParagraph(max_length=max_seq[1]).from_segment(segment)
        return chunked_paragraph


class ThreePiecesToChunkedParagraph(ToChunkedParagraph):
    """truncate for 3 pieces: head + middle + tail"""
    ratios = (0.4, 0.2, 0.4)

    def from_segment(self, segment):
        if sum(len(s) for s in segment) <= self.max_length:
            return ''.join(segment)

        max_seq = [int(self.max_length * r) for r in self.ratios]
        chunked_paragraph = ''
        chunked_paragraph += HeadToChunkedParagraph(max_length=max_seq[0]).from_segment(segment)
        chunked_paragraph += MiddleToChunkedParagraph(max_length=max_seq[1]).from_segment(segment)
        chunked_paragraph += TailToChunkedParagraph(max_length=max_seq[2]).from_segment(segment)
        return chunked_paragraph


class KPiecesToChunkedParagraph(ToChunkedParagraph):
    """average truncate for k pieces, each piece keep the head"""
    k: int = 5

    def from_segment(self, segment):
        word_n = [len(s) for s in segment]
        total_n = sum(word_n)

        if total_n <= self.max_length:
            return ''.join(segment)

        mean_n = total_n // self.k
        _max_length = self.max_length // self.k

        chunked_paragraph = ''
        tmp_n = 0
        tmp_segment = []
        for n, s in zip(word_n, segment):
            if tmp_n >= mean_n:
                chunked_paragraph += HeadToChunkedParagraph(max_length=_max_length).from_segment(tmp_segment)
                tmp_n = 0
                tmp_segment = []

            tmp_n += n
            tmp_segment.append(s)
        return chunked_paragraph
