"""chunk: merge short sequences to one long sequence

paragraphs (List[str]):
    all original lines, each item in list is a str line
    e.g.: ['hello world!', 'hello python!']
chunked_paragraphs (List[str]):
    each line has the same length as paragraphs as possibly, each item in list is a str line
    e.g.: ['hello world! hello python!']
segments (List[List[str]]):
    all lines after cut, each item in list is a cut word list
    e.g.: [['hello', 'world!'], ['hello', 'python!']]
chunked_segments (List[List[str]]):
    each line has the same length as segments as possibly, each item in list is a cut word list
    e.g.: [['hello', 'world!', 'hello', 'python!']]
"""
import re
from typing import List

import numpy as np
from tqdm import tqdm

from . import spliter


class ToChunked:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def from_paragraphs(self, paragraphs: List[str]):
        """same to from_segment"""
        raise NotImplemented

    def from_paragraph(self, paragraph: str):
        raise NotImplemented

    def from_segments(self, segments: List[List[str]]):
        raise NotImplemented

    def from_segment(self, segment: List[str]):
        """same to from_segment"""
        raise NotImplemented

    @staticmethod
    def count_paragraphs_len(paragraphs):
        return sum([len(p) for p in paragraphs])


class ToChunkedParagraphs(ToChunked):
    """chunk without dropping any context"""
    max_len = 512

    def from_segment(self, segment: List[str]) -> List[str]:
        return self.from_paragraphs(segment)


class WindowToChunkedParagraphs(ToChunkedParagraphs):
    """simple concatenate by window sliding"""

    def from_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Usage:
            >>> paragraphs = [f'hello! world{i}!' for i in range(5)]
            >>> WindowToChunkedParagraphs(max_len=30).from_paragraphs(paragraphs)
            ['hello! world0!hello! world1!', 'hello! world2!hello! world3!', 'hello! world4!']

        """
        chunked_paragraphs = []
        chunk = ''

        for p in paragraphs:
            assert len(p) < self.max_len, f'all input text length must no more than {self.max_len}, or try to use `RetentionToChunkedParagraphs`'

            if len(chunk + p) > self.max_len:
                chunked_paragraphs.append(chunk)
                chunk = ''

            chunk += p

        if chunk:
            chunked_paragraphs.append(chunk)

        return chunked_paragraphs

    def from_paragraph(self, paragraph: str) -> List[str]:
        """
        Usage:
            >>> paragraph = ''.join([f'hello! world{i}!' for i in range(5)])
            >>> WindowToChunkedParagraphs(max_len=30).from_paragraph(paragraph)
            ['hello! world0!hello! world1!he', 'llo! world2!hello! world3!hell', 'o! world4!']
        """
        chunked_paragraphs = []
        s = 0

        while True:
            chunked_paragraphs.append(paragraph[s: s + self.max_len])
            s += self.max_len
            if s > len(paragraph):
                break

        return chunked_paragraphs


class RetentionToChunkedParagraphs(ToChunkedParagraphs):
    """chunk by truncate to keep all chunks' length close to the `max_len` as far as possible

    Args:
        max_len:
            chunked will be stopped before len(seq) < max_len
        min_len:
            chunked stopped after len(seq) > min_len

    """

    full_stop_rx = re.compile(r'.*[。\.!?！？]', re.DOTALL)
    half_stop_rx = re.compile(r'.*[\];；,，、》）}]', re.DOTALL)
    newline_stop_rx = re.compile(r'.+\n', re.DOTALL)
    min_len: int

    def from_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Usage:
            >>> paragraphs = [f'hello! world{i}!' for i in range(5)]
            >>> RetentionToChunkedParagraphs(max_len=25).from_paragraphs(paragraphs)
            ['hello! world0!hello!', ' world1!hello! world2!', 'hello! world3!hello!', ' world4!']

            >>> RetentionToChunkedParagraphs(max_len=5, min_len=10).from_paragraphs(['abc', 'def', 'ghi', 'jk', 'lmn'])
            ['hello! world0!', 'hello! world1!', 'hello! world2!', 'hello! world3!', 'hello! world4!']
        """
        chunked_paragraphs = []
        chunk = ''

        for p in paragraphs:
            chunk += p

            if len(chunk) > self.max_len:
                chunks = self.from_paragraph(chunk)
                chunked_paragraphs.extend(chunks[:-1])
                chunk = chunks[-1]
            elif hasattr(self, 'min_len') and self.count_paragraphs_len(chunk) >= self.min_len:
                chunked_paragraphs.append(chunk)
                chunk = ''

        if chunk:
            chunked_paragraphs.append(chunk)

        return chunked_paragraphs

    def from_paragraph(self, paragraph: str) -> List[str]:
        """
        Usage:
            >>> paragraph = ''.join([f'hello! world{i}!' for i in range(5)])
            >>> RetentionToChunkedParagraphs(max_len=30).from_paragraph(paragraph)
            ['hello! world0!hello! world1!', 'hello! world2!hello! world3!', 'hello! world4!']
        """
        chunked_paragraphs = []
        rest = paragraph

        old_rest = rest
        while True:
            if len(rest) <= self.max_len:
                if rest:
                    chunked_paragraphs.append(rest)
                break

            tail = rest[self.max_len:]

            left_f, right_f, is_matched_f = self.truncate_by_stop_symbol(rest[:self.max_len], self.full_stop_rx)
            left_n, right_n, is_matched_n = self.truncate_by_stop_symbol(rest[:self.max_len], self.newline_stop_rx)

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
                left_h, right_h, is_matched_h = self.truncate_by_stop_symbol(rest[:self.max_len], self.half_stop_rx)
                if is_matched_h:
                    sect, rest = left_h, right_h + tail
                else:
                    sect, rest = rest[:self.max_len], rest[self.max_len:]

            chunked_paragraphs.append(sect)
            assert rest != old_rest, paragraph
            old_rest = rest

        return chunked_paragraphs

    def truncate_by_stop_symbol(self, line, pattern: re.Pattern) -> tuple:
        m = re.match(pattern, line)

        if m:
            left = line[:m.span()[1]]
            right = line[m.span()[1]:]
            if hasattr(self, 'min_len') and self.count_paragraphs_len(left) < self.min_len:
                is_matched = False
            else:
                is_matched = True
        else:
            left, right = line, ''
            is_matched = False

        return left, right, is_matched


class RandomRetentionToChunkedParagraphs(RetentionToChunkedParagraphs):
    """split paragraphs in pieces randomly, and then chunk them one by one"""

    min_choices = 2
    max_choices = None

    def from_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
         Usage:
            >>> paragraphs = [f'hello! world{i}!' for i in range(5)]
            >>> RandomRetentionToChunkedParagraphs(max_len=25).from_paragraphs(paragraphs)
            ['hello! world0!hello!', ' world1!', 'hello! world2!hello!', ' world3!hello! world4!']
        """
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


class DuplicatedToChunkedParagraphs(ToChunkedParagraphs):
    """chunk by window sliding with duplicated context"""
    duplicate_len: int
    duplicate_ratio: float = 0.2

    def from_paragraphs(self, paragraphs) -> List[str]:
        """
         Usage:
            >>> paragraphs = [f'hello! world{i}!' for i in range(5)]
            >>> DuplicatedToChunkedParagraphs(max_len=30, duplicate_len=5).from_paragraphs(paragraphs)
            ['hello! world0!hello! world1!', 'hello! world1!hello! world2!', 'hello! world3!hello! world4!']
        """
        if hasattr(self, 'duplicate_len'):
            duplicate_len = self.duplicate_len
        else:
            duplicate_len = int(self.max_len * self.duplicate_ratio)

        keep_length = self.max_len - duplicate_len

        chunked_paragraphs = []
        chunk = ''
        next_chunk = ''

        for p in paragraphs:
            assert len(p) < self.max_len, f'all input text length must no more than {self.max_len}, or try to use `RetentionToChunkedParagraphs`'

            if len(chunk + p) > self.max_len:
                chunked_paragraphs.append(chunk)
                chunk = next_chunk
                next_chunk = ''
            elif len(chunk + p) > keep_length:
                next_chunk += p

            chunk += p

        if chunk:
            chunked_paragraphs.append(chunk)

        return chunked_paragraphs

    def from_paragraph(self, paragraph: str) -> List[str]:
        """
         Usage:
            >>> paragraph = ''.join([f'hello! world{i}!' for i in range(5)])
            >>> DuplicatedToChunkedParagraphs(max_len=30).from_paragraph(paragraph)
            ['hello! world0!hello! world1!he', 'ld1!hello! world2!hello! world', ' world3!hello! world4!']
        """
        if hasattr(self, 'duplicate_len'):
            duplicate_len = self.duplicate_len
        else:
            duplicate_len = int(self.max_len * self.duplicate_ratio)

        chunked_paragraphs = []
        s = 0

        while True:
            chunked_paragraphs.append(paragraph[s: s + self.max_len])
            s += self.max_len - duplicate_len
            if s > len(paragraph):
                break

        return chunked_paragraphs


class KPiecesToChunkedParagraphs(ToChunkedParagraphs):
    """chunk to k pieces whose text length as same as possible"""
    k: int = 5
    spliter = spliter.ToSegment(
        sep_pattern=r'.*?[。\.!?！？\];；,，、》）}]',
        is_split_punctuation=False
    )

    def from_paragraphs(self, paragraphs: List[str]) -> List[str]:
        chunked_paragraphs = []
        for paragraph in paragraphs:
            chunked_paragraphs += self.from_paragraph(paragraph)
        return chunked_paragraphs

    def from_segment(self, segment: List[str]) -> List[str]:
        # todo, bugs when no sep_pattern, such as, ['hello', ' ', 'world'] -> ['hello world'], ['hello', 'world'] -> ['helloworld']
        return self.from_paragraph(''.join(segment))

    def from_paragraph(self, paragraph: str) -> List[str]:
        total_n = len(paragraph)
        mean_n = total_n / self.k
        segments = self.spliter.from_paragraph(paragraph)

        diff = 0
        chunked_paragraphs = []
        chunk = ''

        for s in segments:
            if len(chunk + s) > mean_n:
                l = len(chunk) - mean_n + diff
                r = len(chunk + s) - mean_n + diff
                if l + r > 0:
                    chunked_paragraphs.append(chunk)
                    chunk = s
                    diff = l
                else:
                    chunk += s
                    chunked_paragraphs.append(chunk)
                    chunk = ''
                    diff = r

            else:
                chunk += s

        if chunk:
            chunked_paragraphs.append(chunk)

        return chunked_paragraphs


class ToChunkedSegments(ToChunked):
    """chunk without dropping any context"""

    def from_paragraphs(self, paragraphs: List[str]) -> List[str]:
        return self.from_segment(paragraphs)


class RetentionToChunkedSegments(ToChunkedSegments):
    """chunk by truncate to keep all chunks' length close to the `max_len` as far as possible"""

    full_stop_tokens = set('。.!?！？')
    half_stop_tokens = set('];；,，、》）}')
    newline_stop_tokens = set('\n')
    max_len = 512
    min_len: int
    verbose = False

    def from_segments(self, segments: List[List[str]]) -> List[List[str]]:
        """
         Usage:
            >>> segments = spliter.ToSegments().from_paragraphs([f'hello{i}! world{i}!' for i in range(5)])
            >>> RetentionToChunkedSegments(max_len=7).from_segments(segments)
            [['hello0', '!', 'world0', '!', 'hello1', '!'], ['world1', '!', 'hello2', '!', 'world2', '!'], ['hello3', '!', 'world3', '!', 'hello4', '!'], ['world4', '!']]
        """
        chunked_segments = []
        chunked_segment = []

        if self.verbose:
            from utils.visualize import TextVisualize
            segments = tqdm(segments, desc=TextVisualize.highlight_str('Chunk segments'))

        for p in segments:
            chunked_segment += p

            if len(chunked_segment) > self.max_len:
                chunks = self.from_segment(chunked_segment)
                chunked_segments.extend(chunks[:-1])
                chunked_segment = chunks[-1]
            elif hasattr(self, 'min_len') and len(chunked_segment) >= self.min_len:
                chunked_segments.append(chunked_segment)
                chunked_segment = []

        if chunked_segment:
            chunked_segments.append(chunked_segment)

        return chunked_segments

    def from_segment(self, segment: List[str]) -> List[List[str]]:
        """
         Usage:
            >>> segment = spliter.ToSegment().from_paragraph(''.join([f'hello{i}! world{i}!' for i in range(5)]))
            >>> RetentionToChunkedSegments(max_len=7).from_segment(segment)
            [['hello0', '!', 'world0', '!', 'hello1', '!'], ['world1', '!', 'hello2', '!', 'world2', '!'], ['hello3', '!', 'world3', '!', 'hello4', '!'], ['world4', '!']]
        """
        chunked_segments = []
        rest = segment

        old_rest = rest
        while True:
            if len(rest) <= self.max_len:
                if rest:
                    chunked_segments.append(rest)
                break

            keep = rest[:self.max_len]
            tail = rest[self.max_len:]

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
            assert rest != old_rest, segment
            old_rest = rest

        return chunked_segments

    def from_paragraphs(self, paragraphs: List[str]) -> List[List[str]]:
        """only counting the len of chunks different from `from_segment`"""
        chunked_segments = []
        rest = paragraphs

        old_rest = rest
        while True:
            if self.count_paragraphs_len(rest) <= self.max_len:
                if rest:
                    chunked_segments.append(rest)
                break

            max_len = 0
            chunk_idx = 0
            for chunk_idx, s in enumerate(rest):
                max_len += len(s)
                if max_len >= self.max_len:
                    break

            keep = rest[:chunk_idx]
            tail = rest[chunk_idx:]

            left_f, right_f, is_matched_f = self.truncate_by_stop_token(keep, self.full_stop_tokens)
            left_n, right_n, is_matched_n = self.truncate_by_stop_token(keep, self.newline_stop_tokens)

            if is_matched_f and is_matched_n:
                if self.count_paragraphs_len(left_f) >= self.count_paragraphs_len(left_n):
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
            assert rest != old_rest, paragraphs
            old_rest = rest

        return chunked_segments

    def truncate_by_stop_token(self, segment, token: set) -> tuple:
        is_matched = False
        left, right = segment, []

        for i, s in enumerate(segment):
            if s in token:
                # except eg. 1.2, 1-2, 1:2
                if (
                        s in '.-:'
                        and i > 0 and segment[i - 1].isdigit()
                        and i < len(segment) - 1 and segment[i + 1].isdigit()
                ):
                    continue

                left = segment[:i + 1]
                right = segment[i + 1:]
                if hasattr(self, 'min_len') and self.count_paragraphs_len(left) < self.min_len:
                    is_matched = False
                else:
                    is_matched = True

        return left, right, is_matched


class RandomToChunkedSegments(RetentionToChunkedSegments):
    """split segments in pieces randomly, and then chunk them one by one"""
    min_choices = 2
    max_choices = None

    def from_segments(self, segments: List[List[str]]) -> List[List[str]]:
        """
         Usage:
            >>> segments = spliter.ToSegments().from_paragraphs([f'hello{i}! world{i}!' for i in range(5)])
            >>> RandomToChunkedSegments(max_len=7).from_segments(segments)
            [['hello0', '!', 'world0', '!', 'hello1', '!'], ['world1', '!'], ['hello2', '!', 'world2', '!'], ['hello3', '!', 'world3', '!', 'hello4', '!'], ['world4', '!']]
        """
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


class ToChunkedParagraph(ToChunked):
    """chunk by dropping some context"""

    max_len: int = 512
    spliter = spliter.ToSegment()

    def from_paragraphs(self, paragraphs: List[str]) -> str:
        return self.from_segment(paragraphs)

    def from_paragraph(self, paragraph) -> str:
        segment = self.paragraph_to_segment(paragraph)
        return self.from_segment(segment)

    def paragraph_to_segment(self, paragraph) -> List[str]:
        """see also `spliter`"""
        return self.spliter.from_paragraph(paragraph)


class HeadToChunkedParagraph(ToChunkedParagraph):
    """only the keep the `max_len` text of the head"""

    def from_segment(self, segment) -> str:
        """
         Usage:
            >>> segment = [f'hello{i}! world{i}!' for i in range(5)]
            >>> HeadToChunkedParagraph(max_len=30).from_segment(segment)
            hello0! world0!hello1! world1!
        """
        chunked_paragraph = ''

        for seg in segment:
            if len(chunked_paragraph + seg) > self.max_len:
                break

            chunked_paragraph += seg

        return chunked_paragraph


class TailToChunkedParagraph(HeadToChunkedParagraph):
    """only the keep the `max_len` text of the tail"""

    def from_segment(self, segment) -> str:
        """
         Usage:
            >>> segment = [f'hello{i}! world{i}!' for i in range(5)]
            >>> TailToChunkedParagraph(max_len=30).from_segment(segment)
            hello3! world3!hello4! world4!
        """
        chunked_paragraph = ''

        for seg in segment[::-1]:
            if len(chunked_paragraph + seg) > self.max_len:
                break

            chunked_paragraph = seg + chunked_paragraph

        return chunked_paragraph


class MiddleToChunkedParagraph(ToChunkedParagraph):
    """only the keep the `max_len` text of the middle"""

    def from_segment(self, segment) -> str:
        """
         Usage:
            >>> segment = [f'hello{i}! world{i}!' for i in range(5)]
            >>> MiddleToChunkedParagraph(max_len=30).from_segment(segment)
            hello1! world1!hello2! world2!
        """
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
                if pl + a > self.max_len:
                    break
                else:
                    keep.add(i)
                    r += 1
                    flag = 'l'
            else:
                i = mid_n - l
                a = word_n[i]
                if pl + a > self.max_len:
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

    def from_segment(self, segment) -> str:
        """
         Usage:
            >>> segment = [f'hello{i}! world{i}!' for i in range(5)]
            >>> TwoPiecesToChunkedParagraph(max_len=30).from_segment(segment)
            hello0! world0!hello4! world4!
        """
        if sum(len(s) for s in segment) <= self.max_len:
            return ''.join(segment)

        max_seq = [int(self.max_len * r) for r in self.ratios]
        chunked_paragraph = ''
        chunked_paragraph += HeadToChunkedParagraph(max_len=max_seq[0]).from_segment(segment)
        chunked_paragraph += TailToChunkedParagraph(max_len=max_seq[1]).from_segment(segment)
        return chunked_paragraph


class ThreePiecesToChunkedParagraph(ToChunkedParagraph):
    """truncate for 3 pieces: head + middle + tail"""
    ratios = (0.4, 0.2, 0.4)

    def from_segment(self, segment) -> str:
        """
         Usage:
            >>> segment = [f'hello{i}! world{i}!' for i in range(5)]
            >>> ThreePiecesToChunkedParagraph(max_len=50, ratios=(0.3, 0.3, 0.3)).from_segment(segment)
            hello0! world0!hello2! world2!hello4! world4!
        """
        if sum(len(s) for s in segment) <= self.max_len:
            return ''.join(segment)

        max_seq = [int(self.max_len * r) for r in self.ratios]
        chunked_paragraph = ''
        chunked_paragraph += HeadToChunkedParagraph(max_len=max_seq[0]).from_segment(segment)
        chunked_paragraph += MiddleToChunkedParagraph(max_len=max_seq[1]).from_segment(segment)
        chunked_paragraph += TailToChunkedParagraph(max_len=max_seq[2]).from_segment(segment)
        return chunked_paragraph


class KPiecesToChunkedParagraph(ToChunkedParagraph):
    """average truncate for k pieces, each piece keep the head"""
    k: int = 5

    def from_segment(self, segment: List[str]) -> str:
        """
         Usage:
            >>> segment = [f'hello{i}! world{i}!' for i in range(10)]
            >>> KPiecesToChunkedParagraph(max_len=75).from_segment(segment)
            hello0! world0!hello2! world2!hello4! world4!hello6! world6!hello8! world8!
        """
        word_n = [len(s) for s in segment]
        total_n = sum(word_n)

        if total_n <= self.max_len:
            return ''.join(segment)

        mean_n = total_n // self.k
        _max_len = self.max_len // self.k

        chunked_paragraph = ''
        tmp_n = 0
        tmp_segment = []
        for n, s in zip(word_n, segment):
            if tmp_n >= mean_n:
                chunked_paragraph += HeadToChunkedParagraph(max_len=_max_len).from_segment(tmp_segment)
                tmp_n = 0
                tmp_segment = []

            tmp_n += n
            tmp_segment.append(s)

        if tmp_n >= mean_n:
            chunked_paragraph += HeadToChunkedParagraph(max_len=_max_len).from_segment(tmp_segment)

        return chunked_paragraph
