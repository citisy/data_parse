import torch


def enum_search(x):
    pass


def greedy_search(x, seq_lens, score_fn, **score_fn_kwargs):
    """

    Args:
        x (torch.Tensor): (batch_size, seq_length, vocab_size) after log softmax

    Returns:
        preds: (batch_size, beam_size, seq_length)

    """
    pass


def beam_search(x, seq_lens, score_fn, eos_ids=(), max_gen_len=100, top_k=1, start_pos=0, **score_fn_kwargs):
    assert seq_lens is not None

    batch_size = len(x)
    eos_flag = [False] * batch_size

    prev_pos = start_pos
    min_pos = min(seq_lens)

    for cur_pos in range(min_pos, min_pos + max_gen_len):
        logits = score_fn(
            x[:, prev_pos: cur_pos],
            start_pos=prev_pos,
            **score_fn_kwargs
        )

        x = torch.cat([x, torch.zeros((batch_size, 1)).to(x)], dim=-1)

        for batch in range(batch_size):
            if eos_flag[batch]:
                continue

            if x[batch][cur_pos] != 0:
                continue

            preds = logits[batch, -1]
            arg = torch.argsort(preds, descending=True)
            keep = arg[:top_k]
            preds = preds[keep]
            preds = preds / preds.sum()

            # random sampling
            next_id = keep[preds.multinomial(1)[0]]
            x[batch][cur_pos] = next_id

            if next_id in eos_ids:
                eos_flag[batch] = True

        if all(eos_flag):
            break

        prev_pos = cur_pos

    return x


def prefix_beam_search(x):
    pass
