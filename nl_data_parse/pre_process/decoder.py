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


def beam_search(x, seq_lens, score_fn, eos_ids=(), max_gen_len=100, top_k=1, start_pos=0, penalty=1.1, **score_fn_kwargs):
    assert seq_lens is not None

    batch_size = len(x)
    eos_flag = [False] * batch_size

    prev_pos = start_pos
    min_pos = min(seq_lens)
    end_pos = [-1] * batch_size

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
            # if cur_preds < 0 then repetition penalty has to be multiplied to reduce the token probabilities
            cur_preds = torch.gather(preds, 0, x[batch])
            cur_preds = torch.where(cur_preds < 0, cur_preds * penalty, cur_preds / penalty)
            preds = preds.scatter(0, x[batch], cur_preds)
            arg = torch.argsort(preds, descending=True)
            keep = arg[:top_k]
            keep_preds = preds[keep]
            keep_preds = keep_preds / (keep_preds.sum() + 1e-8) + 1e-8    # avoid underflow

            # random sampling
            next_id = keep[keep_preds.multinomial(1)[0]]
            x[batch][cur_pos] = next_id

            if next_id in eos_ids:
                eos_flag[batch] = True
                end_pos[batch] = cur_pos

        if all(eos_flag):
            break

        prev_pos = cur_pos

    return dict(
        preds=x,
        end_pos=end_pos
    )


def prefix_beam_search(x):
    pass
