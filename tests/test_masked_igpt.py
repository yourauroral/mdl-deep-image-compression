import pytest
import torch

from mdlic.masking_scheduler import MaskSchedule
from mdlic.models.cc_igpt import CCIGPT
from mdlic.models.cc_mdlm import CCMDLM
from mdlic.models.igpt import IGPT
from mdlic.models.layers import MultiHeadAttentionBlock
from mdlic.models.masked_igpt import MaskedIGPT


def _tiny_masked_model():
    return MaskedIGPT(
        image_size=2,
        in_channels=1,
        vocab_size=256,
        d_model=8,
        N=1,
        h=1,
        d_ff=16,
        dropout=0.0,
    )


def _tiny_cc_kwargs():
    return dict(
        image_size=4,
        in_channels=1,
        vocab_size=256,
        pool_factor=2,
        fine_d_model=8,
        fine_N=1,
        fine_h=1,
        fine_d_ff=16,
        coarse_d_model=8,
        coarse_N=1,
        coarse_h=1,
        coarse_d_ff=16,
        dropout=0.0,
    )


def test_attention_explicitly_separates_causal_and_bidirectional_paths():
    torch.manual_seed(31)
    attention = MultiHeadAttentionBlock(8, 1, dropout=0.0).eval()
    original = torch.randn(1, 4, 8)
    changed = original.clone()
    changed[:, -1] += 3.0
    position_ids = torch.arange(4)

    causal_a = attention(
        original, original, original,
        position_ids=position_ids,
        is_causal=True,
    )
    causal_b = attention(
        changed, changed, changed,
        position_ids=position_ids,
        is_causal=True,
    )
    bidirectional_a = attention(
        original, original, original,
        position_ids=position_ids,
        is_causal=False,
    )
    bidirectional_b = attention(
        changed, changed, changed,
        position_ids=position_ids,
        is_causal=False,
    )

    assert torch.allclose(causal_a[:, 0], causal_b[:, 0], atol=1e-6, rtol=0)
    assert not torch.allclose(
        bidirectional_a[:, 0], bidirectional_b[:, 0], atol=1e-6, rtol=0,
    )


def test_forward_masked_uses_full_sequence_and_only_scores_current_group():
    torch.manual_seed(32)
    model = _tiny_masked_model().train()
    inputs = torch.tensor([[256, 7, 256, 9]], dtype=torch.long)
    targets = torch.tensor([[4, 7, 6, 9]], dtype=torch.long)
    loss_mask = torch.tensor([[True, False, False, False]])

    output = model.forward_masked(
        inputs,
        target_tokens=targets,
        loss_mask=loss_mask,
    )

    assert output["logits"].shape == (1, 4, 256)
    assert output["num_loss_tokens"] == 1
    assert output["masked_positions"].tolist() == [[True, False, True, False]]
    assert model.token_embed.num_embeddings == 256
    assert model.head.out_features == 256
    assert model.mask_token_id == 256
    output["loss"].backward()
    assert model.mask_embedding.grad is not None
    assert model.mask_embedding.grad.abs().sum().item() > 0
    assert model.mask_ratio_proj.weight.grad is not None

    with pytest.raises(RuntimeError, match="forward_masked"):
        model(torch.rand(1, 1, 2, 2))


def test_forward_masked_rejects_scoring_a_revealed_token():
    model = _tiny_masked_model().eval()
    inputs = torch.tensor([[256, 7, 256, 9]], dtype=torch.long)
    targets = torch.tensor([[4, 7, 6, 9]], dtype=torch.long)
    bad_mask = torch.tensor([[False, True, False, False]])

    with pytest.raises(ValueError, match="subset"):
        model.forward_masked(
            inputs,
            target_tokens=targets,
            loss_mask=bad_mask,
        )


def test_masked_attention_checkpointing_propagates_noncausal_flag():
    torch.manual_seed(36)
    model = MaskedIGPT(
        image_size=2,
        in_channels=1,
        d_model=8,
        N=1,
        h=1,
        d_ff=16,
        activation_checkpointing=True,
    ).train()
    inputs = torch.tensor([[256, 7, 256, 9]], dtype=torch.long)
    targets = torch.tensor([[4, 7, 6, 9]], dtype=torch.long)

    output = model.forward_masked(inputs, target_tokens=targets)
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert model.blocks[0].attn.w_q.weight.grad is not None


def test_ar_weights_are_init_compatible_but_not_strict_resume_compatible():
    torch.manual_seed(33)
    ar_model = IGPT(
        image_size=2, in_channels=1, d_model=8, N=1, h=1, d_ff=16,
    )
    masked_model = _tiny_masked_model()

    incompatible = masked_model.load_state_dict(ar_model.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "mask_embedding",
        "mask_ratio_proj.weight",
        "mask_ratio_proj.bias",
    }
    assert incompatible.unexpected_keys == []
    with pytest.raises(RuntimeError, match="Missing key"):
        _tiny_masked_model().load_state_dict(ar_model.state_dict(), strict=True)


def test_cc_mdlm_group_objective_is_explicit_and_backward_finite():
    torch.manual_seed(34)
    model = CCMDLM(**_tiny_cc_kwargs()).train()
    schedule = MaskSchedule.raster_chunks(model.fine.seq_len, 4)
    x = torch.rand(2, 1, 4, 4)

    output = model.forward_group(x, schedule=schedule, group_index=1)

    assert output["objective"] == "fixed_group_conditional_ce"
    assert output["num_groups"] == 4
    assert output["num_group_tokens"] == 4
    assert output["bpd"] is None
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.fine.mask_embedding.grad is not None
    assert model.ctx_alpha.grad is not None

    with pytest.raises(TypeError, match="schedule"):
        model(x)


def test_cc_mdlm_preserves_ar_parameter_names_for_init_from():
    ar_model = CCIGPT(**_tiny_cc_kwargs())
    masked_model = CCMDLM(**_tiny_cc_kwargs())

    incompatible = masked_model.load_state_dict(ar_model.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "fine.mask_embedding",
        "fine.mask_ratio_proj.weight",
        "fine.mask_ratio_proj.bias",
    }
    assert incompatible.unexpected_keys == []
