import torch
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import SkillEmbedding, InputEmbedding

def test_skill_interp_shape():
    c = ModelConfig()
    s = SkillEmbedding(c)
    out = s(torch.tensor([0, 5000, 2500]))
    assert out.shape == (3, c.dim_emb)

def test_skill_clamps():
    c = ModelConfig()
    s = SkillEmbedding(c)
    assert torch.allclose(s(torch.tensor([9000])), s(torch.tensor([5000])), atol=1e-6)

def test_input_embedding_shape():
    c = ModelConfig()
    emb = InputEmbedding(c)
    board = torch.zeros(4, 8, 8, 12)
    history = torch.zeros(4, 7, 8, 8, 12)
    elo_self = torch.tensor([1500, 1500, 1500, 1500])
    elo_opp = torch.tensor([1400, 1400, 1400, 1400])
    out = emb(board, history, elo_self, elo_opp)
    assert out.shape == (4, 64, c.dim_vit)
