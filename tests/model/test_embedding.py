import torch
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding, SkillEmbedding

def test_input_embedding_shape():
    c = ModelConfig()
    emb = InputEmbedding(c)
    board = torch.zeros(4, 8, 8, 12)
    history = torch.zeros(4, 7, 8, 8, 12)
    out = emb(board, history)
    assert out.shape == (4, 64, c.d_model)

def test_skill_embedding_shape_and_interpolation():
    c = ModelConfig()
    skill = SkillEmbedding(c)
    elo = torch.tensor([0, 5000, 2500])
    out = skill(elo)
    assert out.shape == (3, c.d_model)

def test_skill_endpoints_differ():
    c = ModelConfig()
    skill = SkillEmbedding(c)
    weak = skill(torch.tensor([0]))
    strong = skill(torch.tensor([5000]))
    assert not torch.allclose(weak, strong)

def test_skill_clamps_out_of_range():
    c = ModelConfig()
    skill = SkillEmbedding(c)
    hi = skill(torch.tensor([9000]))
    strong = skill(torch.tensor([5000]))
    assert torch.allclose(hi, strong, atol=1e-6)
