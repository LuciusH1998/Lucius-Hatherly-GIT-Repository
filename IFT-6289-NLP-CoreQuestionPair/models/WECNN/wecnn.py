import torch
import torch.nn as nn
import torch.nn.functional as F

class WECNN(nn.Module):
  
  def __init__(self, vocab_size, embedding_dim=300, filters=[2,3,4,6,8], num_filters=200, pad_idx=0):
    super(WECNN, self).__init__()
    
    # Embedding Layer Question Pair & Unique Words
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

    # Keyword Embdding and Freezing Weights - Section 3.5
    self.keyword_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
    self.keyword_embedding.weight.requires_grad = False
    
    self.filters = filters
    self.num_filters = num_filters 
    
    self.convs = nn.ModuleList([
      nn.Conv1d(in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=h)
      for h in filters
    ])
    
    # Initialization of conv layers with Xavier Uniform
    for conv in self.convs:
      nn.init.xavier_uniform_(conv.weight)
      if conv.bias is not None:
        nn.init.zeros_(conv.bias)
      
    # lambda
    self.lam_raw = nn.Parameter(torch.tensor(10.0))
    
    # Dropout=0.1 as paper specs  
    self.dropout = nn.Dropout(0.1)
  
    self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.7)
    
    # Concatenation layer with n_filters * 3  * (cos(c_min), cos(c_max), cos(c_max_i))
    # Output is 2 since they are similar or not
    self.fc = nn.Linear(len(self.filters) * 3 * 3, 2)
    
  def _embed(self, x, is_keyword=False):
    emb = self.keyword_embedding(x) if is_keyword else self.embedding(x)  # (B, L, D)
    emb = emb.transpose(1, 2)                                             # (B, D, L)
    emb = self.bn(emb)
    return emb
    
  def get_features(self, x, isKeyword=False):
    x = self._embed(x, is_keyword=isKeyword)
    
    all_pools = []
    for conv in self.convs:
      c = self.dropout(F.relu(conv(x))) # (B, num_filters, L')
      
      c_max = torch.max(c, dim=2)[0]
      c_min = torch.min(c, dim=2)[0]
      
      # chunked max over 4 equal chunks along time dimension
      chunks = torch.chunk(c, 4, dim=2)
      chunk_maxes = [torch.max(chunk, dim=2)[0] for chunk in chunks] # (B, num_filters)
      c_max_i = torch.cat(chunk_maxes, dim=1) # (B, 4 * num_filters)
      
      all_pools.append((c_max, c_min, c_max_i))
    return all_pools
  
  def cosine_sim(self, f1, f2):
    dot = torch.sum(f1 * f2, dim=1)
    norm = torch.norm(f1, 2, 1) * torch.norm(f2, 2, 1)
    epsilon = 1e-8
    lam = torch.sigmoid(self.lam_raw)
    # Not compute division by zero
    return (dot / (norm + epsilon))  + lam
  
  def forward(self, p, q, p_u, q_u, p_k, q_k, probs=False):
    
    # Get all values for each embeddings  
    feat_p   = self.get_features(p)
    feat_p_u = self.get_features(p_u)
    feat_p_k = self.get_features(p_k, True)
    
    feat_q   = self.get_features(q)
    feat_q_u = self.get_features(q_u)
    feat_q_k = self.get_features(q_k, True)
    
    combined_scores = []
    for i in range(len(self.filters)):
      for p_feat, q_feat in [(feat_p, feat_q), (feat_p_u, feat_q_u), (feat_p_k, feat_q_k)]:
        combined_scores.append(self.cosine_sim(p_feat[i][0], q_feat[i][0])) # c_max
        combined_scores.append(self.cosine_sim(p_feat[i][1], q_feat[i][1])) # c_min
        combined_scores.append(self.cosine_sim(p_feat[i][2], q_feat[i][2])) # c_max_i

    x = torch.stack(combined_scores, dim=1) # (B, 45)
    
    logits = self.fc(x)                     # (B, 2)
    
    if probs == True:
      return F.softmax(logits, dim=1)
    
    return logits