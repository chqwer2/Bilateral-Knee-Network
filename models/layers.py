import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class PHConv(nn.Module):

  def __init__(self, n, in_features, out_features, kernel_size, padding=0, stride=1, cuda=True):
    super(PHConv, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.padding = padding
    self.stride = stride
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))
    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
    self.F = nn.Parameter(torch.nn.init.xavier_uniform_(
        torch.zeros((n, self.out_features//n, self.in_features//n, kernel_size, kernel_size))))
    self.weight = torch.zeros((self.out_features, self.in_features))
    self.kernel_size = kernel_size

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, A, F):
    siz1 = torch.Size(torch.tensor(A.shape[-2:]) * torch.tensor(F.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(F.shape[-2:]))
    res = A.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * F.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size))
    if self.cuda:
        H = H.cuda()
    for i in range(self.n):
        kron_prod = torch.kron(self.A[i], self.F[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.F), dim=0)
    # self.weight = self.kronecker_product2()
    #if self.cuda:
    #    self.weight = self.weight.cuda()

    input = input.type(dtype=self.weight.type())
        
    return F.conv2d(input, weight=self.weight, stride=self.stride, padding=self.padding)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.F, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)
    

class SELayer(nn.Module):
  def __init__(self, channel, reduction=16) -> None:
     super().__init__()
     self.avg_pool = nn.AdaptiveAvgPool2d(1)
     self.fc = nn.Sequential(
       nn.Linear(channel, channel//reduction, bias=False), 
       nn.ReLU(inplace=True), 
       nn.Linear(channel//reduction, channel, bias=False), 
       nn.Sigmoid()
     )
  
  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.0)   # attention_probs_dropout_prob

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        x = x.unsqueeze(1)   # view(batch_size, -1, self.num_heads, self.dim)
        # print("x.view shape:", x.shape)   # 8 all_head_size 384
        return x.permute(0, 2, 1, 3)

    def forward_feature(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        return query_layer, key_layer, value_layer
    def forward(self, query_layer, key_layer, value_layer ):

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        # residual
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
