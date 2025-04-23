import math
from CourseDataset import *
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from BarlowTwinsLoss import BarlowTwinsLoss


class CLGADN(nn.Module):
    def __init__(self, args, Graph1, Graph2):
        super().__init__()
        super(CLGADN, self).__init__()

        # self.args = args
        self.num_student = args.num_student
        self.num_category = args.num_category
        self.num_course = args.num_course

        self.Graph1 = Graph1.to(args.device)
        self.Graph2 = Graph2.to(args.device)

        self.device = args.device
        self.sequence_length = args.sequence_length
        self.keep_prob = args.keep_prob
        self.latent_dim = args.recdim
        self.dropout = args.dropout

        self.sigmoid = nn.Sigmoid()
        self.transformer = args.transformer
        self.n_layers = args.layer  # the layer num of lightGCN
        self.latent_dim = args.recdim
        self.nhead = args.nhead
        self.cl = BarlowTwinsLoss()

        self.stu_embedding = nn.Embedding(int(self.num_student), self.latent_dim)
        self.course_embedding = nn.Embedding(int(self.num_course) + 1, self.latent_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(int(self.num_category) + 1, self.latent_dim, padding_idx=0)

        # Network
        if self.transformer:
            self.embeddings_position = nn.Embedding(self.sequence_length + 1, (self.latent_dim))
            self.transfomerlayer = TransformerLayer(d_model=(self.latent_dim), d_ff=512, n_heads=self.nhead,
                                                    dropout=0.2)

            # user + hist + target
            self.liner = nn.Sequential(
                nn.Linear(2200, 1024),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(256, 1),
            )

        self._init_weight_()
        self.to(self.device)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob, Graph):
        graph = self.__dropout_x(Graph, keep_prob)
        return graph

    def computer(self, Graph):
        """
        propagate methods for lightGCN
        """
        users_emb = self.stu_embedding.weight  # torch.Size([58907, 100])
        items_emb = self.course_embedding.weight  # torch.Size([2584, 100])
        all_emb = torch.cat([users_emb, items_emb])  # torch.Size([61491, 100])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob, Graph)
            else:
                g_droped = Graph
        else:
            g_droped = Graph

        for layer in range(self.n_layers):
            # torch.sparse.mm 二维矩阵的乘法 前者是稀疏矩阵，后者是稀疏矩阵或者密集矩阵
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)  # torch.Size([61491, 4, 100])
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_student, self.num_course + 1])  # torch.Size([61491, 100])
        return users, items  # torch.Size([58907, 100]) torch.Size([2584, 100])

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.stu_embedding.weight, std=0.001)
        nn.init.normal_(self.course_embedding.weight, std=0.001)
        nn.init.normal_(self.category_embedding.weight, std=0.001)
        nn.init.normal_(self.embeddings_position.weight, std=0.001)
        for m in self.liner:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    # （batch maxlen）
    def generate_src_key_padding_mask(self, courses):
        return courses == 0

    def getEmbedding(self, student, courses, candidate):
        all_users1, all_items1 = self.computer(self.Graph1)  # 得到传播后的embedding
        all_users2, all_items2 = self.computer(self.Graph2)

        uloss = self.cl(all_users1, all_users2)
        iloss = self.cl(all_items1, all_items2)

        all_users = 0.5 * all_users1 + 0.5 * all_users2
        all_items = 0.5 * all_items1 + 0.5 * all_items2

        student_emb = all_users[student]
        hist_emb = all_items[courses]
        candidate_emb = all_items[candidate.long()]

        return student_emb, hist_emb, candidate_emb, uloss + iloss

    def forward(self, student, courses, category, candidate, candidate_cate):
        '''
        :param student: batch
        :param courses: batch, 20
        :param candidate: batch, 20
        :param candidate: batch
        :param candidate_cate: batch
        :return: stu_id, candidate, out  都是 batch
        '''
        ##### user features
        # compute embedding

        # user_features (batch, 100)
        # courses_history (batch, 20, 100)
        # candidate_course (batch,100)
        (user_features_emb, courses_history_emb, candidate_course_emb, cl_loss) = self.getEmbedding(student,
                                                                                                    courses,
                                                                                                    candidate)

        # category_history (batch, 20, 100)
        category_history = self.category_embedding(category)

        # course_history_feature [batch,20,100]
        course_history_feature = courses_history_emb + category_history

        ##### courses features
        # candidate_cate (batch,5)
        candidate_cate = self.category_embedding(candidate_cate)
        # candidate_course_feature (batch,105)
        candidate_course_feature = candidate_course_emb + candidate_cate

        if self.transformer:
            positions = torch.arange(0, self.sequence_length + 1, 1, dtype=int, device=self.device)
            positions = self.embeddings_position(positions)  # (20+1, 100)

            # transfomer_features (batch,21,100)
            sequence_features = torch.cat((course_history_feature, candidate_course_feature.unsqueeze(1)), dim=1)

            # encoded_courses_history_with_poistion_and_rating (batch,21,100)
            encoded_courses_history_with_poistion_and_rating = (
                    sequence_features + positions
            )
            # transfomer_features (batch,21,105)
            sequence_output, att_weight = self.transfomerlayer(encoded_courses_history_with_poistion_and_rating,
                                                               encoded_courses_history_with_poistion_and_rating,
                                                               encoded_courses_history_with_poistion_and_rating,
                                                               )
            # sequence_output (batch, 21*100)
            sequence_output = torch.flatten(sequence_output, start_dim=1)
        else:
            # sequence_output (batch, 21*100)
            sequence_features = torch.cat((course_history_feature, candidate_course_feature.unsqueeze(1)), dim=1)
            sequence_output = torch.flatten(sequence_features, start_dim=1)

        # Concat with other features
        # features
        features = torch.cat(
            (user_features_emb, sequence_output), dim=1
        )
        # print('features.shape:', features.shape)
        out = self.liner(features)

        return student, candidate, self.sigmoid(out), att_weight, cl_loss


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        d_feature = d_model // n_heads
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.
        """
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=1).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        query2, att_weight = self.masked_attn_head(query, key, values, mask=src_mask)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query, att_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        xavier_uniform_(self.gammas)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # (batch, len , dim) -> (batch, len , head, dim//head)
        q = self.k_linear(q).view(bs, -1, self.h, self.d_k)  # d_model = dim//head
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)  # (batch, head, len, d_model)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores, att_weight = attention(q, k, v, self.d_k, mask, self.dropout, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, att_weight


def attention(q, k, v, d_k, mask, dropout, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, head, seqlen, seqlen)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()
    with torch.no_grad():
        position_effect = torch.abs((x1 - x2) / 21)[None, None, :, :].type(torch.FloatTensor).to(
            device)  # 1, 1, seqlen, seqlen
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    total_effect = torch.clamp(torch.clamp((-position_effect * gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores

if __name__ == '__main__':
    pass
