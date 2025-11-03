import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

from utils import get_func


multisub_mol2vec_dims = {
    "a":342,
    "s":327,
    "f":415,
    "p":306,
    "m":300,
}


class PyramidMPNN_MultiSub(nn.Module):
    def __init__(self, hid_dim=300, num_heads=8, act=nn.PReLU()):
        """
        MultiViewMassagePassing
        view: a, ap, apj
        suffix: filed to save the nodes' hidden state in dgl.graph.
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        self.mp: view='a',suffix = 'h'
        self.mp_aug: view='ap',suffix='aug'

        all
        node_type: a,p,junc
        edge_type: (a,b,a),(p,r,p),(a,j,p)(p,j,a)

        view : 'a'
        node_type: a,p
        edge_type: (a,b,a)

        view: 'ap'
        node_type: a,p
        edge_type: (a,b,a),(p,r,p)
        """
        super(PyramidMPNN_MultiSub, self).__init__()
        # self.view = view
        # self.depth = depth
        # self.suffix = suffix
        # self.msg_func = msg_func
        self.act = act
        # self.homo_etypes = [('a', 'b', 'a'),("s","c","s")]
        # self.hetero_etypes = [("a","j","s"),("s","d","m")]m

        self.mpnn_encoder = dglnn.HeteroGraphConv({
            "b": dglnn.SAGEConv(hid_dim, hid_dim, aggregator_type="lstm",activation=act),
            "c": dglnn.SAGEConv(hid_dim, hid_dim, aggregator_type = "lstm",activation = act),
            "d": dglnn.SAGEConv(hid_dim, hid_dim, aggregator_type = "lstm",activation = act),
            "e": dglnn.SAGEConv(hid_dim, hid_dim, aggregator_type="lstm", activation=act),
            "j_s": dglnn.SAGEConv(hid_dim, hid_dim, aggregator_type="lstm", activation=act),
            "j_f": dglnn.SAGEConv(hid_dim, hid_dim, aggregator_type="lstm", activation=act),
            "j_p": dglnn.SAGEConv(hid_dim, hid_dim, aggregator_type="lstm", activation=act),
            "i_s": dglnn.GATConv(hid_dim, hid_dim,num_heads = num_heads,activation = act),
            "i_f": dglnn.GATConv(hid_dim, hid_dim, num_heads=num_heads, activation=act),
            "i_p": dglnn.GATConv(hid_dim, hid_dim, num_heads=num_heads, activation=act)
        },aggregate='sum')
        self.node_embeds = nn.ModuleDict()
        self.node_embeds["a"] = nn.Linear(hid_dim, hid_dim).cuda()
        self.node_embeds["s"] = nn.Linear(hid_dim, hid_dim).cuda()
        self.node_embeds["f"] = nn.Linear(hid_dim, hid_dim).cuda()
        self.node_embeds["p"] = nn.Linear(hid_dim, hid_dim).cuda()
        self.node_embeds["m"] = nn.Linear(hid_dim, hid_dim).cuda()

    def updata_feature(self, node_features):
        for node_type,f_NN in self.node_embeds.items():
            node_features[node_type] = self.act(f_NN(node_features[node_type]))
        return node_features

    def get_feature(self, bg,f_index = "f_"):
        return bg.ndata[f_index]

    def forward(self, bg,node_features):

        '''
        bg:
        node
        bg.nodes['a'].data['f']
        bg.nodes['p'].data['f']
        bg.nodes['a'].data['f_junc']
        bg.nodes['p'].data['f_junc']

        edge
        bg.edges[('a','b','a')].data['x']
        bg.edges[('p','r','p')].data['x']

        '''
        '''
        view:a,suffix:h

        '''
        # node_features = self.get_feature(bg)  # 三个视图节点-边的特征Hidden
        node_features = self.mpnn_encoder(bg,{node_type: node_features[node_type] for node_type in bg.ntypes})
        node_features["m"] = torch.mean(node_features["m"], dim=1)
        node_features = self.updata_feature(node_features)
        return node_features
# enhance_sub by common feature
class CommonFeatureExtractor(nn.Module):
    def __init__(self, hidden_size, similarity_threshold=0.6): # default = 0.6
        super(CommonFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.similarity_threshold = similarity_threshold


        self.fp_encoders = nn.ModuleDict({
            'brics': nn.Sequential(
                nn.Linear(hidden_size, 300),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(300, hidden_size)
            ),
            'function_group': nn.Sequential(
                nn.Linear(hidden_size, 300),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(300, hidden_size)
            ),
            'pharmacophore': nn.Sequential(
                nn.Linear(hidden_size, 300),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(300, hidden_size)
            )

        })

        self.commonality_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        self.enhancement_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.weight_generator = nn.Linear(hidden_size * 3, 3)

        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, brics, function_group, pharmacophore, contrastive=False):
        batch_size = brics.size(0)

        encoded_fps = {
            'brics': self.fp_encoders['brics'](brics),
            'function_group': self.fp_encoders['function_group'](function_group),
            'pharmacophore': self.fp_encoders['pharmacophore'](pharmacophore),
        }

        all_fp_common_features = []
        fp_keys = list(encoded_fps.keys())

        for b in range(batch_size):
            sample_fps = torch.stack([encoded_fps[k][b] for k in fp_keys])  # [3, hidden_size]

            sample_fps_norm = F.normalize(sample_fps, dim=1)
            similarity_matrix = torch.mm(sample_fps_norm, sample_fps_norm.t())  # [3, 3]

            mask = torch.eye(3, device=brics.device)  # 生成对角线全1，其余部分全0的二维数组
            similarity_matrix = similarity_matrix * (1 - mask)

            pairwise_similarities = []
            for i in range(2):
                for j in range(i + 1, 3):
                    pairwise_similarities.append((i, j, similarity_matrix[i, j]))

            pairwise_similarities.sort(key=lambda x: x[2], reverse=True)

            common_features = []
            pair_weights = []

            for i, j, sim in pairwise_similarities:
                if sim > self.similarity_threshold:
                    fp1, fp2 = sample_fps[i], sample_fps[j]
                    fp1_norm, fp2_norm = sample_fps_norm[i], sample_fps_norm[j]

                    element_sim = fp1_norm * fp2_norm

                    high_sim_mask = (element_sim > self.similarity_threshold).float()
                    common_feature = ((fp1 + fp2) / 2) * high_sim_mask

                    common_features.append(common_feature)
                    pair_weights.append(sim)

            if not common_features:
                sample_common = torch.mean(sample_fps, dim=0)
            else:
                pair_weights = torch.tensor(pair_weights, device=brics.device)
                pair_weights = F.softmax(pair_weights, dim=0)
                common_features = torch.stack(common_features)
                sample_common = torch.sum(common_features * pair_weights.unsqueeze(1), dim=0)

            all_fp_common_features.append(sample_common)

        common_features = torch.stack(all_fp_common_features)  # [batch_size, hidden_size]

        all_fps_concat = torch.cat([encoded_fps[k] for k in fp_keys], dim=1)  # [batch_size, hidden_size*3]
        fp_weights = F.softmax(self.weight_generator(all_fps_concat), dim=1)  # [batch_size, 3]

        weighted_fp_sum = torch.zeros(batch_size, self.hidden_size, device=brics.device)
        for i, k in enumerate(fp_keys):
            weighted_fp_sum += encoded_fps[k] * fp_weights[:, i].unsqueeze(1)

        enhancement_factors = self.enhancement_layer(common_features)
        enhanced_common = common_features * enhancement_factors

        fused_features = self.fusion_layer(torch.cat([weighted_fp_sum, enhanced_common], dim=1))

        if contrastive:
            projections = self.projection_head(fused_features)
            projections = F.normalize(projections, dim=1)

            fp_projections = {}
            for k in fp_keys:
                proj = self.projection_head(encoded_fps[k])
                fp_projections[k] = F.normalize(proj, dim=1)

            return fused_features, projections, fp_projections

        return fused_features


# only enhance subgraph with contrastive loss
class Version3_MultiSub_Contrastive(nn.Module):
    def __init__(self, args):
        super(Version3_MultiSub_Contrastive, self).__init__()
        hid_dim = args['hid_dim']
        self.act = get_func(args['act'])
        self.depth = args['depth']
        self.similarity_sub = args['similarity_sub']

        # init
        self.node_embeds = nn.ModuleDict()
        self.node_embeds["a"] = nn.Linear(multisub_mol2vec_dims["a"], hid_dim).to(args["device"])
        self.node_embeds["s"] = nn.Linear(multisub_mol2vec_dims["s"], hid_dim).to(args["device"])
        self.node_embeds["f"] = nn.Linear(multisub_mol2vec_dims["f"], hid_dim).to(args["device"])
        self.node_embeds["p"] = nn.Linear(multisub_mol2vec_dims["p"], hid_dim).to(args["device"])
        self.node_embeds["m"] = nn.Linear(multisub_mol2vec_dims["m"], hid_dim).to(args["device"])

        self.mpnn_layers = nn.ModuleList()
        for i_layer in range(self.depth):
            self.mpnn_layers.append(PyramidMPNN_MultiSub(hid_dim))

        ## predict
        self.out = nn.Sequential(nn.Linear(hid_dim*3, hid_dim),
                                 self.act,
                                 nn.Linear(hid_dim, hid_dim),
                                 self.act,
                                 nn.Linear(hid_dim, args['num_task'])
                                 )
        self.initialize_weights()
        self.enhance_encoder = CommonFeatureExtractor(hid_dim,self.similarity_sub)
        self.graph_proj_head = nn.Sequential(
            nn.Linear(hid_dim*3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def init_feature(self, bg):
        for node_type,f_NN in self.node_embeds.items():
            bg.nodes[node_type].data['f_'] = self.act(f_NN(bg.nodes[node_type].data['f']))
        return bg.ndata["f_"]

    def split_batch(self, bg, node_f, ntype, device):
        hidden = node_f[ntype]
        node_size = bg.batch_num_nodes(ntype)# 获得每个分子中ntype类型的节点数
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])# 每个类型节点的开始节点下标
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)# 截取该i个目标分子的节点p,a特征向量
            cur_hidden = torch.mean(cur_hidden, dim=0)
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)
        '''
        [batch_size,d_dim]
        '''
        return hidden_lst

    def forward(self, bg, contrastive=True):

        """
        Args:
            bg: a batch of graphs
        """
        device = bg.device
        node_features = self.init_feature(bg)  # 三个视图节点-边的特征Hidden
        for mpnn_layer in self.mpnn_layers:
            node_features = mpnn_layer(bg,{node_type: node_features[node_type] for node_type in bg.ntypes})
        node_level = self.split_batch(bg,node_features,"a",device)
        sungraph_s = self.split_batch(bg,node_features,"s",device)
        sungraph_f = self.split_batch(bg, node_features, "f", device)
        sungraph_p = self.split_batch(bg, node_features, "p", device)
        mol_level = node_features["m"]
        # enhance feature module
        if contrastive:
            enhance_sub, fp_projections, fp_view_projections = self.enhance_encoder(
                sungraph_s,sungraph_f, sungraph_p, contrastive=True
            )  # # fp_x融合后的特征，fp_projections是融合后特征经过两层fc,fp_view_projections是字典，包含3个fp,对应原始fp经过两侧fc.
            embed_m_ = torch.cat([node_level, enhance_sub, mol_level], dim=1)
            graph_projections = self.graph_proj_head(embed_m_)
            graph_projections = F.normalize(graph_projections, dim=1)

            self.contrastive_data = {
                'graph_proj': graph_projections,
                'fp_proj': fp_projections,
                'fp_view_projs': fp_view_projections
            }
            self.contrastive_data2 = {
                'atom_f': node_level,
                'sub_f': enhance_sub,
                'mol_f': mol_level
            }
        else:
            enhance_sub = self.enhance_encoder(sungraph_s,sungraph_f, sungraph_p)  # 900



        embed_m = torch.cat([node_level, enhance_sub, mol_level], dim=1)
        out = self.out(embed_m)
        return out

    def compute_contrastive_loss2(self, temperature=0.2): # default 0.1
        """Calculating Comparative Learning Losses"""
        if not hasattr(self, 'contrastive_data2'):
            return torch.tensor(0.0, device=self.device)
        batch_size = self.contrastive_data2['atom_f'].size(0)
        total_contrastive_loss = 0
        for i_num,i_f in enumerate(["atom_f","sub_f","mol_f"]):
            for j_num,j_f in enumerate(["atom_f","sub_f","mol_f"]):
                if j_num > i_num:
                    sim_matrix = torch.mm(self.contrastive_data2[i_f], self.contrastive_data2[j_f].t()) / temperature
                    labels = torch.arange(batch_size).cuda()  # 生成指定范围值的一维张量
                    loss_graph_fp = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(),
                                                                                          labels)  ) /2# 交叉熵损失函数，用于在全连接层之后，做loss的计算
                    total_contrastive_loss += loss_graph_fp

        return total_contrastive_loss


