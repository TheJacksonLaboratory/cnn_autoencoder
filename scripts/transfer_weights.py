import argparse
import torch


cae_replace_keys = [
    ('quantiles', '', 0),
    ('_matrices.', '_matrix%i', 1),
    ('_biases.', '_bias%i', 1),
    ('_factors.', '_factor%i', 1),
    ('.weight', '%i.model.%i.weight', 2),
    ('.bias', '%i.model.%i.bias', 2),
    ('.gamma', '%i.model.%i.gamma', 2),
    ('.beta', '%i.model.%i.beta', 2),
]

cai_replace_keys = [
    ('quantiles', '', 0),
    ('_matrix', '', 0),
    ('_bias', '', 0),
    ('_factor', '', 0),
    ('.weight', '%i.weight', 1),
    ('.bias', '%i.bias', 1),
    ('.gamma', '%i.gamma', 1),
    ('.beta', '%i.beta', 1),
]

cae_replace_module = [
    ('encoder', 'g_a.', 'analysis_track.'),
    ('decoder', 'g_s.', 'synthesis_track.'),
    ('fact_entropy', 'entropy_bottleneck.', ''),
]

cai_replace_module = [
    ('encoder', 'analysis_track.', 'g_a.'),
    ('decoder', 'synthesis_track.', 'g_s.'),
    ('fact_entropy', 'fact_ent', 'entropy_bottleneck.'),
]


def ext_idx_cae(k, k_s, n_idx):
    idx, rem = k.split(k_s)
    if len(idx) == 0:
        idx = rem
        rem = ''

    if n_idx == 1:
        idx = int(idx)
    else:
        idx = (int(idx), int(idx))
    return idx, rem


def ext_idx_cai(k, k_s, n_idx):
    k = k.split('.model.')
    k1 = int(k[0].split('.')[-1])
    k2 = int(k[1].split('.')[0])
    rem = k[1].split(k_s)[1]
    return k1 * 2 + k2, rem


def transfer_weights(chk_src, cai2cae=True):
    chk = {}
    if cai2cae:
        replace_module = cae_replace_module
        replace_keys = cae_replace_keys
        ext_idx = ext_idx_cae

    else:
        replace_module = cai_replace_module
        replace_keys = cai_replace_keys
        ext_idx = ext_idx_cai

    # Iterate over the checkpoint modules
    for m_name, m_src, m_dst in replace_module:
        print('Replacing keys in module', m_name)
        chk[m_name] = {}

        chk_new = dict([(k.split(m_src)[1], w)
                        for k, w in chk_src.items() if m_src in k])

        # Replace the keys in the source checkpoint
        chk_keys = list(chk_new.keys())
        for k in chk_keys:
            new_key = None

            for k_s, k_d, n_idx in replace_keys:
                if k_s in k:
                    if n_idx == 0:
                        new_key = m_dst + k
                        trans_w = chk_new.pop(k)

                    elif n_idx > 0:
                        print(k, k_s, k.split(k_s), len(k.split(k_s)))
                        idx, rem = ext_idx(k, k_s, n_idx)
                        new_key = m_dst + k_d % idx + rem
                        trans_w = chk_new.pop(k)

                    break

            if new_key is not None:
                chk_new[new_key] = trans_w
            elif cai2cae:
                print('Removing', k)
                chk_new.pop(k)

        chk[m_name].update(chk_new)

    return chk


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transfer weights from Compres AI to CAE "
                                     "code")
    parser.add_argument("-d", "--destination", help="CAE model checkpoint")
    parser.add_argument("-s", "--source", help="Compress AI model checkpoint")
    parser.add_argument("-o", "--output", help="Output model filename")
    parser.add_argument("-cai", "--cae2cai", dest="cai2cae",
                        help="Convert CAE to Compress AI",
                        action="store_false",
                        default=True)

    args = parser.parse_args()

    chk_dst = torch.load(args.destination, map_location='cpu')
    chk_src = torch.load(args.source, map_location='cpu')

    if args.cai2cae:
        chk_transfer = transfer_weights(chk_src, cai2cae=True)

        chk_dst["fact_ent"] = chk_transfer["fact_entropy"]
        chk_dst["encoder"] = chk_transfer["encoder"]
        chk_dst["decoder"] = chk_transfer["decoder"]
    else:
        chk_src_model = {}
        chk_src_model.update(chk_src["decoder"])
        chk_src_model.update(chk_src["encoder"])
        for k in chk_src["fact_ent"].keys():
            chk_src_model["fact_ent." + k] = chk_src["fact_ent"][k]

        chk_transfer = transfer_weights(chk_src_model, cai2cae=False)
        chk_dst = {}
        chk_dst.update(chk_transfer["fact_entropy"])
        chk_dst.update(chk_transfer["encoder"])
        chk_dst.update(chk_transfer["decoder"])

    torch.save(chk_dst, args.output)