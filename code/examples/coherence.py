diz_coherence_tot = defaultdict(dict)
list_coherence_types = ['c_v', 'u_mass', 'c_uci', 'c_npmi']
for coh_type in list_coherence_types:
    coh = tp.coherence.Coherence(mdl, coherence=coh_type, top_n = 10)
    s = 0
    for k_i in range(16):
        diz_coherence_tot[coh_type][k_i] = coh.get_score(topic_id = k_i)
        s += diz_coherence_tot[coh_type][k_i]
    diz_coherence_tot[coh_type]['avg'] = s/16