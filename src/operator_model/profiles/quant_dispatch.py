"""Vector issue-gap terms for a single quantization tile."""


def applies(p):
    # A full row fragment emits >=32 scale bytes. This is a conservative
    # modelling guard, not proof of cache-line alignment or no write conflicts.
    return p['iterations']==1 and p['cols_per_core']>=1024


def correction(p,weights):
    return weights[0]+weights[1]*p['active_aiv']/64
