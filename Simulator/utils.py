import os
import gurobipy as gp

def set_environment(threads=24, output_flag=0, node_method=2, concurrent_mip=1):
    options = {
        "WLSACCESSID": os.getenv("WLSACCESSID"),
        "WLSSECRET": os.getenv("WLSSECRET"),
        "LICENSEID": int(os.getenv("LICENSEID")),
    }

    env = gp.Env(params=options)
    env.setParam("OutputFlag", output_flag)
    env.setParam("Threads", threads)
    env.setParam('NodeMethod', node_method)
    env.setParam('ConcurrentMIP', concurrent_mip)

    return env