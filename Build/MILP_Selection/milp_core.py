# ==========================
# milp_core.py  
# ==========================
import numpy as np
import pyomo.environ as pyo

DEBUG_LAST = {}       # <-- Global debug container


def compute_latencies(batch_df, edge_nodes, cloud_nodes,
                      r_bits=5000 * 8, dbh=0.03):

    N = len(batch_df)
    J = len(edge_nodes)
    K = len(cloud_nodes)

    T_edge = np.full((N, J), np.inf)
    T_cloud = np.full((N, K), np.inf)

    debug = {
        "edge_calc": [],
        "cloud_calc": []
    }

    for i in range(N):
        b = float(batch_df.iloc[i]["data_mb"] * 8e6)
        c = float(batch_df.iloc[i]["task_flops"])
        bw = float(batch_df.iloc[i].get("bandwidth_mbps", 20) * 1e6)

        ue_deadline = batch_df.iloc[i]["deadline_ms"] / 1000.0

        # EDGE LATENCY
        for j in range(J):
            R = bw * edge_nodes[j].get("R_scale", 1.0)
            f = edge_nodes[j]["f_j"]
            util = max(edge_nodes[j].get("util", 0.05), 0.05)

            eff_cpu = f * (1 - util)
            if eff_cpu <= 0:
                continue

            uplink = b / R
            compute = c / eff_cpu
            resp = r_bits / R
            total = uplink + compute + resp

            T_edge[i, j] = total

            debug["edge_calc"].append({
                "edge": edge_nodes[j]["id"],
                "upload": uplink,
                "compute": compute,
                "response": resp,
                "total_edge_latency": total,
                "util": util,
                "cpu_hz": f
            })

        # CLOUD LATENCY
        for k in range(K):
            R = bw * cloud_nodes[k].get("R_scale", 1.0)
            f = cloud_nodes[k]["f_k"]
            util = max(cloud_nodes[k].get("util", 0.05), 0.05)

            eff_cpu = f * (1 - util)
            if eff_cpu <= 0:
                continue

            uplink = b / (R * 0.5)
            compute = c / eff_cpu
            resp = r_bits / (R * 0.5)
            backhaul = 0.06

            total = uplink + compute + resp + backhaul
            T_cloud[i, k] = total

            debug["cloud_calc"].append({
                "cloud": cloud_nodes[k]["id"],
                "upload": uplink,
                "compute": compute,
                "response": resp,
                "backhaul": backhaul,
                "total_cloud_latency": total,
                "util": util,
                "cpu_hz": f
            })

    DEBUG_LAST["edge_matrix"] = T_edge.tolist()
    DEBUG_LAST["cloud_matrix"] = T_cloud.tolist()
    DEBUG_LAST["debug_latency_steps"] = debug

    return T_edge, T_cloud


def milp_scheduler(batch_df, edge_nodes, cloud_nodes,
                   local_penalty=9999, util_limit=0.95):

    global DEBUG_LAST
    DEBUG_LAST["ue_input"] = batch_df.to_dict()

    N = len(batch_df)
    J = len(edge_nodes)
    K = len(cloud_nodes)

    T_edge, T_cloud = compute_latencies(batch_df, edge_nodes, cloud_nodes)

    m = pyo.ConcreteModel()
    I = range(N)
    E = range(J)
    C = range(K)

    m.alpha = pyo.Var(I, domain=pyo.Binary)
    m.xe = pyo.Var(I, E, domain=pyo.Binary) if J>0 else None
    m.xc = pyo.Var(I, C, domain=pyo.Binary) if K>0 else None

    def obj_rule(m):
        total = 0
        for i in I:
            edge_term = sum(m.xe[i,j] * T_edge[i,j] for j in E) if J>0 else 0
            cloud_term = sum(m.xc[i,k] * T_cloud[i,k] for k in C) if K>0 else 0
            local_cost = (1 - m.alpha[i]) * local_penalty
            total += edge_term + cloud_term + local_cost
        return total

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def assign_rule(m,i):
        offload_sum = 0
        if J>0:
            offload_sum += sum(m.xe[i,j] for j in E)
        if K>0:
            offload_sum += sum(m.xc[i,k] for k in C)
        return offload_sum == m.alpha[i]

    m.assign = pyo.Constraint(I, rule=assign_rule)

    # RELAXED DEADLINE + explanation logging
    def deadline_rule(m,i):
        edge_part = sum(m.xe[i,j] * T_edge[i,j] for j in E) if J>0 else 0
        cloud_part = sum(m.xc[i,k] * T_cloud[i,k] for k in C) if K>0 else 0
        D = batch_df.iloc[i]["deadline_ms"] / 1000.0
        DEBUG_LAST["deadline_seconds"] = D
        return edge_part + cloud_part <= D + 3

    m.deadline = pyo.Constraint(I, rule=deadline_rule)

    try:
        solver = pyo.SolverFactory("highs")
        solver.solve(m)
    except:
        solver = pyo.SolverFactory("cbc")
        solver.solve(m)

    decisions = []

    for i in I:
        if pyo.value(m.alpha[i]) < 0.5:
            decisions.append("Local")
            DEBUG_LAST["reason"] = "Local chosen because MILP could not find feasible edge/cloud below cost."
            continue

        assigned = None

        if J>0:
            for j in E:
                if pyo.value(m.xe[i,j]) > 0.5:
                    assigned = f"Edge:{edge_nodes[j]['id']}"
                    DEBUG_LAST["reason"] = f"Edge selected because it has min latency and satisfied relaxed deadline."
                    break

        if assigned is None and K>0:
            for k in C:
                if pyo.value(m.xc[i,k]) > 0.5:
                    assigned = f"Cloud:{cloud_nodes[k]['id']}"
                    DEBUG_LAST["reason"] = f"Cloud selected because edge latency higher than cloud while still feasible."
                    break

        if assigned is None:
            assigned = "Local"
            DEBUG_LAST["reason"] = "Fallback Local decision"

        decisions.append(assigned)

    DEBUG_LAST["final_decision"] = decisions[0]
    return decisions
