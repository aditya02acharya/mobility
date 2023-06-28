import numpy as np
import pandas
import collections
import cvxopt
import warnings


#--------------------------------------------------------
# UTILITY MAXIMIZATION

'''
The main routine computes the competitive assignment.

We need to first split the graph into strongly connected components.
Each vertex belongs to a SCC, and from a vertex it's possible to reach
any other vertex in the SCC. In any feasible flow of vehicles, the
flow between SCCs must be 0. Within each SSC, solve the flow problem
independently. 

Finally, fill in rebalancing costs and profit on each edge.
On edges between SSCs, set alpha=beta=0, rebalancing_cost=infty, profit=0.
Also, for niceness, let's normalize the xi within each component --
add a constant it so the median amount paid for every rideshare trip is 0.
This makes no difference to rebalancing costs (= xi[dst] - xi[src]),
but it makes for saner plots.

Optional arguments control the cvxopt solver. 
See https://cvxopt.org/userguide/solvers.html#algorithm-parameters
'''

def maxprofit(routes, **kwargs):
    return assignment_in_graph(routes, include_profit=True, **kwargs)
def maxutility(routes, **kwargs):
    return assignment_in_graph(routes, include_profit=False, **kwargs)

def assignment_in_graph(routes, include_profit, show_progress=False, **kwargs):
    # sanity check on types
    for colname in ['src','dst','demand','demand','vehicle_cost','bus_price']:
        assert colname in routes, f"routes should have column {colname}"
    for colname in ['u_rideshare_bias','u_rideshare_pricesen','u_bus_bias','u_bus_pricesen']:
        assert colname in routes, f"routes should have column {colname}"
    routes = pandas.DataFrame(routes)

    
    # ec = [SCC_id of edge 1, SCC_id of edge 2, ...], one entry for each edge
    # vc = [[node label, node label, ...], ...], one list for each SSC
    (ec, vc) = strongly_connected_components(routes['src'], routes['dst'])

    all_alpha = np.full(len(routes), np.nan)
    all_beta = np.full(len(routes), np.nan)
    all_xi = {}

    # cvxopt has stateful option setting. Override it,
    # and remember the old options so we can restore them afterwards.
    kwargs['show_progress'] = show_progress
    opts = {}
    for k,v in kwargs.items():
        if k in cvxopt.solvers.options:
            opts[k] = cvxopt.solvers.options[k]
        cvxopt.solvers.options[k] = v

    # Find the optimal assignment, in each graph component in turn
    for scc_id, nodes_in_component in enumerate(vc):
        if show_progress:
            print("Solving network component", scc_id, "with nodes", nodes_in_component)
        if len(nodes_in_component)==1:
            (v,) = nodes_in_component
            all_xi[v] = 0
            continue
        (e,) = np.nonzero(ec==scc_id)
        routes_in_component = routes.iloc[e]
        (alpha, beta, xi, status) = assignment_in_component(routes = routes_in_component,
                                                            nodes = nodes_in_component,
                                                            include_profit = include_profit)
        if status != 'optimal':
            warnings.warn(f"Failed to converge on component {ssc_id}")
        all_alpha[e] = alpha
        all_beta[e] = beta
        for k,v in zip(nodes_in_component, xi):
            all_xi[k] = v

    # Restore the old cvxopt options
    for k in kwargs.keys():
        if k in opts:
            cvxopt.solvers.options[k] = opts[k]
        else:
            del cvxopt.solvers.options[k]
            
    no_component = (ec == -1)
    all_alpha[no_component] = 0
    all_beta[no_component] = 0
    res = pandas.DataFrame({'component': ec, 'alpha': all_alpha, 'beta': all_beta})
    if include_profit:
        res['profit'] = np.where(no_component, 0, 1/(routes['u_rideshare_pricesen'].values*(1-res['alpha'])))
    res['shadow_price_src'] = [all_xi[s] for s in routes['src']]
    res['shadow_price_dst'] = [all_xi[d] for d in routes['dst']]
    res['rebalancing_cost'] = np.where(no_component, np.inf, res['shadow_price_dst'] - res['shadow_price_src'])
    res['price'] = routes['vehicle_cost'].values + res['rebalancing_cost']
    if include_profit:
        res['price'] = res['price'] + res['profit']
    res.set_index(routes.index, inplace=True)
    
    return res



def assignment_in_component(routes, nodes, include_profit):
    '''Find the optimal assignment, in a strongly connected graph.
    The graph is specified by routes (a list of edges) and nodes (a list of vertices)
    where routes is a dataframe with the following columns:
        src, dst: origin and destination, given as node labels
        demand, vehicle_cost, bus_price: attributes of the route
        u_rideshare_bias, u_rideshare_pricesen, u_bus_bias, u_bus_pricesen: utility model coef
    and nodes is a list of node labels.
    Return: (alpha, beta, xi)
        alpha, beta: vectors with one entry per route
        xi: vector with one entry per node
    '''

    # Convenient notation
    src,dst = routes['src'], routes['dst']
    d,vc,bp = routes['demand'], routes['vehicle_cost'], routes['bus_price']
    bias,psen = routes['u_rideshare_bias'], routes['u_rideshare_pricesen']
    biasp,psenp = routes['u_bus_bias'], routes['u_bus_pricesen']
    V,E = len(nodes), len(src)
    assert E == len(src) == len(dst) == len(d) == len(vc) == len(bp)
    assert E == len(bias) == len(psen) == len(biasp) == len(psenp)
    node_idx = {v:i for i,v in enumerate(nodes)}

    # Find a feasible circulation given rideshare demand
    # Note that when we construct an flow-balance incidence matrix A
    # (one row per vertex, one column per route, A @ circ = 0)
    # that one row is always redundant, wlog the last row.
    alpha0 = np.full_like(d, 0.5, dtype=np.float32)
    A = np.zeros((V, E))
    for e,(_src,_dst) in enumerate(zip(src,dst)):
        A[node_idx[_src], e] = 1
        A[node_idx[_dst], e] = -1
    G = - np.identity(E)
    h = np.zeros_like(d)
    b = - (A @ (d*alpha0))
    res = cvxopt.solvers.lp(c=m(vc), G=m(G), h=m(h), A=m(A[:-1]), b=m(b[:-1]))
    # I won't bother with a check for optimality... all I really care about is feasibility.
    beta0 = np.array(res['x'])[:,0]
    if not np.all(np.isclose(A @ beta0, b)):
        raise Exception("Error finding a feasible circulation")
    beta0[np.isclose(beta0,0)] = 0  # tidy

    # Specify the objective function of the optimization.
    # See cvxopt documentation for details of what this function should accept+return.
    # The argument x is a vector with all the optimization variables, here alpha and beta.
    # The objective function is negative implied_utility
    def F(x=None, z=None):
        if x is None and z is None:
            x0 = np.concatenate([alpha0,beta0])
            return (0, m(x0))
        x = np.array(x).reshape(-1)
        alpha,beta = x[:E], x[E:]
        (entr,Dentr,D2entr) = entropy(alpha)
        (prof,Dprof,D2prof) = profit(alpha) if include_profit else np.zeros((3,E))
        u1 = bias - psen * vc
        u2 = biasp - psenp * bp
        implied_utility = d/psen * (alpha*u1 + (1-alpha)*u2 + entr + prof) - beta*vc
        Dalpha = d/psen * (u1 - u2 + Dentr + Dprof)
        Dbeta = - vc
        f = - np.sum(implied_utility)
        Df = - np.concatenate([Dalpha,Dbeta])
        if (z is None):
            return (m(f), m(Df[None,:]))
        [[z]] = np.array(z)
        D2alpha = d/psen * (D2entr + D2prof)
        D2f = np.concatenate([- D2alpha, np.zeros(E)])
        H = cvxopt.spmatrix(D2f, np.arange(2*E), np.arange(2*E))
        return (m(f), m(Df[None,:]), m(H))

    # Specify the region of the optimization, as G @ x <= h
    # This specifies: alpha >= 0, alpha <= 1, beta >= 0
    i = np.eye(E)
    z = np.zeros((E,E))
    G = np.block([[-i, z], [i, z], [z, -i]])
    h = np.concatenate([np.zeros(E), np.ones(E), np.zeros(E)])

    # Specify the constraints on the optimization, as A @ x = b
    # This specifies the circulation constraint,
    # that at every vertex the total flow in (alpha*d + beta)
    # is equal to the total flow out.
    # One equation is redundant, wlog the last one.
    A = np.zeros((V, 2*E))
    for e,(_d,_src,_dst) in enumerate(zip(d,src,dst)):
        A[node_idx[_src], e] = - _d
        A[node_idx[_dst], e] = _d
        A[node_idx[_src], e+E] = - 1
        A[node_idx[_dst], e+E] = 1
    A = A[:-1]
    b = np.zeros(V-1)

    # Run the optimization!
    res = cvxopt.solvers.cp(F, G=m(G), h=m(h[:,None]), A=m(A), b=m(b[:,None]))
    x = np.array(res['x'])[:,0]
    x[np.isclose(x,0)] = 0  # tidy
    alpha_opt, beta_opt = x[:E], x[E:]

    xi_opt = np.append(np.array(res['y']), 0)
    # Let's normalize xi so that the average xi at src is 0.
    # (We can add or subtract an arbitrary constant to the xi vector; it makes no difference
    # what the zero level is; so we might as well set it to be informative.)
    avg_xi_at_source = np.average([xi_opt[node_idx[_src]] for _src in src], weights=d*alpha_opt)
    xi_opt = xi_opt - avg_xi_at_source
    
    return (alpha_opt, beta_opt, xi_opt, res['status'])


def entropy(p):
    '''Computes the entropy H(p,1-p), and its first and second derivatives'''
    p = np.array(p)
    # To avoid warnings when p=0 or p-1, use np.where(safe, ..., ...)
    safe = (p>0) & (p<1)
    p = np.where(safe, p, np.where(p<0.5, 0.1, 0.9))
    f = np.where(safe, - p*np.log(p) - (1-p)*np.log(1-p), 0)
    Df = np.where(safe, np.log((1-p)/p), np.where(p<.5, np.inf, -np.inf))
    D2f = np.where(safe, - 1/p - 1/(1-p), -np.inf)
    return (f,Df,D2f)

def profit(p):
    '''Computes the profit term log(1-p), and its first and second derivatives'''
    p = np.array(p)
    # To avoid warnings when p=1, use escape clauses
    safe = p<1
    p = np.where(safe, p, 0.5)
    f = np.where(safe, np.log(1-p), -np.inf)
    Df = np.where(safe, -1/(1-p), -np.inf)
    D2f = np.where(safe, -np.power(1-p,-2), -np.inf)
    return (f,Df,D2f)


# cvxopt has its own matrix type; I'll do all my code in numpy and convert to cvxopt at the last minute
def m(x): return cvxopt.matrix(x, tc='d')



#---------------------------------------------------------------------
# GRAPH ALGORITHMS
#
# If there's a route i -> j, and no path back from j to i, then we don't want
# to assign any vehicles on that route, since if we did then it'd be impossible
# to achieve a circulation. For such edges i -> j, we should set
# rebalancing_cost = infty,  alpha = 0
# In the language of graph theory, we need to find the strongly connected components,
# and solve the assignment problem within each such component.
# The following function is for finding strongly connected components,
# given a list of edges.

class Vertex:
    def __init__(self):
        (self._to, self._from) = ([], [])

def strongly_connected_components(src,dst):
    '''Find strongly connected components.
    Arguments: 
        src,dst are lists, one item per edge, with labels of start and end vertices
    Returns: (n, e, v)
        n is the number of SSCs, and each is assigned an id {0,...,n-1}
        e is a list, one entry per edge, None if the edge is between SCCs and SCC_id otherwise
        v is a list, one entry per SSC, with a list of vertex labels
    '''
    vertices = collections.defaultdict(Vertex)
    for s,d in zip(src,dst):
        s,d = vertices[s], vertices[d]
        s._to.append(d)
        d._from.append(s)

    for k,v in vertices.items():
        v.label = k
        v.visited = False
        v.component_id = None
    ordered_vertices = []

    def visit(v):
        v.visited = True
        for u in v._to:
            if not u.visited:
                visit(u)
        ordered_vertices.insert(0, v)

    def assign(v, component_id):
        v.component_id = component_id
        for u in v._from:
            if u.component_id is None:
                assign(u, component_id)

    for v in vertices.values():
        if not v.visited:
            visit(v)
    num_components = 0
    for v in ordered_vertices:
        if v.component_id is None:
            assign(v, num_components)
            num_components = num_components + 1

    edge_component = np.full_like(src, -1)
    for i,(s,d) in enumerate(zip(src,dst)):
        s,d = vertices[s].component_id, vertices[d].component_id
        edge_component[i] = s if s==d else 0
    components = [[] for _ in range(num_components)]
    for v in vertices.values():
        components[v.component_id].append(v.label)
    
    return (edge_component, components)



#-----------------------------------------------------------
# TEST

if __name__ == '__main__':
    Route = collections.namedtuple('Route', ['src','dst','demand','vehicle_cost','bus_price'])

    routes = pandas.DataFrame([Route(src='A', dst='B', demand=10, vehicle_cost=1.5, bus_price=1),
                               Route(src='B', dst='A', demand=2, vehicle_cost=1.5, bus_price=6)])
    routes['u_rideshare_bias'] = 1.55393376
    routes['u_rideshare_pricesen'] = 0.3034984
    routes['u_bus_bias'] = 0
    routes['u_bus_pricesen'] = 0.09368271

    print("Finding allocation for routes", routes, sep='\n')

    res = assignment(routes)

    print(res)
