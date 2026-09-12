import numpy as np


cost = np.array([
    [4, 6, 8, 6],
    [3, 5, 2, 5],
    [3, 9, 6, 5]
], dtype=float)


supply = [
    50,
    60,
    25
]


demand = [
    30,
    40,
    50,
    15
]


def vam(cost, supply, demand):

    supply = supply.copy()

    demand = demand.copy()


    m, n = cost.shape


    allocation = np.zeros((m, n))


    rows = set(range(m))

    cols = set(range(n))


    while rows and cols:


        penalties = []


        # Row penalties

        for i in rows:

            values = sorted(
                cost[i, j] for j in cols
            )


            if len(values) > 1:

                penalty = values[1] - values[0]

            else:

                penalty = values[0]


            penalties.append(
                (penalty, "row", i)
            )


        # Column penalties

        for j in cols:

            values = sorted(
                cost[i, j] for i in rows
            )


            if len(values) > 1:

                penalty = values[1] - values[0]

            else:

                penalty = values[0]


            penalties.append(
                (penalty, "col", j)
            )


        _, kind, index = max(penalties)


        if kind == "row":

            i = index

            j = min(
                cols,
                key=lambda x: cost[i, x]
            )


        else:

            j = index

            i = min(
                rows,
                key=lambda x: cost[x, j]
            )


        quantity = min(
            supply[i],
            demand[j]
        )


        allocation[i, j] = quantity


        supply[i] -= quantity

        demand[j] -= quantity


        if supply[i] == 0:

            rows.remove(i)


        if demand[j] == 0:

            cols.remove(j)


    return allocation



def modi_optimality_test(cost, allocation):


    m, n = cost.shape


    u = [None] * m

    v = [None] * n


    u[0] = 0


    basic = []


    for i in range(m):

        for j in range(n):

            if allocation[i, j] > 0:

                basic.append(
                    (i, j)
                )


    changed = True


    while changed:


        changed = False


        for i, j in basic:


            if u[i] is not None and v[j] is None:


                v[j] = cost[i, j] - u[i]

                changed = True


            elif v[j] is not None and u[i] is None:


                u[i] = cost[i, j] - v[j]

                changed = True


    delta = np.zeros((m, n))


    for i in range(m):

        for j in range(n):

            delta[i, j] = cost[i, j] - (
                u[i] + v[j]
            )


    return u, v, delta



# VAM Solution

allocation = vam(
    cost,
    supply,
    demand
)


print(
    "Initial Basic Feasible Solution using VAM:"
)


print(allocation)


initial_cost = np.sum(
    allocation * cost
)


print(
    "\nTransportation Cost =",
    initial_cost
)



# MODI Test

u, v, delta = modi_optimality_test(
    cost,
    allocation
)


print("\nMODI Potentials:")


print("u =", u)

print("v =", v)


print(
    "\nOpportunity Cost Matrix:"
)


print(delta)



if np.all(delta >= -1e-9):

    print(
        "\nAll opportunity costs are non-negative."
    )

    print(
        "Therefore, the VAM solution is optimal."
    )


else:

    print(
        "\nNegative opportunity cost found."
    )

    print(
        "Further MODI improvement is required."
    )



print(
    "\nOptimal Allocation:"
)


print(allocation)


print(
    "\nMinimum Transportation Cost =",
    initial_cost
)