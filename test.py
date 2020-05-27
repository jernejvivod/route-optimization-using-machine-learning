perm = [(9, 0), (8, 2), (6, 4), (2, 5)]

def merge_perm(perm_list):
    perm_list_copy = perm_list.copy()
    found = True
    while found:
        for idx in range(len(perm_list_copy)):
            subperm = perm_list_copy[idx]
            del(perm_list_copy[idx])
            perm_list_copy, flg = merge_aux(subperm, perm_list_copy)
            if flg:
                break
            else:
                perm_list_copy.insert(idx, subperm)
                if idx == len(perm_list_copy)-1:
                    found = False
                else:
                    pass
    
    return perm_list_copy


def merge_aux(subperm, perm_list):
    perm_list_aux = perm_list.copy()
    for idx in range(len(perm_list_aux)):
        if subperm[-1] == perm_list_aux[idx][0]:
            res = subperm + perm_list_aux[idx][1:]
            del(perm_list_aux[idx])
            perm_list_aux.append(res)
            return perm_list_aux, True
        if subperm[0] == perm_list_aux[idx][-1]:
            res = perm_list_aux[idx][:-1] + subperm
            del(perm_list_aux[idx])
            perm_list_aux.append(res)
            return perm_list_aux, True
    return perm_list, False

