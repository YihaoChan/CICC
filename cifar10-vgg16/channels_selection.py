def assign_prune_channels(prune_channels: list,
                          entity0_prune,
                          entity1_prune,
                          entity2_prune,
                          entity3_prune):
    for i in range(13):  # 13¸öconv
        if i in range(0, 2):
            curr_prune = entity0_prune
        elif i in range(2, 4):
            curr_prune = entity1_prune
        elif i in range(4, 7):
            curr_prune = entity2_prune
        elif i in range(7, 14):
            curr_prune = entity3_prune
        else:
            raise RuntimeError("out of range!")

        prune_channels[i] = curr_prune
