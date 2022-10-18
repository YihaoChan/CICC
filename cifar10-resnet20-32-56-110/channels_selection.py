def assign_prune_channels(prune_channels: list,
                          layer1_prune,
                          layer2_prune,
                          layer3_prune,
                          num_block):
    for i in range(3):  # 3 layers
        for j in range(num_block):
            curr_prune = 0

            if i == 0:
                curr_prune = layer1_prune
            elif i == 1:
                curr_prune = layer2_prune
            elif i == 2:
                curr_prune = layer3_prune

            prune_channels[2 * num_block * i + 2 * j] = curr_prune


def layer_wise_assign_prune_channels(prune_channels: list,
                                     layer1_prune,
                                     layer2_prune,
                                     layer3_prune,
                                     num_block):
    for i in range(3):  # 3 layers
        for j in range(2 * num_block):
            curr_prune = 0

            if i == 0:
                curr_prune = layer1_prune
            elif i == 1:
                curr_prune = layer2_prune
            elif i == 2:
                curr_prune = layer3_prune

            prune_channels[2 * num_block * i + j] = curr_prune
