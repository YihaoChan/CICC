from densenet import BasicBlock


def assign_prune_channels(prune_channels: list,
                          network,
                          layer1_prune,
                          layer2_prune,
                          layer3_prune):
    prune_ptr = 0

    for module in network.modules():
        if isinstance(module, BasicBlock):
            if module.bn1.num_features <= 156:
                prune_channels[prune_ptr] = layer1_prune
            elif module.bn1.num_features <= 300:
                prune_channels[prune_ptr] = layer2_prune
            elif module.bn1.num_features <= 444:
                prune_channels[prune_ptr] = layer3_prune

            prune_ptr += 1
