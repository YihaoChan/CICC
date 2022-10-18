def assign_prune_channels_resnet18(prune_channels: list,
                                   layer1_prune,
                                   layer2_prune,
                                   layer3_prune,
                                   layer4_prune):
    assign_prune_channels(prune_channels=prune_channels,
                          layer1_prune=layer1_prune,
                          layer2_prune=layer2_prune,
                          layer3_prune=layer3_prune,
                          layer4_prune=layer4_prune,
                          layer1_num_block=2,
                          layer2_num_block=2,
                          layer3_num_block=2,
                          layer4_num_block=2,
                          conv_each_block=2)


def assign_prune_channels_resnet34(prune_channels: list,
                                   layer1_prune,
                                   layer2_prune,
                                   layer3_prune,
                                   layer4_prune):
    assign_prune_channels(prune_channels=prune_channels,
                          layer1_prune=layer1_prune,
                          layer2_prune=layer2_prune,
                          layer3_prune=layer3_prune,
                          layer4_prune=layer4_prune,
                          layer1_num_block=3,
                          layer2_num_block=4,
                          layer3_num_block=6,
                          layer4_num_block=3,
                          conv_each_block=2)


def assign_prune_channels_resnet50(prune_channels: list,
                                   layer1_prune,
                                   layer2_prune,
                                   layer3_prune,
                                   layer4_prune):
    assign_prune_channels(prune_channels=prune_channels,
                          layer1_prune=layer1_prune,
                          layer2_prune=layer2_prune,
                          layer3_prune=layer3_prune,
                          layer4_prune=layer4_prune,
                          layer1_num_block=3,
                          layer2_num_block=4,
                          layer3_num_block=6,
                          layer4_num_block=3,
                          conv_each_block=3)


def assign_prune_channels_resnet101(prune_channels: list,
                                    layer1_prune,
                                    layer2_prune,
                                    layer3_prune,
                                    layer4_prune):
    assign_prune_channels(prune_channels=prune_channels,
                          layer1_prune=layer1_prune,
                          layer2_prune=layer2_prune,
                          layer3_prune=layer3_prune,
                          layer4_prune=layer4_prune,
                          layer1_num_block=3,
                          layer2_num_block=4,
                          layer3_num_block=23,
                          layer4_num_block=3,
                          conv_each_block=3)


def assign_prune_channels(prune_channels: list,
                          layer1_prune,
                          layer2_prune,
                          layer3_prune,
                          layer4_prune,
                          layer1_num_block,
                          layer2_num_block,
                          layer3_num_block,
                          layer4_num_block,
                          conv_each_block):
    ptr = 0

    for i in range(layer1_num_block * conv_each_block):
        prune_channels[ptr] = layer1_prune
        ptr += 1

    for i in range(layer2_num_block * conv_each_block):
        prune_channels[ptr] = layer2_prune
        ptr += 1

    for i in range(layer3_num_block * conv_each_block):
        prune_channels[ptr] = layer3_prune
        ptr += 1

    for i in range(layer4_num_block * conv_each_block):
        prune_channels[ptr] = layer4_prune
        ptr += 1
