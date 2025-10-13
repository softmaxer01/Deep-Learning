def inception_model_arch():
    return {
        "inception_net": [
            # Conv layers before inception blocks
            (3, 64, 7, 2, 3),    # Conv1: 7x7/2
            "M",                  # MaxPool: 3x3/2
            (64, 64, 1, 1, 0),   # Conv2a: 1x1/1
            (64, 192, 3, 1, 1),  # Conv2b: 3x3/1
            "M",                  # MaxPool: 3x3/2
            
            # Inception block 3a
            [
                # branch 1: 1x1
                (192, 64, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(192, 96, 1, 1, 0),
                (96, 128, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(192, 16, 1, 1, 0),
                (16, 32, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (192, 32, 1, 1, 0)
            ],
            
            # Inception block 3b
            [
                # branch 1: 1x1
                (256, 128, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(256, 128, 1, 1, 0),
                (128, 192, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(256, 32, 1, 1, 0),
                (32, 96, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (256, 64, 1, 1, 0)
            ],
            
            "M",  # MaxPool: 3x3/2
            
            # Inception block 4a
            [
                # branch 1: 1x1
                (480, 192, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(480, 96, 1, 1, 0),
                (96, 208, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(480, 16, 1, 1, 0),
                (16, 48, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (480, 64, 1, 1, 0)
            ],
            
            # Inception block 4b
            [
                # branch 1: 1x1
                (512, 160, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(512, 112, 1, 1, 0),
                (112, 224, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(512, 24, 1, 1, 0),
                (24, 64, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (512, 64, 1, 1, 0)
            ],
            
            # Inception block 4c
            [
                # branch 1: 1x1
                (512, 128, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(512, 128, 1, 1, 0),
                (128, 256, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(512, 24, 1, 1, 0),
                (24, 64, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (512, 64, 1, 1, 0)
            ],
            
            # Inception block 4d
            [
                # branch 1: 1x1
                (512, 112, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(512, 144, 1, 1, 0),
                (144, 288, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(512, 32, 1, 1, 0),
                (32, 64, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (512, 64, 1, 1, 0)
            ],
            
            # Inception block 4e
            [
                # branch 1: 1x1
                (528, 256, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(528, 160, 1, 1, 0),
                (160, 320, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(528, 32, 1, 1, 0),
                (32, 128, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (528, 128, 1, 1, 0)
            ],
            
            "M",  # MaxPool: 3x3/2
            
            # Inception block 5a
            [
                # branch 1: 1x1
                (832, 256, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(832, 160, 1, 1, 0),
                (160, 320, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(832, 32, 1, 1, 0),
                (32, 128, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (832, 128, 1, 1, 0)
            ],
            
            # Inception block 5b
            [
                # branch 1: 1x1
                (832, 384, 1, 1, 0),
                # branch 2: 1x1 -> 3x3
                [(832, 192, 1, 1, 0),
                (192, 384, 3, 1, 1)],
                # branch 3: 1x1 -> 5x5
                [(832, 48, 1, 1, 0),
                (48, 128, 5, 1, 2)],
                # branch 4: pool -> 1x1
                "M", (832, 128, 1, 1, 0)
            ],
        ],

        "linear_layers": [
            "A",
            "D",
            (1024, 512),
            (512,10)
        ]
    }