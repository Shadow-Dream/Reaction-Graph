REACTION_GRAPH_USPTO_YIELD_CONFIG = {
    "device": "cuda",
    "progress_bar": True,
    "loss_display_buffer": 100,
    "log_delta": 100,
    "network_config": {
        "graph_type": "reaction_graph",
        "dim_edge_length": 0,
        "dim_hidden_features": 64,
        "message_passing_step": 3,
        "pooling_step": 3,
        "num_layers_pooling": 1,
        "dim_hidden_regression": 512,
        "aggregation_level": "molecule",
        "aggregation_method": "sum",
        "dropout":0.1
    },
    "training_config": {
        "epoches":50,
        "save_delta":5,
    },
    "optimizer_config": {
        "lr":1e-3, 
        "weight_decay":1e-5
    },
    "scheduler_config": {
        "milestones":[400, 450], 
        "gamma":0.1,
        "verbose":True
    },
    "criterion_config": {
        "kl":0.1,
        "max_gradient": 1e2,
        "accumulation_steps": 4
    },
    "validate_config":{
        "validate_type":"val",
        "detail_level": "results",
        "num_inference_pass": 30
    },
    "experiments":{
        "gram":{
            "dataset_dir":"datasets/uspto_yield/gram",
            "dataset_name":"ReactionGraphUSPTOYieldGram",
            "model_dir": "checkpoints/reaction_graph/uspto_yield/gram",
            "gpu": "7",
            "seed": "2",
            "mean":0.7311065677322701,
            "std":0.20931794018637168,
            "var":0.04381400008386547,
            "network_config":{
                "dim_node_attribute": 132,
                "dim_edge_attribute": 13,
                "dim_hidden": 512,
            },
            "checkpoint": "checkpoints/reaction_graph/uspto_yield/gram/uspto_yield_gram.ckpt"
        },
        "subgram":{
            "dataset_dir":"datasets/uspto_yield/subgram",
            "dataset_name":"ReactionGraphUSPTOYieldSubgram",
            "model_dir": "checkpoints/reaction_graph/uspto_yield/subgram",
            "gpu": "7",
            "seed": "2",
            "mean":0.568238250428795,
            "std":0.26657960625978777,
            "var":0.07106468647362349,
            "network_config":{
                "dim_node_attribute": 121,
                "dim_edge_attribute": 14,
                "dim_hidden": 1024,
            },
            "checkpoint": "checkpoints/reaction_graph/uspto_yield/subgram/uspto_yield_subgram.ckpt"
        },
    }
}